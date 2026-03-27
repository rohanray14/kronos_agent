"""
Position Sizing Environment for V3.

The RL agent does NOT decide direction — Claude does that.
The RL agent only decides HOW MUCH of Claude's trade to execute.

Flow per step:
  1. Claude decides BUY/SELL/HOLD (replayed from historical logs or rule-based)
  2. RL agent observes: market features + Claude's action + portfolio state
  3. RL agent outputs: sizing fraction (0%, 25%, 50%, 75%, 100%)
  4. Fractional trade is executed

This replaces the current all-in/all-out execute() with graduated positions.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from v3.config import (
    SIZING_OBS_DIM, N_SIZING_ACTIONS, SIZING_LEVELS,
    INITIAL_CASH, FORECAST_STEPS, LOOKBACK_DAYS, SIDEWAYS,
)
from v3.reward_sizing import sizing_reward


# Claude action encoding
CLAUDE_SELL = -1.0
CLAUDE_HOLD = 0.0
CLAUDE_BUY = 1.0

ACTION_MAP = {"SELL": CLAUDE_SELL, "HOLD": CLAUDE_HOLD, "BUY": CLAUDE_BUY}


class PositionSizingEnv(gym.Env):
    """
    Gymnasium environment for learning position sizing.

    Observation space (24 dims):
        [0:18]  Market features from MacroFeatureProvider
        [18]    Claude's action (-1=SELL, 0=HOLD, 1=BUY)
        [19]    Position fraction (0-1, current allocation)
        [20]    Unrealized PnL (normalized by initial cash)
        [21]    Days since last trade (normalized by 50)
        [22]    Claude's rolling win rate (last 10 decisions)
        [23]    vol_20d (direct access for sizer)

    Action space: Discrete(5) — maps to [0%, 25%, 50%, 75%, 100%]
        This is the fraction of Claude's requested trade to execute.
        If Claude says HOLD, action is ignored (no trade).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        feature_provider,
        claude_actions: np.ndarray,
        regime_labels: np.ndarray | None = None,
        test_start: str = "2020-01-01",
        test_end: str = "2024-12-31",
        initial_cash: float = INITIAL_CASH,
        forecast_steps: int = FORECAST_STEPS,
        lookback_days: int = LOOKBACK_DAYS,
        transaction_cost: float = 0.001,
        random_start: bool = False,
        min_episode_steps: int = 20,
    ):
        super().__init__()

        self.df = df
        self.feature_provider = feature_provider
        self.initial_cash = initial_cash
        self.forecast_steps = forecast_steps
        self.lookback_days = lookback_days
        self.transaction_cost = transaction_cost
        self.random_start = random_start
        self.min_episode_steps = min_episode_steps

        # Regime labels (optional, for regime-aware reward)
        self._regime_labels = regime_labels

        # Pre-compute step indices
        dates = df.index
        start_ts = pd.Timestamp(test_start)
        end_ts = pd.Timestamp(test_end)
        test_indices = [i for i, d in enumerate(dates)
                        if d >= start_ts and d <= end_ts]
        test_indices = [i for i in test_indices if i >= lookback_days]
        self.step_indices = test_indices[::forecast_steps]

        # Claude's decisions (pre-computed, one per step)
        # Must be same length as step_indices
        assert len(claude_actions) == len(self.step_indices), (
            f"claude_actions length {len(claude_actions)} != "
            f"step_indices length {len(self.step_indices)}"
        )
        self._claude_actions = claude_actions  # array of -1, 0, 1

        # Spaces
        self.action_space = spaces.Discrete(N_SIZING_ACTIONS)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(SIZING_OBS_DIM,), dtype=np.float32
        )

        # Portfolio state
        self._step_idx = 0
        self._cash = initial_cash
        self._shares = 0.0
        self._entry_price = 0.0
        self._peak_pv = initial_cash
        self._days_since_trade = 0
        self._episode_start = 0

        # Claude performance tracking
        self._claude_outcomes: list[bool] = []  # True = profitable decision

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.random_start and len(self.step_indices) > self.min_episode_steps:
            max_start = len(self.step_indices) - self.min_episode_steps
            self._episode_start = self.np_random.integers(0, max_start)
        else:
            self._episode_start = 0

        self._step_idx = self._episode_start
        self._cash = self.initial_cash
        self._shares = 0.0
        self._entry_price = 0.0
        self._peak_pv = self.initial_cash
        self._days_since_trade = 0
        self._claude_outcomes = []

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action):
        action = int(action)
        alloc_frac = SIZING_LEVELS[action]

        idx = self.step_indices[self._step_idx]
        price = float(self.df["close"].iloc[idx])
        pv_before = self._portfolio_value(price)
        position_frac_before = self._position_fraction(price)

        claude_action = self._claude_actions[self._step_idx]

        # ── Execute fractional trade ──
        traded = False
        if claude_action == CLAUDE_BUY and self._cash > 0 and alloc_frac > 0:
            # Buy with alloc_frac of available cash
            trade_cash = self._cash * alloc_frac
            cost = trade_cash * self.transaction_cost
            new_shares = (trade_cash - cost) / price
            self._shares += new_shares
            self._cash -= trade_cash
            self._entry_price = price
            traded = True

        elif claude_action == CLAUDE_SELL and self._shares > 0 and alloc_frac > 0:
            # Sell alloc_frac of current shares
            sell_shares = self._shares * alloc_frac
            proceeds = sell_shares * price
            cost = proceeds * self.transaction_cost
            self._cash += proceeds - cost
            self._shares -= sell_shares
            traded = True

        # HOLD or alloc_frac == 0: do nothing

        if traded:
            self._days_since_trade = 0
        else:
            self._days_since_trade += 1

        # ── Advance to next step ──
        self._step_idx += 1
        terminated = self._step_idx >= len(self.step_indices)
        truncated = False

        if not terminated:
            next_idx = self.step_indices[self._step_idx]
            next_price = float(self.df["close"].iloc[next_idx])
        else:
            next_price = price

        pv_after = self._portfolio_value(next_price)
        self._peak_pv = max(self._peak_pv, pv_after)
        position_frac_after = self._position_fraction(next_price)

        # Track Claude's accuracy
        if claude_action != CLAUDE_HOLD:
            was_profitable = (
                (claude_action == CLAUDE_BUY and next_price > price) or
                (claude_action == CLAUDE_SELL and next_price < price)
            )
            self._claude_outcomes.append(was_profitable)

        # ── Compute reward ──
        regime = self._get_regime()
        vol_20d = float(self.feature_provider._vol_20d[idx])

        reward = sizing_reward(
            pv_before, pv_after, self._peak_pv,
            position_frac_before, vol_20d, regime, action,
        )

        obs = self._get_obs() if not terminated else np.zeros(SIZING_OBS_DIM, dtype=np.float32)
        info = self._get_info()
        info["portfolio_value"] = pv_after
        info["sizing_action"] = alloc_frac
        info["claude_action"] = ["SELL", "HOLD", "BUY"][int(claude_action) + 1]
        info["position_frac"] = position_frac_after

        return obs, reward, terminated, truncated, info

    def _portfolio_value(self, price: float) -> float:
        return self._cash + self._shares * price

    def _position_fraction(self, price: float) -> float:
        pv = self._portfolio_value(price)
        if pv <= 0:
            return 0.0
        return (self._shares * price) / pv

    def _get_regime(self) -> int:
        if self._regime_labels is not None:
            pos = min(self._step_idx, len(self._regime_labels) - 1)
            return int(self._regime_labels[pos])
        return SIDEWAYS

    def _rolling_win_rate(self) -> float:
        if len(self._claude_outcomes) < 3:
            return 0.5  # neutral prior
        recent = self._claude_outcomes[-10:]
        return sum(recent) / len(recent)

    def _get_obs(self) -> np.ndarray:
        if self._step_idx >= len(self.step_indices):
            return np.zeros(SIZING_OBS_DIM, dtype=np.float32)

        idx = self.step_indices[self._step_idx]
        price = float(self.df["close"].iloc[idx])

        # Market features (18 dims)
        market_feats = self.feature_provider.get_features(idx)

        # Agent context features (6 dims)
        claude_action = self._claude_actions[self._step_idx]
        position_frac = self._position_fraction(price)

        # Unrealized PnL
        if self._shares > 0 and self._entry_price > 0:
            unrealized_pnl = (price - self._entry_price) * self._shares / self.initial_cash
        else:
            unrealized_pnl = 0.0

        days_norm = self._days_since_trade / 50.0
        win_rate = self._rolling_win_rate()
        vol_20d = float(self.feature_provider._vol_20d[idx])

        obs = np.zeros(SIZING_OBS_DIM, dtype=np.float32)
        obs[0:18] = market_feats
        obs[18] = claude_action
        obs[19] = position_frac
        obs[20] = unrealized_pnl
        obs[21] = days_norm
        obs[22] = win_rate
        obs[23] = vol_20d

        return obs

    def _get_info(self) -> dict:
        if self._step_idx >= len(self.step_indices):
            return {}
        idx = self.step_indices[self._step_idx]
        price = float(self.df["close"].iloc[idx])
        return {
            "date": str(self.df.index[idx].date()),
            "price": price,
            "cash": self._cash,
            "shares": self._shares,
            "step": self._step_idx,
            "total_steps": len(self.step_indices),
        }


class MultiTickerSizingEnv(gym.Env):
    """
    Multi-ticker wrapper for position sizing training.
    Samples a random ticker each episode.
    """

    metadata = {"render_modes": []}

    def __init__(self, ticker_envs: dict[str, PositionSizingEnv]):
        super().__init__()
        self.ticker_envs = ticker_envs
        self.ticker_names = list(ticker_envs.keys())
        self._current_ticker = None
        self._current_env = None

        sample_env = next(iter(ticker_envs.values()))
        self.action_space = sample_env.action_space
        self.observation_space = sample_env.observation_space

    @property
    def step_indices(self):
        return [idx for env in self.ticker_envs.values()
                for idx in env.step_indices]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        ticker_idx = self.np_random.integers(0, len(self.ticker_names))
        self._current_ticker = self.ticker_names[ticker_idx]
        self._current_env = self.ticker_envs[self._current_ticker]
        obs, info = self._current_env.reset(seed=seed)
        info["ticker"] = self._current_ticker
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._current_env.step(action)
        info["ticker"] = self._current_ticker
        return obs, reward, terminated, truncated, info
