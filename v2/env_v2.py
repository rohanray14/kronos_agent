"""
Enhanced Trading Environments for the Two-Tier RL System.

EnhancedTradingEnv: 22-dim observation with macro features + pluggable reward.
RegimeFilteredEnv: wraps EnhancedTradingEnv, filters steps to a single regime.
MultiTickerRegimeEnv: samples random ticker per episode (for multi-ticker training).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from v2.config import (
    OBS_DIM, INITIAL_CASH, FORECAST_STEPS, LOOKBACK_DAYS, SIDEWAYS,
)
from v2.reward import fallback_reward


class EnhancedTradingEnv(gym.Env):
    """
    Gymnasium environment with 22-dim observation vector.

    Observation layout:
         0: position (0/1)
         1: ret_5d
         2: ret_20d
         3: ret_50d
         4: vol_20d
         5: price_vs_50ma
         6: price_vs_200ma
         7: kronos_expected_ret
         8: kronos_trend
         9: kronos_high_vol
        10: vix_level
        11: vix_change_5d
        12: yield_curve
        13: gld_ret_20d
        14: tlt_ret_20d
        15: uso_ret_20d
        16: iwm_spy_spread_20d
        17: xlu_spy_spread_20d
        18: portfolio_return
        19: days_in_position
        20: regime_id
        21: regime_confidence
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        df,
        feature_provider,
        regime_labels=None,
        regime_confidences=None,
        reward_fn=None,
        test_start="2020-01-01",
        test_end="2024-12-31",
        initial_cash=INITIAL_CASH,
        forecast_steps=FORECAST_STEPS,
        lookback_days=LOOKBACK_DAYS,
        transaction_cost=0.0,
        random_start=False,
        min_episode_steps=20,
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
        self.reward_fn = reward_fn or fallback_reward

        # Regime info (pre-computed per step)
        self._regime_labels = regime_labels  # array of int
        self._regime_confidences = regime_confidences  # array of float

        # Pre-compute step indices
        dates = df.index
        test_indices = [i for i, d in enumerate(dates) if d >= pd.Timestamp(test_start)]
        test_indices = [i for i in test_indices if i >= lookback_days]
        self.step_indices = test_indices[::forecast_steps]

        # Spaces
        self.action_space = spaces.Discrete(3)  # HOLD, BUY, SELL
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )

        # State
        self._step_idx = 0
        self._cash = initial_cash
        self._shares = 0.0
        self._peak_pv = initial_cash
        self._days_in_position = 0
        self._episode_start = 0

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
        self._peak_pv = self.initial_cash
        self._days_in_position = 0

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action):
        action = int(action)
        idx = self.step_indices[self._step_idx]
        price = float(self.df["close"].iloc[idx])
        pv_before = self._portfolio_value(price)

        # Execute trade
        traded = False
        if action == 1 and self._cash > 0:  # BUY
            cost = self._cash * self.transaction_cost
            self._shares = (self._cash - cost) / price
            self._cash = 0.0
            traded = True
        elif action == 2 and self._shares > 0:  # SELL
            proceeds = self._shares * price
            cost = proceeds * self.transaction_cost
            self._cash = proceeds - cost
            self._shares = 0.0
            traded = True

        self._days_in_position = 0 if traded else self._days_in_position + 1

        # Advance
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

        # Get current regime
        regime = self._get_regime(idx)
        position = 1 if self._shares > 0 else 0
        vol_20d = float(self.feature_provider._vol_20d[idx])

        reward = self.reward_fn(pv_before, pv_after, self._peak_pv, position, vol_20d)

        obs = self._get_obs() if not terminated else np.zeros(OBS_DIM, dtype=np.float32)
        info = self._get_info()
        info["portfolio_value"] = pv_after
        info["action_taken"] = ["HOLD", "BUY", "SELL"][action]
        info["regime"] = regime

        return obs, reward, terminated, truncated, info

    def _portfolio_value(self, price: float) -> float:
        return self._cash + self._shares * price

    def _get_regime(self, idx: int) -> int:
        if self._regime_labels is not None:
            # Find index of idx in step_indices context
            # regime_labels is aligned to step_indices
            pos = self._step_idx if self._step_idx < len(self.step_indices) else -1
            if pos >= 0 and pos < len(self._regime_labels):
                return int(self._regime_labels[pos])
        return SIDEWAYS

    def _get_regime_confidence(self, idx: int) -> float:
        if self._regime_confidences is not None:
            pos = self._step_idx if self._step_idx < len(self.step_indices) else -1
            if pos >= 0 and pos < len(self._regime_confidences):
                return float(self._regime_confidences[pos])
        return 0.0

    def _get_obs(self) -> np.ndarray:
        if self._step_idx >= len(self.step_indices):
            return np.zeros(OBS_DIM, dtype=np.float32)

        idx = self.step_indices[self._step_idx]
        price = float(self.df["close"].iloc[idx])

        # Market features (18 dims)
        market_feats = self.feature_provider.get_features(idx)

        # Agent state features
        position = 1.0 if self._shares > 0 else 0.0
        pv = self._portfolio_value(price)
        portfolio_return = (pv - self.initial_cash) / self.initial_cash
        days_norm = self._days_in_position / 50.0

        regime = self._get_regime(idx)
        confidence = self._get_regime_confidence(idx)

        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[0] = position
        obs[1:19] = market_feats
        obs[18] = portfolio_return
        obs[19] = days_norm
        obs[20] = float(regime)
        obs[21] = confidence

        return obs

    def _get_info(self) -> dict:
        if self._step_idx >= len(self.step_indices):
            return {}
        idx = self.step_indices[self._step_idx]
        return {
            "date": str(self.df.index[idx].date()),
            "price": float(self.df["close"].iloc[idx]),
            "cash": self._cash,
            "shares": self._shares,
            "step": self._step_idx,
            "total_steps": len(self.step_indices),
        }


# Need pandas for Timestamp
import pandas as pd


class RegimeFilteredEnv(gym.Env):
    """
    Wraps EnhancedTradingEnv but only exposes steps matching a target regime.
    If the filtered episode is too short, pads with the last available step.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        base_env: EnhancedTradingEnv,
        target_regime: int,
        regime_labels: np.ndarray,
    ):
        super().__init__()
        self.base_env = base_env
        self.target_regime = target_regime

        # Filter step indices to only matching regime
        matching = [
            i for i, r in enumerate(regime_labels)
            if r == target_regime and i < len(base_env.step_indices)
        ]
        self._filtered_positions = matching

        # Store original step_indices for the matching positions
        self._original_step_indices = base_env.step_indices
        self.step_indices = [base_env.step_indices[i] for i in matching]

        self.action_space = base_env.action_space
        self.observation_space = base_env.observation_space

        self._pos = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.base_env._step_idx = 0
        self.base_env._cash = self.base_env.initial_cash
        self.base_env._shares = 0.0
        self.base_env._peak_pv = self.base_env.initial_cash
        self.base_env._days_in_position = 0
        self._pos = 0

        if len(self._filtered_positions) == 0:
            return np.zeros(OBS_DIM, dtype=np.float32), {}

        # Set base env to first filtered position
        self.base_env._step_idx = self._filtered_positions[0]
        obs = self.base_env._get_obs()
        info = self.base_env._get_info()
        return obs, info

    def step(self, action):
        if self._pos >= len(self._filtered_positions):
            return (np.zeros(OBS_DIM, dtype=np.float32),
                    0.0, True, False, {})

        # Set base env to current filtered position
        self.base_env._step_idx = self._filtered_positions[self._pos]

        obs, reward, _, _, info = self.base_env.step(action)

        self._pos += 1
        terminated = self._pos >= len(self._filtered_positions)

        if terminated:
            obs = np.zeros(OBS_DIM, dtype=np.float32)

        return obs, reward, terminated, False, info


class MultiTickerRegimeEnv(gym.Env):
    """
    Multi-ticker training environment for regime-specific policies.
    Samples a random ticker each episode.
    """

    metadata = {"render_modes": []}

    def __init__(self, ticker_envs: dict[str, gym.Env]):
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
                for idx in (env.step_indices if hasattr(env, 'step_indices') else [])]

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
