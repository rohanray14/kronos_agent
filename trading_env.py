"""
Trading Environment (Gymnasium)
================================
Wraps the Kronos-based backtest loop into a standard Gym environment
for training RL agents (PPO, DQN, etc.).

State: numeric features derived from market data + Kronos forecasts
Action: 0=HOLD, 1=BUY, 2=SELL
Reward: change in portfolio value (with optional risk shaping)

Usage:
    from trading_env import TradingEnv
    env = TradingEnv(df, predictor)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from agent import (
    LOOKBACK_DAYS, FORECAST_STEPS, INITIAL_CASH,
    VOLATILITY_MULTIPLIER,
    get_forecast,
)


class TradingEnv(gym.Env):
    """
    Gymnasium environment for S&P 500 trading with Kronos forecasts.

    Observation space (10 features):
        0: position          — 0.0 (cash) or 1.0 (invested)
        1: return_20d         — 20-day price return
        2: return_50d         — 50-day price return
        3: volatility_20d     — 20-day realized volatility
        4: price_vs_50ma      — price relative to 50-day MA (ratio - 1)
        5: kronos_expected_ret — Kronos 1-step expected return
        6: kronos_trend        — fraction of forecast steps that are up
        7: kronos_high_vol     — 1.0 if forecast vol > recent vol * multiplier
        8: portfolio_return    — current portfolio return since start
        9: days_in_position    — normalized days held in current position

    Action space: Discrete(3)
        0 = HOLD, 1 = BUY, 2 = SELL
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        predictor,
        test_start: str = "2020-01-01",
        test_end: str = "2024-12-31",
        initial_cash: float = INITIAL_CASH,
        forecast_steps: int = FORECAST_STEPS,
        lookback_days: int = LOOKBACK_DAYS,
        reward_type: str = "log_return",  # "log_return", "pnl", "sharpe"
        transaction_cost: float = 0.0,    # fraction, e.g. 0.001 = 10bps
        random_start: bool = False,       # randomize episode start for data augmentation
        min_episode_steps: int = 20,      # minimum steps per episode when random_start=True
    ):
        super().__init__()

        self.df = df
        self.predictor = predictor
        self.test_start = test_start
        self.test_end = test_end
        self.initial_cash = initial_cash
        self.forecast_steps = forecast_steps
        self.lookback_days = lookback_days
        self.reward_type = reward_type
        self.transaction_cost = transaction_cost
        self.random_start = random_start
        self.min_episode_steps = min_episode_steps

        # Pre-compute step indices
        dates = df.index
        test_indices = [i for i, d in enumerate(dates) if d >= pd.Timestamp(test_start)]
        test_indices = [i for i in test_indices if i >= lookback_days]
        self.step_indices = test_indices[::forecast_steps]

        # Spaces
        self.action_space = spaces.Discrete(3)  # HOLD, BUY, SELL
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )

        # State variables (set in reset)
        self._step_idx = 0
        self._cash = initial_cash
        self._shares = 0.0
        self._prev_portfolio_value = initial_cash
        self._days_in_position = 0
        self._recent_returns = []  # for Sharpe reward
        self._prev_action = 0  # track previous action for penalty
        self._episode_start = 0  # for random start

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Random start: begin episode at a random point in the data
        # This creates diverse training episodes from the same data
        if self.random_start and len(self.step_indices) > self.min_episode_steps:
            max_start = len(self.step_indices) - self.min_episode_steps
            self._episode_start = self.np_random.integers(0, max_start)
        else:
            self._episode_start = 0

        self._step_idx = self._episode_start
        self._cash = self.initial_cash
        self._shares = 0.0
        self._prev_portfolio_value = self.initial_cash
        self._days_in_position = 0
        self._recent_returns = []
        self._prev_action = 0

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action):
        idx = self.step_indices[self._step_idx]
        price = float(self.df["close"].iloc[idx])

        # Previous portfolio value
        pv_before = self._portfolio_value(price)

        # Execute action
        traded = False
        if action == 1:  # BUY
            if self._cash > 0:
                cost = self._cash * self.transaction_cost
                self._shares = (self._cash - cost) / price
                self._cash = 0.0
                traded = True
        elif action == 2:  # SELL
            if self._shares > 0:
                proceeds = self._shares * price
                cost = proceeds * self.transaction_cost
                self._cash = proceeds - cost
                self._shares = 0.0
                traded = True

        if traded:
            self._days_in_position = 0
        else:
            self._days_in_position += 1

        # Advance to next step
        self._step_idx += 1
        terminated = self._step_idx >= len(self.step_indices)
        truncated = False

        # Compute reward based on new portfolio value
        if not terminated:
            next_idx = self.step_indices[self._step_idx]
            next_price = float(self.df["close"].iloc[next_idx])
        else:
            next_price = price

        pv_after = self._portfolio_value(next_price)
        reward = self._compute_reward(pv_before, pv_after)
        self._prev_portfolio_value = pv_after
        self._recent_returns.append(reward)

        obs = self._get_obs() if not terminated else np.zeros(10, dtype=np.float32)
        info = self._get_info()
        info["portfolio_value"] = pv_after
        info["action_taken"] = ["HOLD", "BUY", "SELL"][action]

        return obs, reward, terminated, truncated, info

    def _portfolio_value(self, price: float) -> float:
        return self._cash + self._shares * price

    def _compute_reward(self, pv_before: float, pv_after: float) -> float:
        if self.reward_type == "log_return":
            if pv_before <= 0:
                return 0.0
            return float(np.log(pv_after / pv_before))
        elif self.reward_type == "pnl":
            return (pv_after - pv_before) / self.initial_cash
        elif self.reward_type == "sharpe":
            # Incremental Sharpe-like: return minus running volatility penalty
            ret = (pv_after - pv_before) / pv_before if pv_before > 0 else 0.0
            if len(self._recent_returns) > 5:
                vol = np.std(self._recent_returns[-20:])
                return float(ret - 0.5 * vol)
            return float(ret)
        else:
            return (pv_after - pv_before) / pv_before if pv_before > 0 else 0.0

    def _get_obs(self) -> np.ndarray:
        """Build the 10-dimensional observation vector."""
        if self._step_idx >= len(self.step_indices):
            return np.zeros(10, dtype=np.float32)

        idx = self.step_indices[self._step_idx]
        prices = self.df["close"].values
        current_price = float(prices[idx])

        # Position
        position = 1.0 if self._shares > 0 else 0.0

        # Recent returns
        recent_20 = prices[max(0, idx - 20):idx]
        recent_50 = prices[max(0, idx - 50):idx]

        ret_20d = (current_price - float(recent_20[0])) / float(recent_20[0]) if len(recent_20) > 0 else 0.0
        ret_50d = (current_price - float(recent_50[0])) / float(recent_50[0]) if len(recent_50) > 0 else 0.0

        # Volatility
        if len(recent_20) > 1:
            daily_rets = np.diff(recent_20) / recent_20[:-1]
            vol_20d = float(np.std(daily_rets))
        else:
            vol_20d = 0.0

        # Price vs 50 MA
        ma_50 = float(np.mean(recent_50)) if len(recent_50) > 0 else current_price
        price_vs_50ma = (current_price - ma_50) / ma_50

        # Kronos forecast features
        kronos_ret, kronos_trend, kronos_high_vol = self._get_kronos_features(idx)

        # Portfolio return
        pv = self._portfolio_value(current_price)
        portfolio_return = (pv - self.initial_cash) / self.initial_cash

        # Days in position (normalized by forecast steps)
        days_norm = self._days_in_position / 50.0  # normalize

        obs = np.array([
            position,
            ret_20d,
            ret_50d,
            vol_20d,
            price_vs_50ma,
            kronos_ret,
            kronos_trend,
            kronos_high_vol,
            portfolio_return,
            days_norm,
        ], dtype=np.float32)

        return obs

    def _get_kronos_features(self, idx: int) -> tuple[float, float, float]:
        """Extract numeric features from Kronos forecast."""
        history_df = self.df.iloc[max(0, idx - self.lookback_days):idx]
        current_date = self.df.index[idx]
        current_price = float(self.df["close"].iloc[idx])

        future_dates = pd.bdate_range(
            start=current_date, periods=self.forecast_steps + 1, freq="B"
        )[1:]

        forecast = get_forecast(self.predictor, history_df, future_dates)

        closes = forecast["close"]
        highs = forecast["high"]
        lows = forecast["low"]

        # Expected return
        expected_ret = (closes[0] - current_price) / current_price

        # Trend: fraction of steps up
        trend = float(np.sum(closes > current_price)) / len(closes)

        # Volatility flag
        forecast_spread = float(np.mean(highs - lows) / current_price)
        recent_spread = float(np.mean(
            history_df["high"].values[-20:] - history_df["low"].values[-20:]
        ) / current_price)
        high_vol = 1.0 if forecast_spread > (recent_spread * VOLATILITY_MULTIPLIER) else 0.0

        return expected_ret, trend, high_vol

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


class GatedTradingEnv(gym.Env):
    """
    Trading environment where the agent chooses WHETHER to call Kronos.

    The key idea: Kronos forecasts are not always reliable. The agent
    learns when to request a forecast vs. when to decide from market
    context alone. Calling the forecast has an optional cost (simulating
    compute latency or API cost).

    Two-phase action per step:
        Phase 1 (gate): observe market context → decide to forecast or not
        Phase 2 (trade): observe full state (with or without forecast) → BUY/SELL/HOLD

    Combined into a single Discrete(6) action space:
        0 = NO_FORECAST + HOLD
        1 = NO_FORECAST + BUY
        2 = NO_FORECAST + SELL
        3 = FORECAST + HOLD
        4 = FORECAST + BUY
        5 = FORECAST + SELL

    Observation space (12 features):
        0-4:  market context (position, ret_20d, ret_50d, vol_20d, price_vs_50ma)
        5-7:  kronos features (expected_ret, trend, high_vol) — ZEROED if not requested
        8:    portfolio_return
        9:    days_in_position (normalized)
        10:   used_forecast (1.0 if forecast was requested this step, else 0.0)
        11:   forecast_accuracy — rolling accuracy of recent Kronos forecasts
    """

    metadata = {"render_modes": []}

    # Action decoding
    ACTION_NAMES = [
        "HOLD (no forecast)", "BUY (no forecast)", "SELL (no forecast)",
        "HOLD (forecast)", "BUY (forecast)", "SELL (forecast)",
    ]
    TRADE_ACTIONS = ["HOLD", "BUY", "SELL", "HOLD", "BUY", "SELL"]

    def __init__(
        self,
        df: pd.DataFrame,
        predictor,
        test_start: str = "2020-01-01",
        test_end: str = "2024-12-31",
        initial_cash: float = INITIAL_CASH,
        forecast_steps: int = FORECAST_STEPS,
        lookback_days: int = LOOKBACK_DAYS,
        reward_type: str = "log_return",
        transaction_cost: float = 0.0,
        forecast_cost: float = 0.0,       # reward penalty per forecast call
        random_start: bool = False,
        min_episode_steps: int = 20,
    ):
        super().__init__()

        self.df = df
        self.predictor = predictor
        self.test_start = test_start
        self.test_end = test_end
        self.initial_cash = initial_cash
        self.forecast_steps = forecast_steps
        self.lookback_days = lookback_days
        self.reward_type = reward_type
        self.transaction_cost = transaction_cost
        self.forecast_cost = forecast_cost
        self.random_start = random_start
        self.min_episode_steps = min_episode_steps

        # Pre-compute step indices
        dates = df.index
        test_indices = [i for i, d in enumerate(dates) if d >= pd.Timestamp(test_start)]
        test_indices = [i for i in test_indices if i >= lookback_days]
        self.step_indices = test_indices[::forecast_steps]

        # Spaces
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )

        # State variables
        self._step_idx = 0
        self._cash = initial_cash
        self._shares = 0.0
        self._prev_portfolio_value = initial_cash
        self._days_in_position = 0
        self._recent_returns = []
        self._episode_start = 0

        # Forecast gating tracking
        self._forecast_calls = 0       # total forecasts requested this episode
        self._total_steps_done = 0     # total steps this episode
        self._forecast_accuracy_history = []  # was kronos directionally correct?
        # Carry forward last forecast so agent has memory
        self._last_kronos_features = (0.0, 0.0, 0.0)
        self._pending_kronos_direction = None

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
        self._prev_portfolio_value = self.initial_cash
        self._days_in_position = 0
        self._recent_returns = []
        self._forecast_calls = 0
        self._total_steps_done = 0
        self._forecast_accuracy_history = []
        self._last_kronos_features = (0.0, 0.0, 0.0)
        self._pending_kronos_direction = None

        obs = self._get_obs(used_forecast=False)
        info = self._get_info()
        return obs, info

    def step(self, action):
        action = int(action)
        used_forecast = action >= 3
        trade_action = action % 3  # 0=HOLD, 1=BUY, 2=SELL

        idx = self.step_indices[self._step_idx]
        price = float(self.df["close"].iloc[idx])
        pv_before = self._portfolio_value(price)

        # If agent requests forecast, fetch it and update memory
        if used_forecast:
            self._last_kronos_features = self._get_kronos_features(idx)
            self._forecast_calls += 1
            kronos_ret = self._last_kronos_features[0]
            self._pending_kronos_direction = 1 if kronos_ret > 0 else -1
        else:
            self._pending_kronos_direction = None

        # Execute trade
        traded = False
        if trade_action == 1 and self._cash > 0:  # BUY
            cost = self._cash * self.transaction_cost
            self._shares = (self._cash - cost) / price
            self._cash = 0.0
            traded = True
        elif trade_action == 2 and self._shares > 0:  # SELL
            proceeds = self._shares * price
            cost = proceeds * self.transaction_cost
            self._cash = proceeds - cost
            self._shares = 0.0
            traded = True

        self._days_in_position = 0 if traded else self._days_in_position + 1
        self._total_steps_done += 1

        # Advance
        self._step_idx += 1
        terminated = self._step_idx >= len(self.step_indices)
        truncated = False

        if not terminated:
            next_idx = self.step_indices[self._step_idx]
            next_price = float(self.df["close"].iloc[next_idx])
            # Track Kronos accuracy: was the direction prediction correct?
            if self._pending_kronos_direction is not None:
                actual_direction = 1 if next_price > price else -1
                correct = 1.0 if actual_direction == self._pending_kronos_direction else 0.0
                self._forecast_accuracy_history.append(correct)
        else:
            next_price = price

        pv_after = self._portfolio_value(next_price)

        # Compute reward
        reward = self._compute_reward(pv_before, pv_after)
        if used_forecast:
            reward -= self.forecast_cost  # pay for using forecast
        self._prev_portfolio_value = pv_after
        self._recent_returns.append(reward)

        # Observation now always includes last known forecast features
        # (zeros if never forecasted, stale if forecasted N steps ago)
        obs = self._get_obs(used_forecast=used_forecast) if not terminated else np.zeros(12, dtype=np.float32)
        info = self._get_info()
        info["portfolio_value"] = pv_after
        info["action_taken"] = self.TRADE_ACTIONS[action]
        info["used_forecast"] = used_forecast
        info["forecast_calls"] = self._forecast_calls
        info["total_steps"] = self._total_steps_done
        info["forecast_rate"] = self._forecast_calls / max(1, self._total_steps_done)

        return obs, reward, terminated, truncated, info

    def _portfolio_value(self, price: float) -> float:
        return self._cash + self._shares * price

    def _compute_reward(self, pv_before: float, pv_after: float) -> float:
        if self.reward_type == "log_return":
            return float(np.log(pv_after / pv_before)) if pv_before > 0 else 0.0
        elif self.reward_type == "pnl":
            return (pv_after - pv_before) / self.initial_cash
        elif self.reward_type == "sharpe":
            ret = (pv_after - pv_before) / pv_before if pv_before > 0 else 0.0
            if len(self._recent_returns) > 5:
                vol = np.std(self._recent_returns[-20:])
                return float(ret - 0.5 * vol)
            return float(ret)
        else:
            return (pv_after - pv_before) / pv_before if pv_before > 0 else 0.0

    def _get_obs(self, used_forecast: bool = False) -> np.ndarray:
        if self._step_idx >= len(self.step_indices):
            return np.zeros(12, dtype=np.float32)

        idx = self.step_indices[self._step_idx]
        prices = self.df["close"].values
        current_price = float(prices[idx])

        # Market context (always available)
        position = 1.0 if self._shares > 0 else 0.0

        recent_20 = prices[max(0, idx - 20):idx]
        recent_50 = prices[max(0, idx - 50):idx]
        ret_20d = (current_price - float(recent_20[0])) / float(recent_20[0]) if len(recent_20) > 0 else 0.0
        ret_50d = (current_price - float(recent_50[0])) / float(recent_50[0]) if len(recent_50) > 0 else 0.0

        if len(recent_20) > 1:
            daily_rets = np.diff(recent_20) / recent_20[:-1]
            vol_20d = float(np.std(daily_rets))
        else:
            vol_20d = 0.0

        ma_50 = float(np.mean(recent_50)) if len(recent_50) > 0 else current_price
        price_vs_50ma = (current_price - ma_50) / ma_50

        # Kronos features: always show the LAST forecast (stale if not refreshed)
        # This gives the agent memory — it can see old forecasts are stale
        # and decide whether to refresh by calling Kronos again
        kronos_ret, kronos_trend, kronos_high_vol = self._last_kronos_features

        # Portfolio return
        pv = self._portfolio_value(current_price)
        portfolio_return = (pv - self.initial_cash) / self.initial_cash
        days_norm = self._days_in_position / 50.0

        # Forecast meta-features
        used_flag = 1.0 if used_forecast else 0.0
        if len(self._forecast_accuracy_history) >= 3:
            forecast_accuracy = float(np.mean(self._forecast_accuracy_history[-10:]))
        else:
            forecast_accuracy = 0.5  # prior: assume 50/50

        return np.array([
            position, ret_20d, ret_50d, vol_20d, price_vs_50ma,
            kronos_ret, kronos_trend, kronos_high_vol,
            portfolio_return, days_norm,
            used_flag, forecast_accuracy,
        ], dtype=np.float32)

    def _get_kronos_features(self, idx: int) -> tuple[float, float, float]:
        history_df = self.df.iloc[max(0, idx - self.lookback_days):idx]
        current_date = self.df.index[idx]
        current_price = float(self.df["close"].iloc[idx])
        future_dates = pd.bdate_range(
            start=current_date, periods=self.forecast_steps + 1, freq="B"
        )[1:]
        forecast = get_forecast(self.predictor, history_df, future_dates)
        closes = forecast["close"]
        highs = forecast["high"]
        lows = forecast["low"]
        expected_ret = (closes[0] - current_price) / current_price
        trend = float(np.sum(closes > current_price)) / len(closes)
        forecast_spread = float(np.mean(highs - lows) / current_price)
        recent_spread = float(np.mean(
            history_df["high"].values[-20:] - history_df["low"].values[-20:]
        ) / current_price)
        high_vol = 1.0 if forecast_spread > (recent_spread * VOLATILITY_MULTIPLIER) else 0.0
        return expected_ret, trend, high_vol

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


class CachedGatedTradingEnv(GatedTradingEnv):
    """GatedTradingEnv with pre-computed Kronos forecast caching."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._forecast_cache = {}
        self._cache_ready = False

    def precompute_forecasts(self, verbose: bool = True):
        total = len(self.step_indices)
        if verbose:
            print(f"Pre-computing Kronos forecasts for {total} steps (gated env)...")
        for i, idx in enumerate(self.step_indices):
            self._forecast_cache[idx] = GatedTradingEnv._get_kronos_features(self, idx)
            if verbose and (i + 1) % 20 == 0:
                print(f"   {i + 1}/{total} forecasts computed")
        self._cache_ready = True
        if verbose:
            print(f"   Done! {total} forecasts cached.")

    def save_cache(self, path: str):
        import json
        serializable = {str(k): list(v) for k, v in self._forecast_cache.items()}
        with open(path, "w") as f:
            json.dump(serializable, f)

    def load_cache(self, path: str):
        import json
        with open(path, "r") as f:
            data = json.load(f)
        self._forecast_cache = {int(k): tuple(v) for k, v in data.items()}
        self._cache_ready = True

    def _get_kronos_features(self, idx: int) -> tuple[float, float, float]:
        if self._cache_ready and idx in self._forecast_cache:
            return self._forecast_cache[idx]
        return GatedTradingEnv._get_kronos_features(self, idx)


class SoftGatedTradingEnv(gym.Env):
    """
    Trading environment where the forecast is ALWAYS provided, but with
    a confidence score that signals how reliable it is in the current regime.

    Unlike the hard-gated env (which chooses forecast or no forecast),
    the soft gate always shows the forecast but attaches a reliability
    signal. The agent learns to weight the forecast by this confidence.

    Confidence is derived from 20-day realized volatility:
        Low volatility  -> high confidence (forecasts tend to be reliable)
        High volatility -> low confidence  (forecasts tend to be noisy)

    Observation space (11 features):
        0:    position
        1:    return_20d
        2:    return_50d
        3:    volatility_20d
        4:    price_vs_50ma
        5:    kronos_expected_ret (ALWAYS present)
        6:    kronos_trend        (ALWAYS present)
        7:    kronos_high_vol     (ALWAYS present)
        8:    portfolio_return
        9:    days_in_position (normalized)
        10:   forecast_confidence (0-1, volatility-derived)

    Action space: Discrete(3) — 0=HOLD, 1=BUY, 2=SELL
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        predictor,
        test_start: str = "2020-01-01",
        test_end: str = "2024-12-31",
        initial_cash: float = INITIAL_CASH,
        forecast_steps: int = FORECAST_STEPS,
        lookback_days: int = LOOKBACK_DAYS,
        reward_type: str = "log_return",
        transaction_cost: float = 0.0,
        random_start: bool = False,
        min_episode_steps: int = 20,
        vol_ceiling: float = 0.02,  # volatility at which confidence = 0
    ):
        super().__init__()

        self.df = df
        self.predictor = predictor
        self.test_start = test_start
        self.test_end = test_end
        self.initial_cash = initial_cash
        self.forecast_steps = forecast_steps
        self.lookback_days = lookback_days
        self.reward_type = reward_type
        self.transaction_cost = transaction_cost
        self.random_start = random_start
        self.min_episode_steps = min_episode_steps
        self.vol_ceiling = vol_ceiling

        # Pre-compute step indices
        dates = df.index
        test_indices = [i for i, d in enumerate(dates) if d >= pd.Timestamp(test_start)]
        test_indices = [i for i in test_indices if i >= lookback_days]
        self.step_indices = test_indices[::forecast_steps]

        # Spaces
        self.action_space = spaces.Discrete(3)  # HOLD, BUY, SELL
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32
        )

        # State variables
        self._step_idx = 0
        self._cash = initial_cash
        self._shares = 0.0
        self._prev_portfolio_value = initial_cash
        self._days_in_position = 0
        self._recent_returns = []
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
        self._prev_portfolio_value = self.initial_cash
        self._days_in_position = 0
        self._recent_returns = []

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
        reward = self._compute_reward(pv_before, pv_after)
        self._prev_portfolio_value = pv_after
        self._recent_returns.append(reward)

        obs = self._get_obs() if not terminated else np.zeros(11, dtype=np.float32)
        info = self._get_info()
        info["portfolio_value"] = pv_after
        info["action_taken"] = ["HOLD", "BUY", "SELL"][action]

        return obs, reward, terminated, truncated, info

    def _portfolio_value(self, price: float) -> float:
        return self._cash + self._shares * price

    def _compute_reward(self, pv_before: float, pv_after: float) -> float:
        if self.reward_type == "log_return":
            return float(np.log(pv_after / pv_before)) if pv_before > 0 else 0.0
        elif self.reward_type == "pnl":
            return (pv_after - pv_before) / self.initial_cash
        elif self.reward_type == "sharpe":
            ret = (pv_after - pv_before) / pv_before if pv_before > 0 else 0.0
            if len(self._recent_returns) > 5:
                vol = np.std(self._recent_returns[-20:])
                return float(ret - 0.5 * vol)
            return float(ret)
        else:
            return (pv_after - pv_before) / pv_before if pv_before > 0 else 0.0

    def _get_obs(self) -> np.ndarray:
        if self._step_idx >= len(self.step_indices):
            return np.zeros(11, dtype=np.float32)

        idx = self.step_indices[self._step_idx]
        prices = self.df["close"].values
        current_price = float(prices[idx])

        # Position
        position = 1.0 if self._shares > 0 else 0.0

        # Recent returns
        recent_20 = prices[max(0, idx - 20):idx]
        recent_50 = prices[max(0, idx - 50):idx]
        ret_20d = (current_price - float(recent_20[0])) / float(recent_20[0]) if len(recent_20) > 0 else 0.0
        ret_50d = (current_price - float(recent_50[0])) / float(recent_50[0]) if len(recent_50) > 0 else 0.0

        # Volatility
        if len(recent_20) > 1:
            daily_rets = np.diff(recent_20) / recent_20[:-1]
            vol_20d = float(np.std(daily_rets))
        else:
            vol_20d = 0.0

        # Price vs 50 MA
        ma_50 = float(np.mean(recent_50)) if len(recent_50) > 0 else current_price
        price_vs_50ma = (current_price - ma_50) / ma_50

        # Kronos features (ALWAYS present)
        kronos_ret, kronos_trend, kronos_high_vol = self._get_kronos_features(idx)

        # Portfolio return
        pv = self._portfolio_value(current_price)
        portfolio_return = (pv - self.initial_cash) / self.initial_cash
        days_norm = self._days_in_position / 50.0

        # Forecast confidence: low vol -> high confidence, high vol -> low
        confidence = max(0.0, min(1.0, 1.0 - (vol_20d / self.vol_ceiling)))

        return np.array([
            position, ret_20d, ret_50d, vol_20d, price_vs_50ma,
            kronos_ret, kronos_trend, kronos_high_vol,
            portfolio_return, days_norm,
            confidence,
        ], dtype=np.float32)

    def _get_kronos_features(self, idx: int) -> tuple[float, float, float]:
        history_df = self.df.iloc[max(0, idx - self.lookback_days):idx]
        current_date = self.df.index[idx]
        current_price = float(self.df["close"].iloc[idx])
        future_dates = pd.bdate_range(
            start=current_date, periods=self.forecast_steps + 1, freq="B"
        )[1:]
        forecast = get_forecast(self.predictor, history_df, future_dates)
        closes = forecast["close"]
        highs = forecast["high"]
        lows = forecast["low"]
        expected_ret = (closes[0] - current_price) / current_price
        trend = float(np.sum(closes > current_price)) / len(closes)
        forecast_spread = float(np.mean(highs - lows) / current_price)
        recent_spread = float(np.mean(
            history_df["high"].values[-20:] - history_df["low"].values[-20:]
        ) / current_price)
        high_vol = 1.0 if forecast_spread > (recent_spread * VOLATILITY_MULTIPLIER) else 0.0
        return expected_ret, trend, high_vol

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


class CachedSoftGatedTradingEnv(SoftGatedTradingEnv):
    """SoftGatedTradingEnv with pre-computed Kronos forecast caching."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._forecast_cache = {}
        self._cache_ready = False

    def precompute_forecasts(self, verbose: bool = True):
        total = len(self.step_indices)
        if verbose:
            print(f"Pre-computing Kronos forecasts for {total} steps (soft gated env)...")
        for i, idx in enumerate(self.step_indices):
            self._forecast_cache[idx] = SoftGatedTradingEnv._get_kronos_features(self, idx)
            if verbose and (i + 1) % 20 == 0:
                print(f"   {i + 1}/{total} forecasts computed")
        self._cache_ready = True
        if verbose:
            print(f"   Done! {total} forecasts cached.")

    def save_cache(self, path: str):
        import json
        serializable = {str(k): list(v) for k, v in self._forecast_cache.items()}
        with open(path, "w") as f:
            json.dump(serializable, f)

    def load_cache(self, path: str):
        import json
        with open(path, "r") as f:
            data = json.load(f)
        self._forecast_cache = {int(k): tuple(v) for k, v in data.items()}
        self._cache_ready = True

    def _get_kronos_features(self, idx: int) -> tuple[float, float, float]:
        if self._cache_ready and idx in self._forecast_cache:
            return self._forecast_cache[idx]
        return SoftGatedTradingEnv._get_kronos_features(self, idx)


class CachedTradingEnv(TradingEnv):
    """
    TradingEnv variant that pre-computes all Kronos forecasts once,
    then replays them during training. Much faster for RL training
    where the env is reset thousands of times.

    Usage:
        env = CachedTradingEnv(df, predictor)
        env.precompute_forecasts()  # run once (~5 min)
        # Now reset/step are fast — no Kronos calls
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._forecast_cache = {}  # idx -> (expected_ret, trend, high_vol)
        self._cache_ready = False

    def precompute_forecasts(self, verbose: bool = True):
        """Run Kronos for every step index and cache the results."""
        total = len(self.step_indices)
        if verbose:
            print(f"Pre-computing Kronos forecasts for {total} steps...")

        for i, idx in enumerate(self.step_indices):
            self._forecast_cache[idx] = super()._get_kronos_features(idx)
            if verbose and (i + 1) % 20 == 0:
                print(f"   {i + 1}/{total} forecasts computed")

        self._cache_ready = True
        if verbose:
            print(f"   Done! {total} forecasts cached.")

    def save_cache(self, path: str):
        """Save forecast cache to disk."""
        import json
        serializable = {str(k): list(v) for k, v in self._forecast_cache.items()}
        with open(path, "w") as f:
            json.dump(serializable, f)

    def load_cache(self, path: str):
        """Load forecast cache from disk."""
        import json
        with open(path, "r") as f:
            data = json.load(f)
        self._forecast_cache = {int(k): tuple(v) for k, v in data.items()}
        self._cache_ready = True

    def _get_kronos_features(self, idx: int) -> tuple[float, float, float]:
        if self._cache_ready and idx in self._forecast_cache:
            return self._forecast_cache[idx]
        # Fallback to live computation
        return super()._get_kronos_features(idx)
