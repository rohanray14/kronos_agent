"""
Signal Quality Scorer Environment for V3.

The RL agent observes Kronos forecast features + macro context and
outputs a quality score (0-1) predicting how reliable the forecast is.

This replaces the crude heuristic (1.0 - vol/0.02) with a learned score
that Claude can reason about.

Key difference from position sizing: this is an UPSTREAM improvement.
The scorer changes what Claude sees, not what happens after Claude decides.

Observation space (22 dims):
    [0:3]   Kronos forecast features (expected_ret, trend, high_vol)
    [3:6]   Forecast track record (rolling 5/10/20 step accuracy)
    [6:9]   Forecast error stats (mean error, std error, recent bias)
    [9:18]  Macro features (VIX, yields, gold, treasuries, oil, spreads)
    [18:22] Cross-asset context (correlation state, risk-on/off signals)

Action space: Box(0, 1) — single continuous quality score
    Or Discrete(11) — {0.0, 0.1, 0.2, ..., 1.0} for PPO compatibility
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from v3.config import (
    INITIAL_CASH, FORECAST_STEPS, LOOKBACK_DAYS,
)
from v3.reward_scorer import scorer_reward


SCORER_OBS_DIM = 22
N_SCORE_LEVELS = 11  # 0.0, 0.1, 0.2, ..., 1.0
SCORE_LEVELS = [i / 10.0 for i in range(N_SCORE_LEVELS)]


class SignalScorerEnv(gym.Env):
    """
    Gymnasium environment for learning forecast quality scoring.

    Each step:
      1. Agent observes forecast features + macro context
      2. Agent outputs a quality score (0.0 to 1.0)
      3. We advance to see the realized price move
      4. Reward = calibration (did score match whether Kronos was right?)

    No portfolio management — this env is purely about forecast evaluation.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        feature_provider,
        kronos_cache: dict[int, tuple],
        test_start: str = "2020-01-01",
        test_end: str = "2024-12-31",
        forecast_steps: int = FORECAST_STEPS,
        lookback_days: int = LOOKBACK_DAYS,
        random_start: bool = False,
        min_episode_steps: int = 20,
    ):
        super().__init__()

        self.df = df
        self.feature_provider = feature_provider
        self.kronos_cache = kronos_cache
        self.forecast_steps = forecast_steps
        self.lookback_days = lookback_days
        self.random_start = random_start
        self.min_episode_steps = min_episode_steps

        # Pre-compute step indices
        dates = df.index
        start_ts = pd.Timestamp(test_start)
        end_ts = pd.Timestamp(test_end)
        all_indices = [i for i, d in enumerate(dates)
                       if d >= start_ts and d <= end_ts]
        all_indices = [i for i in all_indices if i >= lookback_days]
        # Only keep steps where we have Kronos forecasts AND room to verify
        self.step_indices = [
            i for i in all_indices[::forecast_steps]
            if i in kronos_cache and i + forecast_steps < len(df)
        ]

        # Spaces
        self.action_space = spaces.Discrete(N_SCORE_LEVELS)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(SCORER_OBS_DIM,), dtype=np.float32
        )

        # State
        self._step_idx = 0
        self._episode_start = 0

        # Rolling accuracy tracking (built up during episode)
        self._accuracy_history: list[float] = []
        self._forecast_errors: list[float] = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.random_start and len(self.step_indices) > self.min_episode_steps:
            max_start = len(self.step_indices) - self.min_episode_steps
            self._episode_start = self.np_random.integers(0, max_start)
        else:
            self._episode_start = 0

        self._step_idx = self._episode_start
        self._accuracy_history = []
        self._forecast_errors = []

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action):
        action = int(action)
        score = SCORE_LEVELS[action]

        idx = self.step_indices[self._step_idx]
        prices = self.df["close"].values
        current_price = float(prices[idx])

        # Get Kronos forecast
        k_ret, k_trend, k_hvol = self.kronos_cache.get(idx, (0.0, 0.0, 0.0))

        # Get realized outcome (look ahead by forecast_steps)
        future_idx = min(idx + self.forecast_steps, len(prices) - 1)
        future_price = float(prices[future_idx])
        realized_return = (future_price - current_price) / current_price

        # Did Kronos get the direction right?
        kronos_direction = 1 if k_ret > 0 else (-1 if k_ret < 0 else 0)
        actual_direction = 1 if realized_return > 0 else (-1 if realized_return < 0 else 0)
        direction_correct = (kronos_direction == actual_direction) and kronos_direction != 0

        # Track accuracy
        self._accuracy_history.append(1.0 if direction_correct else 0.0)
        forecast_error = abs(k_ret - realized_return)
        self._forecast_errors.append(forecast_error)

        # Compute reward
        reward = scorer_reward(
            score=score,
            kronos_direction_correct=direction_correct,
            forecast_return=k_ret,
            realized_return=realized_return,
        )

        # Advance
        self._step_idx += 1
        terminated = self._step_idx >= len(self.step_indices)
        truncated = False

        obs = self._get_obs() if not terminated else np.zeros(SCORER_OBS_DIM, dtype=np.float32)
        info = self._get_info()
        info["score"] = score
        info["kronos_ret"] = k_ret
        info["realized_ret"] = realized_return
        info["direction_correct"] = direction_correct
        info["reward"] = reward

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        if self._step_idx >= len(self.step_indices):
            return np.zeros(SCORER_OBS_DIM, dtype=np.float32)

        idx = self.step_indices[self._step_idx]

        # ── Kronos forecast features (3 dims) ──
        k_ret, k_trend, k_hvol = self.kronos_cache.get(idx, (0.0, 0.0, 0.0))

        # ── Forecast track record (3 dims) ──
        acc_5 = self._rolling_accuracy(5)
        acc_10 = self._rolling_accuracy(10)
        acc_20 = self._rolling_accuracy(20)

        # ── Forecast error stats (3 dims) ──
        if len(self._forecast_errors) >= 3:
            recent_errors = self._forecast_errors[-10:]
            mean_error = float(np.mean(recent_errors))
            std_error = float(np.std(recent_errors))
            # Bias: is Kronos consistently over/under-predicting?
            recent_acc = self._accuracy_history[-10:]
            bias = float(np.mean(recent_acc)) - 0.5  # >0 = Kronos has been accurate
        else:
            mean_error = 0.01  # neutral prior
            std_error = 0.01
            bias = 0.0

        # ── Macro features (9 dims) ──
        # Subset of the 18-dim MacroFeatureProvider output
        market_feats = self.feature_provider.get_features(idx)
        vol_20d = market_feats[3]       # vol_20d
        vix_level = market_feats[9]     # vix_level
        vix_change = market_feats[10]   # vix_change_5d
        yield_curve = market_feats[11]  # yield_curve
        gld_ret = market_feats[12]      # gld_ret_20d
        tlt_ret = market_feats[13]      # tlt_ret_20d
        uso_ret = market_feats[14]      # uso_ret_20d
        iwm_spread = market_feats[15]   # iwm_spy_spread
        xlu_spread = market_feats[16]   # xlu_spy_spread

        # ── Cross-asset context (4 dims) ──
        ret_20d = market_feats[1]
        ret_50d = market_feats[2]

        # Risk-on/off signal: gold up + treasuries up + equities down = risk-off
        risk_off = float(gld_ret > 0 and tlt_ret > 0 and ret_20d < 0)

        # Momentum divergence: short-term vs long-term
        momentum_div = ret_20d - ret_50d

        obs = np.array([
            # Kronos forecast (3)
            k_ret, k_trend, k_hvol,
            # Track record (3)
            acc_5, acc_10, acc_20,
            # Error stats (3)
            mean_error, std_error, bias,
            # Macro (9)
            vol_20d, vix_level, vix_change, yield_curve,
            gld_ret, tlt_ret, uso_ret, iwm_spread, xlu_spread,
            # Cross-asset context (4)
            risk_off, momentum_div, ret_20d, ret_50d,
        ], dtype=np.float32)

        return obs

    def _rolling_accuracy(self, window: int) -> float:
        if len(self._accuracy_history) < 3:
            return 0.5  # neutral prior
        recent = self._accuracy_history[-window:]
        return float(np.mean(recent))

    def _get_info(self) -> dict:
        if self._step_idx >= len(self.step_indices):
            return {}
        idx = self.step_indices[self._step_idx]
        return {
            "date": str(self.df.index[idx].date()),
            "price": float(self.df["close"].iloc[idx]),
            "step": self._step_idx,
            "total_steps": len(self.step_indices),
        }


class MultiTickerScorerEnv(gym.Env):
    """Multi-ticker wrapper for signal scorer training."""

    metadata = {"render_modes": []}

    def __init__(self, ticker_envs: dict[str, SignalScorerEnv]):
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
