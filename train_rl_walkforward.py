"""
Walk-Forward CV + Multi-Ticker Training for RL Trading Agent
==============================================================
Combines two enhancements to the RL agent:

1. MULTI-TICKER TRAINING: Train on a pool of correlated ETFs
   (SPY, QQQ, DIA, IWM, XLF, XLE, XLK) for ~7x more training data.
   The agent sees normalized features so it learns general market
   dynamics, not ticker-specific patterns.

2. WALK-FORWARD CV: Evaluate across 6 sliding windows over 10 years
   to test robustness across market regimes (bull, bear, crash, recovery).

Windows:
    Fold 1: Train 2014-2018 -> Test 2019  (Late Bull)
    Fold 2: Train 2015-2019 -> Test 2020  (COVID Crash + Recovery)
    Fold 3: Train 2016-2020 -> Test 2021  (Post-COVID Bull)
    Fold 4: Train 2017-2021 -> Test 2022  (Bear Market)
    Fold 5: Train 2018-2022 -> Test 2023  (Recovery)
    Fold 6: Train 2019-2023 -> Test 2024  (AI Rally)

Usage:
    python train_rl_walkforward.py                          # all folds, multi-ticker
    python train_rl_walkforward.py --single-ticker          # SPY only (baseline)
    python train_rl_walkforward.py --folds 1,2,3            # specific folds
    python train_rl_walkforward.py --timesteps 200000       # more training
    python train_rl_walkforward.py --seeds 5                # multi-seed eval
    python train_rl_walkforward.py --precompute-only        # just cache forecasts
    python train_rl_walkforward.py --gated                  # include gated forecast agent (hard gate)
    python train_rl_walkforward.py --soft-gated             # include soft gated agent (forecast + confidence)
    python train_rl_walkforward.py --tickers SPY,QQQ,DIA    # custom ticker pool
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import os
import json
import argparse
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Kronos"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import gymnasium as gym
from gymnasium import spaces
import yfinance as yf

from agent import (
    LOOKBACK_DAYS, FORECAST_STEPS, INITIAL_CASH,
    VOLATILITY_MULTIPLIER,
    load_kronos, get_forecast, execute, decide,
    AgentState,
)
from trading_env import CachedTradingEnv, CachedGatedTradingEnv, CachedSoftGatedTradingEnv


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

DATA_START = "2012-01-01"   # buffer for 60-day lookback into 2014
DATA_END   = "2024-12-31"

# Ticker pool: S&P 500 + correlated major ETFs / indices
DEFAULT_TICKERS = ["SPY", "QQQ", "DIA", "IWM", "XLF", "XLE", "XLK"]
TEST_TICKER = "SPY"  # always test on SPY for comparability

FOLDS = [
    {"fold": 1, "train_start": "2014-01-01", "train_end": "2018-12-31",
     "test_start": "2019-01-01", "test_end": "2019-12-31",
     "regime": "Late Bull", "notes": "Pre-COVID, low vol, steady growth"},
    {"fold": 2, "train_start": "2015-01-01", "train_end": "2019-12-31",
     "test_start": "2020-01-01", "test_end": "2020-12-31",
     "regime": "COVID Crash + Recovery", "notes": "-34% crash then V-shaped recovery"},
    {"fold": 3, "train_start": "2016-01-01", "train_end": "2020-12-31",
     "test_start": "2021-01-01", "test_end": "2021-12-31",
     "regime": "Post-COVID Bull", "notes": "Stimulus-driven bull, meme stocks, low rates"},
    {"fold": 4, "train_start": "2017-01-01", "train_end": "2021-12-31",
     "test_start": "2022-01-01", "test_end": "2022-12-31",
     "regime": "Bear Market", "notes": "Fed hikes, -25% drawdown, inflation"},
    {"fold": 5, "train_start": "2018-01-01", "train_end": "2022-12-31",
     "test_start": "2023-01-01", "test_end": "2023-12-31",
     "regime": "Recovery", "notes": "SVB collapse then AI-driven recovery"},
    {"fold": 6, "train_start": "2019-01-01", "train_end": "2023-12-31",
     "test_start": "2024-01-01", "test_end": "2024-12-31",
     "regime": "AI Rally", "notes": "Mag-7 led bull market, rate cuts begin"},
]


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Walk-forward CV with multi-ticker RL training")
    parser.add_argument("--folds", type=str, default=None,
                        help="Comma-separated fold numbers (e.g., '1,2,3'). Default: all")
    parser.add_argument("--timesteps", type=int, default=100_000,
                        help="PPO training timesteps per fold")
    parser.add_argument("--seeds", type=int, default=3,
                        help="Random seeds per fold for robustness")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--reward", default="log_return",
                        choices=["log_return", "pnl", "sharpe"])
    parser.add_argument("--tx-cost", type=float, default=0.0)
    parser.add_argument("--precompute-only", action="store_true",
                        help="Cache forecasts for all folds/tickers, then exit")
    parser.add_argument("--gated", action="store_true",
                        help="Include gated forecast agent")
    parser.add_argument("--forecast-cost", type=float, default=0.001)
    parser.add_argument("--single-ticker", action="store_true",
                        help="Train on SPY only (baseline, no multi-ticker)")
    parser.add_argument("--soft-gated", action="store_true",
                        help="Include soft gated forecast agent (always forecast + confidence)")
    parser.add_argument("--tickers", type=str, default=None,
                        help="Custom ticker pool (comma-separated, e.g., 'SPY,QQQ,DIA')")
    return parser.parse_args()


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_ticker_data(ticker: str, max_retries: int = 3) -> pd.DataFrame:
    """Load OHLCV data for a single ticker with retry logic."""
    import time
    for attempt in range(max_retries):
        try:
            df = yf.download(ticker, start=DATA_START, end=DATA_END,
                             progress=False, auto_adjust=True)
            if df is None or len(df) == 0:
                raise ValueError(f"Empty data for {ticker}")
            df.columns = df.columns.get_level_values(0)
            df = df.rename(columns={
                "Open": "open", "High": "high", "Low": "low",
                "Close": "close", "Volume": "volume",
            })
            df = df[["open", "high", "low", "close", "volume"]]
            df.index = pd.to_datetime(df.index)
            df = df.dropna()
            if len(df) == 0:
                raise ValueError(f"All NaN data for {ticker}")
            return df
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"   Retry {attempt + 1}/{max_retries} for {ticker} "
                      f"(waiting {wait}s): {e}")
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"Failed to download {ticker} after {max_retries} attempts: {e}")


def load_all_tickers(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Load data for all tickers in the pool."""
    import time
    print(f"Loading {len(tickers)} tickers: {', '.join(tickers)}...")
    ticker_data = {}
    for i, ticker in enumerate(tickers):
        df = load_ticker_data(ticker)
        ticker_data[ticker] = df
        print(f"   {ticker}: {len(df)} trading days "
              f"({df.index[0].date()} -> {df.index[-1].date()})")
        # Small delay between tickers to avoid rate limiting
        if i < len(tickers) - 1:
            time.sleep(1)
    return ticker_data


# ─────────────────────────────────────────────
# MULTI-TICKER TRAINING ENVIRONMENT
# ─────────────────────────────────────────────

class MultiTickerTrainEnv(gym.Env):
    """
    Training environment that samples a random ticker each episode.

    On each reset(), a ticker is drawn uniformly from the pool and
    a random starting point within the training window is selected.
    This gives the PPO agent diverse market dynamics to learn from
    while keeping the observation space identical (all features are
    normalized returns/ratios).

    The agent learns a GENERAL trading policy, not a ticker-specific one.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        ticker_envs: dict[str, CachedTradingEnv],
    ):
        super().__init__()
        self.ticker_envs = ticker_envs
        self.ticker_names = list(ticker_envs.keys())
        self._current_ticker = None
        self._current_env = None

        # All envs share the same observation/action space
        sample_env = next(iter(ticker_envs.values()))
        self.action_space = sample_env.action_space
        self.observation_space = sample_env.observation_space

    @property
    def step_indices(self):
        """Total steps across all tickers (for PPO n_steps calculation)."""
        return [idx for env in self.ticker_envs.values()
                for idx in env.step_indices]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Sample a random ticker
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


class MultiTickerGatedTrainEnv(gym.Env):
    """Same as MultiTickerTrainEnv but for gated forecast envs."""

    metadata = {"render_modes": []}

    def __init__(self, ticker_envs: dict[str, CachedGatedTradingEnv]):
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


class MultiTickerSoftGatedTrainEnv(gym.Env):
    """Same as MultiTickerTrainEnv but for soft gated forecast envs."""

    metadata = {"render_modes": []}

    def __init__(self, ticker_envs: dict[str, CachedSoftGatedTradingEnv]):
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


# ─────────────────────────────────────────────
# NO-FORECAST ENV
# ─────────────────────────────────────────────

class NoForecastEnv(CachedTradingEnv):
    """TradingEnv that zeros out Kronos features."""
    def _get_kronos_features(self, idx):
        return 0.0, 0.0, 0.0


class MultiTickerNoForecastEnv(gym.Env):
    """Multi-ticker wrapper for no-forecast training."""

    metadata = {"render_modes": []}

    def __init__(self, ticker_envs: dict[str, NoForecastEnv]):
        super().__init__()
        self.ticker_envs = ticker_envs
        self.ticker_names = list(ticker_envs.keys())
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
        self._current_env = self.ticker_envs[self.ticker_names[ticker_idx]]
        return self._current_env.reset(seed=seed)

    def step(self, action):
        return self._current_env.step(action)


# ─────────────────────────────────────────────
# FORECAST CACHING
# ─────────────────────────────────────────────

def get_cache_dir():
    return "walkforward_cache"


def get_cache_path(ticker: str, fold_num: int, split: str, gated: bool = False) -> str:
    prefix = "gated_" if gated else ""
    return os.path.join(get_cache_dir(),
                        f"{prefix}{ticker}_fold{fold_num}_{split}.json")


def ensure_cache(env, cache_path: str) -> bool:
    """Load or compute forecast cache. Returns True if newly computed."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.exists(cache_path):
        env.load_cache(cache_path)
        return False
    else:
        env.precompute_forecasts(verbose=True)
        env.save_cache(cache_path)
        return True


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

def compute_fold_metrics(portfolio_values):
    """Compute return, Sharpe, max drawdown."""
    pv = np.array(portfolio_values)
    if len(pv) < 2:
        return {"return": 0.0, "sharpe": 0.0, "max_dd": 0.0, "final": INITIAL_CASH}
    total_return = (pv[-1] - INITIAL_CASH) / INITIAL_CASH * 100
    returns = np.diff(pv) / pv[:-1]
    sharpe = ((returns.mean() / returns.std()) * np.sqrt(252 / FORECAST_STEPS)
              if returns.std() > 0 else 0.0)
    peak = np.maximum.accumulate(pv)
    max_dd = float(((pv - peak) / peak).min()) * 100
    return {
        "return": round(total_return, 2),
        "sharpe": round(sharpe, 3),
        "max_dd": round(max_dd, 2),
        "final": round(float(pv[-1]), 2),
    }


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate_rl(model, env, action_names=None):
    """Run trained RL model through env."""
    obs, info = env.reset()
    done = False
    actions = []
    portfolio_values = [INITIAL_CASH]
    dates = [info["date"]]
    forecast_calls = 0
    total_steps = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if action_names:
            actions.append(action_names[action])
        else:
            actions.append(["HOLD", "BUY", "SELL"][action])

        if "portfolio_value" in info:
            portfolio_values.append(info["portfolio_value"])
        if not done and "date" in info:
            dates.append(info["date"])

        if "forecast_calls" in info:
            forecast_calls = info["forecast_calls"]
            total_steps = info["total_steps"]

    return {
        "actions": actions,
        "portfolio_values": portfolio_values[:len(dates)],
        "dates": dates,
        "forecast_calls": forecast_calls,
        "total_steps": total_steps,
        "forecast_rate": forecast_calls / max(1, total_steps),
    }


def evaluate_buy_and_hold(df, test_start, test_end):
    """Buy and hold baseline at step frequency."""
    test_indices = [i for i, d in enumerate(df.index) if d >= pd.Timestamp(test_start)]
    test_indices = [i for i in test_indices if i >= LOOKBACK_DAYS]
    step_indices = test_indices[::FORECAST_STEPS]

    if not step_indices:
        return [], []

    initial_price = float(df["close"].iloc[step_indices[0]])
    bh_shares = INITIAL_CASH / initial_price
    dates = [str(df.index[i].date()) for i in step_indices]
    values = [float(bh_shares * df["close"].iloc[i]) for i in step_indices]
    return values, dates


def evaluate_rule_based(df, predictor, test_start, test_end):
    """Rule-based agent baseline."""
    dates = df.index
    state = AgentState()
    test_indices = [i for i, d in enumerate(dates) if d >= pd.Timestamp(test_start)]
    test_indices = [i for i in test_indices if i >= LOOKBACK_DAYS]
    step_indices = test_indices[::FORECAST_STEPS]

    for idx in step_indices:
        current_date = dates[idx]
        current_price = float(df["close"].iloc[idx])
        history_df = df.iloc[max(0, idx - LOOKBACK_DAYS):idx]
        future_dates = pd.bdate_range(
            start=current_date, periods=FORECAST_STEPS + 1, freq="B"
        )[1:]
        forecast = get_forecast(predictor, history_df, future_dates)
        action = decide(forecast, current_price, history_df)
        execute(action, state, current_price)
        state.portfolio_values.append(state.portfolio_value(current_price))
        state.actions.append(action)
        state.dates.append(current_date)

    return state


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

def train_ppo(env, timesteps, lr, seed):
    """Train a PPO agent."""
    from stable_baselines3 import PPO

    n_steps = min(len(env.step_indices), 2048)
    # Ensure n_steps is at least 1
    n_steps = max(n_steps, 64)

    model = PPO(
        "MlpPolicy", env,
        learning_rate=lr,
        n_steps=n_steps,
        batch_size=min(64, n_steps),
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=0,
        seed=seed,
    )
    model.learn(total_timesteps=timesteps, progress_bar=False)
    return model


# ─────────────────────────────────────────────
# SEED AGGREGATION
# ─────────────────────────────────────────────

def aggregate_seeds(seed_results):
    """Aggregate metrics across multiple random seeds."""
    returns = [r["metrics"]["return"] for r in seed_results]
    sharpes = [r["metrics"]["sharpe"] for r in seed_results]
    max_dds = [r["metrics"]["max_dd"] for r in seed_results]
    finals = [r["metrics"]["final"] for r in seed_results]

    # Use median seed's results for plotting
    median_idx = int(np.argmin(np.abs(np.array(returns) - np.median(returns))))
    best = seed_results[median_idx]

    forecast_rates = [r.get("forecast_rate", 0.0) for r in seed_results]

    return {
        "metrics": {
            "return_mean": round(float(np.mean(returns)), 2),
            "return_std": round(float(np.std(returns)), 2),
            "sharpe_mean": round(float(np.mean(sharpes)), 3),
            "sharpe_std": round(float(np.std(sharpes)), 3),
            "max_dd_mean": round(float(np.mean(max_dds)), 2),
            "max_dd_std": round(float(np.std(max_dds)), 2),
            "final_mean": round(float(np.mean(finals)), 2),
            "return": round(float(np.mean(returns)), 2),
            "sharpe": round(float(np.mean(sharpes)), 3),
            "max_dd": round(float(np.mean(max_dds)), 2),
            "final": round(float(np.mean(finals)), 2),
        },
        "forecast_rate_mean": round(float(np.mean(forecast_rates)), 3),
        "n_seeds": len(seed_results),
        "dates": best.get("dates", []),
        "portfolio_values": best.get("portfolio_values", []),
        "actions": best.get("actions", []),
    }


# ─────────────────────────────────────────────
# RUN ONE FOLD
# ─────────────────────────────────────────────

def run_fold(fold_config, ticker_data, predictor, args, train_tickers):
    """Train and evaluate all agents for one walk-forward fold."""
    fold_num = fold_config["fold"]
    train_start = fold_config["train_start"]
    train_end = fold_config["train_end"]
    test_start = fold_config["test_start"]
    test_end = fold_config["test_end"]
    regime = fold_config["regime"]

    test_df = ticker_data[TEST_TICKER]

    print(f"\n{'='*70}")
    print(f"  FOLD {fold_num}: Train {train_start[:4]}-{train_end[:4]} "
          f"-> Test {test_start[:4]}")
    print(f"  Regime: {regime} -- {fold_config['notes']}")
    print(f"  Train tickers: {', '.join(train_tickers)} "
          f"| Test ticker: {TEST_TICKER}")
    print(f"{'='*70}")

    fold_results = {
        "fold": fold_num, "regime": regime,
        "test_start": test_start, "test_end": test_end,
        "train_start": train_start, "train_end": train_end,
        "agents": {},
    }

    # ─── Build per-ticker training envs ───
    print(f"\n  Setting up training environments...")
    train_envs = {}
    train_envs_nf = {}     # no-forecast variants
    train_envs_gated = {}  # gated variants
    train_envs_soft = {}   # soft gated variants

    total_train_steps = 0
    for ticker in train_tickers:
        df = ticker_data[ticker]
        common = dict(
            df=df, predictor=predictor,
            test_start=train_start, test_end=train_end,
            reward_type=args.reward, transaction_cost=args.tx_cost,
            random_start=True,
        )

        # Standard env (always forecast)
        env = CachedTradingEnv(**common)
        cache_path = get_cache_path(ticker, fold_num, "train")
        ensure_cache(env, cache_path)
        train_envs[ticker] = env
        total_train_steps += len(env.step_indices)

        # No-forecast env
        nf_env = NoForecastEnv(**common)
        train_envs_nf[ticker] = nf_env

        # Gated env (if needed)
        if args.gated:
            g_env = CachedGatedTradingEnv(
                **common, forecast_cost=args.forecast_cost)
            g_cache = get_cache_path(ticker, fold_num, "train", gated=True)
            ensure_cache(g_env, g_cache)
            train_envs_gated[ticker] = g_env

        # Soft gated env (if needed)
        if args.soft_gated:
            sg_env = CachedSoftGatedTradingEnv(**common)
            sg_cache = get_cache_path(ticker, fold_num, "train", gated=False)
            # Reuse the standard forecast cache (same forecasts, different obs)
            ensure_cache(sg_env, sg_cache)
            train_envs_soft[ticker] = sg_env

    print(f"  Total training steps across tickers: {total_train_steps}")

    # ─── Build test env (SPY only) ───
    test_common = dict(
        df=test_df, predictor=predictor,
        test_start=test_start, test_end=test_end,
        reward_type=args.reward, transaction_cost=args.tx_cost,
        random_start=False,
    )

    test_env = CachedTradingEnv(**test_common)
    test_cache = get_cache_path(TEST_TICKER, fold_num, "test")
    ensure_cache(test_env, test_cache)

    test_env_nf = NoForecastEnv(**test_common)

    if args.gated:
        test_env_gated = CachedGatedTradingEnv(
            **test_common, forecast_cost=args.forecast_cost)
        g_test_cache = get_cache_path(TEST_TICKER, fold_num, "test", gated=True)
        ensure_cache(test_env_gated, g_test_cache)

    if args.soft_gated:
        test_env_soft = CachedSoftGatedTradingEnv(**test_common)
        sg_test_cache = get_cache_path(TEST_TICKER, fold_num, "test", gated=False)
        ensure_cache(test_env_soft, sg_test_cache)

    test_steps = len(test_env.step_indices)
    print(f"  Test steps ({TEST_TICKER}): {test_steps}")

    if args.precompute_only:
        return None

    # ─── Agent 1: Buy & Hold ───
    print(f"  [1] Buy & Hold...")
    bh_values, bh_dates = evaluate_buy_and_hold(test_df, test_start, test_end)
    bh_metrics = compute_fold_metrics(bh_values)
    fold_results["agents"]["Buy & Hold"] = {
        "metrics": bh_metrics, "dates": bh_dates,
        "portfolio_values": bh_values,
    }
    print(f"      Return: {bh_metrics['return']:+.1f}%")

    # ─── Agent 2: Rule-Based ───
    print(f"  [2] Rule-Based...")
    rule_state = evaluate_rule_based(test_df, predictor, test_start, test_end)
    rule_metrics = compute_fold_metrics(rule_state.portfolio_values)
    fold_results["agents"]["Rule-Based"] = {
        "metrics": rule_metrics,
        "dates": [str(d.date()) for d in rule_state.dates],
        "portfolio_values": rule_state.portfolio_values,
    }
    print(f"      Return: {rule_metrics['return']:+.1f}%")

    # ─── Agent 3: RL (no forecast) — multi-seed ───
    print(f"  [3] RL (no forecast) x{args.seeds} seeds...")
    nf_seed_results = []
    for seed in range(args.seeds):
        if len(train_tickers) > 1:
            multi_env = MultiTickerNoForecastEnv(train_envs_nf)
        else:
            multi_env = train_envs_nf[train_tickers[0]]

        nf_model = train_ppo(multi_env, args.timesteps, args.lr, seed=42 + seed)
        nf_res = evaluate_rl(nf_model, test_env_nf)
        nf_metrics = compute_fold_metrics(nf_res["portfolio_values"])
        nf_seed_results.append({"metrics": nf_metrics, **nf_res})
        print(f"      Seed {seed}: {nf_metrics['return']:+.1f}%")

    nf_agg = aggregate_seeds(nf_seed_results)
    fold_results["agents"]["RL (no forecast)"] = nf_agg
    print(f"      Mean: {nf_agg['metrics']['return_mean']:+.1f}% "
          f"+/- {nf_agg['metrics']['return_std']:.1f}%")

    # ─── Agent 4: RL (always forecast) — multi-seed ───
    print(f"  [4] RL (always forecast) x{args.seeds} seeds...")
    af_seed_results = []
    for seed in range(args.seeds):
        if len(train_tickers) > 1:
            multi_env = MultiTickerTrainEnv(train_envs)
        else:
            multi_env = train_envs[train_tickers[0]]

        af_model = train_ppo(multi_env, args.timesteps, args.lr, seed=42 + seed)
        af_res = evaluate_rl(af_model, test_env)
        af_metrics = compute_fold_metrics(af_res["portfolio_values"])
        af_seed_results.append({"metrics": af_metrics, **af_res})
        print(f"      Seed {seed}: {af_metrics['return']:+.1f}%")

    af_agg = aggregate_seeds(af_seed_results)
    fold_results["agents"]["RL (always forecast)"] = af_agg
    print(f"      Mean: {af_agg['metrics']['return_mean']:+.1f}% "
          f"+/- {af_agg['metrics']['return_std']:.1f}%")

    # ─── Agent 5: RL (gated forecast) — optional ───
    if args.gated:
        print(f"  [5] RL (gated forecast) x{args.seeds} seeds...")
        gated_seed_results = []
        for seed in range(args.seeds):
            if len(train_tickers) > 1:
                multi_env = MultiTickerGatedTrainEnv(train_envs_gated)
            else:
                multi_env = train_envs_gated[train_tickers[0]]

            gated_model = train_ppo(
                multi_env, args.timesteps, args.lr, seed=42 + seed)
            gated_res = evaluate_rl(
                gated_model, test_env_gated,
                action_names=CachedGatedTradingEnv.ACTION_NAMES)
            gated_metrics = compute_fold_metrics(gated_res["portfolio_values"])
            gated_seed_results.append({"metrics": gated_metrics, **gated_res})
            print(f"      Seed {seed}: {gated_metrics['return']:+.1f}% "
                  f"(fcst rate: {gated_res['forecast_rate']:.0%})")

        gated_agg = aggregate_seeds(gated_seed_results)
        fold_results["agents"]["RL (gated forecast)"] = gated_agg
        print(f"      Mean: {gated_agg['metrics']['return_mean']:+.1f}% "
              f"+/- {gated_agg['metrics']['return_std']:.1f}%")

    # ─── Agent 6: RL (soft gated forecast) — optional ───
    if args.soft_gated:
        agent_num = 6 if args.gated else 5
        print(f"  [{agent_num}] RL (soft gated forecast) x{args.seeds} seeds...")
        soft_seed_results = []
        for seed in range(args.seeds):
            if len(train_tickers) > 1:
                multi_env = MultiTickerSoftGatedTrainEnv(train_envs_soft)
            else:
                multi_env = train_envs_soft[train_tickers[0]]

            soft_model = train_ppo(
                multi_env, args.timesteps, args.lr, seed=42 + seed)
            soft_res = evaluate_rl(soft_model, test_env_soft)
            soft_metrics = compute_fold_metrics(soft_res["portfolio_values"])
            soft_seed_results.append({"metrics": soft_metrics, **soft_res})
            print(f"      Seed {seed}: {soft_metrics['return']:+.1f}%")

        soft_agg = aggregate_seeds(soft_seed_results)
        fold_results["agents"]["RL (soft gated)"] = soft_agg
        print(f"      Mean: {soft_agg['metrics']['return_mean']:+.1f}% "
              f"+/- {soft_agg['metrics']['return_std']:.1f}%")

    return fold_results


# ─────────────────────────────────────────────
# CROSS-FOLD AGGREGATION
# ─────────────────────────────────────────────

def aggregate_folds(all_fold_results):
    """Compute cross-fold summary statistics."""
    agent_names = set()
    for fold in all_fold_results:
        agent_names.update(fold["agents"].keys())

    summary = {}
    for agent in sorted(agent_names):
        fold_returns = []
        fold_sharpes = []
        fold_max_dds = []
        fold_forecast_rates = []
        per_fold = []

        for fold in all_fold_results:
            if agent not in fold["agents"]:
                continue
            a = fold["agents"][agent]
            m = a["metrics"]
            ret = m.get("return_mean", m.get("return", 0.0))
            sha = m.get("sharpe_mean", m.get("sharpe", 0.0))
            mdd = m.get("max_dd_mean", m.get("max_dd", 0.0))
            fold_returns.append(ret)
            fold_sharpes.append(sha)
            fold_max_dds.append(mdd)
            if "forecast_rate_mean" in a:
                fold_forecast_rates.append(a["forecast_rate_mean"])
            per_fold.append({
                "fold": fold["fold"], "regime": fold["regime"],
                "return": ret, "sharpe": sha, "max_dd": mdd,
            })

        summary[agent] = {
            "return_mean": round(float(np.mean(fold_returns)), 2),
            "return_std": round(float(np.std(fold_returns)), 2),
            "sharpe_mean": round(float(np.mean(fold_sharpes)), 3),
            "sharpe_std": round(float(np.std(fold_sharpes)), 3),
            "max_dd_mean": round(float(np.mean(fold_max_dds)), 2),
            "max_dd_std": round(float(np.std(fold_max_dds)), 2),
            "win_rate_vs_bh": 0.0,
            "n_folds": len(fold_returns),
            "per_fold": per_fold,
        }
        if fold_forecast_rates:
            summary[agent]["forecast_rate_mean"] = round(
                float(np.mean(fold_forecast_rates)), 3)

    # Win rate vs buy & hold
    if "Buy & Hold" in summary:
        bh_folds = {pf["fold"]: pf["return"]
                    for pf in summary["Buy & Hold"]["per_fold"]}
        for agent in summary:
            if agent == "Buy & Hold":
                continue
            wins = sum(1 for pf in summary[agent]["per_fold"]
                       if pf["fold"] in bh_folds
                       and pf["return"] > bh_folds[pf["fold"]])
            total = sum(1 for pf in summary[agent]["per_fold"]
                        if pf["fold"] in bh_folds)
            summary[agent]["win_rate_vs_bh"] = round(
                wins / max(1, total) * 100, 1)

    return summary


# ─────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────

def plot_results(all_fold_results, summary, train_tickers, args):
    """Generate walk-forward CV visualizations."""
    n_folds = len(all_fold_results)
    agent_names = list(summary.keys())

    colors = {
        "Buy & Hold": "#9E9E9E",
        "Rule-Based": "#FF9800",
        "RL (no forecast)": "#E91E63",
        "RL (always forecast)": "#2196F3",
        "RL (gated forecast)": "#4CAF50",
        "RL (soft gated)": "#9C27B0",
    }

    ticker_label = (f"Multi-Ticker ({', '.join(train_tickers)})"
                    if len(train_tickers) > 1 else f"Single-Ticker ({train_tickers[0]})")

    # ─── Figure 1: Per-fold returns + summary ───
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    fig.suptitle(f"Walk-Forward CV: 5yr Train / 1yr Test — {ticker_label}",
                 fontsize=14, fontweight="bold")

    # Panel 1: Per-fold bar chart
    ax1 = axes[0]
    x = np.arange(n_folds)
    width = 0.8 / len(agent_names)
    for i, agent in enumerate(agent_names):
        returns = []
        stds = []
        for fold in all_fold_results:
            if agent in fold["agents"]:
                m = fold["agents"][agent]["metrics"]
                returns.append(m.get("return_mean", m.get("return", 0.0)))
                stds.append(m.get("return_std", 0.0))
            else:
                returns.append(0.0)
                stds.append(0.0)

        offset = (i - len(agent_names) / 2 + 0.5) * width
        ax1.bar(x + offset, returns, width, label=agent,
                color=colors.get(agent, "#666"), alpha=0.85,
                yerr=stds if max(stds) > 0 else None,
                capsize=3, error_kw={"linewidth": 1})

    fold_labels = [f"Fold {f['fold']}\n{f['test_start'][:4]}\n({f['regime']})"
                   for f in all_fold_results]
    ax1.set_xticks(x)
    ax1.set_xticklabels(fold_labels, fontsize=8)
    ax1.set_ylabel("Test Return (%)")
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.legend(fontsize=7, loc="upper left")
    ax1.set_title("Per-Fold Test Returns (error bars = std across seeds)")
    ax1.grid(True, alpha=0.2, axis="y")

    # Panel 2: Summary bar chart
    ax2 = axes[1]
    metric_keys = ["return_mean", "sharpe_mean", "max_dd_mean"]
    metric_labels = ["Mean Return (%)", "Mean Sharpe", "Mean Max DD (%)"]
    x2 = np.arange(len(metric_keys))
    width2 = 0.8 / len(agent_names)

    for i, agent in enumerate(agent_names):
        s = summary[agent]
        values = [s[k] for k in metric_keys]
        offset = (i - len(agent_names) / 2 + 0.5) * width2
        ax2.bar(x2 + offset, values, width2, label=agent,
                color=colors.get(agent, "#666"), alpha=0.85)

    ax2.set_xticks(x2)
    ax2.set_xticklabels(metric_labels)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.legend(fontsize=7, loc="upper left")
    ax2.set_title(f"Aggregate Metrics Across {n_folds} Folds")
    ax2.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    out = "walkforward_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nChart saved -> {out}")
    plt.close()

    # ─── Figure 2: Per-fold portfolio curves ───
    n_cols = 3
    n_rows = (n_folds + n_cols - 1) // n_cols
    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    fig2.suptitle(f"Portfolio Curves Per Fold — {ticker_label}",
                  fontsize=14, fontweight="bold")
    if n_rows == 1:
        axes2 = [axes2]  # normalize to 2D

    for idx, fold in enumerate(all_fold_results):
        row, col = divmod(idx, n_cols)
        ax = axes2[row][col]

        for agent_name in agent_names:
            if agent_name not in fold["agents"]:
                continue
            a = fold["agents"][agent_name]
            dates = pd.to_datetime(a.get("dates", []))
            pv = a.get("portfolio_values", [])
            if len(dates) > 0 and len(pv) > 0:
                ax.plot(dates[:len(pv)], pv[:len(dates)],
                        label=agent_name, color=colors.get(agent_name, "#666"),
                        linewidth=1.5,
                        linestyle="--" if "Hold" in agent_name else "-")

        ax.axhline(INITIAL_CASH, color="gray", linestyle=":", alpha=0.4)
        ax.set_title(f"Fold {fold['fold']}: {fold['test_start'][:4]} "
                     f"({fold['regime']})", fontsize=10)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.grid(True, alpha=0.2)
        if idx == 0:
            ax.legend(fontsize=6)

    # Hide empty subplots
    for idx in range(n_folds, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes2[row][col].set_visible(False)

    plt.tight_layout()
    out2 = "walkforward_portfolios.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Chart saved -> {out2}")
    plt.close()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    args = parse_args()

    # Determine ticker pool
    if args.tickers:
        train_tickers = [t.strip().upper() for t in args.tickers.split(",")]
    elif args.single_ticker:
        train_tickers = [TEST_TICKER]
    else:
        train_tickers = DEFAULT_TICKERS

    # Ensure test ticker is in the pool
    if TEST_TICKER not in train_tickers:
        train_tickers.append(TEST_TICKER)

    # Select folds
    if args.folds:
        selected = [int(f.strip()) for f in args.folds.split(",")]
        folds_to_run = [f for f in FOLDS if f["fold"] in selected]
    else:
        folds_to_run = FOLDS

    print("=" * 70)
    print("  WALK-FORWARD CV + MULTI-TICKER RL TRAINING")
    print(f"  {len(folds_to_run)} folds | {args.timesteps:,} timesteps | "
          f"{args.seeds} seeds")
    print(f"  Train tickers: {', '.join(train_tickers)} "
          f"({len(train_tickers)} tickers)")
    print(f"  Test ticker: {TEST_TICKER}")
    print(f"  Reward: {args.reward} | Tx cost: {args.tx_cost}")
    if args.gated:
        print(f"  Gated forecast: ON (cost={args.forecast_cost})")
    if args.soft_gated:
        print(f"  Soft gated forecast: ON (always forecast + confidence score)")
    print("=" * 70)

    # 1. Load all ticker data
    ticker_data = load_all_tickers(train_tickers)

    # 2. Load Kronos
    predictor = load_kronos()

    # 3. Run each fold
    all_fold_results = []
    for fold_config in folds_to_run:
        result = run_fold(fold_config, ticker_data, predictor, args,
                          train_tickers)
        if result is not None:
            all_fold_results.append(result)

    if args.precompute_only:
        print(f"\nAll forecasts cached for {len(train_tickers)} tickers "
              f"x {len(folds_to_run)} folds.")
        print("Run again without --precompute-only to train.")
        return

    if not all_fold_results:
        print("\nNo folds completed.")
        return

    # 4. Aggregate
    summary = aggregate_folds(all_fold_results)

    # 5. Print results
    print(f"\n{'='*100}")
    print(f"  WALK-FORWARD RESULTS — "
          f"{'Multi-Ticker' if len(train_tickers) > 1 else 'Single-Ticker'} "
          f"({', '.join(train_tickers)})")
    print(f"{'='*100}")

    header = (f"  {'Agent':<25} {'Return':>16} {'Sharpe':>16} "
              f"{'Max DD':>16} {'Win vs B&H':>12} {'Folds':>6}")
    print(header)
    print("  " + "-" * 95)

    for agent in ["Buy & Hold", "Rule-Based", "RL (no forecast)",
                  "RL (always forecast)", "RL (gated forecast)",
                  "RL (soft gated)"]:
        if agent not in summary:
            continue
        s = summary[agent]
        ret_str = f"{s['return_mean']:+.1f} +/- {s['return_std']:.1f}%"
        sha_str = f"{s['sharpe_mean']:.2f} +/- {s['sharpe_std']:.2f}"
        mdd_str = f"{s['max_dd_mean']:.1f} +/- {s['max_dd_std']:.1f}%"
        win_str = f"{s['win_rate_vs_bh']:.0f}%" if agent != "Buy & Hold" else "--"
        print(f"  {agent:<25} {ret_str:>16} {sha_str:>16} "
              f"{mdd_str:>16} {win_str:>12} {s['n_folds']:>6}")

    print(f"{'='*100}")

    # Per-fold breakdown
    print(f"\n  PER-FOLD BREAKDOWN (test returns):")
    agents_for_table = [a for a in ["Buy & Hold", "RL (no forecast)",
                                     "RL (always forecast)", "RL (gated forecast)",
                                     "RL (soft gated)"]
                        if a in summary]
    header2 = f"  {'Fold':<6} {'Regime':<25}"
    for a in agents_for_table:
        short = (a.replace("RL (", "").replace(")", "")
                  .replace("Buy & Hold", "B&H"))
        header2 += f" {short:>16}"
    print(header2)
    print("  " + "-" * (30 + 17 * len(agents_for_table)))

    for fold in all_fold_results:
        line = f"  {fold['fold']:<6} {fold['regime']:<25}"
        for agent in agents_for_table:
            if agent in fold["agents"]:
                m = fold["agents"][agent]["metrics"]
                ret = m.get("return_mean", m.get("return", 0.0))
                line += f" {ret:>+15.1f}%"
            else:
                line += f" {'--':>16}"
        print(line)

    # 6. Save results
    save_data = {
        "config": {
            "train_tickers": train_tickers,
            "test_ticker": TEST_TICKER,
            "timesteps": args.timesteps,
            "seeds": args.seeds,
            "reward": args.reward,
            "tx_cost": args.tx_cost,
            "gated": args.gated,
            "soft_gated": args.soft_gated,
            "forecast_cost": args.forecast_cost,
            "n_folds": len(all_fold_results),
        },
        "summary": summary,
        "folds": [],
    }
    for fold in all_fold_results:
        fold_save = {
            "fold": fold["fold"], "regime": fold["regime"],
            "agents": {},
        }
        for agent_name, agent_data in fold["agents"].items():
            fold_save["agents"][agent_name] = {
                "metrics": agent_data["metrics"],
                "forecast_rate_mean": agent_data.get("forecast_rate_mean", None),
            }
        save_data["folds"].append(fold_save)

    out_json = "walkforward_results.json"
    with open(out_json, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved -> {out_json}")

    # 7. Plot
    plot_results(all_fold_results, summary, train_tickers, args)

    print("\nDone!")


if __name__ == "__main__":
    main()
