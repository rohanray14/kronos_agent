"""
Training script for the V3 Position Sizing RL agent.

Per fold:
  1. Load ticker data + macro features (reuse v2 infrastructure)
  2. Generate Claude's decisions (replay from logs or rule-based proxy)
  3. Generate regime labels
  4. Build PositionSizingEnv per ticker
  5. Train PPO sizer on multi-ticker env
  6. Evaluate: compare "Claude + sizer" vs "Claude + all-in/all-out"

Usage:
    python -m v3.train_sizing                          # all folds
    python -m v3.train_sizing --folds 4                # single fold
    python -m v3.train_sizing --timesteps 200000       # more training
    python -m v3.train_sizing --use-logs               # replay Claude reasoning logs
    python -m v3.train_sizing --all-in-baseline        # also eval all-in/all-out
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import os
import json
import argparse
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Kronos"))

import numpy as np
import pandas as pd

from v3.config import (
    FOLDS, DEFAULT_TICKERS, TEST_TICKER, MACRO_TICKERS,
    INITIAL_CASH, FORECAST_STEPS, LOOKBACK_DAYS,
    SIZING_OBS_DIM, N_SIZING_ACTIONS, SIZING_LEVELS,
    DEFAULT_TIMESTEPS, PPO_LR, PPO_ENT_COEF,
    MODEL_DIR, CACHE_DIR, MACRO_CACHE_DIR,
    DATA_START, DATA_END,
)
from v3.env_position_sizing import PositionSizingEnv, MultiTickerSizingEnv, ACTION_MAP

from v2.features import MacroFeatureProvider, download_macro_data
from v2.regime import label_regimes_rule_based


def parse_args():
    parser = argparse.ArgumentParser(
        description="V3 Position Sizing RL Training")
    parser.add_argument("--folds", type=str, default=None,
                        help="Comma-separated fold numbers (default: all)")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--tx-cost", type=float, default=0.001)
    parser.add_argument("--use-logs", action="store_true",
                        help="Replay Claude reasoning logs instead of rule-based proxy")
    parser.add_argument("--log-dir", type=str, default="outputs/reasoning_logs",
                        help="Directory with Claude reasoning log JSON files")
    parser.add_argument("--single-ticker", action="store_true",
                        help="Train on SPY only")
    parser.add_argument("--tickers", type=str, default=None)
    parser.add_argument("--all-in-baseline", action="store_true",
                        help="Also evaluate all-in/all-out as baseline")
    return parser.parse_args()


# ─────────────────────────────────────────────
# DATA LOADING (reuse v2 loaders)
# ─────────────────────────────────────────────

def load_ticker_data(ticker: str, max_retries: int = 3) -> pd.DataFrame:
    import yfinance as yf
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
            return df
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))
            else:
                raise RuntimeError(f"Failed to download {ticker}: {e}")


def load_all_tickers(tickers: list[str]) -> dict[str, pd.DataFrame]:
    print(f"Loading {len(tickers)} tickers: {', '.join(tickers)}...")
    ticker_data = {}
    for i, ticker in enumerate(tickers):
        df = load_ticker_data(ticker)
        ticker_data[ticker] = df
        print(f"   {ticker}: {len(df)} days")
        if i < len(tickers) - 1:
            time.sleep(1)
    return ticker_data


# ─────────────────────────────────────────────
# KRONOS CACHE
# ─────────────────────────────────────────────

def get_kronos_cache_path(ticker: str, fold_num: int, split: str) -> str:
    return os.path.join(CACHE_DIR, f"{ticker}_fold{fold_num}_{split}.json")


def load_kronos_cache(path: str) -> dict[int, tuple]:
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    return {int(k): tuple(v) for k, v in data.items()}


# ─────────────────────────────────────────────
# CLAUDE ACTION GENERATION
# ─────────────────────────────────────────────

def generate_rule_based_actions(
    df: pd.DataFrame,
    predictor,
    step_indices: list[int],
) -> np.ndarray:
    """
    Generate Claude-proxy actions using the rule-based agent.
    Returns array of -1 (SELL), 0 (HOLD), 1 (BUY) per step.
    """
    from agent import decide, get_forecast

    actions = np.zeros(len(step_indices), dtype=np.float32)

    for i, idx in enumerate(step_indices):
        if idx < LOOKBACK_DAYS:
            actions[i] = 0.0  # HOLD
            continue

        current_price = float(df["close"].iloc[idx])
        history_df = df.iloc[max(0, idx - LOOKBACK_DAYS):idx]
        current_date = df.index[idx]

        future_dates = pd.bdate_range(
            start=current_date, periods=FORECAST_STEPS + 1, freq="B"
        )[1:]

        try:
            forecast = get_forecast(predictor, history_df, future_dates)
            action_str = decide(forecast, current_price, history_df)
            actions[i] = ACTION_MAP[action_str]
        except Exception:
            actions[i] = 0.0  # HOLD on error

    return actions


def load_claude_actions_from_logs(
    log_path: str,
    df: pd.DataFrame,
    step_indices: list[int],
) -> np.ndarray:
    """
    Load Claude's actual decisions from reasoning log JSON.
    Maps log entries to step indices by date matching.
    """
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Reasoning log not found: {log_path}")

    with open(log_path, "r") as f:
        log = json.load(f)

    # Build date -> action map from log
    date_to_action = {}
    for entry in log:
        date = entry.get("date", "")
        action = entry.get("action", "HOLD").upper()
        date_to_action[date] = ACTION_MAP.get(action, 0.0)

    # Map to step indices
    actions = np.zeros(len(step_indices), dtype=np.float32)
    dates = df.index
    for i, idx in enumerate(step_indices):
        date_str = str(dates[idx].date())
        actions[i] = date_to_action.get(date_str, 0.0)

    matched = sum(1 for a in actions if a != 0.0)
    print(f"      Matched {matched}/{len(actions)} actions from reasoning log")

    return actions


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

def compute_fold_metrics(portfolio_values: list[float]) -> dict:
    pv = np.array(portfolio_values)
    if len(pv) < 2:
        return {"return": 0.0, "sharpe": 0.0, "max_dd": 0.0,
                "calmar": 0.0, "final": INITIAL_CASH}

    total_return = (pv[-1] - INITIAL_CASH) / INITIAL_CASH * 100
    returns = np.diff(pv) / pv[:-1]
    sharpe = ((returns.mean() / returns.std()) * np.sqrt(252 / FORECAST_STEPS)
              if returns.std() > 0 else 0.0)
    peak = np.maximum.accumulate(pv)
    max_dd = float(((pv - peak) / peak).min()) * 100

    # Calmar ratio: annualized return / max drawdown
    calmar = abs(total_return / max_dd) if max_dd != 0 else 0.0

    return {
        "return": round(total_return, 2),
        "sharpe": round(sharpe, 3),
        "max_dd": round(max_dd, 2),
        "calmar": round(calmar, 2),
        "final": round(float(pv[-1]), 2),
    }


# ─────────────────────────────────────────────
# ALL-IN/ALL-OUT BASELINE
# ─────────────────────────────────────────────

def run_all_in_baseline(
    df: pd.DataFrame,
    step_indices: list[int],
    claude_actions: np.ndarray,
    tx_cost: float = 0.001,
) -> list[float]:
    """
    Run Claude's actions with all-in/all-out execution (current behavior).
    Returns portfolio value series for comparison.
    """
    cash = INITIAL_CASH
    shares = 0.0
    portfolio_values = [INITIAL_CASH]

    for i, idx in enumerate(step_indices):
        price = float(df["close"].iloc[idx])
        action = claude_actions[i]

        if action == 1.0 and cash > 0:  # BUY
            cost = cash * tx_cost
            shares = (cash - cost) / price
            cash = 0.0
        elif action == -1.0 and shares > 0:  # SELL
            proceeds = shares * price
            cost = proceeds * tx_cost
            cash = proceeds - cost
            shares = 0.0

        pv = cash + shares * price
        portfolio_values.append(pv)

    return portfolio_values


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate_sizer(
    model,
    env: PositionSizingEnv,
    n_seeds: int = 3,
) -> tuple[dict, list[float], list[dict]]:
    """
    Evaluate trained sizer on a test environment.
    Returns (metrics, portfolio_values, step_log).
    """
    all_pvs = []
    all_logs = []

    for seed in range(n_seeds):
        obs, info = env.reset(seed=seed)
        pvs = [INITIAL_CASH]
        step_log = []
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if "portfolio_value" in info:
                pvs.append(info["portfolio_value"])
                step_log.append({
                    "date": info.get("date", ""),
                    "claude_action": info.get("claude_action", ""),
                    "sizing": info.get("sizing_action", 1.0),
                    "position_frac": round(info.get("position_frac", 0.0), 3),
                    "portfolio_value": round(info["portfolio_value"], 2),
                })

        all_pvs.append(pvs)
        all_logs.append(step_log)

    # Use median seed for metrics
    lengths = [len(pv) for pv in all_pvs]
    median_idx = np.argsort([pv[-1] for pv in all_pvs])[len(all_pvs) // 2]
    best_pvs = all_pvs[median_idx]

    metrics = compute_fold_metrics(best_pvs)
    return metrics, best_pvs, all_logs[median_idx]


# ─────────────────────────────────────────────
# FOLD RUNNER
# ─────────────────────────────────────────────

def run_fold(fold_config, ticker_data, macro_data, predictor, args, train_tickers):
    fold_num = fold_config["fold"]
    train_start = fold_config["train_start"]
    train_end = fold_config["train_end"]
    test_start = fold_config["test_start"]
    test_end = fold_config["test_end"]
    regime_name = fold_config["regime"]

    print(f"\n{'='*60}")
    print(f"  FOLD {fold_num}: {regime_name}")
    print(f"  Train: {train_start} -> {train_end}")
    print(f"  Test:  {test_start} -> {test_end}")
    print(f"{'='*60}")

    results = {
        "fold": fold_num,
        "regime": regime_name,
        "test_start": test_start,
        "test_end": test_end,
    }

    # ── Build feature providers and generate actions ──
    print("\n   Building feature providers and generating actions...")

    train_envs = {}
    test_env = None

    for ticker in train_tickers:
        df = ticker_data[ticker]

        # Load Kronos cache
        cache_path = get_kronos_cache_path(ticker, fold_num, "train")
        kronos_cache = load_kronos_cache(cache_path)

        # Build feature provider
        provider = MacroFeatureProvider(df, macro_data, kronos_cache)

        # Try loading cached macro features
        macro_cache_path = os.path.join(
            MACRO_CACHE_DIR, f"{ticker}_fold{fold_num}_macro.json")
        if os.path.exists(macro_cache_path):
            provider.load_cache(macro_cache_path)

        # Generate regime labels (over all indices in the dataframe)
        all_indices = list(range(len(df)))
        regime_labels = label_regimes_rule_based(provider, all_indices)

        # Compute step indices matching the env's logic exactly
        dates = df.index
        train_indices = [i for i, d in enumerate(dates)
                         if d >= pd.Timestamp(train_start) and d <= pd.Timestamp(train_end)]
        train_indices = [i for i in train_indices if i >= LOOKBACK_DAYS]
        train_step_indices = train_indices[::FORECAST_STEPS]

        # Generate Claude proxy actions
        print(f"      {ticker}: generating rule-based actions "
              f"({len(train_step_indices)} steps)...")
        claude_actions = generate_rule_based_actions(
            df, predictor, train_step_indices)

        # Map regime labels to step indices
        step_regime_labels = np.array([
            regime_labels[idx] if idx < len(regime_labels) else 4
            for idx in train_step_indices
        ])

        # Build training env (test_start/test_end control which indices env uses)
        env = PositionSizingEnv(
            df=df,
            feature_provider=provider,
            claude_actions=claude_actions,
            regime_labels=step_regime_labels,
            test_start=train_start,
            test_end=train_end,
            transaction_cost=args.tx_cost,
            random_start=True,
        )
        train_envs[ticker] = env
        print(f"      {ticker}: {len(env.step_indices)} train steps")

    # ── Build test environment (SPY only) ──
    spy_df = ticker_data[TEST_TICKER]
    test_cache_path = get_kronos_cache_path(TEST_TICKER, fold_num, "test")
    test_kronos_cache = load_kronos_cache(test_cache_path)

    test_provider = MacroFeatureProvider(spy_df, macro_data, test_kronos_cache)
    test_macro_cache = os.path.join(
        MACRO_CACHE_DIR, f"{TEST_TICKER}_fold{fold_num}_macro.json")
    if os.path.exists(test_macro_cache):
        test_provider.load_cache(test_macro_cache)

    test_all_indices = list(range(len(spy_df)))
    test_regime_labels = label_regimes_rule_based(test_provider, test_all_indices)

    # Test step indices
    test_dates = spy_df.index
    test_idx_list = [i for i, d in enumerate(test_dates)
                     if d >= pd.Timestamp(test_start) and d <= pd.Timestamp(test_end)]
    test_idx_list = [i for i in test_idx_list if i >= LOOKBACK_DAYS]
    test_step_indices = test_idx_list[::FORECAST_STEPS]

    # Generate Claude actions for test period
    print(f"\n   Generating test actions for {TEST_TICKER} ({len(test_step_indices)} steps)...")

    if args.use_logs:
        # Try to load from reasoning logs
        log_path = os.path.join(args.log_dir, f"reasoning_log_hybrid_full.json")
        try:
            test_claude_actions = load_claude_actions_from_logs(
                log_path, spy_df, test_step_indices)
        except FileNotFoundError:
            print(f"      Log not found, falling back to rule-based")
            test_claude_actions = generate_rule_based_actions(
                spy_df, predictor, test_step_indices)
    else:
        test_claude_actions = generate_rule_based_actions(
            spy_df, predictor, test_step_indices)

    test_step_regimes = np.array([
        test_regime_labels[idx] if idx < len(test_regime_labels) else 4
        for idx in test_step_indices
    ])

    test_env = PositionSizingEnv(
        df=spy_df,
        feature_provider=test_provider,
        claude_actions=test_claude_actions,
        regime_labels=test_step_regimes,
        test_start=test_start,
        test_end=test_end,
        transaction_cost=args.tx_cost,
        random_start=False,
    )

    # ── Train ──
    print(f"\n   Training PPO sizer ({args.timesteps} timesteps)...")
    from stable_baselines3 import PPO

    multi_env = MultiTickerSizingEnv(train_envs)
    total_train_steps = len(multi_env.step_indices)
    print(f"   Multi-ticker training pool: {total_train_steps} total steps")

    model = PPO(
        "MlpPolicy",
        multi_env,
        learning_rate=PPO_LR,
        ent_coef=PPO_ENT_COEF,
        n_steps=min(2048, max(64, total_train_steps // 4)),
        batch_size=64,
        n_epochs=10,
        verbose=0,
        seed=42,
    )

    t0 = time.time()
    model.learn(total_timesteps=args.timesteps)
    train_time = time.time() - t0
    print(f"   Training complete in {train_time:.1f}s")

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"ppo_sizer_fold{fold_num}")
    model.save(model_path)
    print(f"   Model saved -> {model_path}")

    # ── Evaluate ──
    print(f"\n   Evaluating on {TEST_TICKER} test period...")

    # RL sizer
    sizer_metrics, sizer_pvs, sizer_log = evaluate_sizer(
        model, test_env, n_seeds=args.seeds)

    results["agents"] = {}
    results["agents"]["Claude + RL Sizer"] = {
        "metrics": sizer_metrics,
        "portfolio_values": [round(v, 2) for v in sizer_pvs],
    }

    print(f"\n   Claude + RL Sizer:")
    print(f"      Return: {sizer_metrics['return']:+.2f}%")
    print(f"      Sharpe: {sizer_metrics['sharpe']:.3f}")
    print(f"      Max DD: {sizer_metrics['max_dd']:.2f}%")
    print(f"      Calmar: {sizer_metrics['calmar']:.2f}")

    # All-in/all-out baseline
    baseline_pvs = run_all_in_baseline(
        spy_df, test_step_indices, test_claude_actions, args.tx_cost)
    baseline_metrics = compute_fold_metrics(baseline_pvs)

    results["agents"]["Claude + All-In/Out"] = {
        "metrics": baseline_metrics,
        "portfolio_values": [round(v, 2) for v in baseline_pvs],
    }

    print(f"\n   Claude + All-In/Out (baseline):")
    print(f"      Return: {baseline_metrics['return']:+.2f}%")
    print(f"      Sharpe: {baseline_metrics['sharpe']:.3f}")
    print(f"      Max DD: {baseline_metrics['max_dd']:.2f}%")
    print(f"      Calmar: {baseline_metrics['calmar']:.2f}")

    # Buy & hold
    bh_pvs = []
    bh_shares = INITIAL_CASH / float(spy_df["close"].iloc[test_step_indices[0]])
    for idx in test_step_indices:
        bh_pvs.append(bh_shares * float(spy_df["close"].iloc[idx]))
    bh_metrics = compute_fold_metrics(bh_pvs)

    results["agents"]["Buy & Hold"] = {
        "metrics": bh_metrics,
    }

    print(f"\n   Buy & Hold:")
    print(f"      Return: {bh_metrics['return']:+.2f}%")
    print(f"      Sharpe: {bh_metrics['sharpe']:.3f}")
    print(f"      Max DD: {bh_metrics['max_dd']:.2f}%")

    # ── Sizing analysis ──
    if sizer_log:
        sizing_dist = {}
        for entry in sizer_log:
            s = entry["sizing"]
            sizing_dist[s] = sizing_dist.get(s, 0) + 1
        print(f"\n   Sizing distribution:")
        for level in SIZING_LEVELS:
            count = sizing_dist.get(level, 0)
            pct = count / len(sizer_log) * 100
            print(f"      {level*100:5.0f}%: {count:4d} ({pct:.1f}%)")

    results["sizing_log"] = sizer_log

    return results


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 60)
    print("  V3: POSITION SIZING RL TRAINING")
    print("=" * 60)

    # Determine tickers
    if args.single_ticker:
        train_tickers = [TEST_TICKER]
    elif args.tickers:
        train_tickers = args.tickers.split(",")
    else:
        train_tickers = DEFAULT_TICKERS

    # Determine folds
    if args.folds:
        fold_nums = [int(f) for f in args.folds.split(",")]
        folds = [f for f in FOLDS if f["fold"] in fold_nums]
    else:
        folds = FOLDS

    # Load data
    all_tickers = list(set(train_tickers + [TEST_TICKER]))
    ticker_data = load_all_tickers(all_tickers)

    # Load macro data
    print("\nDownloading macro data...")
    macro_data = download_macro_data()
    print(f"Downloaded {len(macro_data)} macro tickers")

    # Load Kronos predictor
    print("\nLoading Kronos forecasting model...")
    from agent import load_kronos
    predictor = load_kronos()

    # Run folds
    all_results = []
    for fold_config in folds:
        result = run_fold(
            fold_config, ticker_data, macro_data, predictor, args, train_tickers)
        all_results.append(result)

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  SUMMARY: POSITION SIZING VS ALL-IN/ALL-OUT")
    print("=" * 60)

    print(f"\n   {'Fold':<6} {'Regime':<25} {'Metric':<10} {'RL Sizer':>10} {'All-In/Out':>12} {'Delta':>8}")
    print(f"   {'-'*6} {'-'*25} {'-'*10} {'-'*10} {'-'*12} {'-'*8}")

    for r in all_results:
        sizer = r["agents"]["Claude + RL Sizer"]["metrics"]
        baseline = r["agents"]["Claude + All-In/Out"]["metrics"]

        for metric in ["return", "sharpe", "max_dd", "calmar"]:
            s_val = sizer[metric]
            b_val = baseline[metric]
            delta = s_val - b_val

            label = r["regime"] if metric == "return" else ""
            print(f"   {r['fold']:<6} {label:<25} {metric:<10} "
                  f"{s_val:>+10.2f} {b_val:>+12.2f} {delta:>+8.2f}")
        print()

    # Save results
    os.makedirs("outputs/results", exist_ok=True)
    results_path = "outputs/results/v3_sizing_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"   Results saved -> {results_path}")


if __name__ == "__main__":
    main()
