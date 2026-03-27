"""
Training script for the V3 Signal Quality Scorer.

Per fold:
  1. Load ticker data + macro features + Kronos forecast caches
  2. Build SignalScorerEnv per ticker (no portfolio, just forecast eval)
  3. Train PPO scorer on multi-ticker env
  4. Evaluate: compare learned score vs volatility heuristic on calibration

The scorer learns when Kronos forecasts are reliable vs noise.
Claude then uses this score to weight the forecast in its reasoning.

Usage:
    python -m v3.train_scorer                          # all folds
    python -m v3.train_scorer --folds 4                # single fold
    python -m v3.train_scorer --timesteps 200000       # more training
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
    FORECAST_STEPS, LOOKBACK_DAYS,
    DEFAULT_TIMESTEPS, PPO_LR,
    MODEL_DIR, CACHE_DIR, MACRO_CACHE_DIR,
    DATA_START, DATA_END,
)
from v3.env_signal_scorer import (
    SignalScorerEnv, MultiTickerScorerEnv,
    SCORE_LEVELS, N_SCORE_LEVELS,
)
from v2.features import MacroFeatureProvider, download_macro_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="V3 Signal Quality Scorer Training")
    parser.add_argument("--folds", type=str, default=None)
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--single-ticker", action="store_true")
    parser.add_argument("--tickers", type=str, default=None)
    return parser.parse_args()


# ─────────────────────────────────────────────
# DATA LOADING
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


def get_kronos_cache_path(ticker: str, fold_num: int, split: str) -> str:
    return os.path.join(CACHE_DIR, f"{ticker}_fold{fold_num}_{split}.json")


def load_kronos_cache(path: str) -> dict[int, tuple]:
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    return {int(k): tuple(v) for k, v in data.items()}


# ─────────────────────────────────────────────
# HEURISTIC BASELINE
# ─────────────────────────────────────────────

def heuristic_score(vol_20d: float, vol_ceiling: float = 0.02) -> float:
    """The current heuristic: 1.0 - vol/ceiling, clamped to [0, 1]."""
    return max(0.0, min(1.0, 1.0 - (vol_20d / vol_ceiling)))


def evaluate_heuristic(
    df: pd.DataFrame,
    feature_provider,
    kronos_cache: dict[int, tuple],
    step_indices: list[int],
    forecast_steps: int = FORECAST_STEPS,
) -> dict:
    """Evaluate the volatility heuristic on calibration metrics."""
    scores = []
    actuals = []
    prices = df["close"].values

    for idx in step_indices:
        if idx not in kronos_cache:
            continue
        k_ret, _, _ = kronos_cache[idx]
        future_idx = min(idx + forecast_steps, len(prices) - 1)
        realized = (float(prices[future_idx]) - float(prices[idx])) / float(prices[idx])

        kronos_dir = 1 if k_ret > 0 else (-1 if k_ret < 0 else 0)
        actual_dir = 1 if realized > 0 else (-1 if realized < 0 else 0)
        correct = (kronos_dir == actual_dir) and kronos_dir != 0

        vol = float(feature_provider._vol_20d[idx])
        score = heuristic_score(vol)

        scores.append(score)
        actuals.append(1.0 if correct else 0.0)

    return compute_calibration_metrics(scores, actuals)


def evaluate_scorer(
    model,
    env: SignalScorerEnv,
    n_seeds: int = 3,
) -> tuple[dict, list[dict]]:
    """Evaluate trained scorer on calibration metrics."""
    all_scores = []
    all_actuals = []
    all_logs = []

    for seed in range(n_seeds):
        obs, info = env.reset(seed=seed)
        step_log = []
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if "score" in info:
                all_scores.append(info["score"])
                all_actuals.append(1.0 if info["direction_correct"] else 0.0)
                step_log.append({
                    "date": info.get("date", ""),
                    "score": round(info["score"], 2),
                    "kronos_ret": round(info.get("kronos_ret", 0), 4),
                    "realized_ret": round(info.get("realized_ret", 0), 4),
                    "correct": info["direction_correct"],
                    "reward": round(info["reward"], 4),
                })

        all_logs.append(step_log)

    metrics = compute_calibration_metrics(all_scores, all_actuals)
    # Use first seed's log
    return metrics, all_logs[0] if all_logs else []


def compute_calibration_metrics(scores: list, actuals: list) -> dict:
    """Compute calibration quality metrics."""
    if not scores:
        return {"mae": 1.0, "accuracy": 0.0, "brier": 1.0,
                "avg_score": 0.5, "avg_actual": 0.5, "n": 0}

    scores = np.array(scores)
    actuals = np.array(actuals)

    # Mean absolute error between score and actual (0 = perfect calibration)
    mae = float(np.mean(np.abs(scores - actuals)))

    # Brier score (lower = better calibrated probability)
    brier = float(np.mean((scores - actuals) ** 2))

    # Direction accuracy of the scorer's recommendation
    # If score > 0.5 and correct, or score < 0.5 and incorrect = good
    high_conf = scores > 0.5
    scorer_accuracy = float(np.mean(
        (high_conf & (actuals == 1.0)) | (~high_conf & (actuals == 0.0))
    ))

    # Calibration by bin (split scores into 5 bins, check avg actual in each)
    bins = {}
    for s, a in zip(scores, actuals):
        bin_key = round(s * 5) / 5  # bin to nearest 0.2
        bins.setdefault(bin_key, []).append(a)

    calibration_table = {}
    for k in sorted(bins.keys()):
        calibration_table[f"{k:.1f}"] = {
            "predicted": round(k, 2),
            "actual": round(float(np.mean(bins[k])), 3),
            "count": len(bins[k]),
        }

    return {
        "mae": round(mae, 4),
        "brier": round(brier, 4),
        "scorer_accuracy": round(scorer_accuracy, 4),
        "avg_score": round(float(np.mean(scores)), 3),
        "avg_actual_accuracy": round(float(np.mean(actuals)), 3),
        "calibration_table": calibration_table,
        "n": len(scores),
    }


# ─────────────────────────────────────────────
# FOLD RUNNER
# ─────────────────────────────────────────────

def run_fold(fold_config, ticker_data, macro_data, args, train_tickers):
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

    # ── Build training environments ──
    print("\n   Building scorer environments...")
    train_envs = {}

    for ticker in train_tickers:
        df = ticker_data[ticker]

        # Load Kronos cache
        cache_path = get_kronos_cache_path(ticker, fold_num, "train")
        kronos_cache = load_kronos_cache(cache_path)

        if not kronos_cache:
            print(f"      {ticker}: no Kronos cache, skipping")
            continue

        # Build feature provider
        provider = MacroFeatureProvider(df, macro_data, kronos_cache)

        macro_cache_path = os.path.join(
            MACRO_CACHE_DIR, f"{ticker}_fold{fold_num}_macro.json")
        if os.path.exists(macro_cache_path):
            provider.load_cache(macro_cache_path)

        env = SignalScorerEnv(
            df=df,
            feature_provider=provider,
            kronos_cache=kronos_cache,
            test_start=train_start,
            test_end=train_end,
            random_start=True,
        )

        if len(env.step_indices) > 10:
            train_envs[ticker] = env
            print(f"      {ticker}: {len(env.step_indices)} train steps "
                  f"({len(kronos_cache)} cached forecasts)")
        else:
            print(f"      {ticker}: too few steps ({len(env.step_indices)}), skipping")

    if not train_envs:
        print("   ERROR: No training environments created. Missing Kronos caches?")
        return results

    # ── Build test environment (SPY) ──
    spy_df = ticker_data[TEST_TICKER]
    test_cache_path = get_kronos_cache_path(TEST_TICKER, fold_num, "test")
    test_kronos_cache = load_kronos_cache(test_cache_path)

    if not test_kronos_cache:
        # Fall back to train cache (overlapping indices)
        train_cache_path = get_kronos_cache_path(TEST_TICKER, fold_num, "train")
        test_kronos_cache = load_kronos_cache(train_cache_path)

    test_provider = MacroFeatureProvider(spy_df, macro_data, test_kronos_cache)
    test_macro_cache = os.path.join(
        MACRO_CACHE_DIR, f"{TEST_TICKER}_fold{fold_num}_macro.json")
    if os.path.exists(test_macro_cache):
        test_provider.load_cache(test_macro_cache)

    test_env = SignalScorerEnv(
        df=spy_df,
        feature_provider=test_provider,
        kronos_cache=test_kronos_cache,
        test_start=test_start,
        test_end=test_end,
        random_start=False,
    )
    print(f"      {TEST_TICKER} test: {len(test_env.step_indices)} steps")

    # ── Train ──
    print(f"\n   Training PPO scorer ({args.timesteps} timesteps)...")
    from stable_baselines3 import PPO

    multi_env = MultiTickerScorerEnv(train_envs)
    total_train_steps = len(multi_env.step_indices)
    print(f"   Multi-ticker pool: {total_train_steps} steps")

    model = PPO(
        "MlpPolicy",
        multi_env,
        learning_rate=PPO_LR,
        ent_coef=0.05,  # higher exploration for calibration
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
    model_path = os.path.join(MODEL_DIR, f"ppo_scorer_fold{fold_num}")
    model.save(model_path)
    print(f"   Model saved -> {model_path}")

    # ── Evaluate ──
    print(f"\n   Evaluating on {TEST_TICKER} test period...")

    # RL scorer
    scorer_metrics, scorer_log = evaluate_scorer(
        model, test_env, n_seeds=args.seeds)

    # Heuristic baseline
    heuristic_metrics = evaluate_heuristic(
        spy_df, test_provider, test_kronos_cache,
        test_env.step_indices, FORECAST_STEPS)

    results["rl_scorer"] = scorer_metrics
    results["heuristic"] = heuristic_metrics
    results["scorer_log"] = scorer_log

    # Print comparison
    print(f"\n   {'Metric':<25} {'RL Scorer':>12} {'Heuristic':>12} {'Delta':>10}")
    print(f"   {'-'*25} {'-'*12} {'-'*12} {'-'*10}")
    for metric in ["mae", "brier", "scorer_accuracy"]:
        s = scorer_metrics.get(metric, 0)
        h = heuristic_metrics.get(metric, 0)
        d = s - h
        better = "+" if (metric == "scorer_accuracy" and d > 0) or \
                        (metric != "scorer_accuracy" and d < 0) else ""
        print(f"   {metric:<25} {s:>12.4f} {h:>12.4f} {d:>+10.4f} {better}")

    print(f"\n   Avg score:    RL={scorer_metrics['avg_score']:.2f}  "
          f"Heuristic=N/A")
    print(f"   Kronos accuracy: {scorer_metrics['avg_actual_accuracy']:.1%} "
          f"({scorer_metrics['n']} forecasts)")

    # Calibration table
    if "calibration_table" in scorer_metrics:
        print(f"\n   RL Scorer Calibration:")
        print(f"   {'Predicted':>10} {'Actual':>10} {'Count':>8}")
        for k, v in scorer_metrics["calibration_table"].items():
            print(f"   {v['predicted']:>10.1f} {v['actual']:>10.3f} {v['count']:>8d}")

    # Score distribution
    if scorer_log:
        score_counts = {}
        for entry in scorer_log:
            s = entry["score"]
            score_counts[s] = score_counts.get(s, 0) + 1
        print(f"\n   Score distribution:")
        for level in SCORE_LEVELS:
            count = score_counts.get(level, 0)
            pct = count / len(scorer_log) * 100 if scorer_log else 0
            bar = "#" * int(pct / 2)
            print(f"      {level:.1f}: {count:4d} ({pct:5.1f}%) {bar}")

    return results


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 60)
    print("  V3: SIGNAL QUALITY SCORER TRAINING")
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

    # Run folds
    all_results = []
    for fold_config in folds:
        result = run_fold(fold_config, ticker_data, macro_data, args, train_tickers)
        all_results.append(result)

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  SUMMARY: RL SCORER VS HEURISTIC")
    print("=" * 60)

    print(f"\n   {'Fold':<6} {'Regime':<25} {'RL MAE':>8} {'H MAE':>8} "
          f"{'RL Brier':>10} {'H Brier':>10} {'RL Acc':>8} {'H Acc':>8}")
    print(f"   {'-'*6} {'-'*25} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")

    for r in all_results:
        s = r.get("rl_scorer", {})
        h = r.get("heuristic", {})
        if not s or not h:
            continue
        print(f"   {r['fold']:<6} {r['regime']:<25} "
              f"{s.get('mae', 0):>8.4f} {h.get('mae', 0):>8.4f} "
              f"{s.get('brier', 0):>10.4f} {h.get('brier', 0):>10.4f} "
              f"{s.get('scorer_accuracy', 0):>8.1%} {h.get('scorer_accuracy', 0):>8.1%}")

    # Averages
    rl_maes = [r["rl_scorer"]["mae"] for r in all_results if "rl_scorer" in r and r["rl_scorer"].get("n", 0) > 0]
    h_maes = [r["heuristic"]["mae"] for r in all_results if "heuristic" in r and r["heuristic"].get("n", 0) > 0]
    if rl_maes and h_maes:
        print(f"\n   Average MAE:    RL={np.mean(rl_maes):.4f}  Heuristic={np.mean(h_maes):.4f}")

    # Save results
    os.makedirs("outputs/results", exist_ok=True)
    results_path = "outputs/results/v3_scorer_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n   Results saved -> {results_path}")


if __name__ == "__main__":
    main()
