"""
Training orchestrator for the Two-Tier RL System.

Per fold:
  1. Download + cache macro features
  2. Generate regime labels (rules or HMM)
  3. Train regime classifier on training period
  4. For each regime with enough data: train per-regime PPO
  5. Train fallback all-data policy
  6. Evaluate two-tier agent on test period

Usage:
    python -m v2.train                           # all folds
    python -m v2.train --folds 4 --timesteps 50000
    python -m v2.train --hmm-regimes
    python -m v2.train --regime-threshold 0.6
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import os
import json
import argparse
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Kronos"))

import numpy as np
import pandas as pd

from v2.config import (
    FOLDS, DEFAULT_TICKERS, TEST_TICKER, MACRO_TICKERS,
    CRASH, BEAR, BULL, RECOVERY, SIDEWAYS, REGIMES,
    MIN_REGIME_STEPS, PPO_PARAMS, FALLBACK_PPO,
    INITIAL_CASH, FORECAST_STEPS, LOOKBACK_DAYS, OBS_DIM,
    REGIME_CONFIDENCE_THRESHOLD, CACHE_DIR, MACRO_CACHE_DIR,
    DATA_START, DATA_END,
)
from v2.features import MacroFeatureProvider, download_macro_data
from v2.regime import (
    RegimeClassifier,
    label_regimes_rule_based,
    label_regimes_hmm,
)
from v2.reward import make_reward_fn, fallback_reward
from v2.env_v2 import EnhancedTradingEnv, RegimeFilteredEnv, MultiTickerRegimeEnv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Two-Tier RL Training with Regime-Aware Policies")
    parser.add_argument("--folds", type=str, default=None,
                        help="Comma-separated fold numbers (default: all)")
    parser.add_argument("--timesteps", type=int, default=100_000,
                        help="PPO training timesteps per regime policy")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--tx-cost", type=float, default=0.0)
    parser.add_argument("--hmm-regimes", action="store_true",
                        help="Use HMM for regime labeling instead of rules")
    parser.add_argument("--regime-threshold", type=float,
                        default=REGIME_CONFIDENCE_THRESHOLD,
                        help="Confidence threshold for regime-specific policy")
    parser.add_argument("--single-ticker", action="store_true")
    parser.add_argument("--tickers", type=str, default=None)
    parser.add_argument("--precompute-only", action="store_true")
    return parser.parse_args()


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_ticker_data(ticker: str, max_retries: int = 3) -> pd.DataFrame:
    """Load OHLCV data for a single ticker with retry logic."""
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
    """Load existing v1 Kronos forecast cache."""
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    return {int(k): tuple(v) for k, v in data.items()}


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

def compute_fold_metrics(portfolio_values):
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
# PPO TRAINING
# ─────────────────────────────────────────────

def train_ppo(env, timesteps, lr, ent_coef=0.01, seed=42):
    from stable_baselines3 import PPO

    n_steps = min(len(env.step_indices) if hasattr(env, 'step_indices') else 2048, 2048)
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
        ent_coef=ent_coef,
        verbose=0,
        seed=seed,
    )
    model.learn(total_timesteps=timesteps, progress_bar=False)
    return model


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate_two_tier(
    regime_policies: dict[int, object],
    fallback_policy,
    test_env: EnhancedTradingEnv,
    classifier: RegimeClassifier,
    feature_provider: MacroFeatureProvider,
    threshold: float,
):
    """
    Run the two-tier inference loop:
      - Classify regime at each step
      - If confidence >= threshold, use regime-specific policy
      - Otherwise, use fallback all-data policy
    """
    obs, info = test_env.reset()
    done = False
    portfolio_values = [INITIAL_CASH]
    dates = [info.get("date", "")]
    actions = []
    regime_log = []

    while not done:
        idx = test_env.step_indices[test_env._step_idx]
        regime_feats = feature_provider.get_regime_inputs(idx)
        regime, confidence = classifier.predict(regime_feats)

        if confidence >= threshold and regime in regime_policies:
            action, _ = regime_policies[regime].predict(obs, deterministic=True)
            used_policy = REGIMES.get(regime, "unknown")
        else:
            action, _ = fallback_policy.predict(obs, deterministic=True)
            used_policy = "fallback"

        action = int(action)
        obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated

        actions.append(["HOLD", "BUY", "SELL"][action])
        regime_log.append({
            "regime": REGIMES.get(regime, "unknown"),
            "confidence": round(confidence, 3),
            "policy_used": used_policy,
        })

        if "portfolio_value" in info:
            portfolio_values.append(info["portfolio_value"])
        if not done and "date" in info:
            dates.append(info["date"])

    return {
        "actions": actions,
        "portfolio_values": portfolio_values[:len(dates)],
        "dates": dates,
        "regime_log": regime_log,
    }


def evaluate_buy_and_hold(df, test_start, test_end):
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


# ─────────────────────────────────────────────
# SEED AGGREGATION
# ─────────────────────────────────────────────

def aggregate_seeds(seed_results):
    returns = [r["metrics"]["return"] for r in seed_results]
    sharpes = [r["metrics"]["sharpe"] for r in seed_results]
    max_dds = [r["metrics"]["max_dd"] for r in seed_results]
    finals = [r["metrics"]["final"] for r in seed_results]
    median_idx = int(np.argmin(np.abs(np.array(returns) - np.median(returns))))
    best = seed_results[median_idx]
    return {
        "metrics": {
            "return_mean": round(float(np.mean(returns)), 2),
            "return_std": round(float(np.std(returns)), 2),
            "sharpe_mean": round(float(np.mean(sharpes)), 3),
            "max_dd_mean": round(float(np.mean(max_dds)), 2),
            "return": round(float(np.mean(returns)), 2),
            "sharpe": round(float(np.mean(sharpes)), 3),
            "max_dd": round(float(np.mean(max_dds)), 2),
            "final": round(float(np.mean(finals)), 2),
        },
        "n_seeds": len(seed_results),
        "dates": best.get("dates", []),
        "portfolio_values": best.get("portfolio_values", []),
        "actions": best.get("actions", []),
    }


# ─────────────────────────────────────────────
# RUN ONE FOLD
# ─────────────────────────────────────────────

def run_fold(fold_config, ticker_data, macro_data, args, train_tickers):
    fold_num = fold_config["fold"]
    train_start = fold_config["train_start"]
    train_end = fold_config["train_end"]
    test_start = fold_config["test_start"]
    test_end = fold_config["test_end"]
    regime_name = fold_config["regime"]

    print(f"\n{'='*70}")
    print(f"  FOLD {fold_num}: Train {train_start[:4]}-{train_end[:4]} "
          f"-> Test {test_start[:4]}")
    print(f"  Regime: {regime_name} -- {fold_config['notes']}")
    print(f"{'='*70}")

    test_df = ticker_data[TEST_TICKER]

    # ─── Build feature providers per ticker ───
    print("  Building macro feature providers...")
    providers = {}
    for ticker in train_tickers + [TEST_TICKER]:
        if ticker in providers:
            continue
        df = ticker_data[ticker]
        # Load existing Kronos cache if available
        kronos_cache = load_kronos_cache(
            get_kronos_cache_path(ticker, fold_num, "train"))
        kronos_cache.update(load_kronos_cache(
            get_kronos_cache_path(ticker, fold_num, "test")))
        providers[ticker] = MacroFeatureProvider(df, macro_data, kronos_cache)
        print(f"    {ticker}: features ready "
              f"(kronos cache: {len(kronos_cache)} steps)")

    # ─── Compute step indices for train/test ───
    def get_step_indices(df, start, end):
        dates = df.index
        indices = [i for i, d in enumerate(dates)
                   if d >= pd.Timestamp(start) and d <= pd.Timestamp(end)]
        indices = [i for i in indices if i >= LOOKBACK_DAYS]
        return indices[::FORECAST_STEPS]

    # ─── Label regimes (training data) ───
    print("  Labeling regimes...")
    train_regime_labels = {}
    for ticker in train_tickers:
        train_steps = get_step_indices(ticker_data[ticker], train_start, train_end)
        provider = providers[ticker]
        if args.hmm_regimes:
            labels = label_regimes_hmm(provider, train_steps)
        else:
            labels = label_regimes_rule_based(provider, train_steps)
        train_regime_labels[ticker] = (train_steps, labels)

        dist = Counter(labels)
        print(f"    {ticker}: " + ", ".join(
            f"{REGIMES[r]}={dist.get(r, 0)}" for r in sorted(REGIMES.keys())))

    # ─── Train regime classifier ───
    print("  Training regime classifier...")
    all_X_train = []
    all_y_train = []
    for ticker in train_tickers:
        steps, labels = train_regime_labels[ticker]
        provider = providers[ticker]
        X = np.array([provider.get_regime_inputs(idx) for idx in steps])
        all_X_train.append(X)
        all_y_train.append(labels)

    X_train = np.vstack(all_X_train)
    y_train = np.concatenate(all_y_train)
    classifier = RegimeClassifier()
    classifier.fit(X_train, y_train)
    train_acc = classifier.accuracy(X_train, y_train)
    print(f"    Classifier train accuracy: {train_acc:.3f}")

    # ─── Label test regime + confidences ───
    test_steps = get_step_indices(test_df, test_start, test_end)
    test_provider = providers[TEST_TICKER]
    X_test = np.array([test_provider.get_regime_inputs(idx) for idx in test_steps])
    test_regimes, test_confidences = classifier.predict_batch(X_test)
    test_dist = Counter(test_regimes)
    print(f"    Test regime distribution: " + ", ".join(
        f"{REGIMES[r]}={test_dist.get(r, 0)}" for r in sorted(REGIMES.keys())))

    if args.precompute_only:
        return None

    fold_results = {
        "fold": fold_num, "regime": regime_name,
        "test_start": test_start, "test_end": test_end,
        "agents": {},
    }

    # ─── Buy & Hold baseline ───
    print("  [1] Buy & Hold...")
    bh_values, bh_dates = evaluate_buy_and_hold(test_df, test_start, test_end)
    bh_metrics = compute_fold_metrics(bh_values)
    fold_results["agents"]["Buy & Hold"] = {
        "metrics": bh_metrics, "dates": bh_dates, "portfolio_values": bh_values,
    }
    print(f"      Return: {bh_metrics['return']:+.1f}%")

    # ─── Per-regime PPO policies ───
    regime_policies = {}
    for regime_id in sorted(REGIMES.keys()):
        regime_name_str = REGIMES[regime_id]

        # Count available steps across all train tickers
        total_regime_steps = 0
        for ticker in train_tickers:
            _, labels = train_regime_labels[ticker]
            total_regime_steps += int(np.sum(labels == regime_id))

        if total_regime_steps < MIN_REGIME_STEPS:
            print(f"  [R] {regime_name_str}: skipped "
                  f"({total_regime_steps} < {MIN_REGIME_STEPS} min steps)")
            continue

        print(f"  [R] Training {regime_name_str} policy "
              f"({total_regime_steps} steps)...")

        ppo_params = PPO_PARAMS[regime_id]
        timesteps = int(args.timesteps * ppo_params["timesteps_mult"])
        reward_fn = make_reward_fn(regime_id)

        # Build per-ticker filtered envs
        regime_ticker_envs = {}
        for ticker in train_tickers:
            df = ticker_data[ticker]
            train_steps, labels = train_regime_labels[ticker]
            provider = providers[ticker]

            base_env = EnhancedTradingEnv(
                df=df,
                feature_provider=provider,
                regime_labels=labels,
                regime_confidences=np.ones(len(labels)),
                reward_fn=reward_fn,
                test_start=train_start,
                test_end=train_end,
                transaction_cost=args.tx_cost,
                random_start=True,
            )

            filtered_env = RegimeFilteredEnv(base_env, regime_id, labels)
            if len(filtered_env.step_indices) >= 10:
                regime_ticker_envs[ticker] = filtered_env

        if not regime_ticker_envs:
            print(f"      No tickers with enough filtered steps, skipping")
            continue

        if len(regime_ticker_envs) > 1:
            multi_env = MultiTickerRegimeEnv(regime_ticker_envs)
        else:
            multi_env = next(iter(regime_ticker_envs.values()))

        seed_results = []
        for seed in range(args.seeds):
            model = train_ppo(
                multi_env, timesteps,
                lr=ppo_params["lr"],
                ent_coef=ppo_params["ent_coef"],
                seed=42 + seed,
            )
            seed_results.append(model)

        # Use first seed model as representative (we'll evaluate in evaluate.py)
        regime_policies[regime_id] = seed_results[0]
        print(f"      Trained {args.seeds} seeds")

    # ─── Fallback (all-data) policy ───
    print("  [F] Training fallback (all-data) policy...")
    fallback_ticker_envs = {}
    for ticker in train_tickers:
        df = ticker_data[ticker]
        steps, labels = train_regime_labels[ticker]
        provider = providers[ticker]

        env = EnhancedTradingEnv(
            df=df,
            feature_provider=provider,
            regime_labels=labels,
            regime_confidences=np.ones(len(labels)),
            reward_fn=fallback_reward,
            test_start=train_start,
            test_end=train_end,
            transaction_cost=args.tx_cost,
            random_start=True,
        )
        fallback_ticker_envs[ticker] = env

    if len(fallback_ticker_envs) > 1:
        fallback_multi = MultiTickerRegimeEnv(fallback_ticker_envs)
    else:
        fallback_multi = next(iter(fallback_ticker_envs.values()))

    fallback_policy = train_ppo(
        fallback_multi, args.timesteps,
        lr=FALLBACK_PPO["lr"],
        ent_coef=FALLBACK_PPO["ent_coef"],
        seed=42,
    )
    print("      Fallback policy trained")

    # ─── Evaluate two-tier agent ───
    print("  [E] Evaluating two-tier agent on test period...")

    seed_results = []
    for seed in range(args.seeds):
        test_env = EnhancedTradingEnv(
            df=test_df,
            feature_provider=test_provider,
            regime_labels=test_regimes,
            regime_confidences=test_confidences,
            reward_fn=fallback_reward,
            test_start=test_start,
            test_end=test_end,
            transaction_cost=args.tx_cost,
            random_start=False,
        )

        result = evaluate_two_tier(
            regime_policies=regime_policies,
            fallback_policy=fallback_policy,
            test_env=test_env,
            classifier=classifier,
            feature_provider=test_provider,
            threshold=args.regime_threshold,
        )
        metrics = compute_fold_metrics(result["portfolio_values"])
        seed_results.append({"metrics": metrics, **result})
        print(f"      Seed {seed}: {metrics['return']:+.1f}%")

    agg = aggregate_seeds(seed_results)
    fold_results["agents"]["RL v2 (two-tier)"] = agg
    print(f"      Mean: {agg['metrics']['return_mean']:+.1f}% "
          f"+/- {agg['metrics']['return_std']:.1f}%")

    # ─── Also evaluate fallback-only for comparison ───
    print("  [E] Evaluating fallback-only agent...")
    fb_seed_results = []
    for seed in range(args.seeds):
        test_env = EnhancedTradingEnv(
            df=test_df,
            feature_provider=test_provider,
            regime_labels=test_regimes,
            regime_confidences=test_confidences,
            reward_fn=fallback_reward,
            test_start=test_start,
            test_end=test_end,
            transaction_cost=args.tx_cost,
            random_start=False,
        )

        # Use fallback for everything (threshold=2.0 means never use regime)
        result = evaluate_two_tier(
            regime_policies={},
            fallback_policy=fallback_policy,
            test_env=test_env,
            classifier=classifier,
            feature_provider=test_provider,
            threshold=2.0,
        )
        metrics = compute_fold_metrics(result["portfolio_values"])
        fb_seed_results.append({"metrics": metrics, **result})

    fb_agg = aggregate_seeds(fb_seed_results)
    fold_results["agents"]["RL v2 (fallback only)"] = fb_agg
    print(f"      Fallback-only mean: {fb_agg['metrics']['return_mean']:+.1f}%")

    # ─── Regime accuracy on test ───
    if len(seed_results) > 0 and "regime_log" in seed_results[0]:
        log = seed_results[0]["regime_log"]
        regime_dist = Counter(entry["policy_used"] for entry in log)
        fold_results["regime_usage"] = dict(regime_dist)
        print(f"      Regime policy usage: {dict(regime_dist)}")

    fold_results["classifier_train_accuracy"] = round(train_acc, 3)

    return fold_results


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 70)
    print("  TWO-TIER RL TRADING SYSTEM (v2)")
    print("=" * 70)

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

    print(f"  Tickers: {', '.join(train_tickers)}")
    print(f"  Folds: {[f['fold'] for f in folds]}")
    print(f"  Timesteps: {args.timesteps}")
    print(f"  Regime labeling: {'HMM' if args.hmm_regimes else 'rule-based'}")
    print(f"  Confidence threshold: {args.regime_threshold}")

    # Load data
    all_tickers = list(set(train_tickers + [TEST_TICKER]))
    ticker_data = load_all_tickers(all_tickers)

    print("\nDownloading macro data...")
    macro_data = download_macro_data()
    print(f"  {len(macro_data)} macro tickers loaded")

    # Run folds
    all_results = []
    for fold_config in folds:
        result = run_fold(fold_config, ticker_data, macro_data, args, train_tickers)
        if result is not None:
            all_results.append(result)

    if not all_results:
        print("\nNo results to summarize (precompute-only or no folds).")
        return

    # ─── Summary ───
    print(f"\n{'='*70}")
    print("  CROSS-FOLD SUMMARY")
    print(f"{'='*70}")

    agent_names = set()
    for fold in all_results:
        agent_names.update(fold["agents"].keys())

    header = f"  {'Agent':<25s}"
    for fold in all_results:
        header += f"  Fold {fold['fold']:>2d}"
    header += "    Mean"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for agent in sorted(agent_names):
        row = f"  {agent:<25s}"
        rets = []
        for fold in all_results:
            if agent in fold["agents"]:
                metrics = fold["agents"][agent]["metrics"]
                ret = metrics.get("return_mean", metrics.get("return", 0))
                row += f"  {ret:>+6.1f}%"
                rets.append(ret)
            else:
                row += "      --"
        if rets:
            row += f"  {np.mean(rets):>+7.1f}%"
        print(row)

    # Save results
    os.makedirs("outputs/results", exist_ok=True)
    output_path = "outputs/results/v2_walkforward_results.json"

    # Serialize (strip non-serializable fields)
    serializable = []
    for fold in all_results:
        s_fold = {k: v for k, v in fold.items() if k != "agents"}
        s_fold["agents"] = {}
        for agent_name, agent_data in fold["agents"].items():
            s_agent = {}
            for k, v in agent_data.items():
                if k == "regime_log":
                    s_agent[k] = v
                elif isinstance(v, dict):
                    s_agent[k] = {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv
                                  for kk, vv in v.items()}
                elif isinstance(v, list):
                    s_agent[k] = [float(x) if isinstance(x, (np.floating, np.integer)) else x
                                  for x in v]
                else:
                    s_agent[k] = v
            s_fold["agents"][agent_name] = s_agent
        serializable.append(s_fold)

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
