"""
Evaluation script for the Two-Tier RL System.

Compares v2 agents against v1 baselines:
  - Buy & Hold
  - Rule-Based (Kronos)
  - RL v1 (no forecast, always forecast, soft gated)
  - RL v2 (two-tier regime-aware)

Outputs same JSON format as walkforward_results.json.

Usage:
    python -m v2.evaluate
    python -m v2.evaluate --folds 4
    python -m v2.evaluate --v1-results outputs/results/walkforward_results.json
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import os
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Kronos"))

import numpy as np
import pandas as pd

from v2.config import (
    FOLDS, DEFAULT_TICKERS, TEST_TICKER, REGIMES,
    INITIAL_CASH, FORECAST_STEPS, LOOKBACK_DAYS,
    REGIME_CONFIDENCE_THRESHOLD, CACHE_DIR,
)
from v2.features import MacroFeatureProvider, download_macro_data
from v2.regime import (
    RegimeClassifier,
    label_regimes_rule_based,
    label_regimes_hmm,
)
from v2.reward import make_reward_fn, fallback_reward
from v2.env_v2 import EnhancedTradingEnv, RegimeFilteredEnv, MultiTickerRegimeEnv
from v2.train import (
    load_ticker_data, load_all_tickers, load_kronos_cache,
    get_kronos_cache_path, compute_fold_metrics,
    train_ppo, evaluate_two_tier, evaluate_buy_and_hold,
    aggregate_seeds,
)
from v2.config import PPO_PARAMS, FALLBACK_PPO, MIN_REGIME_STEPS


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate v2 two-tier RL system")
    parser.add_argument("--folds", type=str, default=None)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--tx-cost", type=float, default=0.0)
    parser.add_argument("--hmm-regimes", action="store_true")
    parser.add_argument("--regime-threshold", type=float,
                        default=REGIME_CONFIDENCE_THRESHOLD)
    parser.add_argument("--v1-results", type=str, default=None,
                        help="Path to v1 walkforward_results.json for comparison")
    parser.add_argument("--single-ticker", action="store_true")
    parser.add_argument("--tickers", type=str, default=None)
    return parser.parse_args()


def load_v1_results(path: str) -> dict:
    """Load v1 results for comparison."""
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    # Handle both formats: list of folds or {"folds": [...]}
    if isinstance(data, dict) and "folds" in data:
        fold_list = data["folds"]
    elif isinstance(data, list):
        fold_list = data
    else:
        return {}
    # Index by fold number
    results = {}
    for fold_data in fold_list:
        if isinstance(fold_data, dict):
            fold_num = fold_data.get("fold")
            if fold_num is not None:
                results[fold_num] = fold_data
    return results


def print_comparison(all_results, v1_results):
    """Print a comprehensive comparison table."""
    print(f"\n{'='*90}")
    print("  COMPREHENSIVE COMPARISON: v1 vs v2")
    print(f"{'='*90}")

    # Collect all agent names
    all_agents_v2 = set()
    for fold in all_results:
        all_agents_v2.update(fold["agents"].keys())

    all_agents_v1 = set()
    for fold_data in v1_results.values():
        if "agents" in fold_data:
            all_agents_v1.update(fold_data["agents"].keys())

    all_agents = sorted(all_agents_v1 | all_agents_v2)

    for fold in all_results:
        fold_num = fold["fold"]
        print(f"\n  Fold {fold_num}: {fold.get('regime', '')}")
        print(f"  {'Agent':<30s} {'Return':>10s} {'Sharpe':>10s} {'Max DD':>10s}")
        print(f"  {'-'*60}")

        v1_fold = v1_results.get(fold_num, {}).get("agents", {})

        for agent in all_agents:
            # Check v2 first, then v1
            if agent in fold["agents"]:
                metrics = fold["agents"][agent]["metrics"]
                ret = metrics.get("return_mean", metrics.get("return", 0))
                sharpe = metrics.get("sharpe_mean", metrics.get("sharpe", 0))
                max_dd = metrics.get("max_dd_mean", metrics.get("max_dd", 0))
                source = ""
            elif agent in v1_fold:
                metrics = v1_fold[agent]["metrics"]
                ret = metrics.get("return_mean", metrics.get("return", 0))
                sharpe = metrics.get("sharpe_mean", metrics.get("sharpe", 0))
                max_dd = metrics.get("max_dd_mean", metrics.get("max_dd", 0))
                source = " (v1)"
            else:
                continue

            print(f"  {agent + source:<30s} {ret:>+9.1f}% {sharpe:>9.3f} {max_dd:>+9.1f}%")

        if "regime_usage" in fold:
            print(f"  Regime usage: {fold['regime_usage']}")
        if "classifier_train_accuracy" in fold:
            print(f"  Classifier accuracy: {fold['classifier_train_accuracy']:.3f}")


def main():
    args = parse_args()

    print("=" * 70)
    print("  TWO-TIER RL EVALUATION (v2)")
    print("=" * 70)

    # Tickers
    if args.single_ticker:
        train_tickers = [TEST_TICKER]
    elif args.tickers:
        train_tickers = args.tickers.split(",")
    else:
        train_tickers = DEFAULT_TICKERS

    # Folds
    if args.folds:
        fold_nums = [int(f) for f in args.folds.split(",")]
        folds = [f for f in FOLDS if f["fold"] in fold_nums]
    else:
        folds = FOLDS

    # Load data
    all_tickers = list(set(train_tickers + [TEST_TICKER]))
    ticker_data = load_all_tickers(all_tickers)

    print("\nDownloading macro data...")
    macro_data = download_macro_data()

    # Load v1 results for comparison
    v1_results = {}
    v1_paths = [
        args.v1_results,
        "outputs/results/walkforward_results.json",
        "walkforward_results.json",
    ]
    for path in v1_paths:
        if path and os.path.exists(path):
            v1_results = load_v1_results(path)
            print(f"  Loaded v1 results from {path} ({len(v1_results)} folds)")
            break

    # Run evaluation (train + eval) per fold
    from v2.train import run_fold

    class EvalArgs:
        def __init__(self, base_args):
            self.timesteps = base_args.timesteps
            self.seeds = base_args.seeds
            self.tx_cost = base_args.tx_cost
            self.hmm_regimes = base_args.hmm_regimes
            self.regime_threshold = base_args.regime_threshold
            self.precompute_only = False
            self.single_ticker = base_args.single_ticker

    eval_args = EvalArgs(args)

    all_results = []
    for fold_config in folds:
        result = run_fold(fold_config, ticker_data, macro_data, eval_args, train_tickers)
        if result is not None:
            all_results.append(result)

    if not all_results:
        print("No results.")
        return

    # Print comparison
    print_comparison(all_results, v1_results)

    # Save combined results
    os.makedirs("outputs/results", exist_ok=True)
    output_path = "outputs/results/v2_evaluation_results.json"

    serializable = []
    for fold in all_results:
        s_fold = {}
        for k, v in fold.items():
            if k == "agents":
                s_fold[k] = {}
                for agent_name, agent_data in v.items():
                    s_agent = {}
                    for ak, av in agent_data.items():
                        if isinstance(av, dict):
                            s_agent[ak] = {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv
                                            for kk, vv in av.items()}
                        elif isinstance(av, list):
                            s_agent[ak] = [float(x) if isinstance(x, (np.floating, np.integer)) else x
                                            for x in av]
                        else:
                            s_agent[ak] = av
                    s_fold[k][agent_name] = s_agent
            else:
                s_fold[k] = v
        serializable.append(s_fold)

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
