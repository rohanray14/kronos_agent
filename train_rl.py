"""
RL Training Script for Kronos Trading Agent
=============================================
Trains a PPO agent to make trading decisions using Kronos forecasts.

Usage:
    python train_rl.py                      # train on full period
    python train_rl.py --bear               # train on bear market
    python train_rl.py --eval-only          # evaluate saved model (no training)
    python train_rl.py --precompute-only    # just cache forecasts, don't train
    python train_rl.py --reward sharpe      # reward type: log_return, pnl, sharpe
    python train_rl.py --timesteps 100000   # training timesteps

Steps:
    1. Load data + Kronos
    2. Pre-compute all forecasts (cached to disk for fast retraining)
    3. Train PPO with stable-baselines3
    4. Evaluate against baselines (rule-based, buy & hold, LLM agent)
    5. Plot results
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import os
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Kronos"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from agent import (
    TICKER, LOOKBACK_DAYS, FORECAST_STEPS, INITIAL_CASH,
    MODE, TRAIN_START, TEST_START, TEST_END,
    BLACK_SWANS,
    load_data, load_kronos, execute,
    AgentState, buy_and_hold, compute_metrics,
)
from trading_env import CachedTradingEnv


# ─────────────────────────────────────────────
# CLI ARGS
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train RL trading agent")
    parser.add_argument("--bear", action="store_true", help="Use bear market period")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate saved model only")
    parser.add_argument("--precompute-only", action="store_true", help="Cache forecasts only")
    parser.add_argument("--reward", default="log_return",
                        choices=["log_return", "pnl", "sharpe"],
                        help="Reward function type")
    parser.add_argument("--timesteps", type=int, default=50_000,
                        help="Total training timesteps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--tx-cost", type=float, default=0.0,
                        help="Transaction cost (fraction, e.g. 0.001 = 10bps)")
    return parser.parse_args()


# ─────────────────────────────────────────────
# TRAIN / EVAL SPLIT
# ─────────────────────────────────────────────

def get_date_ranges(bear: bool):
    """Split data into RL train and test periods.
    Uses walk-forward: train on earlier data, test on later."""
    if bear:
        # Train on 2022, test on 2023
        return {
            "train_start": "2022-01-01",
            "train_end":   "2022-12-31",
            "test_start":  "2023-01-01",
            "test_end":    "2023-06-30",
            "data_start":  "2020-01-01",  # need lookback history
        }
    else:
        # Train on 2020-2022, test on 2023-2024
        return {
            "train_start": "2020-01-01",
            "train_end":   "2022-12-31",
            "test_start":  "2023-01-01",
            "test_end":    "2024-12-31",
            "data_start":  "2018-01-01",
        }


# ─────────────────────────────────────────────
# FORECAST CACHING
# ─────────────────────────────────────────────

def get_or_create_cache(env, cache_path: str):
    """Load cached forecasts from disk, or compute and save them."""
    if os.path.exists(cache_path):
        print(f"Loading cached forecasts from {cache_path}...")
        env.load_cache(cache_path)
    else:
        env.precompute_forecasts(verbose=True)
        env.save_cache(cache_path)
        print(f"Forecasts cached to {cache_path}")


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate_model(model, env, deterministic=True):
    """Run the trained model through the env and collect results."""
    obs, info = env.reset()
    done = False
    actions = []
    portfolio_values = [INITIAL_CASH]
    dates = [info["date"]]

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated

        if not done:
            actions.append(["HOLD", "BUY", "SELL"][int(action)])
            portfolio_values.append(info["portfolio_value"])
            dates.append(info["date"])
        else:
            actions.append(["HOLD", "BUY", "SELL"][int(action)])
            if "portfolio_value" in info:
                portfolio_values.append(info["portfolio_value"])

    return {
        "actions": actions,
        "portfolio_values": portfolio_values[:len(actions)],
        "dates": dates[:len(actions)],
    }


def evaluate_baselines(df, predictor, test_start, test_end):
    """Run rule-based and buy-and-hold baselines for comparison."""
    from agent import decide, get_forecast

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

        pv = state.portfolio_value(current_price)
        state.portfolio_values.append(pv)
        state.actions.append(action)
        state.dates.append(current_date)

    # Buy and hold
    test_df = df[(df.index >= test_start) & (df.index <= test_end)]["close"]
    initial_price = float(test_df.iloc[0])
    bh_shares = INITIAL_CASH / initial_price
    bh_values = (test_df * bh_shares).values

    return state, bh_values, test_df.index


# ─────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────

def plot_comparison(rl_results, rule_state, bh_values, bh_dates, mode, test_start):
    """Plot RL agent vs baselines."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    mode_label = "Bear Market" if mode == "bear" else "Full Period"
    fig.suptitle(f"RL Agent vs Baselines — S&P 500 ({mode_label})",
                 fontsize=15, fontweight="bold")

    # Panel 1: Portfolio values
    ax1 = axes[0]
    rl_dates = pd.to_datetime(rl_results["dates"])
    ax1.plot(rl_dates, rl_results["portfolio_values"],
             label="RL Agent (PPO)", color="#2196F3", linewidth=2)
    ax1.plot(pd.to_datetime(rule_state.dates), rule_state.portfolio_values,
             label="Rule-Based", color="#FF9800", linewidth=2, linestyle="--")
    ax1.plot(bh_dates, bh_values,
             label="Buy & Hold", color="#9E9E9E", linewidth=2, linestyle=":")
    ax1.axhline(INITIAL_CASH, color="gray", linestyle=":", linewidth=1, alpha=0.5)

    for label, date_str in BLACK_SWANS.items():
        d = pd.Timestamp(date_str)
        if d >= pd.Timestamp(test_start):
            ax1.axvline(d, color="red", alpha=0.3, linewidth=1.5)

    ax1.set_ylabel("Portfolio Value ($)")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.legend()
    ax1.set_title("Portfolio Performance")
    ax1.grid(True, alpha=0.3)

    # Panel 2: RL actions
    ax2 = axes[1]
    action_colors = {"BUY": "#4CAF50", "SELL": "#F44336", "HOLD": "#9E9E9E"}
    for date, action in zip(rl_dates, rl_results["actions"]):
        ax2.axvline(date, color=action_colors.get(action, "#9E9E9E"),
                    alpha=0.4, linewidth=0.8)

    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=c, label=a) for a, c in action_colors.items()]
    ax2.legend(handles=legend_els, loc="upper left")
    ax2.set_ylabel("RL Agent Actions")
    ax2.set_yticks([])
    ax2.set_title("RL Agent Decisions Over Time")
    ax2.grid(True, alpha=0.2)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    plt.tight_layout()
    chart_file = f"rl_results_{mode}.png"
    plt.savefig(chart_file, dpi=150, bbox_inches="tight")
    print(f"Chart saved -> {chart_file}")
    plt.close()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    args = parse_args()
    mode = "bear" if args.bear else "full"
    ranges = get_date_ranges(args.bear)

    print("=" * 60)
    print(f"  KRONOS RL TRADING AGENT ({mode.upper()} MODE)")
    print("=" * 60)

    # 1. Load data (full range for lookback)
    print(f"\nLoading {TICKER} data...")
    df = load_data()

    # 2. Load Kronos
    predictor = load_kronos()

    # 3. Create environments
    print("\nSetting up environments...")
    train_env = CachedTradingEnv(
        df, predictor,
        test_start=ranges["train_start"],
        test_end=ranges["train_end"],
        reward_type=args.reward,
        transaction_cost=args.tx_cost,
        random_start=True,   # data augmentation: random episode starts
    )
    test_env = CachedTradingEnv(
        df, predictor,
        test_start=ranges["test_start"],
        test_end=ranges["test_end"],
        reward_type=args.reward,
        transaction_cost=args.tx_cost,
        random_start=False,  # deterministic for evaluation
    )

    # 4. Cache forecasts
    train_cache = f"forecast_cache_train_{mode}.json"
    test_cache = f"forecast_cache_test_{mode}.json"
    get_or_create_cache(train_env, train_cache)
    get_or_create_cache(test_env, test_cache)

    if args.precompute_only:
        print("\nForecasts cached. Exiting (--precompute-only).")
        return

    # 5. Train or load model
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback

    model_path = f"ppo_trading_{mode}"

    if args.eval_only:
        print(f"\nLoading saved model from {model_path}.zip...")
        model = PPO.load(model_path, env=test_env)
    else:
        print(f"\nTraining PPO agent...")
        print(f"   Timesteps: {args.timesteps:,}")
        print(f"   Reward: {args.reward}")
        print(f"   Learning rate: {args.lr}")
        print(f"   Seed: {args.seed}")
        print(f"   Transaction cost: {args.tx_cost}")

        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=args.lr,
            n_steps=min(len(train_env.step_indices), 2048),
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,  # encourage exploration
            verbose=1,
            seed=args.seed,
        )

        # Train
        model.learn(
            total_timesteps=args.timesteps,
            progress_bar=True,
        )

        # Save
        model.save(model_path)
        print(f"\nModel saved -> {model_path}.zip")

    # 6. Evaluate on test set
    print(f"\nEvaluating on test period ({ranges['test_start']} -> {ranges['test_end']})...")
    rl_results = evaluate_model(model, test_env)

    # Action distribution
    from collections import Counter
    action_counts = Counter(rl_results["actions"])
    print(f"\n   RL Action distribution: {dict(action_counts)}")

    # Portfolio metrics
    pv = np.array(rl_results["portfolio_values"])
    rl_return = (pv[-1] - INITIAL_CASH) / INITIAL_CASH * 100
    returns = np.diff(pv) / pv[:-1]
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252 / FORECAST_STEPS) if returns.std() > 0 else 0
    peak = np.maximum.accumulate(pv)
    max_dd = ((pv - peak) / peak).min()

    print(f"   RL Agent Return: {rl_return:.1f}%")
    print(f"   RL Sharpe Ratio: {sharpe:.2f}")
    print(f"   RL Max Drawdown: {max_dd*100:.1f}%")

    # 7. Baselines on same test period
    print(f"\nRunning baselines on test period...")
    rule_state, bh_values, bh_dates = evaluate_baselines(
        df, predictor, ranges["test_start"], ranges["test_end"]
    )

    rule_pv = np.array(rule_state.portfolio_values)
    rule_return = (rule_pv[-1] - INITIAL_CASH) / INITIAL_CASH * 100
    bh_return = (bh_values[-1] - INITIAL_CASH) / INITIAL_CASH * 100

    # 8. Results table
    print("\n" + "=" * 50)
    print("  RESULTS (Test Period)")
    print("=" * 50)
    print(f"  {'Metric':<28} {'RL (PPO)':<15} {'Rule-Based':<15} {'Buy & Hold':<15}")
    print(f"  {'-'*28} {'-'*15} {'-'*15} {'-'*15}")
    print(f"  {'Return':<28} {rl_return:>+.1f}%{'':>9} {rule_return:>+.1f}%{'':>9} {bh_return:>+.1f}%")
    print(f"  {'Sharpe Ratio':<28} {sharpe:>.2f}{'':>12} {'—':<15} {'—':<15}")
    print(f"  {'Max Drawdown':<28} {max_dd*100:>.1f}%{'':>11} {'—':<15} {'—':<15}")
    print(f"  {'Final Value':<28} ${pv[-1]:>,.0f}{'':>5} ${rule_pv[-1]:>,.0f}{'':>5} ${bh_values[-1]:>,.0f}")
    print("=" * 50)

    # 9. Save results
    results = {
        "mode": mode,
        "reward_type": args.reward,
        "timesteps": args.timesteps,
        "seed": args.seed,
        "test_start": ranges["test_start"],
        "test_end": ranges["test_end"],
        "rl_return": round(rl_return, 2),
        "rl_sharpe": round(sharpe, 3),
        "rl_max_drawdown": round(max_dd * 100, 2),
        "rl_final_value": round(float(pv[-1]), 2),
        "rule_return": round(rule_return, 2),
        "bh_return": round(bh_return, 2),
        "rl_actions": dict(action_counts),
    }
    results_file = f"rl_results_{mode}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved -> {results_file}")

    # 10. Plot
    plot_comparison(rl_results, rule_state, bh_values, bh_dates, mode,
                    ranges["test_start"])

    print("\nDone!")


if __name__ == "__main__":
    main()
