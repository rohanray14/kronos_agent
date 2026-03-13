"""
Gated Forecast RL Agent — Full Ablation Study
===============================================
Trains and evaluates RL agents that learn WHEN to call Kronos.

This is the core contribution: the agent decides whether a forecast
is worth requesting at each step, learning which market regimes
benefit from Kronos predictions vs. which are better handled by
market context alone.

Ablation table:
    1. Buy & Hold (baseline)
    2. Rule-Based (Kronos always, fixed thresholds)
    3. RL — no forecast (market context only)
    4. RL — always forecast (Kronos always, learned policy)
    5. RL — gated forecast (learns when to call Kronos)

Usage:
    python train_rl_gated.py                        # full ablation
    python train_rl_gated.py --bear                 # bear market
    python train_rl_gated.py --timesteps 100000     # more training
    python train_rl_gated.py --forecast-cost 0.001  # penalize forecast calls
    python train_rl_gated.py --seeds 5              # multi-seed evaluation
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

from agent import (
    TICKER, LOOKBACK_DAYS, FORECAST_STEPS, INITIAL_CASH,
    BLACK_SWANS,
    load_data, load_kronos, execute, decide, get_forecast,
    AgentState,
)
from trading_env import CachedTradingEnv, CachedGatedTradingEnv


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Gated forecast RL ablation study")
    parser.add_argument("--bear", action="store_true")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seeds", type=int, default=1,
                        help="Number of random seeds for evaluation")
    parser.add_argument("--forecast-cost", type=float, default=0.001,
                        help="Reward penalty per forecast call")
    parser.add_argument("--tx-cost", type=float, default=0.0)
    parser.add_argument("--precompute-only", action="store_true")
    return parser.parse_args()


def get_date_ranges(bear: bool):
    if bear:
        return {
            "train_start": "2022-01-01", "train_end": "2022-12-31",
            "test_start": "2023-01-01", "test_end": "2023-06-30",
        }
    else:
        return {
            "train_start": "2020-01-01", "train_end": "2022-12-31",
            "test_start": "2023-01-01", "test_end": "2024-12-31",
        }


# ─────────────────────────────────────────────
# EVALUATION HELPERS
# ─────────────────────────────────────────────

def compute_metrics(portfolio_values):
    """Compute return, Sharpe, max drawdown from portfolio value series."""
    pv = np.array(portfolio_values)
    if len(pv) < 2:
        return {"return": 0.0, "sharpe": 0.0, "max_dd": 0.0, "final": INITIAL_CASH}
    total_return = (pv[-1] - INITIAL_CASH) / INITIAL_CASH * 100
    returns = np.diff(pv) / pv[:-1]
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252 / FORECAST_STEPS) if returns.std() > 0 else 0.0
    peak = np.maximum.accumulate(pv)
    max_dd = float(((pv - peak) / peak).min()) * 100
    return {
        "return": round(total_return, 2),
        "sharpe": round(sharpe, 3),
        "max_dd": round(max_dd, 2),
        "final": round(float(pv[-1]), 2),
    }


def evaluate_rl(model, env, action_names=None):
    """Run trained model through env, return results dict."""
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


def evaluate_rule_based(df, predictor, test_start, test_end):
    """Run rule-based agent."""
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


def evaluate_buy_and_hold(df, test_start, test_end):
    """Buy and hold baseline."""
    test_df = df[(df.index >= test_start) & (df.index <= test_end)]["close"]
    initial_price = float(test_df.iloc[0])
    bh_shares = INITIAL_CASH / initial_price
    return (test_df * bh_shares).values, test_df.index


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

def train_agent(EnvClass, train_env_kwargs, cache_path, timesteps, lr, seed,
                model_name, env_kwargs_extra=None):
    """Train a PPO agent and return the model."""
    from stable_baselines3 import PPO

    env = EnvClass(**train_env_kwargs)
    if hasattr(env, 'load_cache') and os.path.exists(cache_path):
        env.load_cache(cache_path)
    elif hasattr(env, 'precompute_forecasts'):
        env.precompute_forecasts(verbose=False)
        env.save_cache(cache_path)

    model = PPO(
        "MlpPolicy", env,
        learning_rate=lr,
        n_steps=min(len(env.step_indices), 2048),
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=0,
        seed=seed,
    )
    model.learn(total_timesteps=timesteps, progress_bar=False)
    model.save(model_name)
    return model


# ─────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────

def plot_ablation(results_dict, mode, test_start):
    """Plot all agents on one chart + gating analysis."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 14))
    mode_label = "Bear Market" if mode == "bear" else "Full Period"
    fig.suptitle(f"Forecast Gating Ablation — S&P 500 ({mode_label})",
                 fontsize=15, fontweight="bold")

    colors = {
        "Buy & Hold": "#9E9E9E",
        "Rule-Based": "#FF9800",
        "RL (no forecast)": "#E91E63",
        "RL (always forecast)": "#2196F3",
        "RL (gated forecast)": "#4CAF50",
    }

    # Panel 1: Portfolio comparison
    ax1 = axes[0]
    for name, res in results_dict.items():
        dates = pd.to_datetime(res["dates"])
        ax1.plot(dates, res["portfolio_values"][:len(dates)],
                 label=name, color=colors.get(name, "black"), linewidth=2,
                 linestyle="--" if "Hold" in name else "-")
    ax1.axhline(INITIAL_CASH, color="gray", linestyle=":", alpha=0.5)
    for label, date_str in BLACK_SWANS.items():
        d = pd.Timestamp(date_str)
        if d >= pd.Timestamp(test_start):
            ax1.axvline(d, color="red", alpha=0.2, linewidth=1.5)
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.legend(fontsize=9)
    ax1.set_title("Portfolio Performance — All Agents")
    ax1.grid(True, alpha=0.3)

    # Panel 2: Gated agent actions — color by forecast usage
    ax2 = axes[1]
    gated = results_dict.get("RL (gated forecast)")
    if gated:
        dates = pd.to_datetime(gated["dates"])
        for i, (date, action) in enumerate(zip(dates, gated["actions"])):
            if "forecast" in action.lower() and "no" not in action.lower():
                color = "#4CAF50"  # green = used forecast
                alpha = 0.6
            else:
                color = "#FF9800"  # orange = no forecast
                alpha = 0.4
            ax2.axvline(date, color=color, alpha=alpha, linewidth=1.2)

        legend_els = [
            Patch(facecolor="#4CAF50", alpha=0.6, label="Used Kronos"),
            Patch(facecolor="#FF9800", alpha=0.4, label="No Forecast"),
        ]
        ax2.legend(handles=legend_els, loc="upper left")
        ax2.set_title(f"Gated Agent: When Does It Call Kronos? "
                      f"(forecast rate: {gated['forecast_rate']:.0%})")
    ax2.set_ylabel("Forecast Gating")
    ax2.set_yticks([])
    ax2.grid(True, alpha=0.2)

    # Panel 3: Forecast rate over time (rolling window)
    ax3 = axes[2]
    if gated:
        dates = pd.to_datetime(gated["dates"])
        used_forecast = [1.0 if ("forecast" in a.lower() and "no" not in a.lower())
                         else 0.0 for a in gated["actions"]]
        # Rolling forecast rate (window=10)
        window = min(10, len(used_forecast))
        rolling_rate = pd.Series(used_forecast).rolling(window, min_periods=1).mean()
        ax3.plot(dates[:len(rolling_rate)], rolling_rate, color="#4CAF50", linewidth=2)
        ax3.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
        ax3.fill_between(dates[:len(rolling_rate)], rolling_rate, alpha=0.2, color="#4CAF50")

        ax3.set_xlabel("Date")

    ax3.set_ylabel("Kronos Call Rate (rolling)")
    ax3.set_ylim(-0.05, 1.05)
    ax3.set_title("Forecast Gating Rate Over Time")
    ax3.grid(True, alpha=0.2)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    plt.tight_layout()
    chart_file = f"ablation_results_{mode}.png"
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
    print(f"  GATED FORECAST ABLATION ({mode.upper()} MODE)")
    print("=" * 60)

    # Load data + model
    df = load_data()
    predictor = load_kronos()

    # ─── Cache forecasts for all envs ───
    # Training caches
    train_cache = f"forecast_cache_train_{mode}.json"
    test_cache = f"forecast_cache_test_{mode}.json"
    gated_train_cache = f"forecast_cache_gated_train_{mode}.json"
    gated_test_cache = f"forecast_cache_gated_test_{mode}.json"

    common_train = dict(
        df=df, predictor=predictor,
        test_start=ranges["train_start"], test_end=ranges["train_end"],
        reward_type="log_return", transaction_cost=args.tx_cost,
        random_start=True,
    )
    common_test = dict(
        df=df, predictor=predictor,
        test_start=ranges["test_start"], test_end=ranges["test_end"],
        reward_type="log_return", transaction_cost=args.tx_cost,
        random_start=False,
    )

    # Pre-compute all caches
    print("\nCaching forecasts...")
    for cache_path, EnvClass, kwargs in [
        (train_cache, CachedTradingEnv, common_train),
        (test_cache, CachedTradingEnv, common_test),
        (gated_train_cache, CachedGatedTradingEnv, {**common_train, "forecast_cost": args.forecast_cost}),
        (gated_test_cache, CachedGatedTradingEnv, {**common_test, "forecast_cost": args.forecast_cost}),
    ]:
        if not os.path.exists(cache_path):
            env = EnvClass(**kwargs)
            env.precompute_forecasts(verbose=True)
            env.save_cache(cache_path)
        else:
            print(f"   Using existing cache: {cache_path}")

    if args.precompute_only:
        print("\nCaches ready. Exiting.")
        return

    from stable_baselines3 import PPO

    all_results = {}
    all_metrics = {}

    # ─── Agent 1: Buy & Hold ───
    print("\n[1/5] Buy & Hold...")
    bh_values, bh_dates = evaluate_buy_and_hold(df, ranges["test_start"], ranges["test_end"])
    # Subsample to match step frequency
    test_indices = [i for i, d in enumerate(df.index) if d >= pd.Timestamp(ranges["test_start"])]
    test_indices = [i for i in test_indices if i >= LOOKBACK_DAYS]
    step_indices = test_indices[::FORECAST_STEPS]
    bh_step_dates = [str(df.index[i].date()) for i in step_indices]
    bh_step_values = [float(INITIAL_CASH / float(df["close"].iloc[step_indices[0]]) * float(df["close"].iloc[i]))
                      for i in step_indices]
    all_results["Buy & Hold"] = {
        "dates": bh_step_dates,
        "portfolio_values": bh_step_values,
        "actions": ["HOLD"] * len(bh_step_dates),
        "forecast_rate": 0.0,
    }
    all_metrics["Buy & Hold"] = compute_metrics(bh_step_values)
    print(f"   Return: {all_metrics['Buy & Hold']['return']:+.1f}%")

    # ─── Agent 2: Rule-Based ───
    print("\n[2/5] Rule-Based (Kronos always, fixed thresholds)...")
    rule_state = evaluate_rule_based(df, predictor, ranges["test_start"], ranges["test_end"])
    all_results["Rule-Based"] = {
        "dates": [str(d.date()) for d in rule_state.dates],
        "portfolio_values": rule_state.portfolio_values,
        "actions": rule_state.actions,
        "forecast_rate": 1.0,
    }
    all_metrics["Rule-Based"] = compute_metrics(rule_state.portfolio_values)
    print(f"   Return: {all_metrics['Rule-Based']['return']:+.1f}%")

    # ─── Agent 3: RL — no forecast ───
    # Uses CachedTradingEnv but we zero out Kronos features
    print(f"\n[3/5] RL (no forecast) — training {args.timesteps:,} steps...")

    class NoForecastEnv(CachedTradingEnv):
        """TradingEnv that always returns zero Kronos features."""
        def _get_kronos_features(self, idx):
            return 0.0, 0.0, 0.0

    nf_train = NoForecastEnv(**common_train)
    nf_test = NoForecastEnv(**common_test)

    for seed in range(args.seeds):
        nf_model = PPO(
            "MlpPolicy", nf_train,
            learning_rate=args.lr,
            n_steps=min(len(nf_train.step_indices), 2048),
            batch_size=64, n_epochs=10, gamma=0.99,
            ent_coef=0.01, verbose=0, seed=42 + seed,
        )
        nf_model.learn(total_timesteps=args.timesteps, progress_bar=False)

    nf_results = evaluate_rl(nf_model, nf_test)
    all_results["RL (no forecast)"] = nf_results
    all_metrics["RL (no forecast)"] = compute_metrics(nf_results["portfolio_values"])
    print(f"   Return: {all_metrics['RL (no forecast)']['return']:+.1f}%")
    print(f"   Actions: {Counter(nf_results['actions'])}")

    # ─── Agent 4: RL — always forecast ───
    print(f"\n[4/5] RL (always forecast) — training {args.timesteps:,} steps...")
    af_train = CachedTradingEnv(**common_train)
    af_test = CachedTradingEnv(**common_test)
    af_train.load_cache(train_cache)
    af_test.load_cache(test_cache)

    af_model = PPO(
        "MlpPolicy", af_train,
        learning_rate=args.lr,
        n_steps=min(len(af_train.step_indices), 2048),
        batch_size=64, n_epochs=10, gamma=0.99,
        ent_coef=0.01, verbose=0, seed=42,
    )
    af_model.learn(total_timesteps=args.timesteps, progress_bar=False)

    af_results = evaluate_rl(af_model, af_test)
    all_results["RL (always forecast)"] = af_results
    all_metrics["RL (always forecast)"] = compute_metrics(af_results["portfolio_values"])
    print(f"   Return: {all_metrics['RL (always forecast)']['return']:+.1f}%")
    print(f"   Actions: {Counter(af_results['actions'])}")

    # ─── Agent 5: RL — gated forecast (THE CONTRIBUTION) ───
    print(f"\n[5/5] RL (gated forecast) — training {args.timesteps:,} steps...")
    print(f"   Forecast cost: {args.forecast_cost}")

    gated_train = CachedGatedTradingEnv(
        **common_train, forecast_cost=args.forecast_cost,
    )
    gated_test = CachedGatedTradingEnv(
        **common_test, forecast_cost=args.forecast_cost,
    )
    gated_train.load_cache(gated_train_cache)
    gated_test.load_cache(gated_test_cache)

    gated_model = PPO(
        "MlpPolicy", gated_train,
        learning_rate=args.lr,
        n_steps=min(len(gated_train.step_indices), 2048),
        batch_size=64, n_epochs=10, gamma=0.99,
        ent_coef=0.01, verbose=0, seed=42,
    )
    gated_model.learn(total_timesteps=args.timesteps, progress_bar=False)

    gated_results = evaluate_rl(
        gated_model, gated_test,
        action_names=CachedGatedTradingEnv.ACTION_NAMES,
    )
    all_results["RL (gated forecast)"] = gated_results
    all_metrics["RL (gated forecast)"] = compute_metrics(gated_results["portfolio_values"])

    # Analyze gating behavior
    gated_actions = gated_results["actions"]
    n_with_forecast = sum(1 for a in gated_actions if "forecast)" in a.lower() and "no" not in a.lower())
    n_without = sum(1 for a in gated_actions if "no forecast" in a.lower())
    total = len(gated_actions)
    forecast_rate = n_with_forecast / max(1, total)

    print(f"   Return: {all_metrics['RL (gated forecast)']['return']:+.1f}%")
    print(f"   Forecast rate: {forecast_rate:.0%} ({n_with_forecast}/{total} steps used Kronos)")
    print(f"   Actions: {Counter(gated_actions)}")
    gated_results["forecast_rate"] = forecast_rate

    # ─── Results Table ───
    print("\n" + "=" * 85)
    print("  ABLATION RESULTS (Test Period)")
    print("=" * 85)
    header = f"  {'Agent':<25} {'Return':>10} {'Sharpe':>10} {'Max DD':>10} {'Final $':>12} {'Fcst Rate':>10}"
    print(header)
    print("  " + "-" * 83)
    for name in ["Buy & Hold", "Rule-Based", "RL (no forecast)",
                 "RL (always forecast)", "RL (gated forecast)"]:
        m = all_metrics[name]
        fr = all_results[name].get("forecast_rate", "—")
        fr_str = f"{fr:.0%}" if isinstance(fr, float) else fr
        print(f"  {name:<25} {m['return']:>+9.1f}% {m['sharpe']:>10.3f} "
              f"{m['max_dd']:>9.1f}% ${m['final']:>10,.0f} {fr_str:>10}")
    print("=" * 85)

    # ─── Save ───
    save_data = {
        "mode": mode,
        "timesteps": args.timesteps,
        "forecast_cost": args.forecast_cost,
        "tx_cost": args.tx_cost,
        "test_start": ranges["test_start"],
        "test_end": ranges["test_end"],
        "metrics": all_metrics,
        "gated_forecast_rate": forecast_rate,
        "gated_action_distribution": dict(Counter(gated_actions)),
    }
    results_file = f"ablation_results_{mode}.json"
    with open(results_file, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved -> {results_file}")

    # ─── Plot ───
    plot_ablation(all_results, mode, ranges["test_start"])

    print("\nDone!")


if __name__ == "__main__":
    main()
