"""
Kronos Trading Agent - MVP
=============================
Uses Kronos (local HuggingFace) to forecast S&P 500 prices,
then makes buy/sell/hold decisions based on forecast signals.

Kronos is a foundation model for financial markets that operates
on OHLCV (open/high/low/close/volume) candlestick data.

Setup:
    git clone https://github.com/shiyu-coder/Kronos.git
    pip install -r requirements.txt

Run:
    python agent.py
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Kronos"))

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Literal


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

TICKER         = "^GSPC"        # S&P 500
LOOKBACK_DAYS  = 60             # days of history fed to Kronos per step
FORECAST_STEPS = 5              # days ahead to forecast
MAX_CONTEXT    = 512            # max context window for Kronos
INITIAL_CASH   = 100_000        # starting portfolio value ($)

# Mode: "full" (2020-2024) or "bear" (2022 bear market)
MODE = "bear" if "--bear" in sys.argv else "full"

# Date ranges per mode
if MODE == "bear":
    TRAIN_START = "2020-01-01"
    TRAIN_END   = "2021-12-31"
    TEST_START  = "2022-01-01"      # Fed rate hikes, ~25% drawdown
    TEST_END    = "2023-06-30"      # through early recovery
else:
    TRAIN_START = "2018-01-01"
    TRAIN_END   = "2019-12-31"
    TEST_START  = "2020-01-01"      # includes COVID crash
    TEST_END    = "2024-12-31"

# Decision thresholds (asymmetric — reluctant to sell in upward-biased markets)
BUY_THRESHOLD  =  0.003         # forecast >+0.3% → BUY  (easy to enter)
SELL_THRESHOLD = -0.015         # forecast < -1.5% → SELL (hard to exit)

# Multi-step trend: fraction of forecast steps that must agree on direction
TREND_AGREEMENT = 0.6           # 60% of steps must agree to confirm trend

# Volatility filter: if forecast high-low spread exceeds this multiple of
# recent average spread, treat as high-volatility regime (more cautious)
VOLATILITY_MULTIPLIER = 1.5

# Black swan events to annotate on chart
if MODE == "bear":
    BLACK_SWANS = {
        "Fed Hike Start":  "2022-03-16",
        "75bp Hike":       "2022-06-15",
        "CPI Shock":       "2022-09-13",
        "SVB Collapse":    "2023-03-13",
    }
else:
    BLACK_SWANS = {
        "COVID Crash":  "2020-03-20",
        "Fed Hike":     "2022-06-15",
        "SVB Collapse": "2023-03-13",
    }


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    print(f"📥 Downloading {TICKER} OHLCV data ({TRAIN_START} → {TEST_END})...")
    df = yf.download(TICKER, start=TRAIN_START, end=TEST_END, progress=False, auto_adjust=True)
    # Flatten MultiIndex columns from yfinance
    df.columns = df.columns.get_level_values(0)
    # Kronos expects lowercase OHLCV columns
    df = df.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })
    df = df[["open", "high", "low", "close", "volume"]]
    df.index = pd.to_datetime(df.index)
    df = df.dropna()
    print(f"   ✓ {len(df)} trading days loaded")
    return df


# ─────────────────────────────────────────────
# KRONOS FORECASTER
# ─────────────────────────────────────────────

def load_kronos():
    """Load Kronos model and tokenizer from HuggingFace."""
    print("🤖 Loading Kronos model (this may take ~60s first run)...")
    from model import Kronos, KronosTokenizer, KronosPredictor

    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=MAX_CONTEXT)

    print("   ✓ Kronos loaded (Kronos-small + Tokenizer-base)")
    return predictor


def get_forecast(predictor, history_df: pd.DataFrame, future_dates: pd.DatetimeIndex) -> dict:
    """
    Run Kronos on recent OHLCV history.
    Returns full OHLCV forecast for next FORECAST_STEPS days.
    """
    x_df = history_df[["open", "high", "low", "close", "volume"]].copy()
    x_timestamp = history_df.index.to_series().reset_index(drop=True)
    y_timestamp = pd.Series(future_dates)
    pred_len = len(future_dates)

    pred_df = predictor.predict(
        df=x_df.reset_index(drop=True),
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T=1.0,
        top_p=0.9,
        sample_count=1,
        verbose=False,
    )

    return {
        "close": pred_df["close"].values,
        "high":  pred_df["high"].values,
        "low":   pred_df["low"].values,
        "open":  pred_df["open"].values,
    }


# ─────────────────────────────────────────────
# AGENT DECISION LOGIC
# ─────────────────────────────────────────────

@dataclass
class AgentState:
    cash: float = INITIAL_CASH
    shares: float = 0.0
    portfolio_values: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    dates: list = field(default_factory=list)
    forecasts: list = field(default_factory=list)

    def portfolio_value(self, price: float) -> float:
        return self.cash + self.shares * price


def decide(forecast: dict, current_price: float, history_df: pd.DataFrame) -> Literal["BUY", "SELL", "HOLD"]:
    """
    Enhanced decision logic using three signals:
    1. Asymmetric thresholds — easy to buy, hard to sell
    2. Multi-step trend — majority of forecast steps must agree on direction
    3. OHLCV volatility — widen sell threshold in high-volatility regimes
    """
    closes = forecast["close"]
    highs = forecast["high"]
    lows = forecast["low"]

    # --- Signal 1: Expected return from 1-step forecast ---
    expected_return = (closes[0] - current_price) / current_price

    # --- Signal 2: Multi-step trend agreement ---
    # Count how many forecast steps are above/below current price
    steps_up = np.sum(closes > current_price)
    steps_down = np.sum(closes < current_price)
    n_steps = len(closes)
    trend_bullish = (steps_up / n_steps) >= TREND_AGREEMENT
    trend_bearish = (steps_down / n_steps) >= TREND_AGREEMENT

    # --- Signal 3: Volatility filter from OHLCV ---
    # Compare forecast volatility (high-low spread) to recent historical volatility
    forecast_spread = np.mean(highs - lows) / current_price
    recent_spread = np.mean(
        (history_df["high"].values[-20:] - history_df["low"].values[-20:])
    ) / current_price
    high_volatility = forecast_spread > (recent_spread * VOLATILITY_MULTIPLIER)

    # --- Combined decision ---
    # In high volatility: be more cautious, require stronger signals
    effective_buy_threshold = BUY_THRESHOLD * (1.5 if high_volatility else 1.0)
    effective_sell_threshold = SELL_THRESHOLD * (0.5 if high_volatility else 1.0)

    # BUY: expected return exceeds threshold AND trend confirms
    if expected_return > effective_buy_threshold and trend_bullish:
        return "BUY"
    # SELL: expected return below threshold AND trend confirms
    elif expected_return < effective_sell_threshold and trend_bearish:
        return "SELL"
    else:
        return "HOLD"


def execute(action: str, state: AgentState, price: float):
    """Execute buy/sell/hold — all-in/all-out for simplicity."""
    if action == "BUY" and state.cash > 0:
        state.shares = state.cash / price
        state.cash = 0.0

    elif action == "SELL" and state.shares > 0:
        state.cash = state.shares * price
        state.shares = 0.0

    # HOLD: do nothing


# ─────────────────────────────────────────────
# BACKTEST LOOP
# ─────────────────────────────────────────────

def run_backtest(df: pd.DataFrame, predictor) -> AgentState:
    dates = df.index

    state = AgentState()

    print(f"\n🔁 Running backtest ({TEST_START} → {TEST_END})...")
    test_count = len(df[df.index >= TEST_START])
    print(f"   Steps: {test_count} trading days | Forecast horizon: {FORECAST_STEPS} days\n")

    # Step every FORECAST_STEPS days (agent re-evaluates each week)
    test_indices = [i for i, d in enumerate(dates) if d >= pd.Timestamp(TEST_START)]
    step_indices = test_indices[::FORECAST_STEPS]

    for step_num, idx in enumerate(step_indices):
        if idx < LOOKBACK_DAYS:
            continue

        current_date = dates[idx]
        current_price = df["close"].iloc[idx]

        # Get OHLCV history window
        history_df = df.iloc[max(0, idx - LOOKBACK_DAYS):idx]

        # Generate future timestamps for forecast
        future_dates = pd.bdate_range(
            start=current_date, periods=FORECAST_STEPS + 1, freq="B"
        )[1:]  # exclude current date

        # Get Kronos forecast
        forecast = get_forecast(predictor, history_df, future_dates)

        # Agent decides (now with history context for volatility)
        action = decide(forecast, current_price, history_df)
        execute(action, state, current_price)

        pv = state.portfolio_value(current_price)
        state.portfolio_values.append(pv)
        state.actions.append(action)
        state.dates.append(current_date)
        state.forecasts.append({
            "date": current_date,
            "actual": current_price,
            "forecast_close": forecast["close"][0],
            "forecast_high": forecast["high"][0],
            "forecast_low": forecast["low"][0],
        })

        if step_num % 10 == 0:
            print(f"   [{current_date.date()}] Price: ${current_price:,.0f} | "
                  f"Forecast: ${forecast['close'][0]:,.0f} | "
                  f"Action: {action:4s} | Portfolio: ${pv:,.0f}")

    return state


# ─────────────────────────────────────────────
# BASELINE: BUY AND HOLD
# ─────────────────────────────────────────────

def buy_and_hold(df: pd.DataFrame) -> pd.Series:
    test_df = df[df.index >= TEST_START]["close"]
    initial_price = test_df.iloc[0]
    shares = INITIAL_CASH / initial_price
    return (test_df * shares).rename("Buy & Hold")


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

def compute_metrics(state: AgentState, bh_series: pd.Series) -> dict:
    pv = np.array(state.portfolio_values)
    returns = np.diff(pv) / pv[:-1]

    sharpe = (returns.mean() / returns.std()) * np.sqrt(252 / FORECAST_STEPS) if returns.std() > 0 else 0

    peak = np.maximum.accumulate(pv)
    drawdowns = (pv - peak) / peak
    max_dd = drawdowns.min()

    total_return = (pv[-1] - INITIAL_CASH) / INITIAL_CASH * 100
    bh_return    = (bh_series.iloc[-1] - INITIAL_CASH) / INITIAL_CASH * 100

    actions = pd.Series(state.actions)
    action_counts = actions.value_counts().to_dict()

    return {
        "Total Return (Agent)":    f"{total_return:.1f}%",
        "Total Return (B&H)":      f"{bh_return:.1f}%",
        "Sharpe Ratio":            f"{sharpe:.2f}",
        "Max Drawdown":            f"{max_dd*100:.1f}%",
        "Final Portfolio":         f"${state.portfolio_values[-1]:,.0f}",
        "Actions — BUY":           action_counts.get("BUY",  0),
        "Actions — SELL":          action_counts.get("SELL", 0),
        "Actions — HOLD":          action_counts.get("HOLD", 0),
    }


# ─────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────

def plot_results(state: AgentState, df: pd.DataFrame, bh_series: pd.Series):
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    mode_label = "Bear Market (2022–2023)" if MODE == "bear" else "Full Period (2020–2024)"
    fig.suptitle(f"Kronos Trading Agent — S&P 500 Backtest ({mode_label})", fontsize=15, fontweight="bold")

    dates = [d for d in state.dates]
    pv    = state.portfolio_values
    bh_aligned = bh_series.reindex(pd.DatetimeIndex(dates), method="nearest")

    # ── Panel 1: Portfolio Value ──────────────────
    ax1 = axes[0]
    ax1.plot(dates, pv,             label="Kronos Agent", color="#2196F3", linewidth=2)
    ax1.plot(dates, bh_aligned,     label="Buy & Hold",    color="#FF9800", linewidth=2, linestyle="--")
    ax1.axhline(INITIAL_CASH, color="gray", linestyle=":", linewidth=1, alpha=0.7)

    for label, date_str in BLACK_SWANS.items():
        d = pd.Timestamp(date_str)
        if pd.Timestamp(TEST_START) <= d <= pd.Timestamp(TEST_END):
            ax1.axvline(d, color="red", alpha=0.4, linewidth=1.5)
            ax1.text(d, ax1.get_ylim()[0] if ax1.get_ylim()[0] != 0 else INITIAL_CASH * 0.8,
                     label, rotation=90, color="red", fontsize=8, alpha=0.8)

    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend()
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.set_title("Portfolio Performance vs Buy & Hold")
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Actions ──────────────────────────
    ax2 = axes[1]
    action_colors = {"BUY": "#4CAF50", "SELL": "#F44336", "HOLD": "#9E9E9E"}
    for date, action in zip(dates, state.actions):
        ax2.axvline(date, color=action_colors[action], alpha=0.4, linewidth=0.8)

    # Price line
    test_price = df[df.index >= TEST_START]["close"]
    ax2_twin = ax2.twinx()
    ax2_twin.plot(test_price.index, test_price.values, color="black", linewidth=1, alpha=0.5)
    ax2_twin.set_ylabel("S&P 500 Price")

    # Legend
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=c, label=a) for a, c in action_colors.items()]
    ax2.legend(handles=legend_els, loc="upper left")
    ax2.set_ylabel("Agent Actions")
    ax2.set_yticks([])
    ax2.set_title("Agent Decisions Over Time")
    ax2.grid(True, alpha=0.2)

    # ── Panel 3: Forecast Accuracy ────────────────
    ax3 = axes[2]
    fc_dates    = [f["date"] for f in state.forecasts]
    fc_actual   = [f["actual"] for f in state.forecasts]
    fc_forecast = [f["forecast_close"] for f in state.forecasts]

    ax3.plot(fc_dates, fc_actual,   label="Actual",           color="black",   linewidth=1.5)
    ax3.plot(fc_dates, fc_forecast, label="Kronos Forecast",  color="#9C27B0", linewidth=1.5, linestyle="--")

    ax3.set_ylabel("Price ($)")
    ax3.legend()
    ax3.set_title("Kronos Forecast vs Actual (1-step ahead)")
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax3.grid(True, alpha=0.3)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator())

    plt.tight_layout()
    chart_file = f"backtest_results_{MODE}.png"
    plt.savefig(chart_file, dpi=150, bbox_inches="tight")
    print(f"\n📊 Chart saved → {chart_file}")
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 55)
    print(f"  KRONOS TRADING AGENT — S&P 500 ({MODE.upper()} MODE)")
    print("=" * 55)

    # 1. Load data
    df = load_data()

    # 2. Load Kronos
    predictor = load_kronos()

    # 3. Run backtest
    state = run_backtest(df, predictor)

    # 4. Baseline
    bh = buy_and_hold(df)

    # 5. Metrics
    metrics = compute_metrics(state, bh)
    print("\n" + "=" * 40)
    print("  RESULTS")
    print("=" * 40)
    for k, v in metrics.items():
        print(f"  {k:<28} {v}")
    print("=" * 40)

    # 6. Save results CSV
    results_df = pd.DataFrame(state.forecasts)
    results_df["action"] = state.actions[:len(results_df)]
    log_file = f"backtest_log_{MODE}.csv"
    results_df.to_csv(log_file, index=False)
    print(f"📁 Detailed log saved → {log_file}")

    # 7. Plot
    plot_results(state, df, bh)

    print("\n✅ Done! Next steps:")
    print("   • Swap rule-based decisions for an RL agent")
    print("   • Add news embeddings as extra context")
    print("   • Test on specific black swan windows")


if __name__ == "__main__":
    main()
