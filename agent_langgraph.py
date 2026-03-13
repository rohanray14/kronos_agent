"""
Kronos Trading Agent — LangGraph + Claude
==========================================
An agentic trading system that uses:
  - Kronos (foundation model) for OHLCV price forecasting
  - Claude (LLM) for reasoning and decision-making
  - LangGraph for the perceive → reason → act → reflect loop

The LLM receives market data and Kronos forecasts as tool calls,
reasons about what to do, then executes a trade decision with
an explanation.

Setup:
    pip install langgraph langchain-anthropic python-dotenv
    cp .env.example .env  # add your ANTHROPIC_API_KEY

Run:
    python agent_langgraph.py              # full 2020-2024 backtest
    python agent_langgraph.py --bear       # 2022-2023 bear market
    python agent_langgraph.py --no-llm     # fallback to rule-based (no API needed)
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Kronos"))

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# Reuse config and utilities from agent.py
# ─────────────────────────────────────────────
from agent import (
    TICKER, LOOKBACK_DAYS, FORECAST_STEPS, MAX_CONTEXT, INITIAL_CASH,
    MODE, TRAIN_START, TEST_START, TEST_END,
    BUY_THRESHOLD, SELL_THRESHOLD, TREND_AGREEMENT, VOLATILITY_MULTIPLIER,
    BLACK_SWANS,
    load_data, load_kronos, get_forecast, execute,
    AgentState as RuleAgentState,
    buy_and_hold, compute_metrics, plot_results,
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

USE_LLM = "--no-llm" not in sys.argv
LLM_MODEL = os.getenv("TRADING_AGENT_MODEL", "claude-haiku-4-5-20251001")

# Resume support: --resume-from STEP replays existing log up to STEP, then continues with LLM
RESUME_FROM_STEP = None
for i, arg in enumerate(sys.argv):
    if arg == "--resume-from" and i + 1 < len(sys.argv):
        RESUME_FROM_STEP = int(sys.argv[i + 1])
        break

# ─────────────────────────────────────────────
# LANGGRAPH AGENT
# ─────────────────────────────────────────────

if USE_LLM:
    from typing import TypedDict, Annotated, Literal
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    from langchain_core.tools import tool
    from langchain_anthropic import ChatAnthropic
    from langgraph.graph import StateGraph, END, add_messages
    from langgraph.prebuilt import ToolNode

    # --- State schema ---
    class TradingState(TypedDict):
        messages: Annotated[list, add_messages]
        # These are set externally before each graph invocation
        current_date: str
        current_price: float
        cash: float
        shares: float
        # Kronos forecast (pre-computed, passed to tools)
        forecast_close: list[float]
        forecast_high: list[float]
        forecast_low: list[float]
        # Historical context
        recent_prices: list[float]
        recent_spread: float
        # Memory
        trade_history: list[dict]
        # Output
        action: str
        reasoning: str

    # --- Global reference for Kronos predictor (can't serialize into state) ---
    _predictor = None
    _current_df = None
    _current_idx = None

    # --- Tools ---
    @tool
    def get_kronos_forecast(dummy: str = "") -> str:
        """Get Kronos model's OHLCV forecast for the next 5 trading days.
        Returns predicted close, high, low prices and derived signals."""
        # Forecast is pre-computed and injected — we reconstruct from global state
        history_df = _current_df.iloc[max(0, _current_idx - LOOKBACK_DAYS):_current_idx]
        current_date = _current_df.index[_current_idx]
        current_price = float(_current_df["close"].iloc[_current_idx])

        future_dates = pd.bdate_range(
            start=current_date, periods=FORECAST_STEPS + 1, freq="B"
        )[1:]
        forecast = get_forecast(_predictor, history_df, future_dates)

        closes = forecast["close"]
        highs = forecast["high"]
        lows = forecast["low"]

        expected_return = (closes[0] - current_price) / current_price
        steps_up = int(np.sum(closes > current_price))
        steps_down = int(np.sum(closes < current_price))

        forecast_spread = float(np.mean(highs - lows) / current_price)
        recent_spread = float(np.mean(
            history_df["high"].values[-20:] - history_df["low"].values[-20:]
        ) / current_price)
        high_vol = forecast_spread > (recent_spread * VOLATILITY_MULTIPLIER)

        return json.dumps({
            "forecast_close": [round(float(c), 1) for c in closes],
            "forecast_high": [round(float(h), 1) for h in highs],
            "forecast_low": [round(float(l), 1) for l in lows],
            "expected_return_pct": round(expected_return * 100, 3),
            "trend": {
                "days_up": steps_up,
                "days_down": steps_down,
                "total_days": len(closes),
                "bullish": steps_up / len(closes) >= TREND_AGREEMENT,
                "bearish": steps_down / len(closes) >= TREND_AGREEMENT,
            },
            "volatility": {
                "forecast_spread_pct": round(forecast_spread * 100, 3),
                "recent_spread_pct": round(recent_spread * 100, 3),
                "high_volatility": high_vol,
            },
        }, indent=2)

    @tool
    def check_portfolio(dummy: str = "") -> str:
        """Check current portfolio state: cash, shares, value, and position status."""
        current_price = float(_current_df["close"].iloc[_current_idx])
        cash = _portfolio_state["cash"]
        shares = _portfolio_state["shares"]
        value = cash + shares * current_price
        invested = shares > 0

        return json.dumps({
            "cash": round(cash, 2),
            "shares": round(shares, 4),
            "portfolio_value": round(value, 2),
            "position": "INVESTED" if invested else "CASH",
            "unrealized_pnl": round(shares * current_price - (INITIAL_CASH - cash), 2) if invested else 0,
        }, indent=2)

    @tool
    def get_market_context(dummy: str = "") -> str:
        """Get recent market statistics: 20-day return, volatility, and price trends."""
        idx = _current_idx
        df = _current_df
        prices = df["close"].values

        recent_20 = prices[max(0, idx - 20):idx]
        recent_50 = prices[max(0, idx - 50):idx]

        ret_20d = (prices[idx] - recent_20[0]) / recent_20[0] if len(recent_20) > 0 else 0
        volatility_20d = float(np.std(np.diff(recent_20) / recent_20[:-1])) if len(recent_20) > 1 else 0
        ma_50 = float(np.mean(recent_50)) if len(recent_50) > 0 else float(prices[idx])

        return json.dumps({
            "current_price": round(float(prices[idx]), 1),
            "20_day_return_pct": round(ret_20d * 100, 2),
            "20_day_volatility_pct": round(volatility_20d * 100, 3),
            "50_day_moving_avg": round(ma_50, 1),
            "price_vs_50ma": "ABOVE" if prices[idx] > ma_50 else "BELOW",
            "price_vs_50ma_pct": round((prices[idx] - ma_50) / ma_50 * 100, 2),
        }, indent=2)

    @tool
    def get_trade_history(n_recent: int = 5) -> str:
        """Review recent trade decisions, their outcomes, and reasoning. Useful for learning from past mistakes."""
        history = _portfolio_state.get("trade_history", [])
        # Cap at 5 most recent to avoid context overflow
        n_recent = min(n_recent, 5)
        recent = history[-n_recent:] if len(history) > n_recent else history
        if not recent:
            return "No trades executed yet."
        # Truncate reasoning to keep context small
        trimmed = []
        for t in recent:
            entry = {k: v for k, v in t.items()}
            if "reasoning" in entry and len(str(entry["reasoning"])) > 100:
                entry["reasoning"] = str(entry["reasoning"])[:100] + "..."
            trimmed.append(entry)
        return json.dumps(trimmed, indent=2, default=str)

    @tool
    def execute_trade(action: str, reasoning: str) -> str:
        """Execute a trade decision. Action must be BUY, SELL, or HOLD. Provide reasoning for the decision."""
        action = action.upper().strip()
        if action not in ("BUY", "SELL", "HOLD"):
            return f"Invalid action '{action}'. Must be BUY, SELL, or HOLD."

        _portfolio_state["pending_action"] = action
        _portfolio_state["pending_reasoning"] = reasoning

        return f"Trade decision recorded: {action}. Reasoning: {reasoning}"

    # --- Portfolio state (mutable, shared across tools) ---
    _portfolio_state = {}

    # --- Build the graph ---
    def build_graph():
        tools = [get_kronos_forecast, check_portfolio, get_market_context,
                 get_trade_history, execute_trade]

        llm = ChatAnthropic(
            model=LLM_MODEL,
            temperature=0,
            max_tokens=500,
        ).bind_tools(tools)

        tool_node = ToolNode(tools)

        def assistant(state: TradingState) -> dict:
            response = llm.invoke(state["messages"])
            return {"messages": [response]}

        def should_continue(state: TradingState) -> Literal["tools", "__end__"]:
            last = state["messages"][-1]
            if hasattr(last, "tool_calls") and last.tool_calls:
                # Check if execute_trade was called
                for tc in last.tool_calls:
                    if tc["name"] == "execute_trade":
                        return "tools"  # process the tool, then end
                return "tools"
            return "__end__"

        def check_done(state: TradingState) -> Literal["assistant", "__end__"]:
            """After tools run, check if execute_trade was the last tool called."""
            if _portfolio_state.get("pending_action"):
                return "__end__"
            return "assistant"

        graph = StateGraph(TradingState)
        graph.add_node("assistant", assistant)
        graph.add_node("tools", tool_node)

        graph.set_entry_point("assistant")
        graph.add_conditional_edges("assistant", should_continue, {
            "tools": "tools",
            "__end__": "__end__",
        })
        graph.add_conditional_edges("tools", check_done, {
            "assistant": "assistant",
            "__end__": "__end__",
        })

        return graph.compile()

    SYSTEM_PROMPT = """You are a quantitative trading agent managing an S&P 500 portfolio.
You have access to Kronos, a financial foundation model that forecasts OHLCV prices.

Your workflow each trading step:
1. Call get_kronos_forecast to get the model's 5-day price prediction
2. Call check_portfolio to see your current position
3. Optionally call get_market_context for technical indicators
4. Optionally call get_trade_history to review past decisions and outcomes
5. Call execute_trade with your decision (BUY, SELL, or HOLD) and a brief reasoning

Decision guidelines:
- Markets have an upward bias. Be willing to buy on modest positive signals.
- Require STRONG evidence to sell (forecast must be clearly negative AND trend must confirm).
- Consider the multi-step trend: do most forecast days agree on the direction?
- In high-volatility regimes, be more cautious — prefer HOLD over risky entries.
- If already invested and forecast is neutral, HOLD (don't churn).
- Keep reasoning concise (2-3 sentences max).

You MUST call execute_trade as your final action. Do not end without making a decision."""


# ─────────────────────────────────────────────
# RULE-BASED FALLBACK (from agent.py)
# ─────────────────────────────────────────────

def decide_rule_based(forecast, current_price, history_df):
    """Original rule-based logic as fallback."""
    from agent import decide
    return decide(forecast, current_price, history_df), "Rule-based decision (no LLM)"


# ─────────────────────────────────────────────
# BACKTEST LOOP
# ─────────────────────────────────────────────

def run_backtest(df, predictor) -> tuple:
    global _predictor, _current_df, _current_idx, _portfolio_state

    _predictor = predictor
    _current_df = df
    dates = df.index

    # Build LangGraph agent
    graph = build_graph() if USE_LLM else None

    # Portfolio tracking (same structure as agent.py for compatibility)
    state = RuleAgentState()
    trade_history = []
    reasoning_log = []

    _portfolio_state = {
        "cash": INITIAL_CASH,
        "shares": 0.0,
        "trade_history": trade_history,
    }

    # --- Resume support: replay existing log up to RESUME_FROM_STEP ---
    if RESUME_FROM_STEP is not None:
        reasoning_file = f"reasoning_log_{MODE}.json"
        if not os.path.exists(reasoning_file):
            # Try the _full variant
            reasoning_file = f"reasoning_log_full.json"
        print(f"\n🔄 Resuming from step {RESUME_FROM_STEP} using {reasoning_file}...")
        with open(reasoning_file, "r") as f:
            existing_log = json.load(f)

        # Replay portfolio state from existing log entries before resume point
        test_indices_all = [i for i, d in enumerate(dates) if d >= pd.Timestamp(TEST_START)]
        step_indices_all = test_indices_all[::FORECAST_STEPS]

        for entry in existing_log:
            if entry["step"] >= RESUME_FROM_STEP:
                break
            step_num = entry["step"]
            idx = step_indices_all[step_num] if step_num < len(step_indices_all) else None
            if idx is None:
                continue

            current_date = dates[idx]
            current_price = float(df["close"].iloc[idx])
            action = entry["action"]

            execute(action, state, current_price)
            _portfolio_state["cash"] = state.cash
            _portfolio_state["shares"] = state.shares

            pv = state.portfolio_value(current_price)
            state.portfolio_values.append(pv)
            state.actions.append(action)
            state.dates.append(current_date)

            # Skip expensive Kronos forecast during replay — use placeholder
            state.forecasts.append({
                "date": current_date,
                "actual": current_price,
                "forecast_close": current_price,
                "forecast_high": current_price,
                "forecast_low": current_price,
            })

            reasoning_log.append(entry)
            trade_history.append({
                "date": entry["date"],
                "price": entry["price"],
                "action": action,
                "portfolio_value": round(pv, 0),
                "reasoning": entry["reasoning"],
            })

        print(f"   ✓ Replayed {len(reasoning_log)} steps, portfolio: ${state.portfolio_value(float(df['close'].iloc[step_indices_all[reasoning_log[-1]['step']]])):.0f}")
        print(f"   Continuing from step {RESUME_FROM_STEP} with LLM...\n")

    test_count = len(df[df.index >= TEST_START])
    print(f"\n🔁 Running backtest ({TEST_START} → {TEST_END})...")
    print(f"   Steps: {test_count} trading days | Forecast horizon: {FORECAST_STEPS} days")
    print(f"   Mode: {'LangGraph + Claude (' + LLM_MODEL + ')' if USE_LLM else 'Rule-based (no LLM)'}\n")

    test_indices = [i for i, d in enumerate(dates) if d >= pd.Timestamp(TEST_START)]
    step_indices = test_indices[::FORECAST_STEPS]

    for step_num, idx in enumerate(step_indices):
        # Skip already-replayed steps when resuming
        if RESUME_FROM_STEP is not None and step_num < RESUME_FROM_STEP:
            continue
        if idx < LOOKBACK_DAYS:
            continue

        _current_idx = idx
        current_date = dates[idx]
        current_price = float(df["close"].iloc[idx])

        if USE_LLM:
            # Reset pending action
            _portfolio_state["pending_action"] = None
            _portfolio_state["pending_reasoning"] = None

            # Invoke LangGraph agent
            initial_state = {
                "messages": [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=(
                        f"Trading date: {current_date.date()}\n"
                        f"S&P 500 price: ${current_price:,.2f}\n"
                        f"Position: {'INVESTED' if _portfolio_state['shares'] > 0 else 'CASH'}\n"
                        f"Portfolio value: ${_portfolio_state['cash'] + _portfolio_state['shares'] * current_price:,.0f}\n\n"
                        f"Analyze the market and make your trading decision."
                    )),
                ],
            }

            try:
                result = graph.invoke(initial_state, {"recursion_limit": 10})
                action = _portfolio_state.get("pending_action", "HOLD")
                reasoning = _portfolio_state.get("pending_reasoning", "No reasoning provided")

                if action is None:
                    action = "HOLD"
                    reasoning = "Agent did not reach a decision — defaulting to HOLD"
            except Exception as e:
                action = "HOLD"
                reasoning = f"LLM error: {e} — defaulting to HOLD"

        else:
            # Rule-based fallback
            history_df = df.iloc[max(0, idx - LOOKBACK_DAYS):idx]
            future_dates = pd.bdate_range(
                start=current_date, periods=FORECAST_STEPS + 1, freq="B"
            )[1:]
            forecast = get_forecast(predictor, history_df, future_dates)
            action, reasoning = decide_rule_based(forecast, current_price, history_df)

        # Execute the trade
        execute(action, state, current_price)

        # Sync portfolio state for tools
        _portfolio_state["cash"] = state.cash
        _portfolio_state["shares"] = state.shares

        pv = state.portfolio_value(current_price)
        state.portfolio_values.append(pv)
        state.actions.append(action)
        state.dates.append(current_date)

        # Get forecast for logging (run if LLM mode since forecast was inside tool)
        history_df = df.iloc[max(0, idx - LOOKBACK_DAYS):idx]
        future_dates = pd.bdate_range(
            start=current_date, periods=FORECAST_STEPS + 1, freq="B"
        )[1:]
        forecast = get_forecast(predictor, history_df, future_dates)

        state.forecasts.append({
            "date": current_date,
            "actual": current_price,
            "forecast_close": forecast["close"][0],
            "forecast_high": forecast["high"][0],
            "forecast_low": forecast["low"][0],
        })

        # Record trade in memory
        trade_record = {
            "date": str(current_date.date()),
            "price": round(current_price, 1),
            "action": action,
            "portfolio_value": round(pv, 0),
            "reasoning": reasoning,
        }
        # Add outcome of previous trade
        if len(trade_history) > 0:
            prev = trade_history[-1]
            prev["outcome_pv"] = round(pv, 0)
            prev["outcome_pnl"] = round(pv - prev["portfolio_value"], 0)

        trade_history.append(trade_record)

        reasoning_log.append({
            "step": step_num,
            "date": str(current_date.date()),
            "price": current_price,
            "action": action,
            "reasoning": reasoning,
            "portfolio_value": pv,
        })

        if step_num % 10 == 0:
            print(f"   [{current_date.date()}] Price: ${current_price:,.0f} | "
                  f"Action: {action:4s} | Portfolio: ${pv:,.0f}")
            if USE_LLM:
                # Print abbreviated reasoning
                short_reason = reasoning[:80] + "..." if len(reasoning) > 80 else reasoning
                print(f"      Reasoning: {short_reason}")

    return state, reasoning_log


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print(f"  KRONOS TRADING AGENT — LANGGRAPH ({MODE.upper()} MODE)")
    print("=" * 60)

    # 1. Load data
    df = load_data()

    # 2. Load Kronos
    predictor = load_kronos()

    # 3. Run backtest
    state, reasoning_log = run_backtest(df, predictor)

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

    # 6. Save results CSV (with reasoning column)
    results_df = pd.DataFrame(state.forecasts)
    results_df["action"] = state.actions[:len(results_df)]
    results_df["reasoning"] = [r["reasoning"] for r in reasoning_log[:len(results_df)]]
    log_file = f"backtest_log_langgraph_{MODE}.csv"
    results_df.to_csv(log_file, index=False)
    print(f"📁 Detailed log saved → {log_file}")

    # 7. Save reasoning log
    reasoning_file = f"reasoning_log_{MODE}.json"
    with open(reasoning_file, "w") as f:
        json.dump(reasoning_log, f, indent=2, default=str)
    print(f"🧠 Reasoning log saved → {reasoning_file}")

    # 8. Plot
    plot_results(state, df, bh)

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
