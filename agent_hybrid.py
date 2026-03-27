"""
Hybrid Agent: RL Gate + Claude Trader
=======================================
Combines the best of both approaches:
  - RL agent (trained on 7 tickers) decides WHETHER to consult Kronos
  - Claude (LLM) decides WHAT to do with the information

The RL gate learned regime-dependent forecast usage:
  - Bull markets: ~100% forecast usage (Kronos is helpful)
  - Bear markets: ~12% forecast usage (Kronos is noise)

Claude gets either:
  - Full context (market + Kronos forecast) when the gate says "use forecast"
  - Market context only when the gate says "skip forecast"

This prevents Claude from being misled by unreliable forecasts in
volatile markets while still giving it Kronos signals when they're useful.

Usage:
    python agent_hybrid.py                    # full 2020-2024 backtest
    python agent_hybrid.py --bear             # 2022-2023 bear market
    python agent_hybrid.py --no-llm           # rule-based fallback (no API)
    python agent_hybrid.py --gate-threshold 0.5  # custom gate threshold
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

# Gate threshold: RL gate outputs probability of using forecast.
# Above this threshold -> use forecast, below -> skip
GATE_THRESHOLD = 0.5
for i, arg in enumerate(sys.argv):
    if arg == "--gate-threshold" and i + 1 < len(sys.argv):
        GATE_THRESHOLD = float(sys.argv[i + 1])


# ─────────────────────────────────────────────
# RL GATE MODEL
# ─────────────────────────────────────────────

class RLGate:
    """
    Uses a trained PPO gated-forecast model to decide whether
    Kronos forecasts are worth requesting at this market state.

    The gate observes market features (returns, volatility, MA position)
    and outputs: True (use forecast) or False (skip forecast).
    """

    def __init__(self, df, predictor, model_path=None):
        self.df = df
        self.predictor = predictor
        self.model = None
        self._shares = 0.0
        self._cash = INITIAL_CASH
        self._days_in_position = 0
        self._last_kronos_features = (0.0, 0.0, 0.0)
        self._forecast_accuracy_history = []
        self._forecast_calls = 0
        self._total_steps = 0

        if model_path and os.path.exists(model_path + ".zip"):
            from stable_baselines3 import PPO
            self.model = PPO.load(model_path)
            print(f"   RL gate loaded from {model_path}.zip")
        else:
            print("   No RL gate model found — using market volatility heuristic")

    def should_use_forecast(self, idx: int) -> tuple[bool, str]:
        """
        Decide whether to call Kronos at this step.
        Returns (use_forecast: bool, gate_reason: str)
        """
        self._total_steps += 1

        if self.model is not None:
            return self._rl_gate(idx)
        else:
            return self._heuristic_gate(idx)

    def get_confidence(self, idx: int) -> tuple[float, str]:
        """
        Return a confidence score (0-1) for how reliable forecasts are right now.
        Used by the soft gate — Claude always gets the forecast but with this score.
        """
        self._total_steps += 1

        if self.model is not None:
            return self._rl_confidence(idx)
        else:
            return self._heuristic_confidence(idx)

    def _rl_confidence(self, idx: int) -> tuple[float, str]:
        """Use trained RL model action probabilities as confidence."""
        obs = self._build_observation(idx)
        action, _ = self.model.predict(obs, deterministic=False)
        # Get action probabilities
        import torch
        obs_tensor = torch.as_tensor(obs).float().unsqueeze(0)
        dist = self.model.policy.get_distribution(obs_tensor)
        probs = dist.distribution.probs.detach().numpy()[0]
        # Actions 3,4,5 = with forecast; sum their probability
        forecast_prob = float(probs[3:].sum())
        self._forecast_calls += 1
        reason = f"RL soft gate: confidence={forecast_prob:.0%}"
        return forecast_prob, reason

    def _heuristic_confidence(self, idx: int) -> tuple[float, str]:
        """
        Map volatility to a 0-1 confidence score.
        Low vol -> high confidence, high vol -> low confidence.
        """
        prices = self.df["close"].values
        recent_20 = prices[max(0, idx - 20):idx]

        if len(recent_20) > 1:
            daily_rets = np.diff(recent_20) / recent_20[:-1]
            vol_20d = float(np.std(daily_rets))
        else:
            vol_20d = 0.0

        # Scale: 0% vol -> 1.0 confidence, 2%+ vol -> 0.0 confidence
        confidence = max(0.0, min(1.0, 1.0 - (vol_20d / 0.02)))
        self._forecast_calls += 1

        reason = (f"Soft gate: confidence={confidence:.0%} "
                  f"(vol={vol_20d*100:.2f}%)")
        return confidence, reason

    def _rl_gate(self, idx: int) -> tuple[bool, str]:
        """Use trained RL model to decide."""
        obs = self._build_observation(idx)
        action, _ = self.model.predict(obs, deterministic=True)
        action = int(action)
        use_forecast = action >= 3  # actions 3,4,5 = with forecast

        if use_forecast:
            self._forecast_calls += 1

        rate = self._forecast_calls / max(1, self._total_steps)
        reason = (f"RL gate: {'USE' if use_forecast else 'SKIP'} forecast "
                  f"(action={action}, rate={rate:.0%})")
        return use_forecast, reason

    def _heuristic_gate(self, idx: int) -> tuple[bool, str]:
        """
        Fallback heuristic based on the RL finding:
        high volatility -> skip forecast, low volatility -> use forecast.
        """
        prices = self.df["close"].values
        recent_20 = prices[max(0, idx - 20):idx]

        if len(recent_20) > 1:
            daily_rets = np.diff(recent_20) / recent_20[:-1]
            vol_20d = float(np.std(daily_rets))
        else:
            vol_20d = 0.0

        # Median daily vol for S&P 500 is roughly 0.8-1.0%
        # In bear markets it spikes to 1.5-2.0%+
        high_vol_threshold = 0.015
        use_forecast = vol_20d < high_vol_threshold

        if use_forecast:
            self._forecast_calls += 1

        rate = self._forecast_calls / max(1, self._total_steps)
        reason = (f"Volatility gate: {'USE' if use_forecast else 'SKIP'} forecast "
                  f"(vol={vol_20d*100:.2f}%, threshold={high_vol_threshold*100:.1f}%, "
                  f"rate={rate:.0%})")
        return use_forecast, reason

    def _build_observation(self, idx: int) -> np.ndarray:
        """Build the 12-dim observation vector for the gated env."""
        prices = self.df["close"].values
        current_price = float(prices[idx])

        position = 1.0 if self._shares > 0 else 0.0

        recent_20 = prices[max(0, idx - 20):idx]
        recent_50 = prices[max(0, idx - 50):idx]
        ret_20d = ((current_price - float(recent_20[0])) / float(recent_20[0])
                   if len(recent_20) > 0 else 0.0)
        ret_50d = ((current_price - float(recent_50[0])) / float(recent_50[0])
                   if len(recent_50) > 0 else 0.0)

        if len(recent_20) > 1:
            daily_rets = np.diff(recent_20) / recent_20[:-1]
            vol_20d = float(np.std(daily_rets))
        else:
            vol_20d = 0.0

        ma_50 = float(np.mean(recent_50)) if len(recent_50) > 0 else current_price
        price_vs_50ma = (current_price - ma_50) / ma_50

        kronos_ret, kronos_trend, kronos_high_vol = self._last_kronos_features

        pv = self._cash + self._shares * current_price
        portfolio_return = (pv - INITIAL_CASH) / INITIAL_CASH
        days_norm = self._days_in_position / 50.0

        # Forecast accuracy
        if len(self._forecast_accuracy_history) >= 3:
            forecast_accuracy = float(np.mean(self._forecast_accuracy_history[-10:]))
        else:
            forecast_accuracy = 0.5

        return np.array([
            position, ret_20d, ret_50d, vol_20d, price_vs_50ma,
            kronos_ret, kronos_trend, kronos_high_vol,
            portfolio_return, days_norm,
            0.0,  # used_forecast flag (not known yet)
            forecast_accuracy,
        ], dtype=np.float32)

    def sync_portfolio(self, cash, shares, days_in_position):
        """Sync portfolio state from the main backtest loop."""
        self._cash = cash
        self._shares = shares
        self._days_in_position = days_in_position


# ─────────────────────────────────────────────
# LANGGRAPH AGENT (modified for gating)
# ─────────────────────────────────────────────

if USE_LLM:
    from typing import TypedDict, Annotated, Literal
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.tools import tool
    from langchain_anthropic import ChatAnthropic
    from langgraph.graph import StateGraph, add_messages
    from langgraph.prebuilt import ToolNode

    class TradingState(TypedDict):
        messages: Annotated[list, add_messages]

    # Global state for tools
    _predictor = None
    _current_df = None
    _current_idx = None
    _portfolio_state = {}
    _forecast_gated = False  # True = forecast available, False = blocked by RL gate
    _forecast_confidence = 1.0  # Soft gate confidence score (0-1)

    @tool
    def get_kronos_forecast(dummy: str = "") -> str:
        """Get Kronos model's OHLCV forecast for the next 5 trading days.
        Returns predicted close, high, low prices, derived signals,
        and an RL-derived reliability score (0-100%)."""
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

        # Confidence label based on soft gate score
        conf = _forecast_confidence
        if conf >= 0.7:
            reliability_label = "HIGH — forecast is likely reliable, weight it heavily"
        elif conf >= 0.4:
            reliability_label = "MEDIUM — forecast may be noisy, use with caution"
        else:
            reliability_label = "LOW — market is volatile, forecast is probably unreliable, treat with heavy skepticism"

        return json.dumps({
            "forecast_reliability": {
                "score_pct": round(conf * 100, 0),
                "label": reliability_label,
            },
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
            "unrealized_pnl": round(
                shares * current_price - (INITIAL_CASH - cash), 2
            ) if invested else 0,
        }, indent=2)

    @tool
    def get_market_context(dummy: str = "") -> str:
        """Get recent market statistics: 20-day return, volatility, and price trends."""
        idx = _current_idx
        df = _current_df
        prices = df["close"].values

        recent_20 = prices[max(0, idx - 20):idx]
        recent_50 = prices[max(0, idx - 50):idx]

        ret_20d = ((prices[idx] - recent_20[0]) / recent_20[0]
                   if len(recent_20) > 0 else 0)
        volatility_20d = (float(np.std(np.diff(recent_20) / recent_20[:-1]))
                          if len(recent_20) > 1 else 0)
        ma_50 = (float(np.mean(recent_50))
                 if len(recent_50) > 0 else float(prices[idx]))

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
        """Review recent trade decisions, their outcomes, and reasoning."""
        history = _portfolio_state.get("trade_history", [])
        n_recent = min(n_recent, 5)
        recent = history[-n_recent:] if len(history) > n_recent else history
        if not recent:
            return "No trades executed yet."
        trimmed = []
        for t in recent:
            entry = {k: v for k, v in t.items()}
            if "reasoning" in entry and len(str(entry["reasoning"])) > 100:
                entry["reasoning"] = str(entry["reasoning"])[:100] + "..."
            trimmed.append(entry)
        return json.dumps(trimmed, indent=2, default=str)

    @tool
    def execute_trade(action: str, reasoning: str) -> str:
        """Execute a trade decision. Action must be BUY, SELL, or HOLD."""
        action = action.upper().strip()
        if action not in ("BUY", "SELL", "HOLD"):
            return f"Invalid action '{action}'. Must be BUY, SELL, or HOLD."
        _portfolio_state["pending_action"] = action
        _portfolio_state["pending_reasoning"] = reasoning
        return f"Trade decision recorded: {action}. Reasoning: {reasoning}"

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
                return "tools"
            return "__end__"

        def check_done(state: TradingState) -> Literal["assistant", "__end__"]:
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
Each forecast comes with an RL-derived RELIABILITY SCORE (0-100%) that tells you how trustworthy the forecast is in the current market regime.

Your workflow each trading step:
1. Call check_portfolio to see your current position
2. Call get_kronos_forecast — pay close attention to the reliability score
3. Optionally call get_market_context for technical indicators
4. Optionally call get_trade_history to review past decisions and outcomes
5. Call execute_trade with your decision (BUY, SELL, or HOLD) and a brief reasoning

Decision guidelines:
- Markets have an upward bias. Your default when invested should be HOLD.
- WEIGHT THE FORECAST BY ITS RELIABILITY SCORE:
  * HIGH reliability (70-100%): Trust the forecast. Act on clear bullish/bearish signals.
  * MEDIUM reliability (40-70%): Forecast is noisy. Only act on very strong signals (e.g., 5/5 day consensus AND >1% expected return). Otherwise HOLD.
  * LOW reliability (0-40%): Nearly ignore the forecast. Fall back to market context (price vs MA, volatility, recent returns). Be very conservative.
- BUYING: Buy when in CASH and either (a) forecast is bullish with HIGH reliability, or (b) market is technically oversold regardless of forecast.
- SELLING: Require STRONG evidence. A bearish forecast alone is NOT enough — Kronos often predicts short-term dips that reverse. To sell, you need:
  * Bearish forecast (negative return, majority days down) WITH high reliability, OR
  * Clear technical breakdown (price well below 50-day MA, negative momentum, rising volatility) regardless of forecast
- Avoid churning: if you recently sold and bought back, you are over-trading. Bias toward HOLD.
- Keep reasoning concise (2-3 sentences max).

You MUST call execute_trade as your final action. Do not end without making a decision."""


# ─────────────────────────────────────────────
# BACKTEST LOOP
# ─────────────────────────────────────────────

def run_backtest(df, predictor) -> tuple:
    global _predictor, _current_df, _current_idx, _portfolio_state, _forecast_gated, _forecast_confidence

    _predictor = predictor
    _current_df = df
    dates = df.index

    # Build LangGraph agent
    graph = build_graph() if USE_LLM else None

    # Initialize RL gate (soft mode)
    gate_model_path = f"ppo_gated_{MODE}"
    rl_gate = RLGate(df, predictor, model_path=gate_model_path)

    # Portfolio tracking
    state = RuleAgentState()
    trade_history = []
    reasoning_log = []
    gate_log = []

    _portfolio_state = {
        "cash": INITIAL_CASH,
        "shares": 0.0,
        "trade_history": trade_history,
    }

    days_in_position = 0

    test_count = len(df[df.index >= TEST_START])
    print(f"\n   Running hybrid backtest ({TEST_START} -> {TEST_END})...")
    print(f"   Steps: {test_count} trading days | Forecast horizon: {FORECAST_STEPS} days")
    if USE_LLM:
        print(f"   Trader: Claude ({LLM_MODEL})")
    else:
        print(f"   Trader: Rule-based fallback")
    print(f"   Gate: SOFT {'(RL model)' if rl_gate.model else '(volatility heuristic)'}")
    print(f"   Mode: Always show forecast + reliability score\n")

    test_indices = [i for i, d in enumerate(dates) if d >= pd.Timestamp(TEST_START)]
    step_indices = test_indices[::FORECAST_STEPS]

    forecast_used_count = 0
    total_steps = 0

    for step_num, idx in enumerate(step_indices):
        if idx < LOOKBACK_DAYS:
            continue

        _current_idx = idx
        current_date = dates[idx]
        current_price = float(df["close"].iloc[idx])
        total_steps += 1

        # ─── SOFT GATE: GET CONFIDENCE SCORE ───
        rl_gate.sync_portfolio(
            _portfolio_state["cash"],
            _portfolio_state["shares"],
            days_in_position,
        )
        confidence, gate_reason = rl_gate.get_confidence(idx)
        _forecast_confidence = confidence
        _forecast_gated = True  # Always provide forecast in soft gate mode

        forecast_used_count += 1

        gate_log.append({
            "step": step_num,
            "date": str(current_date.date()),
            "confidence": round(confidence, 3),
            "gate_reason": gate_reason,
        })

        # ─── CLAUDE TRADING DECISION ───
        if USE_LLM:
            _portfolio_state["pending_action"] = None
            _portfolio_state["pending_reasoning"] = None

            conf_pct = round(confidence * 100)
            if confidence >= 0.7:
                conf_label = "HIGH"
            elif confidence >= 0.4:
                conf_label = "MEDIUM"
            else:
                conf_label = "LOW"

            initial_state = {
                "messages": [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=(
                        f"Trading date: {current_date.date()}\n"
                        f"S&P 500 price: ${current_price:,.2f}\n"
                        f"Position: {'INVESTED' if _portfolio_state['shares'] > 0 else 'CASH'}\n"
                        f"Portfolio value: ${_portfolio_state['cash'] + _portfolio_state['shares'] * current_price:,.0f}\n"
                        f"Forecast reliability: {conf_pct}% ({conf_label})\n\n"
                        f"Analyze the market and make your trading decision."
                    )),
                ],
            }

            try:
                result = graph.invoke(initial_state, {"recursion_limit": 10})
                action = _portfolio_state.get("pending_action", "HOLD")
                reasoning = _portfolio_state.get("pending_reasoning",
                                                  "No reasoning provided")
                if action is None:
                    action = "HOLD"
                    reasoning = "Agent did not reach a decision — defaulting to HOLD"
            except Exception as e:
                action = "HOLD"
                reasoning = f"LLM error: {e} — defaulting to HOLD"

        else:
            # Rule-based fallback (always uses forecast in soft gate mode)
            history_df = df.iloc[max(0, idx - LOOKBACK_DAYS):idx]
            future_dates = pd.bdate_range(
                start=current_date, periods=FORECAST_STEPS + 1, freq="B"
            )[1:]

            from agent import decide
            forecast = get_forecast(predictor, history_df, future_dates)
            action = decide(forecast, current_price, history_df)
            reasoning = f"Rule-based with forecast (confidence: {confidence:.0%})"

        # ─── EXECUTE ───
        prev_invested = _portfolio_state["shares"] > 0
        execute(action, state, current_price)

        _portfolio_state["cash"] = state.cash
        _portfolio_state["shares"] = state.shares

        # Track days in position
        if action in ("BUY", "SELL"):
            days_in_position = 0
        else:
            days_in_position += 1

        pv = state.portfolio_value(current_price)
        state.portfolio_values.append(pv)
        state.actions.append(action)
        state.dates.append(current_date)

        # Log forecast for charting
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

        # Trade history for Claude's memory
        trade_record = {
            "date": str(current_date.date()),
            "price": round(current_price, 1),
            "action": action,
            "portfolio_value": round(pv, 0),
            "reasoning": reasoning,
            "confidence": round(confidence, 3),
        }
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
            "confidence": round(confidence, 3),
            "gate_reason": gate_reason,
        })

        conf_sym = f"{round(confidence*100)}%"
        if step_num % 10 == 0:
            print(f"   [{current_date.date()}] Price: ${current_price:,.0f} | "
                  f"Conf: {conf_sym:>4s} | Action: {action:4s} | "
                  f"Portfolio: ${pv:,.0f}")
            if USE_LLM:
                short_reason = (reasoning[:70] + "..."
                                if len(reasoning) > 70 else reasoning)
                print(f"      Reasoning: {short_reason}")

    # Compute average confidence
    confidences = [g["confidence"] for g in gate_log]
    avg_confidence = float(np.mean(confidences)) if confidences else 0.0
    low_conf_steps = sum(1 for c in confidences if c < 0.4)
    print(f"\n   Avg forecast confidence: {avg_confidence:.0%}")
    print(f"   Low confidence steps (<40%): {low_conf_steps}/{total_steps}")

    return state, reasoning_log, gate_log, avg_confidence


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print(f"  HYBRID AGENT: SOFT GATE + CLAUDE TRADER ({MODE.upper()})")
    print("=" * 60)

    # 1. Load data
    df = load_data()

    # 2. Load Kronos
    predictor = load_kronos()

    # 3. Run backtest
    state, reasoning_log, gate_log, avg_confidence = run_backtest(df, predictor)

    # 4. Baseline
    bh = buy_and_hold(df)

    # 5. Metrics
    metrics = compute_metrics(state, bh)
    print("\n" + "=" * 50)
    print("  HYBRID AGENT RESULTS (SOFT GATE)")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:<28} {v}")
    print(f"  {'Avg Forecast Confidence':<28} {avg_confidence:.0%}")
    print("=" * 50)

    # 6. Save results
    results_df = pd.DataFrame(state.forecasts)
    results_df["action"] = state.actions[:len(results_df)]
    results_df["reasoning"] = [r["reasoning"] for r in reasoning_log[:len(results_df)]]
    results_df["forecast_used"] = [r["forecast_used"] for r in reasoning_log[:len(results_df)]]
    log_file = f"backtest_log_hybrid_{MODE}.csv"
    results_df.to_csv(log_file, index=False)
    print(f"\n   Detailed log saved -> {log_file}")

    reasoning_file = f"reasoning_log_hybrid_{MODE}.json"
    with open(reasoning_file, "w") as f:
        json.dump(reasoning_log, f, indent=2, default=str)
    print(f"   Reasoning log saved -> {reasoning_file}")

    gate_file = f"gate_log_hybrid_{MODE}.json"
    with open(gate_file, "w") as f:
        json.dump(gate_log, f, indent=2, default=str)
    print(f"   Gate log saved -> {gate_file}")

    # 7. Plot
    plot_results(state, df, bh)

    print("\nDone!")


if __name__ == "__main__":
    main()
