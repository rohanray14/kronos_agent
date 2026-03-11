# Kronos Trading Agent — Presentation Guide

Use this document as your source material when building slides.
Every slide's content, key numbers, and talking points are here.

---

## Slide 1: Title

**Title:** Kronos Trading Agent: An Agentic System for Financial Market Forecasting

**Subtitle:** Using Foundation Models + LLM Reasoning to Beat the Market

**Your name, course, date**

---

## Slide 2: Problem Statement

**What we're asking:**
> Can an AI agent that combines a financial forecasting model with LLM-based reasoning outperform the market — especially during crashes?

**Why it matters:**
- Traditional trading algorithms use fixed rules that can't adapt
- Foundation models can now forecast financial data with high accuracy
- But a forecast alone isn't a decision — you need reasoning about *when and how* to act
- The 2020–2024 period includes COVID, rate hikes, and bank collapses — a real stress test

---

## Slide 3: Architecture Overview

**Use this diagram on the slide:**

```
┌─────────────────────────────────────────────────────┐
│                   LangGraph Agent Loop               │
│                                                      │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│   │ PERCEIVE │───>│  REASON  │───>│   ACT    │──┐   │
│   │ (Tools)  │    │ (Claude) │    │ (Trade)  │  │   │
│   └──────────┘    └──────────┘    └──────────┘  │   │
│        ^                                         │   │
│        │              ┌──────────┐               │   │
│        └──────────────│ REFLECT  │<──────────────┘   │
│                       │ (Memory) │                    │
│                       └──────────┘                    │
│                                                      │
│   Tools available:                                   │
│   - get_kronos_forecast()  (Kronos model)            │
│   - check_portfolio()      (current state)           │
│   - get_market_context()   (technical indicators)    │
│   - get_trade_history()    (memory of past trades)   │
│   - execute_trade()        (BUY / SELL / HOLD)       │
└─────────────────────────────────────────────────────┘
```

**Talking points:**
- Every 5 trading days, the agent wakes up and runs this loop
- It calls tools to gather information, reasons about what to do, then acts
- Past trades and outcomes are stored in memory — it can learn from mistakes
- Two foundation models: Kronos (forecasting) + Claude (reasoning)

---

## Slide 4: Kronos — The Forecasting Model

**What is Kronos?**
- A foundation model built specifically for financial markets
- Trained on 12 billion+ candlestick records from 45 global exchanges
- Predicts full OHLCV (Open, High, Low, Close, Volume) — not just price

**Architecture (two stages):**

| Stage | What it does | How |
|-------|-------------|-----|
| **Tokenizer** | Converts OHLCV prices into discrete tokens | Encoder → Binary Spherical Quantizer → Decoder |
| **Predictor** | Autoregressively generates future tokens | 12-layer decoder-only Transformer (like GPT) |

**Key specs:**
- Model used: Kronos-small (24.7M parameters)
- Context window: 512 tokens
- Input: 60 days of OHLCV history
- Output: 5 days of predicted OHLCV

**Why OHLCV matters:**
- Close price tells you *where* the market ended
- High-Low spread tells you *how volatile* it was
- The agent uses both — volatility signals help it avoid risky entries

---

## Slide 5: Three Approaches Compared

**Show this as a progression:**

```
Level 1: Buy & Hold         →  No model, no decisions
Level 2: Rule-Based Agent   →  Kronos forecasts + if/else rules
Level 3: LangGraph Agent    →  Kronos forecasts + LLM reasoning + memory
```

| Feature | Buy & Hold | Rule-Based | LangGraph Agent |
|---------|-----------|------------|-----------------|
| Forecasting model | None | Kronos | Kronos |
| Decision-making | None | Hardcoded thresholds | Claude LLM reasoning |
| Adapts to context | No | No | Yes |
| Explains decisions | No | No | Yes (natural language) |
| Memory of past trades | No | No | Yes |
| Tool use | No | No | Yes (4 tools) |

---

## Slide 6: The Decision Engine — Rule-Based vs LLM

**Rule-based (if/else):**
```
IF forecast > +0.3% AND trend confirms → BUY
IF forecast < -1.5% AND trend confirms → SELL
OTHERWISE → HOLD
```
Fixed. Can't weigh conflicting signals. Doesn't learn.

**LLM-based (Claude):**
> "Kronos forecasts -2.86% return with price 5.38% below 50-day MA,
> confirming downtrend. Lock in +$387 gain before forecasted weakness
> materializes. Bearish technical setup warrants exiting to cash."

Considers multiple signals, weighs context, explains reasoning.

**Key difference:** When Kronos says "-1.2% expected return," the rule-based agent sees it's above the -1.5% threshold and holds. Claude sees the same number and thinks: *"That's borderline, but all 5 days are down, volatility is spiking, and my last two holds lost money — I should sell."*

---

## Slide 7: Results — Bear Market (2022–2023)

**Use the chart:** `backtest_results_bear.png`

| Metric | LangGraph + Claude | Rule-Based | Buy & Hold |
|--------|-------------------|------------|------------|
| **Total Return** | **+18.5%** | -1.1% | -8.3% |
| **Sharpe Ratio** | **0.83** | 0.05 | — |
| **Max Drawdown** | **-11.3%** | -20.1% | ~-25% |
| **Final Portfolio** | **$118,493** | $98,893 | $91,700 |

**Talking points:**
- The market lost 8.3%. The LangGraph agent *gained* 18.5%. That's a 26.8pp spread.
- Max drawdown was halved — the agent protected capital during the worst drops
- Sharpe ratio of 0.83 is strong risk-adjusted performance
- The agent sold early in downturns, stayed in cash, and bought near bottoms

---

## Slide 8: Results — Full Period (2020–2024)

**Use the chart:** `backtest_results_full.png`

| Metric | LangGraph + Claude | Rule-Based | Buy & Hold |
|--------|-------------------|------------|------------|
| **Total Return** | **+134.4%** | +39.3% | +81.3% |
| **Sharpe Ratio** | **1.18** | 0.48 | — |
| **Max Drawdown** | -22.0% | -25.5% | ~-34% |
| **Final Portfolio** | **$234,431** | $139,254 | $181,300 |

**Talking points:**
- The agent turned $100K into $234K — beating the market by 53 percentage points
- Sharpe ratio above 1.0 is considered excellent
- It outperformed in *both* bull and bear markets — rare for any strategy
- The rule-based agent underperformed buy & hold (39% vs 81%), showing that rules alone aren't enough

---

## Slide 9: How the Agent Thinks — Example Reasoning

**Pick 3-4 of these quotes to show on the slide:**

**Selling before COVID crash (March 2, 2020):**
> "Kronos forecasts -2.86% return with price 5.38% below 50-day MA, confirming downtrend. Lock in +$387 gain before forecasted weakness materializes."

**Buying the COVID bottom (March 9, 2020):**
> "Kronos forecasts unanimous 5-day uptrend (+7.5% return). Market is deeply oversold (15.5% below 50-day MA), providing attractive entry. Strong bullish consensus warrants deploying cash to capture forecasted recovery."

**Holding through volatility (March 16, 2020):**
> "Kronos forecasts unanimous 5-day uptrend (+12.6% return). Market is deeply oversold (25% below 50-day MA), and strong bullish consensus supports holding through the forecasted recovery phase."

**Recognizing mixed signals (July 5, 2022):**
> "Kronos forecasts 4 down days out of 5 with price 4.35% below 50-day MA in confirmed downtrend. Despite +0.263% expected return, the bearish consensus is clear. Remain in cash to avoid repeating recent losses."

**Learning from mistakes (Feb 8, 2022):**
> "Bearish trend with weak conviction despite +0.654% expected return. Recent bullish forecasts failed; market remains below 50-day MA. Waiting for clearer bullish signals before deploying capital."

---

## Slide 10: Why the LLM Agent Won

**Three key advantages over rule-based:**

1. **Contextual reasoning** — Weighs multiple conflicting signals instead of checking one threshold. Can say "the forecast is slightly positive but everything else looks bad — hold."

2. **Memory and adaptation** — Reviews past trades. When recent buys lost money, it becomes more cautious. The rule-based agent repeats the same mistakes.

3. **Nuanced risk management** — Understands concepts like "oversold," "mean reversion," "catching a falling knife." These are hard to encode as rules but natural for an LLM.

**The core insight:**
> The value isn't just in *predicting* the market (Kronos does that).
> It's in *reasoning about how to act* on those predictions (Claude does that).
> Each layer of intelligence improved performance: 81% → 39% → 134%.

---

## Slide 11: Technical Stack

| Component | Technology | Role |
|-----------|-----------|------|
| Forecasting | Kronos-small (24.7M params) | Predicts 5-day OHLCV prices |
| Reasoning | Claude Haiku 4.5 | Analyzes signals, makes decisions |
| Agent framework | LangGraph | State machine, tool routing, agent loop |
| Market data | yfinance | S&P 500 OHLCV download |
| Language | Python | All implementation |
| Data | S&P 500 (2018–2024) | 1,760 trading days |

**Run commands:**
```bash
python agent_langgraph.py            # full 2020-2024
python agent_langgraph.py --bear     # bear market 2022-2023
python agent_langgraph.py --no-llm   # rule-based fallback
```

---

## Slide 12: Limitations & Future Work

**Limitations:**
- No cross-run learning — Claude doesn't update weights between sessions
- Backtest only — not live trading (no slippage, fees, or execution risk)
- LLM non-determinism — results may vary slightly between runs
- API cost — each backtest run costs ~$0.15–0.30 (Haiku)

**Future work:**
- Replace rule-based fallback with reinforcement learning (PPO/DQN)
- Add news sentiment embeddings as additional context for Claude
- Test on individual stocks and crypto markets
- Use Kronos-base (102M params) for better forecast accuracy
- Add position sizing (fractional trades instead of all-in/all-out)

---

## Slide 13: Conclusion

> We built an agentic trading system that combines two foundation models —
> Kronos for financial forecasting and Claude for reasoning — connected
> through a LangGraph agent loop with tool use and memory.
>
> The agent outperformed buy & hold in **both** bull (+134% vs +81%) and
> bear (+18.5% vs -8.3%) markets, demonstrating that intelligent reasoning
> on top of predictions is more valuable than predictions alone.

---

## Appendix: Files in the Project

| File | What it is |
|------|-----------|
| `agent.py` | Rule-based agent (Kronos + if/else thresholds) |
| `agent_langgraph.py` | LangGraph agent (Kronos + Claude reasoning) |
| `Kronos/` | Cloned Kronos model repo |
| `backtest_results_full.png` | Chart — full period results |
| `backtest_results_bear.png` | Chart — bear market results |
| `backtest_log_langgraph_full.csv` | Trade log with forecasts and actions |
| `backtest_log_langgraph_bear.csv` | Trade log — bear market |
| `reasoning_log_full.json` | Every Claude reasoning explanation (full) |
| `reasoning_log_bear.json` | Every Claude reasoning explanation (bear) |
| `requirements.txt` | Python dependencies |
| `.env` | Anthropic API key (do not share) |
