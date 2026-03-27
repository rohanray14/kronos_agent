# Kronos Trading Agent: An Agentic System for Financial Market Forecasting

**Rohan Ray | COSC 89.35 | Final Project Report**

## 1. Research Question

Can an AI agent that combines a domain-specific foundation model for financial forecasting with LLM-based reasoning outperform passive market strategies — particularly during periods of high volatility and market downturns?

Traditional algorithmic trading systems rely on fixed, hand-crafted rules that cannot adapt to changing market conditions. Recent advances in foundation models have produced systems capable of accurate financial time series forecasting, but a forecast alone does not constitute a decision. This project investigates whether adding an explicit reasoning layer — an LLM that interprets forecasts, considers context, and learns from past trades — produces meaningfully better trading outcomes than either passive investing or rule-based strategies built on the same forecasts.

## 2. Methodology

### System Architecture

The system combines two foundation models within a LangGraph agent loop:

**Kronos** (forecasting): A decoder-only transformer trained on 12 billion+ K-line (candlestick) records from 45 global financial exchanges (Shi et al., 2025). Unlike general-purpose time series models, Kronos operates on full OHLCV (Open, High, Low, Close, Volume) data through a two-stage architecture: (1) a tokenizer that encodes continuous OHLCV values into hierarchical discrete tokens via Binary Spherical Quantization (BSQ), and (2) a 24.7M-parameter autoregressive transformer (Kronos-small) that generates future tokens. This produces rich multi-dimensional forecasts — not just predicted prices, but predicted volatility through the high-low spread.

**Claude Haiku 4.5** (reasoning): An LLM that serves as the agent's decision-making brain. Rather than applying fixed thresholds to Kronos outputs, Claude receives the forecast alongside portfolio state, technical indicators, and a memory of past trades, then reasons in natural language about whether to BUY, SELL, or HOLD.

**LangGraph** (orchestration): A state machine framework that implements the agent loop: Perceive (call tools) → Reason (Claude analyzes signals) → Act (execute trade) → Reflect (store outcome in memory). The agent has access to five tools: `get_kronos_forecast`, `check_portfolio`, `get_market_context`, `get_trade_history`, and `execute_trade`. Every 5 trading days, the agent wakes up, gathers information through these tools, and makes one trading decision with an explicit explanation.

### Experimental Setup

We backtested on the S&P 500 index using daily OHLCV data from Yahoo Finance. Two test periods were evaluated: a full period (2020–2024) spanning both bull and bear markets including COVID, Fed rate hikes, and the SVB collapse; and an isolated bear market period (Jan 2022 – Jun 2023) covering the Fed tightening cycle. The agent begins with $100,000 in cash and executes all-in/all-out trades for simplicity. We compared three approaches: (1) Buy & Hold as a passive baseline, (2) a Rule-Based agent using Kronos forecasts with hand-tuned asymmetric thresholds, trend confirmation, and volatility filtering, and (3) the full LangGraph agent with Claude reasoning.

## 3. Results and Analysis

| Metric | LangGraph + Claude | Rule-Based | Buy & Hold |
|---|---|---|---|
| Bear Return (2022–23) | **+18.5%** | −1.1% | −8.3% |
| Full Return (2020–24) | **+134.4%** | +39.3% | +81.3% |
| Bear Sharpe Ratio | **0.83** | 0.05 | — |
| Full Sharpe Ratio | **1.18** | 0.48 | — |
| Bear Max Drawdown | **−11.3%** | −20.1% | ~−25% |
| Full Max Drawdown | −22.0% | −25.5% | ~−34% |

The LangGraph agent outperformed both baselines across all metrics and market conditions. In the bear market, it returned +18.5% while the market lost 8.3% — a 26.8 percentage point spread. Over the full period, it returned 134.4% versus 81.3% for buy and hold, with a Sharpe ratio of 1.18 (considered excellent). Crucially, it achieved lower maximum drawdowns in both periods, indicating superior risk management.

The rule-based agent presents an interesting contrast: it beat buy and hold in the bear market (−1.1% vs −8.3%) but significantly underperformed in the full period (39.3% vs 81.3%). This reveals a fundamental limitation of fixed rules — they cannot simultaneously optimize for capital preservation in downturns and participation in rallies. The asymmetric thresholds that prevented unnecessary selling also caused the agent to miss re-entry points during recoveries.

Analysis of Claude's reasoning logs reveals three key advantages. First, **contextual signal integration**: when Kronos produced a marginally positive forecast but technical indicators and trend analysis were bearish, Claude held rather than buying — something the threshold-based rules could not express. Second, **adaptive risk management**: after a series of losing trades, Claude explicitly referenced past failures in its reasoning ("recent bullish forecasts failed; waiting for clearer signals"), effectively adjusting its strategy within the session. Third, **regime awareness**: Claude recognized market concepts like "oversold," "catching a falling knife," and "mean reversion" — concepts that are natural for an LLM but nearly impossible to encode as rules.

## 4. Limitations

Several limitations warrant discussion. The LLM has prior knowledge of these market events in its training data, introducing potential lookahead bias — though Claude does not have access to future prices, its general knowledge of events like COVID or SVB may influence reasoning. The backtest does not account for transaction costs, slippage, or market impact. The all-in/all-out position sizing is unrealistic for real trading. Claude's non-determinism means results vary slightly between runs. Finally, the agent exhibits bounded rationality — it adapts within a session through memory but does not update model weights across sessions, distinguishing it from a true reinforcement learning agent.

## 5. Connection to Course Concepts

This project directly implements several core concepts from the course. The LangGraph agent embodies the **agentic AI paradigm** — an autonomous system that perceives, reasons, acts, and reflects in a loop, using tools to interact with its environment. The architecture demonstrates **tool use** as a key capability of modern LLM agents, with five distinct tools providing structured access to forecasting, portfolio management, and memory. The **ReAct pattern** (Yao et al., 2023) is realized through Claude's interleaved reasoning and tool-calling behavior. The memory mechanism connects to work on **retrieval-augmented generation** and agent memory systems, where past experiences inform future decisions. Finally, the two-model architecture illustrates the emerging pattern of **compound AI systems** (Zaharia et al., 2024) — combining specialized models (Kronos for domain-specific forecasting) with general-purpose reasoning (Claude) to achieve capabilities neither could alone.

## References

- Shi, S., et al. (2025). Kronos: A Foundation Model for the Language of Financial Markets. *arXiv:2508.02739*.
- Yao, S., et al. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. *ICLR 2023*.
- Zaharia, M., et al. (2024). The Shift from Models to Compound AI Systems. *Berkeley AI Research Blog*.
- LangGraph Documentation. https://github.com/langchain-ai/langgraph
- Anthropic. (2024). Building Effective Agents. https://www.anthropic.com/research/building-effective-agents
