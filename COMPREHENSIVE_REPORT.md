# Kronos Trading Agent: An Agentic System for Financial Market Forecasting

**Rohan Ray | COSC 89.35 | Final Project — Comprehensive Report**

---

## 1. Research Question

Can an AI agent that combines a domain-specific foundation model for financial forecasting with LLM-based reasoning outperform passive market strategies — particularly during periods of high volatility and market downturns? And can a reinforcement learning agent learn *when* to consult the forecasting model, discovering that forecast reliability is regime-dependent?

Traditional algorithmic trading systems rely on fixed, hand-crafted rules that cannot adapt to changing market conditions. Recent advances in foundation models have produced systems capable of accurate financial time series forecasting, but a forecast alone does not constitute a decision. This project investigates whether adding an explicit reasoning layer produces meaningfully better trading outcomes — and whether an RL agent can learn to gate forecast usage based on market regime.

---

## 2. System Architecture

The project implements a **compound AI system** — multiple specialized models orchestrated to achieve capabilities none could alone. Three progressively sophisticated agent architectures were built and evaluated.

### 2.1 Foundation Models

**Kronos (Financial Forecasting Model)**
- Decoder-only transformer with 24.7M parameters (Kronos-small)
- Trained on 12 billion+ K-line candlestick records from 45 global financial exchanges
- Two-stage architecture:
  - **Tokenizer:** Binary Spherical Quantization (BSQ) converts continuous OHLCV values into hierarchical discrete tokens
  - **Predictor:** Autoregressive transformer generates future tokens
- Input: 60 days of OHLCV (Open, High, Low, Close, Volume) history
- Output: 5-day multi-dimensional forecast — predicted prices *and* predicted volatility through the high-low spread
- Key advantage: Produces rich forecasts including volatility information, not just point price predictions

**Claude Haiku 4.5 (LLM Reasoning Layer)**
- Serves as the decision-making brain of the agentic system
- Receives Kronos forecasts + portfolio state + technical indicators + trade history
- Reasons in natural language about BUY/SELL/HOLD decisions
- Temperature: 0 (deterministic when possible), max tokens: 500 per decision
- Key advantage: Can weigh conflicting signals contextually, unlike fixed thresholds

### 2.2 Agent Architectures

#### Level 1: Buy & Hold (Passive Baseline)
No model, no decisions. Starting $100K compounded at market returns. Serves as the baseline that any active strategy must beat to justify its complexity.

#### Level 2: Rule-Based Agent
Uses Kronos forecasts with hand-tuned thresholds:
- **BUY** if forecast return > +0.3% AND trend confirms (60% of 5-day forecast steps up)
- **SELL** if forecast return < −1.5% AND trend confirms (60% of 5-day forecast steps down)
- Asymmetric thresholds reflect the upward bias of equity markets (reluctant to sell)
- **Volatility filter:** If forecast high-low spread exceeds 1.5× recent average spread, thresholds are scaled up (more cautious in volatile regimes)

#### Level 3: LangGraph + Claude Agent
Same Kronos forecasts, but Claude provides intelligent reasoning via a LangGraph state machine implementing the perceive → reason → act → reflect loop.

**Tool Suite (5 structured tools):**

| Tool | Function |
|------|----------|
| `get_kronos_forecast()` | Fetch 5-day OHLCV prediction from Kronos |
| `check_portfolio()` | Check current cash, shares, portfolio value |
| `get_market_context()` | Get 20-day return, volatility, 50-day MA position |
| `get_trade_history()` | Review 5 most recent trades and their outcomes |
| `execute_trade()` | Execute BUY/SELL/HOLD with natural language reasoning |

The agent wakes up every 5 trading days, gathers information through these tools, and makes one trading decision with an explicit explanation.

#### Level 4: RL Agent (PPO)
A small neural network trained via Proximal Policy Optimization to learn the optimal trading policy directly from reward signals, replacing both the fixed rules and the LLM reasoning layer.

#### Level 5: Gated Forecast RL Agent
Extends the RL agent with the ability to choose *whether* to consult Kronos at each step, learning which market regimes benefit from forecasts versus which are better handled by market context alone.

---

## 3. Experimental Setup

### 3.1 Data

S&P 500 (^GSPC) daily OHLCV data from Yahoo Finance.

**Test Periods:**
- **Full period (2020–2024):** Spans COVID crash, recovery, Fed rate hikes, SVB collapse, AI rally
- **Bear market (Jan 2022 – Jun 2023):** Isolated Fed tightening cycle with ~25% drawdown

**Key Market Events Covered:**

| Event | Date | Impact |
|-------|------|--------|
| COVID Crash | Mar 2020 | −34% drawdown in 23 days |
| COVID Recovery | Apr–Dec 2020 | V-shaped recovery |
| Fed Rate Hikes Begin | Mar 2022 | Start of tightening cycle |
| 75bp Rate Hike | Jun 2022 | Aggressive monetary tightening |
| CPI Shock | Sep 2022 | Higher-than-expected inflation |
| SVB Collapse | Mar 2023 | Banking crisis |
| AI Rally | 2023–2024 | Technology-led bull market |

### 3.2 Common Parameters

| Parameter | Value |
|-----------|-------|
| Initial capital | $100,000 |
| Lookback window | 60 trading days |
| Forecast horizon | 5 trading days |
| Decision frequency | Every 5 trading days |
| Position sizing | All-in / all-out |
| Transaction costs | 0 (default; configurable) |

---

## 4. LLM Agent Results

### 4.1 Performance Summary

#### Full Period (2020–2024)

| Metric | LangGraph + Claude | Rule-Based | Buy & Hold |
|--------|-------------------|------------|------------|
| **Total Return** | **+134.4%** | +39.3% | +81.3% |
| **Sharpe Ratio** | **1.18** | 0.48 | — |
| **Max Drawdown** | −22.0% | −25.5% | ~−34% |
| **Final Portfolio** | **$234,431** | $139,254 | $181,300 |

#### Bear Market (2022–2023)

| Metric | LangGraph + Claude | Rule-Based | Buy & Hold |
|--------|-------------------|------------|------------|
| **Total Return** | **+18.5%** | −1.1% | −8.3% |
| **Sharpe Ratio** | **0.83** | 0.05 | — |
| **Max Drawdown** | **−11.3%** | −20.1% | ~−25% |
| **Final Portfolio** | **$118,493** | $98,893 | $91,700 |

### 4.2 Why the LLM Agent Won

Analysis of Claude's reasoning logs reveals three concrete advantages over fixed rules:

**1. Contextual Signal Integration**
When Kronos produced a marginally positive forecast but technical indicators were bearish, Claude held rather than buying. The rule-based agent cannot express conditional logic like "the forecast is slightly positive but the broader trend is deteriorating, so I'll wait."

**2. Adaptive Risk Management**
After a series of losing trades, Claude explicitly referenced past failures in its reasoning: *"Recent bullish forecasts failed; waiting for clearer signals."* It effectively adjusted its strategy within the session based on outcomes — something the rule-based agent repeats mechanically regardless of results.

**3. Regime Awareness**
Claude recognized market concepts like "oversold" (price 25% below 50-day MA), "catching a falling knife," and "mean reversion opportunity." These concepts are natural for an LLM trained on financial text but nearly impossible to encode as threshold-based rules.

**Example Reasoning Entries:**

> **Selling into COVID crash (Mar 2, 2020):** "Kronos forecasts −2.86% return with price 5.38% below 50-day MA, confirming downtrend. Lock in +$387 gain before forecasted weakness materializes."

> **Buying COVID bottom (Mar 9, 2020):** "Kronos forecasts unanimous 5-day uptrend (+7.5% return). Market deeply oversold (15.5% below 50-MA), providing attractive entry."

### 4.3 Rule-Based Agent Failure Mode

The rule-based agent presents a revealing failure: it beat buy & hold in the bear market (−1.1% vs −8.3%) but significantly underperformed in the full period (39.3% vs 81.3%). This exposes a fundamental limitation of fixed thresholds — they cannot simultaneously optimize for capital preservation in downturns and participation in rallies. The asymmetric sell threshold (−1.5%) that prevented unnecessary selling also caused the agent to miss re-entry points during recoveries.

---

## 5. Reinforcement Learning Extension

### 5.1 Motivation

The LLM agent demonstrated that intelligent reasoning about forecasts matters more than the forecasts themselves. But it has limitations: API costs, non-determinism, potential lookahead bias from training data, and no ability to learn across sessions. The RL extension asks: can a small neural network trained purely from reward signals achieve similar or better performance?

### 5.2 Trading Environment

A Gymnasium-compliant environment wraps the backtest into a standard RL interface.

**Observation Space (10 features):**

| # | Feature | Description |
|---|---------|-------------|
| 0 | `position` | 0.0 = cash, 1.0 = invested |
| 1 | `return_20d` | 20-day price return |
| 2 | `return_50d` | 50-day price return |
| 3 | `volatility_20d` | 20-day realized volatility (std of daily returns) |
| 4 | `price_vs_50ma` | Price relative to 50-day moving average (ratio − 1) |
| 5 | `kronos_expected_ret` | Kronos 1-step expected return |
| 6 | `kronos_trend` | Fraction of 5 forecast steps that are up |
| 7 | `kronos_high_vol` | 1.0 if forecast vol > recent vol × multiplier |
| 8 | `portfolio_return` | Current portfolio return since start |
| 9 | `days_in_position` | Normalized days held in current position |

**Action Space:** Discrete(3) — HOLD (0), BUY (1), SELL (2). All-in/all-out.

**Reward Functions (selectable):**
- `log_return` (default): `log(pv_after / pv_before)` — encourages compounding
- `pnl`: raw dollar change normalized by initial cash
- `sharpe`: return minus a running volatility penalty (risk-adjusted)

**Stepping:** The agent acts every 5 trading days, matching the Kronos forecast horizon. At each step, Kronos features are computed from the past 60 days of OHLCV data.

**Data Augmentation:** `random_start=True` during training — each episode begins at a random point in the training window, creating diverse trajectories from limited data.

### 5.3 Forecast Caching

Kronos inference takes ~seconds per call. Running it thousands of times during PPO training is infeasible. The solution:

1. **Pre-compute** all Kronos forecasts for every step index in train/test periods
2. **Save to disk** as JSON files (`forecast_cache_train_full.json`, etc.)
3. `CachedTradingEnv` loads these at startup and returns cached values instead of calling the model

This makes training fast (no model inference in the loop) and reproducible (identical forecasts every run).

### 5.4 PPO Training

**Hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Algorithm | PPO (Proximal Policy Optimization) |
| Policy | MlpPolicy (2-layer MLP) |
| Learning rate | 3 × 10⁻⁴ |
| Batch size | 64 |
| Epochs per update | 10 |
| Discount factor (γ) | 0.99 |
| GAE λ | 0.95 |
| Clip range | 0.2 |
| Entropy coefficient | 0.01 (encourages exploration) |
| Timesteps | 50,000–100,000 |

**Walk-Forward Data Split:**
- Full mode: Train on 2020–2022, test on 2023–2024
- Bear mode: Train on 2022, test on Jan–Jun 2023

### 5.5 Basic RL Results

#### Full Period Test (2023–2024)

| Metric | RL (PPO) | Rule-Based | Buy & Hold |
|--------|----------|------------|------------|
| **Return** | +50.72% | +28.49% | +54.46% |
| **Sharpe Ratio** | 1.71 | — | — |
| **Max Drawdown** | −9.41% | — | — |
| **Final Value** | $150,716 | — | $154,465 |
| **Action Dist.** | 91 BUY, 10 HOLD | — | — |

The RL agent nearly matched buy & hold (+50.7% vs +54.5%) with a strong Sharpe ratio of 1.71 and identical max drawdown. It significantly outperformed the rule-based agent (+28.5%). The action distribution shows the agent learned a mostly-invested strategy — buying and holding through the 2023–2024 bull market.

#### Bear Market Test (Jan–Jun 2023)

| Metric | RL (PPO) | Rule-Based | Buy & Hold |
|--------|----------|------------|------------|
| **Return** | +3.58% | +7.29% | +14.97% |
| **Sharpe Ratio** | 1.98 | — | — |
| **Max Drawdown** | 0.0% | — | — |
| **Final Value** | $103,585 | — | $114,494 |
| **Action Dist.** | 2 BUY, 10 SELL, 13 HOLD | — | — |

In the bear-trained variant, the RL agent was very conservative — mostly holding cash with 0% drawdown but capturing only +3.6% of the recovery. The agent learned risk aversion from the 2022 training period, which was appropriate for the bear market but caused it to miss the early 2023 rally.

---

## 6. Gated Forecast Ablation Study

### 6.1 Core Contribution

The central question: **When should the agent call Kronos?** Forecasting models are not equally reliable across all market conditions. The gated agent learns to decide at each step whether requesting a Kronos forecast is worthwhile, or whether market context alone is sufficient.

### 6.2 Gated Environment Design

The `GatedTradingEnv` expands the action space to **Discrete(6)**:

| Action | Forecast? | Trade |
|--------|-----------|-------|
| 0 | No | HOLD |
| 1 | No | BUY |
| 2 | No | SELL |
| 3 | Yes | HOLD |
| 4 | Yes | BUY |
| 5 | Yes | SELL |

The observation space grows to **12 features** (adds `used_forecast` flag and `forecast_accuracy` — rolling accuracy of recent Kronos directional predictions).

Key design decisions:
- When the agent doesn't request a forecast, Kronos features are zeroed in the observation
- The environment carries forward the *last* forecast (stale data), letting the agent see that predictions are aging
- An optional `forecast_cost` (default 0.001) is subtracted from the reward each time Kronos is called, simulating compute/API cost
- `forecast_accuracy` tracks rolling directional accuracy of Kronos over the last 10 calls, giving the agent a meta-signal about forecast reliability

### 6.3 Five-Way Ablation Design

| # | Agent | Forecast Access | Decision Method |
|---|-------|-----------------|-----------------|
| 1 | Buy & Hold | None | Passive |
| 2 | Rule-Based | Always (Kronos) | Fixed thresholds |
| 3 | RL (no forecast) | Never | Learned policy (market context only) |
| 4 | RL (always forecast) | Always (Kronos) | Learned policy |
| 5 | RL (gated forecast) | Agent decides | Learned policy + learned gating |

Agent 3 uses a `NoForecastEnv` subclass that zeros out all Kronos features, forcing reliance on market context (returns, volatility, MA position).

### 6.4 Ablation Results

#### Full Period Test (2023–2024)

| Agent | Return | Sharpe | Max DD | Final Value | Forecast Rate |
|-------|--------|--------|--------|-------------|---------------|
| Buy & Hold | +54.46% | 1.80 | −9.41% | $154,465 | — |
| Rule-Based | +32.63% | 1.33 | −10.99% | $132,632 | 100% |
| RL (no forecast) | +50.35% | 1.70 | −9.41% | $150,347 | 0% |
| RL (always forecast) | +50.72% | 1.71 | −9.41% | $150,716 | 100% |
| **RL (gated forecast)** | **+54.46%** | **1.80** | **−9.41%** | **$154,465** | **100%** |

In the bull market test period, the gated agent matched buy & hold exactly (+54.46%) by learning to **always use Kronos** (100% forecast rate) and adopting a mostly-invested strategy (90 BUY + 11 HOLD, no SELL). The forecast was consistently helpful in a trending bull market.

#### Bear Market Test (Jan–Jun 2023)

| Agent | Return | Sharpe | Max DD | Final Value | Forecast Rate |
|-------|--------|--------|--------|-------------|---------------|
| Buy & Hold | +14.49% | 2.98 | −5.53% | $114,494 | — |
| Rule-Based | +3.32% | 0.96 | −5.53% | $103,317 | 100% |
| **RL (no forecast)** | **+19.28%** | **5.25** | **−1.09%** | **$119,279** | **0%** |
| RL (always forecast) | +3.58% | 1.98 | 0.0% | $103,585 | 100% |
| RL (gated forecast) | +11.74% | 4.09 | −1.09% | $111,742 | **12%** |

The bear market results reveal the key insight:

1. **RL (no forecast) was the best performer** — +19.28% return with a 5.25 Sharpe and only −1.09% max drawdown. It beat even buy & hold by relying purely on market context features.

2. **RL (always forecast) was the worst RL variant** — +3.58%, barely above flat. Kronos forecasts actively hurt during the bear market, likely because the model's training distribution doesn't represent extreme bear conditions well.

3. **RL (gated forecast) landed in between** — +11.74% with a 4.09 Sharpe. It learned to call Kronos only 12% of the time (3 out of 25 steps), mostly holding cash without consulting the model.

4. **The gated action distribution** (22 HOLD without forecast, 3 BUY with forecast) shows the agent learned a conservative, forecast-skeptical strategy — exactly what the regime called for.

### 6.5 Key Finding: Regime-Dependent Forecast Gating

| Market Regime | Gated Forecast Rate | Interpretation |
|---------------|---------------------|----------------|
| Bull market (2023–2024) | **100%** | Kronos forecasts are helpful; always consult |
| Bear market (2022–2023) | **12%** | Kronos forecasts are noise; rely on market context |

This regime-dependent gating behavior **emerged naturally** from the reward signal — it was not hand-coded. The agent discovered through trial and error that:

- In trending bull markets, Kronos provides useful directional signals that improve trading
- In volatile bear markets, Kronos predictions become unreliable and are better ignored

This finding has broader implications: **foundation model forecasts should be consumed strategically, not unconditionally.** A smart consumer of AI predictions knows when to trust them and when to rely on simpler signals.

---

## 7. Cross-Agent Comparison

### 7.1 Head-to-Head Summary

| Agent | Full Return | Bear Return | Key Strength |
|-------|-------------|-------------|--------------|
| Buy & Hold | +81.3% (LLM test) / +54.5% (RL test) | −8.3% / +14.5% | Zero effort, captures market beta |
| Rule-Based | +39.3% / +32.6% | −1.1% / +3.3% | Simple, interpretable |
| LangGraph + Claude | **+134.4%** | **+18.5%** | Contextual reasoning, adaptive risk mgmt |
| RL (no forecast) | +50.4% | +19.3% | Robust in bear markets, no model dependency |
| RL (always forecast) | +50.7% | +3.6% | Slightly better than no-forecast in bull |
| RL (gated forecast) | +54.5% | +11.7% | Learns regime-dependent forecast usage |

*Note: LLM agent and RL agents were tested on different periods (LLM: 2020–2024, RL: 2023–2024) due to different train/test splits, so direct comparison requires caution.*

### 7.2 Insight Hierarchy

Each layer of the system contributed a distinct insight:

1. **Kronos alone (rule-based) underperformed buy & hold.** Having a forecast is not enough — and with bad rules, it can hurt.

2. **Adding LLM reasoning (Claude) turned the forecast into alpha.** The value isn't in *predicting* the market but in *reasoning about how to act* on predictions. Return jumped from 39.3% → 134.4%.

3. **RL can match LLM performance without API costs or lookahead bias.** PPO agents achieved competitive returns with deterministic, reproducible behavior.

4. **Forecast value is regime-dependent.** The gated agent's emergent behavior — 100% forecast usage in bull markets, 12% in bear markets — reveals that model reliability varies with market conditions.

5. **Sometimes no model is best.** In bear markets, the RL agent with no forecast at all outperformed every other approach (+19.3%), suggesting that during extreme conditions, simple features (returns, volatility, MA) are more reliable than model predictions.

---

## 8. Technical Implementation

### 8.1 Codebase Structure

| File | Lines | Description |
|------|-------|-------------|
| `agent.py` | ~550 | Shared utilities, data loading, Kronos integration, rule-based agent, backtest loop |
| `agent_langgraph.py` | ~600 | LangGraph + Claude agent with 5-tool system, system prompt, resume support |
| `trading_env.py` | ~680 | Gymnasium environments: `TradingEnv`, `GatedTradingEnv`, cached variants |
| `train_rl.py` | ~400 | PPO training script with walk-forward split, evaluation, plotting |
| `train_rl_gated.py` | ~530 | 5-way ablation study with gated forecast analysis |

### 8.2 Key Engineering Decisions

**Forecast Caching:** Kronos inference (~seconds/call) is pre-computed for all step indices and saved to JSON. This makes PPO training (thousands of env resets) feasible and ensures reproducibility. Cache files: `forecast_cache_{train|test}_{full|bear}.json`.

**Random Episode Starts:** During training, `random_start=True` begins each episode at a random point in the training window. This is critical data augmentation given the limited number of trading steps (~252/year at weekly frequency).

**Stale Forecast Memory:** In the gated environment, the last Kronos forecast is carried forward even when not refreshed. This lets the agent observe that predictions are aging and decide whether to update — a form of learned information management.

**Walk-Forward Split:** Training always occurs on earlier data, testing on later data — no future information leakage. Full: train 2020–2022, test 2023–2024. Bear: train 2022, test Jan–Jun 2023.

### 8.3 Dependencies

```
chronos-forecasting>=2.1.0    # Kronos model
torch>=2.2.0                  # PyTorch backend
yfinance>=0.2.40              # Market data
pandas>=2.0.0, numpy>=1.24.0  # Data manipulation
matplotlib>=3.7.0             # Visualization
gymnasium>=0.29.0             # RL environment
stable-baselines3>=2.3.0      # PPO implementation
langchain, langgraph           # Agent orchestration
langchain-anthropic            # Claude integration
```

### 8.4 Running the Experiments

```bash
# LLM Agent
python agent_langgraph.py              # Full 2020-2024 backtest
python agent_langgraph.py --bear       # Bear market 2022-2023
python agent_langgraph.py --no-llm     # Rule-based fallback (no API needed)

# RL Agent (basic)
python train_rl.py                     # Train PPO, full period
python train_rl.py --bear              # Train PPO, bear market
python train_rl.py --eval-only         # Evaluate saved model
python train_rl.py --reward sharpe     # Alternative reward function

# Gated Forecast Ablation
python train_rl_gated.py               # Full 5-way ablation
python train_rl_gated.py --bear        # Bear market ablation
python train_rl_gated.py --timesteps 100000  # More training
python train_rl_gated.py --forecast-cost 0.001  # Penalize forecast calls
```

---

## 9. Limitations

### 9.1 Potential Lookahead Bias
Claude's training data includes knowledge of events in the test period (COVID, SVB, etc.). While it doesn't have access to future prices, its general knowledge may influence reasoning. The RL agents do not suffer from this limitation.

### 9.2 Simplified Trading Mechanics
- **No transaction costs** in default runs (configurable via `--tx-cost`)
- **All-in/all-out** position sizing — unrealistic for real portfolio management
- **No slippage or market impact** — assumes trades execute at closing price
- **Single asset** — only S&P 500 (^GSPC)

### 9.3 Limited Training Data
PPO trains on ~252 weekly trading steps per year — extremely small by RL standards. The random episode start augmentation helps but cannot fully compensate.

### 9.4 Non-Determinism
Claude's responses vary slightly between runs. The RL agents are deterministic given the same seed but sensitive to hyperparameters.

### 9.5 No Cross-Session Learning
The LLM agent adapts within a session (via trade history memory) but does not update weights. The RL agent learns during training but is frozen at test time. Neither agent truly learns online.

### 9.6 Single Seed Evaluation
Default RL evaluations use a single seed (42). Multi-seed evaluation is supported (`--seeds N`) but was not run for all configurations, limiting statistical confidence.

---

## 10. Future Work

1. **Head-to-head comparison:** Run the LLM agent and RL agents on identical test periods for direct comparison
2. **Two-stage gated agent:** Separate the gate decision (forecast or not) from the trade decision (BUY/SELL/HOLD) into sequential steps for cleaner credit assignment
3. **More training data:** Multiple tickers (SPY, QQQ, DIA), longer history (2015–2022), daily frequency instead of weekly
4. **Multi-seed evaluation:** 5–10 seeds per configuration, report mean ± standard deviation
5. **Walk-forward cross-validation:** Sliding train/test windows instead of a single split
6. **Larger Kronos model:** Upgrade from Kronos-small (24.7M) to Kronos-base (102M parameters)
7. **Hybrid agent:** Use RL for the gate decision and Claude for the trade decision, combining learned gating with LLM reasoning
8. **Live paper trading:** Deploy to a paper trading account with real-time data

---

## 11. Conclusions

This project demonstrates three core findings:

**1. Reasoning about predictions matters more than predictions alone.** A rule-based agent with Kronos forecasts underperformed buy & hold (39.3% vs 81.3%), while adding Claude's contextual reasoning produced 134.4% returns. The forecast is necessary but not sufficient — the reasoning layer is what creates value.

**2. LLM agents can exhibit sophisticated financial reasoning.** Claude demonstrated contextual signal integration, adaptive risk management based on trade history, and market regime awareness — behaviors that emerged naturally from the LLM's training rather than being explicitly programmed.

**3. Forecast reliability is regime-dependent.** The gated RL agent discovered through reward-driven learning that Kronos forecasts should be consumed strategically: always in bull markets (100% usage), rarely in bear markets (12% usage). In bear markets, an RL agent with no forecast at all was the top performer (+19.3%). This finding has implications beyond trading — any system consuming AI predictions should consider whether the prediction model's reliability varies with operating conditions.

Together, these findings support the compound AI systems paradigm: the future of AI applications lies not in single monolithic models but in orchestrated systems where specialized components — forecasting, reasoning, gating, memory — each contribute their strength.

---

## References

- Shi, S., et al. (2025). Kronos: A Foundation Model for the Language of Financial Markets. *arXiv:2508.02739*.
- Yao, S., et al. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. *ICLR 2023*.
- Zaharia, M., et al. (2024). The Shift from Models to Compound AI Systems. *Berkeley AI Research Blog*.
- Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. *arXiv:1707.06347*.
- LangGraph Documentation. https://github.com/langchain-ai/langgraph
- Anthropic. (2024). Building Effective Agents. https://www.anthropic.com/research/building-effective-agents

---

## Appendix A: Output Files

| File | Description |
|------|-------------|
| `backtest_results_full.png` | LLM agent portfolio, actions, forecast accuracy (2020–2024) |
| `backtest_results_bear.png` | Same for bear market (2022–2023) |
| `rl_results_full.png` | RL agent vs baselines (2023–2024) |
| `rl_results_bear.png` | RL agent vs baselines (bear test) |
| `ablation_results_full.png` | 5-agent ablation with gating analysis (full) |
| `ablation_results_bear.png` | 5-agent ablation with gating analysis (bear) |
| `backtest_log_langgraph_full.csv` | Every forecast + action + reasoning (2020–2024) |
| `backtest_log_langgraph_bear.csv` | Bear market reasoning log |
| `reasoning_log_full.json` | Full Claude reasoning outputs (100+ entries) |
| `reasoning_log_bear.json` | Bear market reasoning outputs |
| `rl_results_full.json` | RL evaluation metrics (full test) |
| `rl_results_bear.json` | RL evaluation metrics (bear test) |
| `ablation_results_full.json` | 5-agent ablation metrics with forecast rates (full) |
| `ablation_results_bear.json` | 5-agent ablation metrics with forecast rates (bear) |
| `forecast_cache_*.json` | Pre-computed Kronos predictions |
| `ppo_trading_full.zip` | Trained RL model (full period) |
| `ppo_trading_bear.zip` | Trained RL model (bear market) |
