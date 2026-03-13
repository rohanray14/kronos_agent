# Kronos Trading Agent — Progress Update for Nikhil

## What's Built

### 1. LLM Agent (Claude + Kronos + LangGraph) — Complete
The agentic system from the original proposal is fully working. LangGraph orchestrates a perceive-reason-act-reflect loop where Claude receives Kronos forecasts via tool calls, reasons about market conditions, and executes trades with natural language explanations.

**Results (S&P 500 backtest):**

| Metric | LLM Agent | Rule-Based | Buy & Hold |
|---|---|---|---|
| Full Period Return (2020-24) | **+134.4%** | +39.3% | +81.3% |
| Bear Market Return (2022-23) | **+18.5%** | -1.1% | -8.3% |
| Full Sharpe Ratio | **1.18** | 0.48 | — |
| Bear Max Drawdown | **-11.3%** | -20.1% | ~-25% |

Claude's reasoning logs show contextual signal integration, adaptive risk management (references past mistakes), and regime awareness — behaviors impossible to encode as fixed rules.

### 2. RL Agent (PPO) — In Progress
Instead of prompting an LLM to reason about trades, we train a small neural network to learn the optimal trading policy directly from reward signals. The setup:

- **Environment:** Gymnasium wrapper around the backtest. Every 5 trading days, the agent observes 10 features (current position, 20d/50d returns, realized volatility, price vs 50-day MA, Kronos expected return, Kronos trend agreement, Kronos volatility flag, portfolio return, time in position).
- **Policy:** 2-layer MLP that maps the 10 features → probability over BUY/SELL/HOLD.
- **Training:** PPO (Proximal Policy Optimization) replays the agent through historical episodes thousands of times. The reward is log portfolio return — actions that grow the portfolio get reinforced, actions that lose money get suppressed.
- **Key engineering:** Kronos forecasts are pre-computed and cached to disk so the environment can reset thousands of times without re-running the model (~5 min to cache, then training runs in seconds).

The RL agent learns through trial and error rather than human-written rules or LLM reasoning — its strategy is entirely shaped by which actions led to positive rewards in which market states.

**Preliminary results (test: 2023-2024):**
- RL agents consistently beat rule-based in both bull and bear regimes
- Best RL run: +54.5% (matched buy & hold) with 1.80 Sharpe and -9.4% max drawdown
- Main challenge: limited training data (252 weekly steps) constrains what PPO can learn

### 3. Gated Forecast Agent — Early Results
This is the core contribution addressing the question: **when should the agent call Kronos?**

Expanded action space to 6 actions (BUY/SELL/HOLD x with/without forecast). The agent learns whether requesting a Kronos prediction is worth it at each step.

**Key finding — regime-dependent gating emerged:**

| Regime | Forecast Rate | Interpretation |
|---|---|---|
| Bull market (2023-24) | 100% | Agent always uses Kronos — forecasts are helpful |
| Bear market (2022-23) | 12% | Agent mostly ignores Kronos — forecasts are noise |

This confirms the hypothesis that forecasts aren't always reliable. However, the gated agent doesn't yet outperform simpler baselines — in the bear market, RL with no forecast at all did best (+19.3% vs +11.7% gated). The gating behavior is emerging but needs more training signal to translate into better returns.

## Architecture

```
Market Data (OHLCV) → Kronos (forecast model) → State Features → PPO Policy Network → Trade Decision
                              ↑                                         |
                      [gated: agent decides                      Reward: Δ portfolio
                       whether to call this]                     value (log return)
```

## Next Steps (Prioritized)

1. **Improve gated agent performance** — simplify to two-stage action (gate decision, then trade), better reward shaping
2. **More training data** — multiple tickers (SPY, QQQ, DIA), longer history (2015-2022), possibly daily frequency
3. **Multi-seed evaluation** — run 5-10 seeds per config, report mean ± std for rigorous comparison
4. **Head-to-head comparison** — RL agent vs Claude agent on identical test periods for the final ablation table
5. **Walk-forward cross-validation** — sliding train/test windows instead of single split

## Files

| File | Description |
|---|---|
| `agent_langgraph.py` | LLM agent (Claude + LangGraph) |
| `agent.py` | Rule-based agent + shared utilities |
| `trading_env.py` | Gym environments: TradingEnv, GatedTradingEnv, cached variants |
| `train_rl.py` | PPO training script (standard) |
| `train_rl_gated.py` | Gated forecast ablation (5-way comparison) |
