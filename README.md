# Kronos Trading Agent

An agentic trading system that combines a financial foundation model (Kronos) with LLM reasoning (Claude) and reinforcement learning to trade the S&P 500. Evaluates whether RL improves trading decisions on top of forecasting + LLM reasoning.

## Quick Start

```bash
pip install -r requirements.txt
python agent.py                    # rule-based agent (no API key needed)
python agent_langgraph.py          # Claude reasoning agent (needs ANTHROPIC_API_KEY)
python agent_hybrid.py             # RL gate + Claude hybrid
```

## Architecture

```
S&P 500 data (yfinance)
        |
Kronos forecast (5-day OHLCV)    <-- financial foundation model (24.7M params)
        |
RL Gate (optional)                <-- learns WHEN forecasts are reliable
        |
Claude reasoning (LangGraph)      <-- decides BUY / SELL / HOLD with explanation
        |
Portfolio tracking + backtest
```

## Agent Variants

| # | Agent | Script | Description |
|---|-------|--------|-------------|
| 1 | Buy & Hold | — | Passive baseline |
| 2 | Rule-Based | `agent.py` | Kronos + hand-tuned thresholds |
| 3 | Claude + Kronos | `agent_langgraph.py` | LLM reasoning via tool use |
| 4 | RL (PPO) | `train_rl.py` | Learned policy from reward signal |
| 5 | RL Gated | `train_rl_gated.py` | Learns when to call Kronos |
| 6 | Walk-Forward CV | `train_rl_walkforward.py` | Multi-ticker, 6-fold evaluation |
| 7 | Hybrid | `agent_hybrid.py` | RL gate + Claude trader |

## Key Results

| Agent | Return (2020-2024) | Sharpe | Max Drawdown |
|-------|-------------------|--------|--------------|
| Buy & Hold | +82.3% | 1.18 | -33.6% |
| Rule-Based | +56.7% | 1.13 | -15.5% |
| RL (walk-forward avg) | +74.3% | 1.23 | -20.2% |
| **Claude + Kronos** | **+134.4%** | **1.18** | **-22.0%** |

**Core finding:** LLM reasoning substantially outperforms both rule-based and RL agents on the same forecasts. RL's best learned behavior (regime-dependent forecast gating) is a pattern Claude infers naturally.

## Repo Structure

```
agent.py                  # Rule-based Kronos agent
agent_langgraph.py        # LangGraph + Claude agent
agent_hybrid.py           # RL gate + Claude hybrid
trading_env.py            # Gymnasium environments
train_rl.py               # Single-period RL training
train_rl_gated.py         # Gated forecast ablation
train_rl_walkforward.py   # Walk-forward CV (main RL script)
v2/                       # Two-tier regime-aware RL system
v3/                       # RL as infrastructure experiments
reports/                  # Detailed reports
Kronos/                   # Forecasting model (submodule)
SETUP.md                  # Detailed 1-pager on setup, models, hyperparameters
```

## Walk-Forward Cross-Validation

6 folds, 5-year train / 1-year test, 7 tickers (SPY, QQQ, DIA, IWM, XLF, XLE, XLK):

| Fold | Test Year | Regime | RL vs B&H |
|------|-----------|--------|-----------|
| 1 | 2019 | Late Bull | +144.6% vs +164.2% |
| 2 | 2020 | COVID Crash | +108.3% vs +97.5% |
| 3 | 2021 | Post-COVID Bull | +68.8% vs +72.6% |
| 4 | 2022 | Bear Market | +36.2% vs +30.1% |
| 5 | 2023 | Recovery | +61.9% vs +58.8% |
| 6 | 2024 | AI Rally | +26.1% vs +26.1% |

## Setup Details

See [SETUP.md](SETUP.md) for the full 1-pager covering task definition, environment, models, training strategy, and hyperparameters.
