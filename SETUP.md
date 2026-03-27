# Kronos Trading Agent — System Setup

**Rohan Ray | COSC 89.35**

---

## Task

Learn a trading policy for the S&P 500 that decides BUY/SELL/HOLD every 5 trading days, using a financial foundation model (Kronos) for price forecasting and an LLM (Claude) for reasoning. Evaluate whether reinforcement learning improves trading decisions on top of the forecasting model.

## Environment

| Parameter | Value |
|-----------|-------|
| Asset | S&P 500 (^GSPC) via Yahoo Finance |
| Action space | Discrete(3): BUY, SELL, HOLD |
| Position type | All-in / all-out (fully invested or fully cash) |
| Initial capital | $100,000 |
| Decision frequency | Every 5 trading days |
| Transaction costs | 0 (base), 0.1% (sensitivity analysis) |
| Test periods | Full: 2020-2024, Bear: 2022-2023 |
| Walk-forward CV | 6 folds, 5-year train / 1-year test, rolling 2019-2024 |
| Ticker pool | SPY, QQQ, DIA, IWM, XLF, XLE, XLK (multi-ticker training) |

**Observation space (RL agents):**

| Variant | Dims | Features |
|---------|------|----------|
| Base RL | 8 | position, ret_20d, ret_50d, vol_20d, price_vs_50ma, kronos_ret, kronos_trend, kronos_hvol |
| Gated RL | 12 | Base + portfolio_return, days_in_position, used_forecast_flag, forecast_accuracy |
| V2 Two-Tier | 22 | Above + macro features (VIX, yields, gold, treasuries, oil, small/large cap spread, defensive rotation) + regime_id + regime_confidence |

## Models

### Kronos (Forecasting)
- **Architecture:** Decoder-only transformer, 24.7M params (Kronos-small)
- **Training data:** 12B+ candlestick records from 45 global exchanges
- **Tokenizer:** Binary Spherical Quantization (BSQ) on OHLCV
- **Input:** 60 days of OHLCV history (max context 512)
- **Output:** 5-day OHLCV forecast (predicted close, high, low prices)
- **Derived signals:** expected return, trend agreement (% days up/down), volatility flag (forecast spread vs recent spread)

### Claude Haiku 4.5 (Reasoning)
- **Role:** Decision-making via tool use (LangGraph state machine)
- **Tools:** `get_kronos_forecast`, `check_portfolio`, `get_market_context`, `get_trade_history`, `execute_trade`
- **Temperature:** 0, max tokens: 500 per decision step
- **Prompt:** Includes forecast reliability score + decision guidelines with regime-aware instructions

### PPO (Reinforcement Learning)
- **Algorithm:** Proximal Policy Optimization (stable-baselines3)
- **Policy:** MlpPolicy (2x64 hidden layers, default)
- **Learning rate:** 3e-4
- **Entropy coefficient:** 0.01 (base), 0.05 (bear/crash regimes)
- **Batch size:** 64, n_epochs: 10
- **Timesteps:** 100K-200K per variant
- **Multi-ticker:** Random ticker sampling per episode (7 tickers, ~7x training data)

## Training Strategy

### Walk-Forward Cross-Validation (6 Folds)

| Fold | Train | Test | Regime |
|------|-------|------|--------|
| 1 | 2014-2018 | 2019 | Late Bull |
| 2 | 2015-2019 | 2020 | COVID Crash + Recovery |
| 3 | 2016-2020 | 2021 | Post-COVID Bull |
| 4 | 2017-2021 | 2022 | Bear Market |
| 5 | 2018-2022 | 2023 | Recovery |
| 6 | 2019-2023 | 2024 | AI Rally |

### Agent Variants Evaluated

1. **Buy & Hold** — passive baseline
2. **Rule-Based** — Kronos forecast + hand-tuned thresholds (asymmetric: BUY > +0.3%, SELL < -1.5%, with volatility filter)
3. **LangGraph + Claude** — Kronos forecast + LLM reasoning via tool use
4. **RL (no forecast)** — PPO on market context only (no Kronos)
5. **RL (always forecast)** — PPO with Kronos forecast always included
6. **RL (gated forecast)** — PPO learns *when* to call Kronos (6-action space: 3 actions x with/without forecast)
7. **Hybrid (RL gate + Claude)** — RL decides whether to show Claude the forecast; Claude makes the trade decision

### Ablation: Gated Forecast
The gated agent's 6-action space decomposes into: {HOLD, BUY, SELL} x {with forecast, without forecast}. A small per-step forecast cost (0.1%) discourages unnecessary calls. The agent learns regime-dependent gating: ~100% forecast usage in bull markets, ~12% in bear markets.

## Key Results

| Agent | Return (2020-2024) | Sharpe | Max Drawdown |
|-------|-------------------|--------|-------------|
| Buy & Hold | +82.3% | 1.18 | -33.6% |
| Rule-Based (Kronos) | +56.7% | 1.13 | -15.5% |
| RL (always forecast, walk-forward avg) | +74.3% | 1.23 | -20.2% |
| **Claude + Kronos** | **+134.4%** | **1.18** | **-22.0%** |

**Core finding:** Claude's LLM reasoning (+134.4%) substantially outperforms both the rule-based agent and RL agents operating on the same Kronos forecasts. RL's primary learned behavior (gating forecasts by regime) is a pattern Claude infers naturally from context. The RL component does not meaningfully improve trading decisions on top of LLM reasoning.

## Repo Structure

```
agent.py                  # Rule-based Kronos agent (Level 2)
agent_langgraph.py        # LangGraph + Claude agent (Level 3)
agent_hybrid.py           # RL gate + Claude hybrid (Level 5)
trading_env.py            # Gymnasium envs (base, gated, soft-gated)
train_rl.py               # Single-period RL training
train_rl_gated.py         # Gated forecast ablation study
train_rl_walkforward.py   # Walk-forward CV training (main RL script)
v2/                       # Two-tier regime-aware RL (macro features, per-regime policies)
v3/                       # RL as infrastructure experiments (position sizing, signal scoring)
reports/                  # Detailed reports and presentation guide
Kronos/                   # Kronos model (git submodule)
```

## Hyperparameter Summary

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Lookback window | 60 days | Kronos context window balance |
| Forecast horizon | 5 days | Weekly rebalancing frequency |
| BUY threshold | +0.3% | Low bar to enter (markets have upward bias) |
| SELL threshold | -1.5% | High bar to exit (avoid whipsaws) |
| Trend agreement | 60% | Majority of forecast steps must confirm direction |
| Volatility multiplier | 1.5x | Widen thresholds when forecast vol > 1.5x recent |
| PPO learning rate | 3e-4 | Standard for stable-baselines3 |
| PPO entropy coef | 0.01-0.05 | Higher in volatile regimes for exploration |
| Walk-forward folds | 6 | Covers bull, bear, crash, recovery regimes |
| Training tickers | 7 | Multi-ticker for generalization |
