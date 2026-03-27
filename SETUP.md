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

## RL Investigation: Progressive Experiments

The RL component was investigated through four progressively more sophisticated approaches, each designed to address limitations discovered in the previous stage.

### Stage 1: Direct RL Trading (PPO on forecast features)
- **Setup:** 8-dim observation (market features + Kronos forecast), Discrete(3) action space, log-return reward
- **Result:** +74.3% return / 1.23 Sharpe (walk-forward avg across 6 folds)
- **Finding:** RL learns a reasonable trading policy that matches buy-and-hold on return but with slightly better risk-adjustment. However, it cannot match the rule-based agent's drawdown control in bear markets.

### Stage 2: Gated Forecast Ablation
- **Setup:** Extended to 12-dim observation + 6-action space ({BUY,SELL,HOLD} x {with/without forecast}). Forecast cost of 0.1% per call.
- **Result:** Agent learns regime-dependent forecast gating — ~100% forecast usage in bull markets, ~12% in bear markets
- **Finding:** This is the key RL contribution. The agent independently discovers that Kronos forecasts are unreliable in high-volatility regimes and learns to ignore them. This regime-dependent behavior was not hand-coded.

### Stage 3: Two-Tier Regime-Aware System (V2)
- **Setup:** 22-dim observation enriched with cross-asset macro features (VIX, yields, gold, treasuries, oil, small/large cap spread, defensive rotation). GradientBoosting regime classifier (5 regimes: bull/bear/crash/recovery/sideways). Per-regime PPO policies with shaped rewards (asymmetric loss penalty, drawdown penalty, inaction bonus in crash/bear, volatility penalty in sideways).
- **Result:** Per-regime policies learn distinct behaviors — conservative in crash (high entropy, inaction bonus), aggressive in bull. Fallback policy with regime confidence gating.
- **Finding:** Richer features and regime-aware training improve interpretability but do not significantly improve out-of-sample performance over Stage 2, suggesting the signal-to-noise ratio in financial markets limits what additional features can capture.

### Stage 4: RL as Infrastructure for LLM (V3)
- **Motivation:** Since Claude's reasoning (+134.4%) outperforms RL's direct decisions (+74.3%), reframe RL to support Claude rather than replace it.
- **Experiments:**
  - **Position Sizing:** RL decides allocation fraction (25/50/75/100%) for each of Claude's trades. 24-dim observation (macro features + Claude's action + position state). Risk-adjusted reward isolating sizing quality from direction quality. Result: +0.79% avg return improvement, +0.04 Sharpe — marginal gains, with sizing distribution adapting by regime (86% full-size in AI rally, 50% half-size in bear).
  - **Signal Quality Scorer:** RL predicts forecast reliability (0-1 score) based on macro context + Kronos track record. Calibration-based reward. Result: Better calibration (MAE 0.449 vs 0.493 heuristic) on 4/6 folds, with the scorer outperforming the volatility heuristic on scorer accuracy (70.6% vs 54.9% on late bull fold).
- **Finding:** RL as infrastructure shows promise on calibration but the gains do not yet translate to meaningful portfolio-level improvements, primarily because Kronos itself is ~50% directionally accurate — there is limited signal for the scorer to learn from.

### Summary of RL Findings

| Stage | What RL Learned | Limitation |
|-------|----------------|------------|
| Direct trading | Reasonable policy matching B&H | Cannot synthesize context like LLM |
| Gated forecast | Regime-dependent forecast reliability | Learned behavior is a simple vol threshold |
| Two-tier regime | Per-regime policies with distinct behaviors | Additional features hit noise floor |
| Infrastructure | Adaptive sizing + better calibration | Marginal portfolio impact given ~50% forecast accuracy |

**Core conclusion:** RL's most valuable contribution is the *discovery* that forecast utility is regime-dependent (Stage 2). This finding informed the hybrid system design where Claude receives a reliability score with each forecast. The progressive RL investigation demonstrates that in low signal-to-noise environments like financial markets, LLM reasoning over the same features consistently outperforms learned policies, but RL can still provide useful meta-information (forecast reliability, regime classification) that improves the LLM's decision context.

## Key Results

| Agent | Return (2020-2024) | Sharpe | Max Drawdown |
|-------|-------------------|--------|--------------|
| Buy & Hold | +82.3% | 1.18 | -33.6% |
| Rule-Based (Kronos) | +56.7% | 1.13 | -15.5% |
| RL (always forecast, walk-forward avg) | +74.3% | 1.23 | -20.2% |
| RL Gated (walk-forward, best fold) | +108.3% | — | — |
| **Claude + Kronos** | **+134.4%** | **1.18** | **-22.0%** |

**RL per-regime highlights:**
- COVID crash (2020): RL (always forecast) beats B&H by +10.8% (+108.3% vs +97.5%)
- Bear market (2022): RL gated beats B&H by +6.1% (+36.2% vs +30.1%)
- RL wins 4/6 folds vs buy-and-hold on risk-adjusted basis (Sharpe)

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
