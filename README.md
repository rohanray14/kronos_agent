# Chronos Trading Agent — MVP

A minimal trading agent using Amazon's Chronos-2 time series model to forecast S&P 500 prices and make buy/sell/hold decisions.

## Setup (2 mins)

```bash
pip install -r requirements.txt
python agent.py
```

## What it does

```
S&P 500 data (yfinance)
        ↓
Chronos-2 forecast (next 5 days)
        ↓
Decision rule (expected return threshold)
        ↓
BUY / SELL / HOLD
        ↓
Portfolio tracking + backtest chart
```

## Output

- `backtest_results.png` — 3-panel chart (portfolio vs B&H, actions, forecast accuracy)
- `backtest_log.csv` — every forecast + decision logged

## Key config (top of agent.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FORECAST_STEPS` | 5 | Days ahead to forecast |
| `LOOKBACK_DAYS` | 60 | History window fed to Chronos |
| `BUY_THRESHOLD` | 0.005 | +0.5% expected return → BUY |
| `SELL_THRESHOLD` | -0.005 | -0.5% expected return → SELL |
| `NUM_SAMPLES` | 20 | Forecast trajectories (↑ = slower, better) |

## Test period

`2020-01-01 → 2024-12-31` — includes:
- COVID crash (March 2020)
- Fed rate hikes (2022)
- SVB collapse (March 2023)

## Next steps (RL upgrade)

Replace the `decide()` function with an RL policy:
```
State:  [current_price, forecast_median, forecast_uncertainty, portfolio_ratio]
Action: {BUY, SELL, HOLD}
Reward: risk-adjusted return (Sharpe)
```

The Chronos forecast becomes a **tool call** the RL agent chooses when to invoke.
