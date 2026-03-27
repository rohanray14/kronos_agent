"""
Configuration for the V3 Position Sizing RL System.

V3 reframes RL's role: Claude decides direction (BUY/SELL/HOLD),
RL decides how much capital to allocate.
"""

import os

# Re-export shared config from v2
from v2.config import (
    DATA_START, DATA_END, DEFAULT_TICKERS, TEST_TICKER, MACRO_TICKERS,
    FOLDS, CRASH, BEAR, BULL, RECOVERY, SIDEWAYS, REGIMES,
    INITIAL_CASH, LOOKBACK_DAYS, FORECAST_STEPS,
    CACHE_DIR, MACRO_CACHE_DIR,
    REGIME_CONFIDENCE_THRESHOLD,
)

# ─────────────────────────────────────────────
# POSITION SIZING
# ─────────────────────────────────────────────

# Discrete allocation levels (fraction of Claude's requested trade)
# No 0% — Claude already decided the direction, sizer only scales it.
# If Claude says BUY/SELL, the sizer must execute at least 25%.
SIZING_LEVELS = [0.25, 0.50, 0.75, 1.0]
N_SIZING_ACTIONS = len(SIZING_LEVELS)

# Observation dimensions
# 18 market features (from MacroFeatureProvider)
#  + 6 agent/sizing context features
# = 24 total
SIZING_OBS_DIM = 24

# Agent context feature indices (appended after 18 market features)
# [18] claude_action: -1=SELL, 0=HOLD, 1=BUY
# [19] position_frac: current position as fraction of portfolio (0-1)
# [20] unrealized_pnl: normalized by initial cash
# [21] days_since_trade: normalized by 50
# [22] rolling_win_rate: Claude's win rate over last 10 decisions
# [23] vol_20d (repeated for sizer's direct access)

# ─────────────────────────────────────────────
# REWARD PARAMS (position sizing)
# ─────────────────────────────────────────────

# Risk-adjusted return weighting
RISK_LAMBDA = 1.5  # drawdown penalty multiplier

# Concentration penalty: penalize being >80% invested when vol is high
CONCENTRATION_VOL_THRESHOLD = 0.015  # 1.5% daily vol
CONCENTRATION_PENALTY = 0.001

# Cash drag penalty: penalize 0% position in confirmed bull regime
CASH_DRAG_PENALTY = 0.0005

# Per-regime sizing reward adjustments
SIZING_REWARD_PARAMS = {
    BULL:     {"risk_lambda": 1.0, "conc_penalty": 0.0005, "drag_penalty": 0.001},
    BEAR:     {"risk_lambda": 2.5, "conc_penalty": 0.002,  "drag_penalty": 0.0},
    CRASH:    {"risk_lambda": 3.0, "conc_penalty": 0.003,  "drag_penalty": 0.0},
    RECOVERY: {"risk_lambda": 1.5, "conc_penalty": 0.001,  "drag_penalty": 0.0005},
    SIDEWAYS: {"risk_lambda": 1.5, "conc_penalty": 0.001,  "drag_penalty": 0.0},
}

# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

DEFAULT_TIMESTEPS = 100_000
PPO_LR = 3e-4
PPO_ENT_COEF = 0.02  # moderate exploration for sizing

# Model save directory
MODEL_DIR = os.path.join(CACHE_DIR, "v3_models")
