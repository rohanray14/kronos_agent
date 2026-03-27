"""
Configuration for the Two-Tier RL Trading System.
"""

import os

# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────

DATA_START = "2012-01-01"
DATA_END = "2024-12-31"

DEFAULT_TICKERS = ["SPY", "QQQ", "DIA", "IWM", "XLF", "XLE", "XLK"]
TEST_TICKER = "SPY"

MACRO_TICKERS = ["^VIX", "^TNX", "^IRX", "GLD", "TLT", "USO", "IWM", "XLU"]

# ─────────────────────────────────────────────
# WALK-FORWARD FOLDS (from train_rl_walkforward.py)
# ─────────────────────────────────────────────

FOLDS = [
    {"fold": 1, "train_start": "2014-01-01", "train_end": "2018-12-31",
     "test_start": "2019-01-01", "test_end": "2019-12-31",
     "regime": "Late Bull", "notes": "Pre-COVID, low vol, steady growth"},
    {"fold": 2, "train_start": "2015-01-01", "train_end": "2019-12-31",
     "test_start": "2020-01-01", "test_end": "2020-12-31",
     "regime": "COVID Crash + Recovery", "notes": "-34% crash then V-shaped recovery"},
    {"fold": 3, "train_start": "2016-01-01", "train_end": "2020-12-31",
     "test_start": "2021-01-01", "test_end": "2021-12-31",
     "regime": "Post-COVID Bull", "notes": "Stimulus-driven bull, meme stocks, low rates"},
    {"fold": 4, "train_start": "2017-01-01", "train_end": "2021-12-31",
     "test_start": "2022-01-01", "test_end": "2022-12-31",
     "regime": "Bear Market", "notes": "Fed hikes, -25% drawdown, inflation"},
    {"fold": 5, "train_start": "2018-01-01", "train_end": "2022-12-31",
     "test_start": "2023-01-01", "test_end": "2023-12-31",
     "regime": "Recovery", "notes": "SVB collapse then AI-driven recovery"},
    {"fold": 6, "train_start": "2019-01-01", "train_end": "2023-12-31",
     "test_start": "2024-01-01", "test_end": "2024-12-31",
     "regime": "AI Rally", "notes": "Mag-7 led bull market, rate cuts begin"},
]

# ─────────────────────────────────────────────
# REGIMES
# ─────────────────────────────────────────────

CRASH = 0
BEAR = 1
BULL = 2
RECOVERY = 3
SIDEWAYS = 4

REGIMES = {
    CRASH: "crash",
    BEAR: "bear",
    BULL: "bull",
    RECOVERY: "recovery",
    SIDEWAYS: "sideways",
}

MIN_REGIME_STEPS = 30  # minimum steps to train a regime-specific policy

# ─────────────────────────────────────────────
# REWARD PARAMS (per-regime)
# ─────────────────────────────────────────────

REWARD_PARAMS = {
    BULL:     {"dd_weight": 0.3, "inaction_bonus": 0.0,   "vol_penalty": 0.0},
    BEAR:     {"dd_weight": 1.5, "inaction_bonus": 0.001, "vol_penalty": 0.0},
    CRASH:    {"dd_weight": 2.0, "inaction_bonus": 0.002, "vol_penalty": 0.0},
    RECOVERY: {"dd_weight": 0.5, "inaction_bonus": 0.0,   "vol_penalty": 0.0},
    SIDEWAYS: {"dd_weight": 0.5, "inaction_bonus": 0.0,   "vol_penalty": 0.5},
}

# ─────────────────────────────────────────────
# PPO HYPERPARAMS (per-regime)
# ─────────────────────────────────────────────

PPO_PARAMS = {
    BULL:     {"lr": 3e-4, "ent_coef": 0.01, "timesteps_mult": 1.0},
    BEAR:     {"lr": 1e-4, "ent_coef": 0.05, "timesteps_mult": 1.0},
    CRASH:    {"lr": 1e-4, "ent_coef": 0.10, "timesteps_mult": 0.5},
    RECOVERY: {"lr": 3e-4, "ent_coef": 0.02, "timesteps_mult": 1.0},
    SIDEWAYS: {"lr": 3e-4, "ent_coef": 0.02, "timesteps_mult": 1.0},
}

# Fallback (all-data) policy uses these
FALLBACK_PPO = {"lr": 3e-4, "ent_coef": 0.01}

# ─────────────────────────────────────────────
# TRADING ENV
# ─────────────────────────────────────────────

INITIAL_CASH = 100_000
LOOKBACK_DAYS = 60
FORECAST_STEPS = 5
OBS_DIM = 22

REGIME_CONFIDENCE_THRESHOLD = 0.5

# ─────────────────────────────────────────────
# CACHING
# ─────────────────────────────────────────────

CACHE_DIR = "walkforward_cache"
MACRO_CACHE_DIR = os.path.join(CACHE_DIR, "macro_features")
