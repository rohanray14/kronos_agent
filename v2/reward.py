"""
Shaped reward functions for per-regime RL training.

Key improvements over v1 log-return:
  - Asymmetric loss penalty (losses hurt 2x)
  - Drawdown penalty (regime-tuned)
  - Inaction bonus in dangerous regimes (CRASH, BEAR)
  - Sharpe-like penalty for SIDEWAYS
"""

import numpy as np

from v2.config import (
    CRASH, BEAR, BULL, RECOVERY, SIDEWAYS,
    REWARD_PARAMS,
)


def shaped_reward(
    pv_before: float,
    pv_after: float,
    peak_pv: float,
    regime: int,
    position: int,
    vol_20d: float,
) -> float:
    """
    Compute shaped reward with regime-specific tuning.

    Args:
        pv_before: portfolio value before step
        pv_after: portfolio value after step
        peak_pv: peak portfolio value so far
        regime: current regime label (CRASH, BEAR, BULL, RECOVERY, SIDEWAYS)
        position: 0 (cash) or 1 (invested)
        vol_20d: 20-day realized volatility
    """
    if pv_before <= 0:
        return 0.0

    base = float(np.log(pv_after / pv_before))

    # Asymmetric: losses hurt 2x more
    if base < 0:
        base *= 2.0

    params = REWARD_PARAMS.get(regime, REWARD_PARAMS[SIDEWAYS])

    # Drawdown penalty
    dd_penalty = 0.0
    if pv_after < peak_pv and peak_pv > 0:
        dd_penalty = params["dd_weight"] * (pv_after - peak_pv) / peak_pv

    # Inaction bonus in dangerous regimes
    inaction_bonus = params["inaction_bonus"] if position == 0 else 0.0

    # Sharpe-like volatility penalty (mainly for sideways)
    vol_penalty = 0.0
    if params["vol_penalty"] > 0 and vol_20d > 0:
        vol_penalty = -params["vol_penalty"] * vol_20d

    return base + dd_penalty + inaction_bonus + vol_penalty


def make_reward_fn(regime: int):
    """Return a reward function bound to a specific regime."""
    def reward_fn(pv_before, pv_after, peak_pv, position, vol_20d):
        return shaped_reward(pv_before, pv_after, peak_pv, regime, position, vol_20d)
    return reward_fn


def fallback_reward(pv_before: float, pv_after: float, peak_pv: float,
                    position: int, vol_20d: float) -> float:
    """Default reward for the all-data fallback policy (uses SIDEWAYS params)."""
    return shaped_reward(pv_before, pv_after, peak_pv, SIDEWAYS, position, vol_20d)
