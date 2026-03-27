"""
Reward functions for position sizing RL.

Key design principle: reward sizing QUALITY, not direction quality.
Claude decides BUY/SELL/HOLD. The sizer decides how much.
The reward isolates whether the chosen SIZE was appropriate for the
risk environment, regardless of whether Claude's direction was right.

Metrics:
  - Return per unit risk (isolates sizing from direction)
  - Drawdown contribution (did the size cause excessive drawdown?)
  - Concentration penalty (too much in high-vol regimes)
  - Cash drag penalty (too little in confirmed bull regimes)
"""

import numpy as np

from v3.config import (
    CRASH, BEAR, BULL, RECOVERY, SIDEWAYS,
    SIZING_REWARD_PARAMS, SIZING_LEVELS,
    CONCENTRATION_VOL_THRESHOLD,
)


def sizing_reward(
    pv_before: float,
    pv_after: float,
    peak_pv: float,
    position_frac: float,
    vol_20d: float,
    regime: int,
    sizing_action: int,
) -> float:
    """
    Compute reward for a position sizing decision.

    Args:
        pv_before: portfolio value before the step
        pv_after: portfolio value after the step
        peak_pv: peak portfolio value (for drawdown)
        position_frac: fraction of portfolio currently invested (0-1)
        vol_20d: 20-day realized volatility
        regime: current market regime
        sizing_action: the discrete sizing action taken (index into SIZING_LEVELS)
    """
    if pv_before <= 0:
        return 0.0

    params = SIZING_REWARD_PARAMS.get(regime, SIZING_REWARD_PARAMS[SIDEWAYS])
    alloc = SIZING_LEVELS[sizing_action]

    # ── Base: risk-adjusted return ──
    # Return per unit of risk exposure. This means:
    # - If Claude was right (positive return), higher allocation = more reward
    # - If Claude was wrong (negative return), lower allocation = less punishment
    # - Scaled by volatility to normalize across regimes
    raw_return = (pv_after - pv_before) / pv_before
    epsilon = 1e-6
    risk_adjusted = raw_return / max(vol_20d, epsilon)

    # Scale down to reasonable reward magnitude
    base = np.clip(risk_adjusted * 0.01, -0.1, 0.1)

    # ── Drawdown penalty ──
    # Penalize sizing that contributes to drawdown
    dd_penalty = 0.0
    if pv_after < peak_pv and peak_pv > 0:
        dd_frac = (peak_pv - pv_after) / peak_pv
        # Penalty proportional to both drawdown AND position size
        # (larger position during drawdown = more penalty)
        dd_penalty = -params["risk_lambda"] * dd_frac * position_frac * 0.01

    # ── Concentration penalty ──
    # Penalize being heavily invested when volatility is high
    conc_penalty = 0.0
    if position_frac > 0.8 and vol_20d > CONCENTRATION_VOL_THRESHOLD:
        excess_vol = (vol_20d - CONCENTRATION_VOL_THRESHOLD) / CONCENTRATION_VOL_THRESHOLD
        conc_penalty = -params["conc_penalty"] * excess_vol

    # ── Cash drag penalty ──
    # Penalize sitting in cash during confirmed bull markets
    drag_penalty = 0.0
    if regime == BULL and position_frac < 0.25 and raw_return > 0:
        # Only penalize if the market actually went up (confirmed bull)
        drag_penalty = -params["drag_penalty"]

    return base + dd_penalty + conc_penalty + drag_penalty


def make_sizing_reward_fn(regime: int):
    """Return a sizing reward function bound to a specific regime."""
    def reward_fn(pv_before, pv_after, peak_pv, position_frac, vol_20d, sizing_action):
        return sizing_reward(
            pv_before, pv_after, peak_pv, position_frac,
            vol_20d, regime, sizing_action,
        )
    return reward_fn
