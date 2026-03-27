"""
V3: RL as Infrastructure for Claude
=====================================
Reframes RL's role — Claude decides trade direction, RL handles sizing.

Phase 1: Position Sizing
    - PositionSizingEnv: RL learns optimal allocation fractions
    - Reward isolates sizing quality from direction quality
    - Trained on replayed Claude decisions (rule-based proxy or actual logs)

Architecture:
    Claude: "BUY" / "SELL" / "HOLD"
        |
        v
    RL Sizer: 0% / 25% / 50% / 75% / 100% of requested trade
        |
        v
    execute_fractional(action, fraction, state, price)
"""
