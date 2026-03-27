"""
Two-Tier RL Trading System (v2)
================================
Regime-aware RL with macro features, per-regime policies, and shaped rewards.

Architecture:
    Tier 1: Regime Classifier (GradientBoosting on macro features)
        → predicts: bull / bear / crash / recovery / sideways + confidence

    Tier 2: Per-Regime RL Policies (PPO, one per regime)
        → each trained only on matching regime data
        → richer 22-dim observation with macro/cross-asset features
        → regime-specific reward shaping

    Fallback: all-data policy used when classifier confidence < threshold
"""
