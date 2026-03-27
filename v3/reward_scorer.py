"""
Reward function for the Signal Quality Scorer.

The scorer outputs a confidence score (0-1) predicting how reliable
the current Kronos forecast is. The reward measures CALIBRATION:
how well the score matches realized forecast accuracy.

A well-calibrated scorer:
  - Outputs 0.8 when Kronos is right ~80% of the time
  - Outputs 0.2 when Kronos is right ~20% of the time

Reward components:
  1. Calibration: -|score - realized_accuracy|
  2. Direction bonus: extra reward if high score + correct, or low score + incorrect
  3. Sharpness: slight bonus for confident predictions (near 0 or 1) that are correct
"""

import numpy as np


def scorer_reward(
    score: float,
    kronos_direction_correct: bool,
    forecast_return: float,
    realized_return: float,
) -> float:
    """
    Compute reward for a signal quality prediction.

    Args:
        score: the predicted quality score (0-1)
        kronos_direction_correct: did Kronos predict the right direction?
        forecast_return: Kronos's predicted return
        realized_return: actual return over the forecast horizon
    """
    actual = 1.0 if kronos_direction_correct else 0.0

    # ── Calibration reward (primary) ──
    # Negative absolute error: perfect calibration = 0, worst = -1
    calibration = -abs(score - actual)

    # ── Direction bonus ──
    # Reward alignment: high score + correct = good, low score + incorrect = good
    # Penalize misalignment: high score + incorrect = bad
    if kronos_direction_correct:
        direction_bonus = score * 0.3  # reward for high confidence when right
    else:
        direction_bonus = (1.0 - score) * 0.3  # reward for low confidence when wrong

    # ── Magnitude awareness ──
    # Bonus for correctly scaling confidence with forecast magnitude
    # Large forecasted moves that are correct deserve high confidence
    magnitude_bonus = 0.0
    if kronos_direction_correct and abs(forecast_return) > 0.01:
        magnitude_bonus = score * 0.1  # reward confidence on large correct moves
    elif not kronos_direction_correct and abs(forecast_return) > 0.01:
        magnitude_bonus = (1.0 - score) * 0.1  # reward skepticism on large wrong moves

    return calibration + direction_bonus + magnitude_bonus
