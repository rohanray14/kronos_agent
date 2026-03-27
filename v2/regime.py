"""
Regime labeling and classification for the Two-Tier RL System.

Two approaches:
  1. Rule-based labeling (default): deterministic thresholds on ret/vol/VIX
  2. HMM-based labeling (--hmm-regimes): data-driven 5-state HMM

Plus a GradientBoosting classifier trained on macro features → regime labels.

Usage:
    python -m v2.regime --visualize
    python -m v2.regime --hmm-regimes --visualize
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from v2.config import CRASH, BEAR, BULL, RECOVERY, SIDEWAYS, REGIMES


def label_regime(ret_20d: float, vol_20d: float, vix: float) -> int:
    """Rule-based regime labeling."""
    if ret_20d < -0.10 and vix > 30:
        return CRASH
    if ret_20d < -0.03 and vol_20d > 0.015:
        return BEAR
    if ret_20d > 0.03 and vol_20d < 0.012:
        return BULL
    if ret_20d > 0.0 and vix < 20 and vol_20d > 0.01:
        return RECOVERY
    return SIDEWAYS


def label_regimes_rule_based(feature_provider, indices: list[int]) -> np.ndarray:
    """Label regime for each index using rule-based approach."""
    labels = np.full(len(indices), SIDEWAYS, dtype=np.int32)
    for i, idx in enumerate(indices):
        ret_20d = float(feature_provider._ret_20d[idx])
        vol_20d = float(feature_provider._vol_20d[idx])
        vix = float(feature_provider._vix_level[idx])
        # Use raw VIX for threshold (vix_level is normalized by 200d mean)
        # Approximate: vix_level > 1.5 ≈ VIX > 30
        vix_raw = vix * 20.0  # rough denormalization (mean VIX ~20)
        labels[i] = label_regime(ret_20d, vol_20d, vix_raw)
    return labels


def label_regimes_hmm(feature_provider, indices: list[int], n_states: int = 5) -> np.ndarray:
    """Label regimes using a Hidden Markov Model (requires hmmlearn)."""
    from hmmlearn import hmm

    # Build feature matrix for HMM
    X = np.zeros((len(indices), 4))
    for i, idx in enumerate(indices):
        X[i, 0] = feature_provider._ret_20d[idx]
        X[i, 1] = feature_provider._vol_20d[idx]
        X[i, 2] = feature_provider._vix_level[idx]
        # 50d MA slope
        ma50 = feature_provider._ma_50
        if idx >= 5 and ma50[idx - 5] > 0:
            X[i, 3] = (ma50[idx] - ma50[idx - 5]) / ma50[idx - 5]

    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=200,
        random_state=42,
    )
    model.fit(X)
    hidden_states = model.predict(X)

    # Map HMM states to our regime labels based on state means
    state_means = model.means_
    # Sort by return (column 0): lowest return = crash, highest = bull
    sorted_states = np.argsort(state_means[:, 0])

    mapping = {}
    if n_states >= 5:
        mapping[sorted_states[0]] = CRASH
        mapping[sorted_states[1]] = BEAR
        mapping[sorted_states[2]] = SIDEWAYS
        mapping[sorted_states[3]] = RECOVERY
        mapping[sorted_states[4]] = BULL
    else:
        # Fallback for fewer states
        for i, s in enumerate(sorted_states):
            mapping[s] = [BEAR, SIDEWAYS, BULL][min(i, 2)]

    labels = np.array([mapping[s] for s in hidden_states], dtype=np.int32)
    return labels


class RegimeClassifier:
    """
    GradientBoosting classifier: macro features → regime label + confidence.
    """

    def __init__(self):
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            random_state=42,
        )
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train on macro features → regime labels."""
        self.model.fit(X, y)
        self._fitted = True

    def predict(self, features: np.ndarray) -> tuple[int, float]:
        """Predict regime and confidence for a single observation."""
        if not self._fitted:
            return SIDEWAYS, 0.0
        if features.ndim == 1:
            features = features.reshape(1, -1)
        proba = self.model.predict_proba(features)[0]
        regime = int(self.model.classes_[np.argmax(proba)])
        confidence = float(np.max(proba))
        return regime, confidence

    def predict_batch(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict regimes and confidences for a batch."""
        if not self._fitted:
            return (np.full(len(X), SIDEWAYS, dtype=np.int32),
                    np.zeros(len(X), dtype=np.float32))
        proba = self.model.predict_proba(X)
        regimes = self.model.classes_[np.argmax(proba, axis=1)].astype(np.int32)
        confidences = np.max(proba, axis=1).astype(np.float32)
        return regimes, confidences

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        if not self._fitted:
            return 0.0
        return float(self.model.score(X, y))


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from v2.config import FOLDS
    from v2.features import MacroFeatureProvider, download_macro_data

    parser = argparse.ArgumentParser(description="Regime labeling and visualization")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--hmm-regimes", action="store_true")
    parser.add_argument("--fold", type=int, default=None,
                        help="Specific fold to visualize (default: all data)")
    args = parser.parse_args()

    from train_rl_walkforward import load_ticker_data
    spy_df = load_ticker_data("SPY")
    print("Downloading macro data...")
    macro = download_macro_data()
    provider = MacroFeatureProvider(spy_df, macro)

    # Label all indices
    all_indices = list(range(200, len(spy_df)))

    if args.hmm_regimes:
        print("Labeling regimes with HMM...")
        labels = label_regimes_hmm(provider, all_indices)
    else:
        print("Labeling regimes (rule-based)...")
        labels = label_regimes_rule_based(provider, all_indices)

    # Distribution
    from collections import Counter
    dist = Counter(labels)
    total = len(labels)
    print("\nRegime distribution:")
    for regime_id in sorted(dist.keys()):
        count = dist[regime_id]
        print(f"  {REGIMES[regime_id]:12s}: {count:5d} ({count/total*100:.1f}%)")

    # Train classifier on first half, test on second
    mid = len(all_indices) // 2
    X_all = np.array([provider.get_regime_inputs(idx) for idx in all_indices])
    X_train, X_test = X_all[:mid], X_all[mid:]
    y_train, y_test = labels[:mid], labels[mid:]

    clf = RegimeClassifier()
    clf.fit(X_train, y_train)
    train_acc = clf.accuracy(X_train, y_train)
    test_acc = clf.accuracy(X_test, y_test)
    print(f"\nClassifier accuracy: train={train_acc:.3f}, test={test_acc:.3f}")

    if args.visualize:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        dates = [spy_df.index[idx] for idx in all_indices]
        prices = [float(spy_df["close"].iloc[idx]) for idx in all_indices]

        regime_colors = {
            CRASH: "#FF0000",
            BEAR: "#FF6600",
            BULL: "#00CC00",
            RECOVERY: "#0066FF",
            SIDEWAYS: "#999999",
        }

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
        fig.suptitle("SPY Regime Classification", fontsize=14, fontweight="bold")

        # Top: price with regime background
        ax1.plot(dates, prices, color="black", linewidth=1)
        for i in range(len(dates) - 1):
            ax1.axvspan(dates[i], dates[i + 1],
                        alpha=0.3, color=regime_colors[labels[i]])
        ax1.set_ylabel("SPY Price")
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=regime_colors[r], alpha=0.5, label=REGIMES[r].capitalize())
            for r in sorted(REGIMES.keys())
        ]
        ax1.legend(handles=legend_elements, loc="upper left")

        # Bottom: regime label over time
        ax2.scatter(dates, labels, c=[regime_colors[l] for l in labels],
                    s=2, alpha=0.7)
        ax2.set_ylabel("Regime ID")
        ax2.set_yticks(list(REGIMES.keys()))
        ax2.set_yticklabels([REGIMES[r].capitalize() for r in sorted(REGIMES.keys())])

        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax2.xaxis.set_major_locator(mdates.YearLocator())
        plt.tight_layout()
        plt.savefig("outputs/plots/regime_classification.png", dpi=150)
        print("\nPlot saved to outputs/plots/regime_classification.png")
        plt.show()
