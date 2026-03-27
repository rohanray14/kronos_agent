"""
Macro Feature Provider for the Two-Tier RL System.

Builds a 22-dim observation vector enriched with cross-asset macro features
(VIX, yields, gold, treasuries, oil, small-cap spread, defensive rotation).

Usage:
    python -m v2.features --fold 1
"""

import os
import json
import time

import numpy as np
import pandas as pd
import yfinance as yf

from v2.config import (
    MACRO_TICKERS, DATA_START, DATA_END, MACRO_CACHE_DIR,
    LOOKBACK_DAYS, FORECAST_STEPS,
)


def download_macro_data(max_retries: int = 3) -> dict[str, pd.DataFrame]:
    """Download all macro tickers, with retry logic."""
    macro_data = {}
    for ticker in MACRO_TICKERS:
        for attempt in range(max_retries):
            try:
                df = yf.download(
                    ticker, start=DATA_START, end=DATA_END,
                    progress=False, auto_adjust=True,
                )
                if df is None or len(df) == 0:
                    raise ValueError(f"Empty data for {ticker}")
                df.columns = df.columns.get_level_values(0)
                df = df.rename(columns={
                    "Open": "open", "High": "high", "Low": "low",
                    "Close": "close", "Volume": "volume",
                })
                df.index = pd.to_datetime(df.index)
                df = df.dropna(subset=["close"])
                macro_data[ticker] = df
                print(f"   {ticker}: {len(df)} days")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    print(f"   Retry {attempt+1}/{max_retries} for {ticker}: {e}")
                    time.sleep(wait)
                else:
                    print(f"   WARNING: Failed to download {ticker}: {e}")
        if ticker not in macro_data:
            time.sleep(1)
        else:
            time.sleep(0.5)
    return macro_data


class MacroFeatureProvider:
    """
    Provides 18 market-level features (indices 1-17 of the 22-dim obs)
    aligned to a primary ticker's trading dates.

    The remaining 4 features (position, portfolio_return, days_in_position,
    regime_id, regime_confidence) are added by the environment.

    Feature vector (market features only, 18 dims):
         0: ret_5d
         1: ret_20d
         2: ret_50d
         3: vol_20d
         4: price_vs_50ma
         5: price_vs_200ma
         6: kronos_expected_ret
         7: kronos_trend
         8: kronos_high_vol
         9: vix_level           (VIX / 200d mean)
        10: vix_change_5d
        11: yield_curve          (10Y - 3M, normalized)
        12: gld_ret_20d
        13: tlt_ret_20d
        14: uso_ret_20d
        15: iwm_spy_spread_20d
        16: xlu_spy_spread_20d
        17: (reserved / padding = 0)
    """

    N_MARKET_FEATURES = 18

    def __init__(
        self,
        ticker_df: pd.DataFrame,
        macro_data: dict[str, pd.DataFrame],
        kronos_cache: dict[int, tuple] | None = None,
    ):
        self.ticker_df = ticker_df
        self.prices = ticker_df["close"].values
        self.dates = ticker_df.index
        self.kronos_cache = kronos_cache or {}

        # Align macro data to primary ticker dates via forward-fill
        self._aligned = {}
        for sym, mdf in macro_data.items():
            aligned = mdf["close"].reindex(self.dates, method="ffill")
            self._aligned[sym] = aligned.values

        # Pre-compute primary ticker derived arrays
        self._precompute()

    def _precompute(self):
        """Pre-compute rolling stats for the primary ticker."""
        prices = self.prices.astype(np.float64)
        n = len(prices)

        # Returns
        self._ret_5d = np.zeros(n)
        self._ret_20d = np.zeros(n)
        self._ret_50d = np.zeros(n)
        for i in range(n):
            if i >= 5 and prices[i - 5] > 0:
                self._ret_5d[i] = (prices[i] - prices[i - 5]) / prices[i - 5]
            if i >= 20 and prices[i - 20] > 0:
                self._ret_20d[i] = (prices[i] - prices[i - 20]) / prices[i - 20]
            if i >= 50 and prices[i - 50] > 0:
                self._ret_50d[i] = (prices[i] - prices[i - 50]) / prices[i - 50]

        # Volatility (20d)
        self._vol_20d = np.zeros(n)
        for i in range(20, n):
            window = prices[i - 20:i]
            rets = np.diff(window) / window[:-1]
            self._vol_20d[i] = np.std(rets)

        # MAs
        self._ma_50 = pd.Series(prices).rolling(50, min_periods=1).mean().values
        self._ma_200 = pd.Series(prices).rolling(200, min_periods=1).mean().values

        # VIX derived
        vix = self._aligned.get("^VIX")
        if vix is not None:
            vix_series = pd.Series(vix)
            vix_ma200 = vix_series.rolling(200, min_periods=1).mean().values
            self._vix_level = np.where(vix_ma200 > 0, vix / vix_ma200, 1.0)
            self._vix_change_5d = np.zeros(n)
            for i in range(5, n):
                if vix[i - 5] > 0:
                    self._vix_change_5d[i] = (vix[i] - vix[i - 5]) / vix[i - 5]
        else:
            self._vix_level = np.ones(n)
            self._vix_change_5d = np.zeros(n)

        # Yield curve: 10Y (^TNX) - 3M (^IRX), normalized by dividing by 4
        tnx = self._aligned.get("^TNX")
        irx = self._aligned.get("^IRX")
        if tnx is not None and irx is not None:
            self._yield_curve = (tnx - irx) / 4.0
        else:
            self._yield_curve = np.zeros(n)

        # Cross-asset 20d returns
        self._macro_ret_20d = {}
        for sym in ["GLD", "TLT", "USO"]:
            arr = self._aligned.get(sym)
            if arr is not None:
                ret = np.zeros(n)
                for i in range(20, n):
                    if arr[i - 20] > 0:
                        ret[i] = (arr[i] - arr[i - 20]) / arr[i - 20]
                self._macro_ret_20d[sym] = ret
            else:
                self._macro_ret_20d[sym] = np.zeros(n)

        # IWM-SPY spread (small vs large cap)
        iwm = self._aligned.get("IWM")
        if iwm is not None:
            self._iwm_spy_spread = np.zeros(n)
            for i in range(20, n):
                iwm_ret = (iwm[i] - iwm[i - 20]) / iwm[i - 20] if iwm[i - 20] > 0 else 0
                spy_ret = self._ret_20d[i]
                self._iwm_spy_spread[i] = iwm_ret - spy_ret
        else:
            self._iwm_spy_spread = np.zeros(n)

        # XLU-SPY spread (defensive rotation)
        xlu = self._aligned.get("XLU")
        if xlu is not None:
            self._xlu_spy_spread = np.zeros(n)
            for i in range(20, n):
                xlu_ret = (xlu[i] - xlu[i - 20]) / xlu[i - 20] if xlu[i - 20] > 0 else 0
                spy_ret = self._ret_20d[i]
                self._xlu_spy_spread[i] = xlu_ret - spy_ret
        else:
            self._xlu_spy_spread = np.zeros(n)

    def get_features(self, idx: int) -> np.ndarray:
        """
        Return 18-dim market feature vector for a given index into the
        primary ticker's dataframe.
        """
        price = float(self.prices[idx])

        # Price vs MAs
        ma50 = self._ma_50[idx]
        price_vs_50ma = (price - ma50) / ma50 if ma50 > 0 else 0.0
        ma200 = self._ma_200[idx]
        price_vs_200ma = (price - ma200) / ma200 if ma200 > 0 else 0.0

        # Kronos features (from cache)
        if idx in self.kronos_cache:
            k_ret, k_trend, k_hvol = self.kronos_cache[idx]
        else:
            k_ret, k_trend, k_hvol = 0.0, 0.0, 0.0

        features = np.array([
            self._ret_5d[idx],              # 0
            self._ret_20d[idx],             # 1
            self._ret_50d[idx],             # 2
            self._vol_20d[idx],             # 3
            price_vs_50ma,                  # 4
            price_vs_200ma,                 # 5
            k_ret,                          # 6
            k_trend,                        # 7
            k_hvol,                         # 8
            self._vix_level[idx],           # 9
            self._vix_change_5d[idx],       # 10
            self._yield_curve[idx],         # 11
            self._macro_ret_20d["GLD"][idx],  # 12
            self._macro_ret_20d["TLT"][idx],  # 13
            self._macro_ret_20d["USO"][idx],  # 14
            self._iwm_spy_spread[idx],      # 15
            self._xlu_spy_spread[idx],      # 16
            0.0,                            # 17 reserved
        ], dtype=np.float32)

        return features

    def get_regime_inputs(self, idx: int) -> np.ndarray:
        """Return subset of features used for regime classification."""
        return np.array([
            self._ret_20d[idx],
            self._vol_20d[idx],
            self._vix_level[idx],
            self._vix_change_5d[idx],
            self._yield_curve[idx],
            self._ret_5d[idx],
            self._ret_50d[idx],
            float(self.prices[idx] - self._ma_50[idx]) / self._ma_50[idx]
            if self._ma_50[idx] > 0 else 0.0,
        ], dtype=np.float32)

    def save_cache(self, path: str):
        """Save pre-computed feature arrays to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "ret_5d": self._ret_5d.tolist(),
            "ret_20d": self._ret_20d.tolist(),
            "ret_50d": self._ret_50d.tolist(),
            "vol_20d": self._vol_20d.tolist(),
            "vix_level": self._vix_level.tolist(),
            "vix_change_5d": self._vix_change_5d.tolist(),
            "yield_curve": self._yield_curve.tolist(),
            "gld_ret_20d": self._macro_ret_20d["GLD"].tolist(),
            "tlt_ret_20d": self._macro_ret_20d["TLT"].tolist(),
            "uso_ret_20d": self._macro_ret_20d["USO"].tolist(),
            "iwm_spy_spread": self._iwm_spy_spread.tolist(),
            "xlu_spy_spread": self._xlu_spy_spread.tolist(),
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load_cache(self, path: str):
        """Load pre-computed feature arrays from disk."""
        with open(path, "r") as f:
            data = json.load(f)
        self._ret_5d = np.array(data["ret_5d"])
        self._ret_20d = np.array(data["ret_20d"])
        self._ret_50d = np.array(data["ret_50d"])
        self._vol_20d = np.array(data["vol_20d"])
        self._vix_level = np.array(data["vix_level"])
        self._vix_change_5d = np.array(data["vix_change_5d"])
        self._yield_curve = np.array(data["yield_curve"])
        self._macro_ret_20d["GLD"] = np.array(data["gld_ret_20d"])
        self._macro_ret_20d["TLT"] = np.array(data["tlt_ret_20d"])
        self._macro_ret_20d["USO"] = np.array(data["uso_ret_20d"])
        self._iwm_spy_spread = np.array(data["iwm_spy_spread"])
        self._xlu_spy_spread = np.array(data["xlu_spy_spread"])


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from v2.config import FOLDS

    parser = argparse.ArgumentParser(description="Verify macro feature pipeline")
    parser.add_argument("--fold", type=int, default=1)
    args = parser.parse_args()

    fold = FOLDS[args.fold - 1]
    print(f"Fold {fold['fold']}: {fold['train_start']} -> {fold['test_end']}")

    # Load primary ticker
    from train_rl_walkforward import load_ticker_data
    spy_df = load_ticker_data("SPY")

    # Load macro data
    print("Downloading macro data...")
    macro = download_macro_data()
    print(f"Downloaded {len(macro)} macro tickers")

    # Build provider
    provider = MacroFeatureProvider(spy_df, macro)

    # Test features at various points
    test_idx = len(spy_df) // 2
    feats = provider.get_features(test_idx)
    print(f"\nFeature vector at index {test_idx} "
          f"(date: {spy_df.index[test_idx].date()}):")
    labels = [
        "ret_5d", "ret_20d", "ret_50d", "vol_20d", "price_vs_50ma",
        "price_vs_200ma", "kronos_ret", "kronos_trend", "kronos_hvol",
        "vix_level", "vix_change_5d", "yield_curve",
        "gld_ret_20d", "tlt_ret_20d", "uso_ret_20d",
        "iwm_spy_spread", "xlu_spy_spread", "reserved",
    ]
    for i, (label, val) in enumerate(zip(labels, feats)):
        print(f"  [{i:2d}] {label:25s} = {val:+.6f}")

    # Cache test
    cache_path = os.path.join(MACRO_CACHE_DIR, f"SPY_fold{args.fold}_macro.json")
    provider.save_cache(cache_path)
    print(f"\nCache saved to {cache_path}")

    # Verify reload
    provider2 = MacroFeatureProvider(spy_df, macro)
    provider2.load_cache(cache_path)
    feats2 = provider2.get_features(test_idx)
    assert np.allclose(feats, feats2), "Cache round-trip mismatch!"
    print("Cache round-trip OK")
