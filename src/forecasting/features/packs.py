from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base import FeaturePack


@dataclass(frozen=True)
class LagsPack(FeaturePack):
    name: str = "lags"
    lags: tuple[int, ...] = (1, 2, 3)

    def transform(self, base: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for col in base.columns:
            for k in self.lags:
                out[f"{col}_lag_{k}"] = base[col].shift(k)
        return pd.DataFrame(out, index=base.index)


@dataclass(frozen=True)
class RollingStatsPack(FeaturePack):
    name: str = "rolling_stats"
    windows: tuple[int, ...] = (3, 6, 12)
    stats: tuple[str, ...] = ("mean", "std")
    min_periods: int = 3

    def transform(self, base: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for col in base.columns:
            for w in self.windows:
                r = base[col].rolling(window=w, min_periods=self.min_periods)
                if "mean" in self.stats:
                    out[f"{col}_rollmean_{w}"] = r.mean()
                if "std" in self.stats:
                    out[f"{col}_rollstd_{w}"] = r.std(ddof=0)
        return pd.DataFrame(out, index=base.index)


@dataclass(frozen=True)
class VolatilityPack(FeaturePack):
    name: str = "volatility"
    windows: tuple[int, ...] = (3, 6, 12)
    min_periods: int = 3

    def transform(self, base: pd.DataFrame) -> pd.DataFrame:
        out = {}

        # Generic volatility proxy: rolling std of past returns.
        # Safety: returns at time t use values at <= t.
        for col in base.columns:
            returns = base[col].pct_change()
            for w in self.windows:
                out[f"{col}_ret_vol_{w}"] = returns.rolling(
                    window=w, min_periods=self.min_periods
                ).std(ddof=0)
        return pd.DataFrame(out, index=base.index)


@dataclass(frozen=True)
class SpreadsPack(FeaturePack):
    name: str = "spreads"

    def transform(self, base: pd.DataFrame) -> pd.DataFrame:
        if "target" not in base.columns:
            raise ValueError("SpreadsPack expects a 'target' column in base")

        other_cols = [c for c in base.columns if c != "target"]
        if len(other_cols) == 0:
            raise ValueError("SpreadsPack requires at least one exogenous column")

        out = {}
        target = base["target"]
        for col in other_cols:
            out[f"spread_target_minus_{col}"] = target - base[col]
            out[f"ratio_target_over_{col}"] = target / base[col].replace(0.0, np.nan)
        return pd.DataFrame(out, index=base.index)
