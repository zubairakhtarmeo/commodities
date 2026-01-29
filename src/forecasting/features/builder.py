from __future__ import annotations

from dataclasses import dataclass
import warnings

import pandas as pd

from forecasting.config import FeaturePackConfig
from .packs import LagsPack, RollingStatsPack, SpreadsPack, VolatilityPack


def _pack_from_config(cfg: FeaturePackConfig):
    if cfg.name == "lags":
        lags = tuple(int(x) for x in cfg.params.get("lags", [1, 2, 3]))
        return LagsPack(lags=lags)
    if cfg.name == "rolling_stats":
        windows = tuple(int(x) for x in cfg.params.get("windows", [3, 6, 12]))
        stats = tuple(str(x) for x in cfg.params.get("stats", ["mean", "std"]))
        min_periods = int(cfg.params.get("min_periods", min(windows) if windows else 1))
        return RollingStatsPack(windows=windows, stats=stats, min_periods=min_periods)
    if cfg.name == "volatility":
        windows = tuple(int(x) for x in cfg.params.get("windows", [3, 6, 12]))
        min_periods = int(cfg.params.get("min_periods", min(windows) if windows else 1))
        return VolatilityPack(windows=windows, min_periods=min_periods)
    if cfg.name == "spreads":
        return SpreadsPack()
    raise ValueError(f"Unknown feature pack: {cfg.name}")


@dataclass
class FeatureBuilder:
    """Builds features from a canonical aligned dataframe."""

    packs: list[tuple[object, bool]]

    @classmethod
    def from_configs(cls, configs: list[FeaturePackConfig]) -> "FeatureBuilder":
        packs: list[tuple[object, bool]] = []
        for cfg in configs:
            if not cfg.enabled:
                continue
            packs.append((_pack_from_config(cfg), bool(cfg.optional)))
        return cls(packs=packs)

    def build(self, base: pd.DataFrame) -> pd.DataFrame:
        # Base columns are always available as features.
        frames = [base.add_prefix("base_")]
        for pack, optional in self.packs:
            try:
                frames.append(pack.transform(base))
            except Exception as e:
                if optional:
                    warnings.warn(
                        f"Skipping optional feature pack '{getattr(pack, 'name', type(pack).__name__)}': {e}",
                        RuntimeWarning,
                    )
                    continue
                raise
        X = pd.concat(frames, axis=1)
        return X
