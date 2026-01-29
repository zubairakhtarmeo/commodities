from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from forecasting.config import DatasetConfig


@dataclass(frozen=True)
class SupervisedDataset:
    X: pd.DataFrame
    y: pd.DataFrame
    times: pd.DatetimeIndex
    horizons: tuple[int, ...]


def build_supervised_dataset(
    *,
    features: pd.DataFrame,
    target: pd.Series,
    cfg: DatasetConfig,
    asof: pd.Timestamp | None = None,
) -> SupervisedDataset:
    """Build a leakage-safe supervised dataset.

    Index semantics:
      - row timestamp t is the feature timestamp (what is known as-of t)
      - label for horizon h is target at t+h (future)

    Safety:
      - targets are created by shifting *backward* (negative shift), never forward.
      - optional as-of filtering removes any rows with feature timestamp > asof.
    """

    horizons = tuple(int(h) for h in cfg.horizon_steps)
    if any(h <= 0 for h in horizons):
        raise ValueError("All horizon_steps must be positive integers")

    X = features.copy()
    y = pd.DataFrame(index=X.index)
    for h in horizons:
        y[f"y_h{h}"] = target.shift(-h)

    if asof is not None:
        X = X.loc[X.index <= asof]
        y = y.loc[y.index <= asof]

    if cfg.drop_na_target:
        mask = ~y.isna().any(axis=1)
        X = X.loc[mask]
        y = y.loc[mask]

    times = pd.DatetimeIndex(X.index)
    return SupervisedDataset(X=X, y=y, times=times, horizons=horizons)
