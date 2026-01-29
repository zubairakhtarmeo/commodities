from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DataQualityReport:
    last_timestamp: pd.Timestamp
    periods: int
    missing_fraction: float
    contiguous_missing_max: int


def assess_data_quality(df: pd.DataFrame) -> DataQualityReport:
    """Basic feed-health checks.

    Safe by design: uses only observed timestamps and missingness.
    """

    if len(df.index) == 0:
        raise ValueError("Empty dataframe")

    missing = float(df.isna().mean().mean())

    # longest contiguous missing run across any column (approx):
    any_missing = df.isna().any(axis=1).astype(int)
    run = 0
    max_run = 0
    for v in any_missing.to_list():
        if v == 1:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0

    return DataQualityReport(
        last_timestamp=pd.Timestamp(df.index.max()),
        periods=int(len(df.index)),
        missing_fraction=missing,
        contiguous_missing_max=int(max_run),
    )
