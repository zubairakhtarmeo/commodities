from __future__ import annotations

import pandas as pd

from forecasting.data.registry import SeriesRegistry


def _resample_to_rule(s: pd.Series, rule: str) -> pd.Series:
    # Assumption: observations are timestamped at the end of the period or within the period.
    # We resample to period-end and take the last observed value in that period.
    # Leakage note: using 'last' is safe (uses <= period end). Never use future-looking aggregations.
    # Pandas deprecates 'M' in favor of 'ME' (month-end). Keep config backward-compatible.
    if rule == "M":
        rule = "ME"
    s = s.dropna()
    return s.resample(rule).last()


def align_asset_frame(registry: SeriesRegistry, rule: str) -> pd.DataFrame:
    """Return a canonical dataframe indexed by the target's resampled index.

    Columns:
      - target
      - exogenous role columns (exo_{k}_{role})

    Safety:
      - No backfill.
      - Exogenous availability lags are applied as shifts on the resampled grid.
    """

    target = _resample_to_rule(registry.target.series, rule)
    df = pd.DataFrame({"target": target})

    for i, exo in enumerate(registry.exogenous):
        s_exo = _resample_to_rule(exo.series, rule).reindex(df.index)
        if exo.availability_lag_steps > 0:
            # If the value stamped at time t is only *known* at t+lag, we must shift it forward.
            # That makes it unavailable at time t, preventing leakage.
            s_exo = s_exo.shift(exo.availability_lag_steps)
        df[f"exo_{i}_{exo.role}"] = s_exo

    if registry.target.availability_lag_steps > 0:
        # Rare, but supported: if target is also published with delay (e.g., official benchmark).
        df["target"] = df["target"].shift(registry.target.availability_lag_steps)

    return df
