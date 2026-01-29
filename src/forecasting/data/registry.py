from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from forecasting.config import AssetSpec, ConnectorConfig
from forecasting.connectors import CSVConnector


@dataclass(frozen=True)
class NamedSeries:
    asset_id: str
    role: str
    series: pd.Series
    availability_lag_steps: int = 0


@dataclass(frozen=True)
class SeriesRegistry:
    """Holds all series for an asset, keyed by role and an internal name."""

    asset_id: str
    target: NamedSeries
    exogenous: list[NamedSeries]


def _connector_from_config(cfg: ConnectorConfig):
    if cfg.type == "csv":
        return CSVConnector(
            path=cfg.path,
            time_col=cfg.time_col,
            value_col=cfg.value_col,
            timezone=cfg.timezone,
        )
    raise ValueError(f"Unsupported connector type: {cfg.type}")


def _apply_availability_lag(s: pd.Series, lag_steps: int) -> pd.Series:
    if lag_steps <= 0:
        return s
    # Shift forward in time index positions (not calendar) after we align to canonical frequency.
    # We apply this later post-resample; here we just record lag.
    return s


def build_series_registry(asset: AssetSpec) -> SeriesRegistry:
    target_conn = _connector_from_config(asset.target.connector)
    target = target_conn.load_series()

    exo_list: list[NamedSeries] = []
    for spec in asset.exogenous:
        conn = _connector_from_config(spec.connector)
        exo_list.append(
            NamedSeries(
                asset_id=asset.asset_id,
                role=spec.role,
                series=conn.load_series(),
                availability_lag_steps=spec.availability_lag_steps,
            )
        )

    return SeriesRegistry(
        asset_id=asset.asset_id,
        target=NamedSeries(
            asset_id=asset.asset_id,
            role=asset.target.role,
            series=target,
            availability_lag_steps=asset.target.availability_lag_steps,
        ),
        exogenous=exo_list,
    )
