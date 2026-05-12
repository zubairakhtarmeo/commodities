"""
Pakistan cotton price connector — domestic lint estimate.

Source:   Derived from ICE Cotton No.2 (FRED PCOTTINDUSDM) with
          a Pakistan-specific quality discount factor.

Market context:
  Pakistan is the 4th-largest cotton producer. Its Bt cotton varieties
  (NIAB-78, FH-142, CIM-598) are medium-staple upland cotton, graded
  below US Strict Middling 1-1/16" due to contamination issues and
  shorter average staple length.

  Domestic Pakistan cotton (lint, ex-gin) typically trades at a 10-14%
  discount to ICE No.2:
    - Quality discount:     ~8-10%  (staple length, contamination)
    - No CIF freight added: ~2-3%   (domestic vs export parity saving)
  Net: price_pk ≈ ICE × 0.88 (historical range: 0.82 - 0.93 depending on crop year)

  Note: Pakistani mills that import US or Brazilian cotton pay ICE + freight,
  which is HIGHER than ICE. This connector represents DOMESTIC Pakistan lint
  (the alternative local supply), NOT the import price.

Unit normalization:
  Output USD/lb and PKR/maund, identical schema to other connectors.
  price_pkr_per_maund = price_usd_per_lb * pkr_rate * 82.2857

Data quality:
  Classification: ESTIMATED
  NOT sourced from live market data — derived from ICE No.2 with
  documented basis. Suitable for trend analysis and relative positioning.
  NOT suitable as a precise daily trading reference.

Update cadence:
  Monthly (follows ICE No.2 / FRED cadence).
"""
from __future__ import annotations

import logging

import pandas as pd

from country_sources._common import (
    fetch_ice_no2_series,
    ice_basis_df,
)

logger = logging.getLogger(__name__)

COUNTRY   = "Pakistan"
SOURCE_ID = "ICE/Pakistan-domestic"

# Domestic Pakistan lint cotton: approximately 88% of ICE No.2 price
# Grounded in: crop-year 2018-2024 KCA spot vs ICE No.2 comparisons
# Range observed: 0.82-0.93; using mid-point 0.88
_QUALITY_MULTIPLIER = 0.88
_BASIS_USD_LB       = 0.00   # no additive spread; multiplier captures the discount


def fetch_pakistan_cotton_monthly(
    start_date: str = "2015-01-01",
    pkr_rate: float = 278.5,
    ice_series: pd.Series | None = None,
    timeout: int = 15,
) -> pd.DataFrame:
    """
    Return monthly Pakistan domestic cotton estimates (USD/lb and PKR/maund).

    Args:
        start_date: ISO date string; rows before this date are dropped.
        pkr_rate:   USD -> PKR rate.
        ice_series: Pre-fetched ICE No.2 monthly Series (avoids extra FRED call).
        timeout:    HTTP timeout if ICE series must be fetched.

    Returns:
        DataFrame: date, country, price_usd_per_lb, price_pkr_per_maund, source
    """
    if ice_series is None:
        try:
            ice_series = fetch_ice_no2_series(start_date=start_date, timeout=timeout)
        except Exception as exc:
            raise RuntimeError(f"Pakistan: ICE No.2 fetch failed: {exc}") from exc

    ice_filtered = ice_series[ice_series.index >= pd.Timestamp(start_date)]
    if ice_filtered.empty:
        raise RuntimeError(f"Pakistan: no ICE data on or after {start_date}")

    logger.info("Pakistan: building domestic estimate (ICE x %.2f)", _QUALITY_MULTIPLIER)
    return ice_basis_df(
        ice_series=ice_filtered,
        country=COUNTRY,
        source_id=SOURCE_ID,
        pkr_rate=pkr_rate,
        multiplier=_QUALITY_MULTIPLIER,
        basis_usd_lb=_BASIS_USD_LB,
    )
