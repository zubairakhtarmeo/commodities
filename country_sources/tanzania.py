"""
Tanzania cotton price connector — East Africa regional estimate.

Source:   Derived from ICE Cotton No.2 with a documented East Africa
          farm-gate discount factor.

Market context:
  Tanzania is East Africa's largest cotton producer (~400k bales/year,
  Shinyanga, Mwanza, Simiyu regions). Tanzanian cotton is mostly
  Upland / Acala type — shorter staple (1-1/8"), lower grade than
  US Strict Middling, with some contamination issues.

  Farm-gate price structure:
    - Grade discount to ICE No.2:         ~15-20% (staple/grade)
    - Transport to Dar es Salaam port:    ~$0.04-0.06/lb
    - Net farm-gate: approximately ICE × 0.80

  Reference: ICAC Cotton This Month, World Bank commodity briefs,
             USDA GAIN Tanzania cotton reports (TZ-1801, TZ-2101).
  Historical range: 0.72 - 0.87 × ICE No.2 (Tanzanian average).

Unit normalization:
  price_usd_per_lb = ICE_No2_price * 0.80
  price_pkr_per_maund = price_usd_per_lb * pkr_rate * 82.2857

Data quality:
  Classification: REGIONAL
  Regional benchmark estimate. Suitable for relative price positioning
  and regional competitiveness analysis. NOT a direct spot market price.

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

COUNTRY   = "Tanzania"
SOURCE_ID = "regional/EastAfrica"

# Tanzania farm-gate: ICE × 0.80 (20% discount for grade + transport)
# Based on: ICAC/USDA GAIN data 2015-2024 average positioning
_MULTIPLIER  = 0.80
_BASIS_USD_LB = 0.00


def fetch_tanzania_cotton_monthly(
    start_date: str = "2015-01-01",
    pkr_rate: float = 278.5,
    ice_series: pd.Series | None = None,
    timeout: int = 15,
) -> pd.DataFrame:
    """
    Return monthly Tanzania cotton regional estimates (USD/lb and PKR/maund).

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
            raise RuntimeError(f"Tanzania: ICE No.2 fetch failed: {exc}") from exc

    ice_filtered = ice_series[ice_series.index >= pd.Timestamp(start_date)]
    if ice_filtered.empty:
        raise RuntimeError(f"Tanzania: no ICE data on or after {start_date}")

    logger.info("Tanzania: building East Africa regional estimate (ICE x %.2f)", _MULTIPLIER)
    return ice_basis_df(
        ice_series=ice_filtered,
        country=COUNTRY,
        source_id=SOURCE_ID,
        pkr_rate=pkr_rate,
        multiplier=_MULTIPLIER,
        basis_usd_lb=_BASIS_USD_LB,
    )
