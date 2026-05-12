"""
Sudan cotton price connector — East Africa regional estimate.

Source:   Derived from ICE Cotton No.2 with a documented Sudan-specific
          farm-gate discount factor.

Market context:
  Sudan historically produced premium extra-long staple (ELS) cotton in the
  Gezira scheme (Blue Nile / White Nile confluence). Since the 1990s,
  production has shifted toward medium-staple Bt/upland cotton.
  Current production: ~250-350k bales/year, concentrated in Blue Nile and Kassala.

  Sudan farm-gate pricing:
    - Infrastructure constraints (limited gin capacity, road access)
    - Significant political/economic instability premium (logistics cost)
    - Currency control issues affect USD conversion
    - Net: farm-gate approximately ICE × 0.78

  Reference: USDA GAIN Sudan reports (SU-2001, SU-2201),
             ICAC Cotton This Month, FAO Sudan agricultural data.
  Historical range: 0.70 - 0.86 × ICE No.2 depending on political stability.

Unit normalization:
  price_usd_per_lb = ICE_No2_price * 0.78
  price_pkr_per_maund = price_usd_per_lb * pkr_rate * 82.2857

Data quality:
  Classification: REGIONAL
  High-uncertainty estimate due to Sudan's economic instability.
  Treat as indicative only. NOT suitable for direct procurement pricing.

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

COUNTRY   = "Sudan"
SOURCE_ID = "regional/EastAfrica-SDN"

# Sudan farm-gate: ICE × 0.78 (infrastructure + stability discount)
# Most uncertain of all regional estimates; treat as indicative.
_MULTIPLIER   = 0.78
_BASIS_USD_LB = 0.00


def fetch_sudan_cotton_monthly(
    start_date: str = "2015-01-01",
    pkr_rate: float = 278.5,
    ice_series: pd.Series | None = None,
    timeout: int = 15,
) -> pd.DataFrame:
    """
    Return monthly Sudan cotton regional estimates (USD/lb and PKR/maund).

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
            raise RuntimeError(f"Sudan: ICE No.2 fetch failed: {exc}") from exc

    ice_filtered = ice_series[ice_series.index >= pd.Timestamp(start_date)]
    if ice_filtered.empty:
        raise RuntimeError(f"Sudan: no ICE data on or after {start_date}")

    logger.info("Sudan: building East Africa regional estimate (ICE x %.2f)", _MULTIPLIER)
    return ice_basis_df(
        ice_series=ice_filtered,
        country=COUNTRY,
        source_id=SOURCE_ID,
        pkr_rate=pkr_rate,
        multiplier=_MULTIPLIER,
        basis_usd_lb=_BASIS_USD_LB,
    )
