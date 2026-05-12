"""
West Africa cotton price connector — regional aggregate estimate.

Source:   Derived from ICE Cotton No.2 with a documented West Africa
          aggregate farm-gate discount factor.

Market context:
  West Africa cotton belt (Burkina Faso, Mali, Benin, Senegal, Togo,
  Niger) produces ~2.5-3.0 million bales/year — the largest cotton-
  producing region in sub-Saharan Africa. Cotton is primarily grown by
  smallholders under contract-farming arrangements with state ginning
  companies (SOFITEX, CMDT, SODECO, etc.).

  West Africa aggregate pricing (non-CIV blended):
    - Government-set farm-gate prices, reviewed annually
    - Typically 78-85% of world price in USD terms
    - Slightly higher infrastructure quality than East Africa
    - Net: approximately ICE × 0.82

  This connector represents the unweighted average for the West Africa
  cotton belt, to be distinguished from the Ivory Coast-specific
  connector (which follows identical methodology but tracks separately
  for country-level granularity in the dashboard).

  Reference: USDA GAIN West Africa Cotton reports,
             ICAC Cotton This Month, World Bank Africa agriculture briefs.

Unit normalization:
  price_usd_per_lb = ICE_No2_price * 0.82
  price_pkr_per_maund = price_usd_per_lb * pkr_rate * 82.2857

Data quality:
  Classification: REGIONAL
  West Africa aggregate estimate. Suitable for regional benchmarking
  and relative price positioning. NOT a single-country spot price.

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

COUNTRY   = "West Africa"
SOURCE_ID = "regional/WestAfrica"

# West Africa aggregate farm gate: ICE × 0.82
# Same basis as Ivory Coast — West Africa belt uses same CFA/government-price structure
_MULTIPLIER   = 0.82
_BASIS_USD_LB = 0.00


def fetch_west_africa_cotton_monthly(
    start_date: str = "2015-01-01",
    pkr_rate: float = 278.5,
    ice_series: pd.Series | None = None,
    timeout: int = 15,
) -> pd.DataFrame:
    """
    Return monthly West Africa cotton regional estimates (USD/lb and PKR/maund).

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
            raise RuntimeError(f"West Africa: ICE No.2 fetch failed: {exc}") from exc

    ice_filtered = ice_series[ice_series.index >= pd.Timestamp(start_date)]
    if ice_filtered.empty:
        raise RuntimeError(f"West Africa: no ICE data on or after {start_date}")

    logger.info("West Africa: building regional aggregate estimate (ICE x %.2f)", _MULTIPLIER)
    return ice_basis_df(
        ice_series=ice_filtered,
        country=COUNTRY,
        source_id=SOURCE_ID,
        pkr_rate=pkr_rate,
        multiplier=_MULTIPLIER,
        basis_usd_lb=_BASIS_USD_LB,
    )
