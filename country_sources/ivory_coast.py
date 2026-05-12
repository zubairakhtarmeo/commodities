"""
Ivory Coast (Cote d'Ivoire) cotton price connector — West Africa regional estimate.

Source:   Derived from ICE Cotton No.2 with a documented West Africa
          CFA-zone farm-gate discount factor.

Market context:
  Ivory Coast is the leading West African cotton producer (~450k bales/year,
  Northern savanna zone). West Africa's cotton sector operates under a
  government-controlled fixed-price system managed by ginning companies
  (CIDT/InterCoton in CIV, SOFITEX in Burkina Faso, CMDT in Mali).

  West African CFA-zone pricing structure:
    - Government-set farm-gate prices are published each crop season
    - Typically set at ~75-85% of the previous year's world price (CFA-linked)
    - With FCFA pegged to EUR, USD fluctuations add an FX layer
    - Net: farm-gate approximately ICE × 0.82 in USD terms

  Reference: ICAC Secretariat, World Bank Cotton Briefs,
             USDA GAIN Cote d'Ivoire (IV-2101, IV-2201).
  Historical range: 0.76 - 0.88 × ICE No.2.

Unit normalization:
  price_usd_per_lb = ICE_No2_price * 0.82
  price_pkr_per_maund = price_usd_per_lb * pkr_rate * 82.2857

Data quality:
  Classification: REGIONAL
  CFA-zone government-set price estimate. Suitable for regional
  competitiveness benchmarking. NOT an exchange-traded spot price.

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

COUNTRY   = "Ivory Coast"
SOURCE_ID = "regional/WestAfrica-CIV"

# West Africa CFA-zone farm gate: ICE × 0.82
# Based on: ICAC/USDA GAIN data 2015-2024, INTERCOTON fixed-price reports
_MULTIPLIER   = 0.82
_BASIS_USD_LB = 0.00


def fetch_ivory_coast_cotton_monthly(
    start_date: str = "2015-01-01",
    pkr_rate: float = 278.5,
    ice_series: pd.Series | None = None,
    timeout: int = 15,
) -> pd.DataFrame:
    """
    Return monthly Ivory Coast cotton regional estimates (USD/lb and PKR/maund).

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
            raise RuntimeError(f"Ivory Coast: ICE No.2 fetch failed: {exc}") from exc

    ice_filtered = ice_series[ice_series.index >= pd.Timestamp(start_date)]
    if ice_filtered.empty:
        raise RuntimeError(f"Ivory Coast: no ICE data on or after {start_date}")

    logger.info("Ivory Coast: building West Africa CFA regional estimate (ICE x %.2f)", _MULTIPLIER)
    return ice_basis_df(
        ice_series=ice_filtered,
        country=COUNTRY,
        source_id=SOURCE_ID,
        pkr_rate=pkr_rate,
        multiplier=_MULTIPLIER,
        basis_usd_lb=_BASIS_USD_LB,
    )
