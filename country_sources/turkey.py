"""
Turkey cotton price connector — import parity estimate.

Source:   Derived from ICE Cotton No.2 (FRED PCOTTINDUSDM) with a
          Turkey-specific import parity premium.

Market context:
  Turkey imports approximately 1.2-1.5 million bales/year of raw cotton,
  making it one of the world's largest cotton importers. Turkish mills
  (primarily in Bursa, Adana, Kahramanmaras) predominantly source from:
    - USA (ICE No.2 reference)
    - Greece (Egean cotton, short-staple premium)
    - Brazil, Australia, Central Asia

  Turkey's import parity price = ICE No.2 + CIF Turkish port premium:
    - Ocean freight (US Gulf / Santos → Iskenderun/Mersin): ~$0.03-0.05/lb
    - Insurance + handling:                                  ~$0.01/lb
    - Net basis: approximately +$0.04/lb over ICE No.2

  Domestic Aegean cotton (Izmir region, Giza-type short-staple) is a
  premium product but produced in very small volumes (~80k bales/year)
  and is not representative of the Turkish cotton market overall.
  We use import parity as the relevant price for Turkish textile mills.

Unit normalization:
  price_usd_per_lb = ICE_No2_price + 0.04
  price_pkr_per_maund = price_usd_per_lb * pkr_rate * 82.2857

Data quality:
  Classification: ESTIMATED
  Derived from ICE No.2 with documented import parity basis.
  Suitable for procurement benchmarking and trend analysis.
  NOT a direct market quotation from Turkey.

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

COUNTRY   = "Turkey"
SOURCE_ID = "ICE/Turkey-import"

# Import parity: ICE No.2 + ~$0.04/lb (CIF Iskenderun/Mersin, freight + insurance)
# Range observed 2019-2024: $0.03-0.07/lb depending on freight market conditions
_MULTIPLIER  = 1.00
_BASIS_USD_LB = 0.04


def fetch_turkey_cotton_monthly(
    start_date: str = "2015-01-01",
    pkr_rate: float = 278.5,
    ice_series: pd.Series | None = None,
    timeout: int = 15,
) -> pd.DataFrame:
    """
    Return monthly Turkey cotton import parity estimates (USD/lb and PKR/maund).

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
            raise RuntimeError(f"Turkey: ICE No.2 fetch failed: {exc}") from exc

    ice_filtered = ice_series[ice_series.index >= pd.Timestamp(start_date)]
    if ice_filtered.empty:
        raise RuntimeError(f"Turkey: no ICE data on or after {start_date}")

    logger.info("Turkey: building import parity estimate (ICE + $%.2f/lb)", _BASIS_USD_LB)
    return ice_basis_df(
        ice_series=ice_filtered,
        country=COUNTRY,
        source_id=SOURCE_ID,
        pkr_rate=pkr_rate,
        multiplier=_MULTIPLIER,
        basis_usd_lb=_BASIS_USD_LB,
    )
