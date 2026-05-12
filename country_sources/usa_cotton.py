"""
USA cotton price connector.

Source:   FRED series PCOTTINDUSDM
          "Cotton, No. 2, New York Board of Trade" (ICE Cotton No.2)
          Published by: IMF Primary Commodity Prices via St. Louis Fed
URL:      https://fred.stlouisfed.org/series/PCOTTINDUSDM

Market context:
  ICE Cotton No.2 (Memphis, TN delivery, Strict Middling 1-1/16")
  is the global benchmark used by US cotton merchants, textile mills,
  and international buyers as the primary US-origin reference price.
  Pakistani textile mills that import US cotton use this as their
  cost basis before adding freight, insurance, and handling.

Unit normalization:
  FRED reports in US cents per pound (¢/lb).
  Output is USD per pound (USD/lb) = FRED value / 100.

Update cadence:
  Monthly — FRED publishes with ~4-6 week lag after month-end.
  As of May 2026, latest available data point: March 2026.

Legal:
  IMF/FRED primary commodity price data is public domain.
  No authentication required. No rate limiting for reasonable use.

Data quality:
  Classification: REAL (live, auto-updating government-backed source)
  NOT synthetic, NOT manually estimated.
"""

from __future__ import annotations

from io import StringIO

import pandas as pd
import requests

SOURCE_ID = "FRED/ICE-No2"
COUNTRY   = "USA"

_FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=PCOTTINDUSDM"
LB_PER_MAUND = 82.2857


def fetch_usa_cotton_monthly(
    start_date: str = "2015-01-01",
    pkr_rate: float = 278.5,
    timeout: int = 15,
) -> pd.DataFrame:
    """
    Fetch USA (ICE Cotton No.2) monthly prices from FRED.

    Args:
        start_date: ISO date string; rows before this date are dropped.
        pkr_rate:   USD → PKR exchange rate for deriving price_pkr_per_maund.
        timeout:    HTTP request timeout in seconds.

    Returns:
        DataFrame with columns:
            date                  YYYY-MM-DD str
            country               "USA"
            price_usd_per_lb      float  (USD/lb)
            price_pkr_per_maund   float  (PKR/maund)
            source                "FRED/ICE-No2"

    Raises:
        RuntimeError: if the FRED request fails or parsing finds no valid rows.
    """
    try:
        resp = requests.get(_FRED_URL, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"FRED request failed: {exc}") from exc

    df = pd.read_csv(StringIO(resp.text))
    if df.shape[1] < 2:
        raise RuntimeError(f"Unexpected FRED CSV shape: {df.shape}")

    df.columns = ["date", "value_cents_lb"]
    df["value_cents_lb"] = pd.to_numeric(df["value_cents_lb"], errors="coerce")
    df = df.dropna(subset=["value_cents_lb"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[df["date"] >= pd.Timestamp(start_date)]

    if df.empty:
        raise RuntimeError(f"No FRED cotton rows on or after {start_date}")

    # Normalize units
    df["price_usd_per_lb"] = (df["value_cents_lb"] / 100.0).round(6)
    df["price_pkr_per_maund"] = (
        df["price_usd_per_lb"] * pkr_rate * LB_PER_MAUND
    ).round(2)

    result = pd.DataFrame({
        "date":                df["date"].dt.strftime("%Y-%m-%d"),
        "country":             COUNTRY,
        "price_usd_per_lb":    df["price_usd_per_lb"],
        "price_pkr_per_maund": df["price_pkr_per_maund"],
        "source":              SOURCE_ID,
    }).reset_index(drop=True)

    return result
