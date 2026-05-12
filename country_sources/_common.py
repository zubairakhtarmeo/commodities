"""
Shared utilities for country cotton price connectors.

All connectors produce a DataFrame with these columns:
    date                  str  YYYY-MM-DD (month-start)
    country               str
    price_usd_per_lb      float  USD per pound
    price_pkr_per_maund   float  PKR per maund
    source                str    connector ID (used for transparency labelling)
"""
from __future__ import annotations

from io import StringIO

import pandas as pd
import requests

# Unit constants
LB_PER_MAUND  = 82.2857   # 1 maund = 82.2857 lb (Pakistani standard)
LB_PER_ARROBA = 33.0693   # 1 arroba = 15 kg = 33.0693 lb (Brazilian standard)

_FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=PCOTTINDUSDM"


def fetch_ice_no2_series(
    start_date: str = "2015-01-01",
    timeout: int = 15,
) -> pd.Series:
    """
    Fetch ICE Cotton No.2 monthly series from FRED (cents/lb → USD/lb).
    Returns a pd.Series indexed by month-start Timestamps.
    """
    resp = requests.get(_FRED_URL, timeout=timeout)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    df.columns = ["date", "value_cents_lb"]
    df["value_cents_lb"] = pd.to_numeric(df["value_cents_lb"], errors="coerce")
    df = df.dropna(subset=["value_cents_lb"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[df["date"] >= pd.Timestamp(start_date)]
    df = df.set_index("date").sort_index()
    return (df["value_cents_lb"] / 100.0).rename("price_usd_per_lb")


def build_output_df(
    dates: pd.DatetimeIndex | pd.Series,
    prices_usd_per_lb: pd.Series,
    country: str,
    source_id: str,
    pkr_rate: float,
) -> pd.DataFrame:
    """Assemble the standard output DataFrame from aligned dates + USD/lb prices."""
    usd = prices_usd_per_lb.round(6)
    pkr = (usd * pkr_rate * LB_PER_MAUND).round(2)

    return pd.DataFrame({
        "date":                pd.to_datetime(dates).strftime("%Y-%m-%d"),
        "country":             country,
        "price_usd_per_lb":    usd.values,
        "price_pkr_per_maund": pkr.values,
        "source":              source_id,
    }).reset_index(drop=True)


def ice_basis_df(
    ice_series: pd.Series,
    country: str,
    source_id: str,
    pkr_rate: float,
    basis_usd_lb: float = 0.0,
    multiplier: float = 1.0,
) -> pd.DataFrame:
    """
    Derive a country price series from ICE No.2 via:
        price = ice_price * multiplier + basis_usd_lb

    multiplier:    quality/grade factor (e.g. 0.90 = 10% discount)
    basis_usd_lb:  additive spread (e.g. +0.04 for freight/insurance)
    """
    adj = (ice_series * multiplier + basis_usd_lb).clip(lower=0.01)
    return build_output_df(
        dates=ice_series.index,
        prices_usd_per_lb=adj,
        country=country,
        source_id=source_id,
        pkr_rate=pkr_rate,
    )
