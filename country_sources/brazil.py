"""
Brazil cotton price connector.

Primary source: CEPEA/Esalq (Centro de Estudos Avancados em Economia Aplicada)
  URL:    https://www.cepea.esalq.usp.br/en/indicator/cotton.aspx
  Series: Algodao 8 (standard Brazilian Cerrado cotton, 8 micronaire)
  Units:  BRL per arroba (15 kg)
  Published by: University of Sao Paulo / Esalq, Brazil's premier
                agricultural economics research center (~30+ years of data)

Market context:
  Brazil is the world's 2nd-largest cotton exporter (Mato Grosso, Bahia).
  CEPEA Esalq cotton (Algodao 8) prices closely track ICE No.2 because
  Brazilian cotton is export-grade and priced at CIF export parity.
  Typical basis to ICE No.2: -$0.02 to +$0.02/lb (near parity).

Unit normalization:
  CEPEA reports BRL/arroba (1 arroba = 15 kg = 33.069 lb).
  price_usd_per_lb = price_brl_per_arroba / brl_usd_rate / LB_PER_ARROBA

Fallback:
  If CEPEA fetch fails, derives from ICE No.2 at parity (basis = 0.0).
  Fallback source ID: "ICE/Brazil-basis"

Update cadence:
  CEPEA: daily (we resample to monthly).
  Fallback: monthly (follows FRED PCOTTINDUSDM cadence).

Data quality:
  LIVE (CEPEA):    official institutional source, 30+ year track record.
  ESTIMATED (fallback): derived from ICE No.2, classified transparently.
"""
from __future__ import annotations

import logging
from io import BytesIO

import pandas as pd
import requests

from country_sources._common import (
    LB_PER_ARROBA,
    LB_PER_MAUND,
    build_output_df,
    fetch_ice_no2_series,
    ice_basis_df,
)

logger = logging.getLogger(__name__)

COUNTRY         = "Brazil"
SOURCE_ID_LIVE  = "CEPEA/ESALQ"
SOURCE_ID_FALLBACK = "ICE/Brazil-basis"

# CEPEA Esalq cotton Excel download (BRL/arroba, Algodao 8)
_CEPEA_URL = (
    "https://www.cepea.esalq.usp.br/upload/kceditor/files/Algodao_en_R$.xlsx"
)


def _fetch_cepea_series(
    start_date: str,
    brl_rate: float,
    timeout: int,
) -> pd.Series | None:
    """
    Try to download and parse CEPEA Algodao 8 Excel.
    Returns monthly pd.Series (USD/lb) indexed by month-start Timestamps,
    or None if the download or parsing fails.
    """
    try:
        resp = requests.get(_CEPEA_URL, timeout=timeout)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("CEPEA download failed: %s", exc)
        return None

    try:
        # CEPEA Excel: first usable row varies; scan for date-like first column
        raw = pd.read_excel(BytesIO(resp.content), header=None, engine="openpyxl")
    except Exception as exc:
        logger.warning("CEPEA Excel parse failed: %s", exc)
        return None

    # Find first row where column 0 looks like a date and column 1 is numeric
    data_rows = []
    for _, row in raw.iterrows():
        try:
            dt = pd.to_datetime(row.iloc[0], dayfirst=True, errors="coerce")
            val = pd.to_numeric(row.iloc[1], errors="coerce")
            if pd.isna(dt) or pd.isna(val) or val <= 0:
                continue
            data_rows.append((dt, float(val)))
        except Exception:
            continue

    if not data_rows:
        logger.warning("CEPEA Excel: no parseable rows found")
        return None

    df = pd.DataFrame(data_rows, columns=["date", "brl_per_arroba"])
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= pd.Timestamp(start_date)].copy()
    df = df.set_index("date").sort_index()

    # Convert BRL/arroba → USD/lb
    df["price_usd_per_lb"] = (df["brl_per_arroba"] / brl_rate / LB_PER_ARROBA)
    series = df["price_usd_per_lb"].resample("MS").mean().dropna()

    if series.empty:
        logger.warning("CEPEA: no rows on or after %s", start_date)
        return None

    logger.info("CEPEA: fetched %d monthly rows (%s → %s)",
                len(series), series.index[0].date(), series.index[-1].date())
    return series


def fetch_brazil_cotton_monthly(
    start_date: str = "2015-01-01",
    pkr_rate: float = 278.5,
    brl_rate: float = 5.0,
    ice_series: pd.Series | None = None,
    timeout: int = 20,
) -> pd.DataFrame:
    """
    Fetch Brazil cotton monthly prices.

    Tries CEPEA Esalq first; falls back to ICE No.2 at parity if CEPEA fails.

    Args:
        start_date: ISO date string; rows before this date are dropped.
        pkr_rate:   USD -> PKR rate for price_pkr_per_maund column.
        brl_rate:   BRL per 1 USD (used to convert CEPEA BRL prices).
        ice_series: Pre-fetched ICE No.2 monthly Series (avoids extra FRED call).
        timeout:    HTTP timeout for CEPEA request.

    Returns:
        DataFrame: date, country, price_usd_per_lb, price_pkr_per_maund, source
    """
    cepea = _fetch_cepea_series(start_date, brl_rate, timeout)

    if cepea is not None and not cepea.empty:
        return build_output_df(
            dates=cepea.index,
            prices_usd_per_lb=cepea,
            country=COUNTRY,
            source_id=SOURCE_ID_LIVE,
            pkr_rate=pkr_rate,
        )

    # Fallback: ICE No.2 at near-parity (Brazil export-grade tracks ICE closely)
    logger.info("Brazil: CEPEA unavailable, using ICE No.2 basis fallback")
    if ice_series is None:
        try:
            ice_series = fetch_ice_no2_series(start_date=start_date, timeout=15)
        except Exception as exc:
            raise RuntimeError(f"Brazil fallback: ICE No.2 fetch failed: {exc}") from exc

    ice_filtered = ice_series[ice_series.index >= pd.Timestamp(start_date)]
    if ice_filtered.empty:
        raise RuntimeError(f"Brazil: no ICE data on or after {start_date}")

    return ice_basis_df(
        ice_series=ice_filtered,
        country=COUNTRY,
        source_id=SOURCE_ID_FALLBACK,
        pkr_rate=pkr_rate,
        multiplier=1.00,   # Brazil cotton ≈ ICE No.2 parity (export-grade Cerrado)
        basis_usd_lb=0.00,
    )
