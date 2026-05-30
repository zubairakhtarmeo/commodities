"""
market_inputs.py
----------------
Market Inputs ingestion layer for procurement intelligence.

Provides four standalone connectors + a normalised dataframe builder:

    fetch_cotton_price()  → ICE Cotton No. 2 front-month futures  (USD/lb)
    fetch_fx_rate()       → USD/PKR spot rate
    fetch_sbp_rate()      → SBP Monetary Policy Rate              (%)
    fetch_psf_price()     → Polyester Staple Fiber spot price     (RMB/ton, SunSirs)

Resilience contract:
    - Each connector tries a primary source then ≥1 fallback source.
    - All network/parse exceptions are caught internally.
    - On any failure: status="FAILED", metric_value=None, error=<reason>.
    - build_market_inputs_dataframe() always returns a DataFrame (never raises).

Dependencies: requests, beautifulsoup4, pandas  (all in requirements.txt)
No extra packages required.

CLI:
    python market_inputs.py
    python market_inputs.py --output "D:/reports/market_inputs.xlsx"
    python market_inputs.py --timeout 20 --psf-id 839
"""

from __future__ import annotations

import argparse
import io
import logging
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_BROWSER_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control":   "no-cache",
}

_JSON_HEADERS: dict[str, str] = {
    **_BROWSER_HEADERS,
    "Accept": "application/json, text/plain, */*",
}


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

@dataclass
class MarketRecord:
    """Normalised container for one market metric.

    Attributes:
        metric_name:          Human-readable label (e.g. "ICE Cotton No. 2")
        metric_value:         Numeric value; None when status == "FAILED"
        unit:                 Unit string (e.g. "USD/lb", "PKR/USD", "%", "RMB/ton")
        source:               Data source label (e.g. "Yahoo Finance (CT=F)")
        retrieval_timestamp:  ISO-8601 UTC timestamp of the fetch attempt
        status:               "OK" | "FAILED" | "STALE"
        error:                Error description when status != "OK"; None otherwise
    """
    metric_name:         str
    metric_value:        Optional[float]
    unit:                str
    source:              str
    retrieval_timestamp: str
    status:              str
    error:               Optional[str]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _failed(metric_name: str, unit: str, source: str, error: str) -> MarketRecord:
    """Convenience constructor for a FAILED record."""
    logger.warning("FAILED %-30s | %s | %s", metric_name, source, error)
    return MarketRecord(
        metric_name=metric_name,
        metric_value=None,
        unit=unit,
        source=source,
        retrieval_timestamp=_now(),
        status="FAILED",
        error=error,
    )


# ---------------------------------------------------------------------------
# 1. Cotton price — ICE Cotton No. 2 (USD/lb)
# ---------------------------------------------------------------------------
# Primary:  Yahoo Finance REST API  (CT=F front-month futures, cents/lb → USD/lb)
# Fallback: FRED PCOTTINDUSDM      (monthly, USD/lb — most recent observation)

_COTTON_METRIC = "ICE Cotton No. 2"
_COTTON_UNIT   = "USD/lb"


def fetch_cotton_price(timeout: int = 15) -> MarketRecord:
    """Fetch ICE Cotton No. 2 futures price (USD/lb).

    Primary:  Yahoo Finance real-time quote  (CT=F)
    Fallback: FRED  PCOTTINDUSDM  (monthly average, last observation)

    Returns:
        MarketRecord with metric_value in USD/lb.
    """
    # ── Primary: Yahoo Finance ────────────────────────────────────────────
    try:
        url = (
            "https://query1.finance.yahoo.com/v8/finance/chart/CT%3DF"
            "?interval=1d&range=5d"
        )
        resp = requests.get(url, headers=_JSON_HEADERS, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()

        result = data["chart"]["result"][0]
        closes = result["indicators"]["quote"][0]["close"]
        closes = [c for c in closes if c is not None]
        if not closes:
            raise ValueError("No close prices in Yahoo Finance response")

        # ICE Cotton futures are quoted in cents/lb on Yahoo Finance
        price_cents_per_lb = float(closes[-1])
        price_usd_per_lb   = round(price_cents_per_lb / 100.0, 4)

        logger.info("Cotton [Yahoo Finance]: %.4f USD/lb (raw %.2f ¢/lb)",
                    price_usd_per_lb, price_cents_per_lb)

        return MarketRecord(
            metric_name=_COTTON_METRIC,
            metric_value=price_usd_per_lb,
            unit=_COTTON_UNIT,
            source="Yahoo Finance (CT=F)",
            retrieval_timestamp=_now(),
            status="OK",
            error=None,
        )
    except Exception as exc:
        logger.warning("Cotton Yahoo Finance failed: %s", exc)

    # ── Fallback: FRED PCOTTINDUSDM ───────────────────────────────────────
    try:
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=PCOTTINDUSDM"
        resp = requests.get(url, headers=_BROWSER_HEADERS, timeout=timeout)
        resp.raise_for_status()

        df = pd.read_csv(io.StringIO(resp.text))
        df.columns = ["date", "value"]
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])
        if df.empty:
            raise ValueError("FRED returned no non-null observations")

        price = round(float(df["value"].iloc[-1]), 4)
        last_date = str(df["date"].iloc[-1])

        logger.info("Cotton [FRED PCOTTINDUSDM]: %.4f USD/lb (as of %s)", price, last_date)

        return MarketRecord(
            metric_name=_COTTON_METRIC,
            metric_value=price,
            unit=_COTTON_UNIT,
            source=f"FRED PCOTTINDUSDM (monthly, as of {last_date})",
            retrieval_timestamp=_now(),
            status="OK",
            error=None,
        )
    except Exception as exc:
        logger.warning("Cotton FRED fallback failed: %s", exc)

    return _failed(_COTTON_METRIC, _COTTON_UNIT, "Yahoo Finance / FRED",
                   "All cotton sources failed — check network or source availability")


# ---------------------------------------------------------------------------
# 2. USD/PKR exchange rate
# ---------------------------------------------------------------------------
# Primary:  ExchangeRate-API (open.er-api.com) — already used in codebase
# Fallback: Frankfurter (ECB-backed, free, no key)

_FX_METRIC = "USD/PKR Exchange Rate"
_FX_UNIT   = "PKR"


def fetch_fx_rate(timeout: int = 15) -> MarketRecord:
    """Fetch USD/PKR spot exchange rate.

    Primary:  https://open.er-api.com/v6/latest/USD  (established in codebase)
    Fallback: https://api.frankfurter.app/latest?from=USD&to=PKR

    Returns:
        MarketRecord with metric_value = PKR per 1 USD.
    """
    # ── Primary: open.er-api.com ──────────────────────────────────────────
    try:
        resp = requests.get(
            "https://open.er-api.com/v6/latest/USD",
            headers=_JSON_HEADERS,
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()

        if payload.get("result") == "error":
            raise ValueError(f"API error: {payload.get('error-type', 'unknown')}")

        pkr = float(payload["rates"]["PKR"])
        if pkr <= 0:
            raise ValueError(f"Non-positive PKR rate: {pkr}")

        logger.info("USD/PKR [open.er-api.com]: %.2f", pkr)

        return MarketRecord(
            metric_name=_FX_METRIC,
            metric_value=round(pkr, 2),
            unit=_FX_UNIT,
            source="open.er-api.com",
            retrieval_timestamp=_now(),
            status="OK",
            error=None,
        )
    except Exception as exc:
        logger.warning("FX open.er-api.com failed: %s", exc)

    # ── Fallback: Frankfurter API ─────────────────────────────────────────
    try:
        resp = requests.get(
            "https://api.frankfurter.app/latest?from=USD&to=PKR",
            headers=_JSON_HEADERS,
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        pkr = float(payload["rates"]["PKR"])
        if pkr <= 0:
            raise ValueError(f"Non-positive PKR rate: {pkr}")

        logger.info("USD/PKR [frankfurter.app]: %.2f", pkr)

        return MarketRecord(
            metric_name=_FX_METRIC,
            metric_value=round(pkr, 2),
            unit=_FX_UNIT,
            source="frankfurter.app (ECB)",
            retrieval_timestamp=_now(),
            status="OK",
            error=None,
        )
    except Exception as exc:
        logger.warning("FX frankfurter.app fallback failed: %s", exc)

    return _failed(_FX_METRIC, _FX_UNIT, "open.er-api.com / frankfurter.app",
                   "All FX sources failed — check network connectivity")


# ---------------------------------------------------------------------------
# 3. SBP Monetary Policy Rate (%)
# ---------------------------------------------------------------------------
# Primary:  Scrape sbp.org.pk/monetary/mpd.asp (MPC decisions page)
# Fallback: sbp.org.pk main monetary page
#
# LIMITATION: SBP has no public JSON API. HTML scraping may break if the
# page layout changes. The scraped rate should be validated by the business
# against SBP press releases before use in production.

_SBP_METRIC = "SBP Policy Rate"
_SBP_UNIT   = "%"

# Regex: matches "11.00" or "11" in the context of a policy rate announcement.
# Accepts patterns like "11.00 percent", "11.00%", "rate: 11.00"
_SBP_RATE_RE = re.compile(
    r"""
    (?:                             # preceding context (non-capturing)
        policy\s*rate[^0-9]*        # "policy rate ..." up to the number
        |rate\s*(?:of\s*)?          # "rate of ..."
        |maintained\s*at\s*         # "maintained at X"
        |unchanged\s*at\s*          # "unchanged at X"
        |revised\s*to\s*            # "revised to X"
        |increased\s*to\s*          # "increased to X"
        |decreased\s*to\s*          # "decreased to X"
        |reduced\s*to\s*            # "reduced to X"
        |cut\s*to\s*                # "cut to X"
    )
    (\d{1,2}(?:\.\d{1,2})?)         # capture: the rate number (e.g. 11 or 11.50)
    \s*(?:percent|%|bps)?           # optional suffix
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Plausible SBP policy rate bounds (sanity check)
_SBP_MIN, _SBP_MAX = 1.0, 30.0


def _parse_sbp_rate(html: str) -> Optional[float]:
    """Extract the policy rate from SBP HTML content.

    Strategy:
        1. Try regex with policy-rate context keywords on the full page text.
        2. If multiple matches, return the most recently mentioned (last match),
           which typically corresponds to the current rate on MPC pages.
    """
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        page_text = soup.get_text(" ", strip=True)
    except Exception:
        page_text = html  # fall back to raw HTML if BS4 fails

    matches = _SBP_RATE_RE.findall(page_text)
    if not matches:
        return None

    # Filter to plausible range and return the last match
    valid = []
    for m in matches:
        try:
            rate = float(m)
            if _SBP_MIN <= rate <= _SBP_MAX:
                valid.append(rate)
        except ValueError:
            continue

    return valid[-1] if valid else None


def fetch_sbp_rate(timeout: int = 20) -> MarketRecord:
    """Fetch SBP Monetary Policy Rate (%) by scraping sbp.org.pk.

    Primary:  https://www.sbp.org.pk/monetary/mpd.asp  (MPC decisions page)
    Fallback: https://www.sbp.org.pk/monetary/          (monetary policy home)

    Returns:
        MarketRecord with metric_value = policy rate as a percentage (e.g. 11.0).

    Note:
        SBP does not provide a public JSON API. This scraper targets text patterns
        on the MPC announcements page. If the site layout changes, status will be
        FAILED until the parser is updated.
    """
    urls_to_try = [
        ("SBP MPC decisions",    "https://www.sbp.org.pk/monetary/mpd.asp"),
        ("SBP monetary home",    "https://www.sbp.org.pk/monetary/"),
    ]

    for source_label, url in urls_to_try:
        try:
            resp = requests.get(url, headers=_BROWSER_HEADERS, timeout=timeout, verify=True)
            resp.raise_for_status()
            rate = _parse_sbp_rate(resp.text)

            if rate is not None:
                logger.info("SBP Policy Rate [%s]: %.2f%%", source_label, rate)
                return MarketRecord(
                    metric_name=_SBP_METRIC,
                    metric_value=rate,
                    unit=_SBP_UNIT,
                    source=f"SBP ({source_label})",
                    retrieval_timestamp=_now(),
                    status="OK",
                    error=None,
                )
            else:
                logger.warning("SBP %s: page fetched but rate pattern not found", source_label)

        except Exception as exc:
            logger.warning("SBP %s failed: %s", source_label, exc)

    return _failed(
        _SBP_METRIC, _SBP_UNIT, "sbp.org.pk",
        "Could not parse policy rate from sbp.org.pk — page layout may have changed "
        "or site is unreachable. Manually confirm the current rate at "
        "https://www.sbp.org.pk/monetary/mpd.asp"
    )


# ---------------------------------------------------------------------------
# 4. PSF price — Polyester Staple Fiber (RMB/ton, SunSirs)
# ---------------------------------------------------------------------------
# Primary:  SunSirs HTML scrape  (same pattern as existing VSF connector)
#
# PSF_COMMODITY_ID configuration note:
#   SunSirs assigns numeric IDs to commodities. The VSF connector uses 1057.
#   For China PSF (Polyester Staple Fiber), the prodetail page is typically
#   at /uk/prodetail-{id}.html.  Default below is 839 — verify at:
#   https://www.sunsirs.com/uk/  by searching "Polyester Staple Fiber".
#   Pass psf_commodity_id=<correct_id> if the default does not return prices.

_PSF_METRIC          = "PSF Price (China)"
_PSF_UNIT            = "RMB/ton"
_PSF_DEFAULT_ID      = 839
_PSF_PRICE_BOUNDS    = (3_000.0, 25_000.0)   # RMB/ton sanity range

# Patterns shared with the VSF connector
_DATE_RE  = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_NUM_RE   = re.compile(r"[-+]?\d+(?:[.,]\d+)?")


def _fetch_html(url: str, timeout: int) -> str:
    """Fetch HTML with browser headers and raise on failure."""
    last_exc: Optional[Exception] = None
    for attempt in range(1, 4):
        try:
            resp = requests.get(url, headers=_BROWSER_HEADERS, timeout=timeout, verify=True)
            resp.raise_for_status()
            return resp.text
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            logger.debug("Attempt %d/3 failed for %s: %s", attempt, url, exc)
    raise ConnectionError(f"Failed to fetch {url} after 3 attempts: {last_exc}")


def _parse_sunsirs_prodetail(html: str, price_bounds: tuple[float, float]) -> Optional[float]:
    """Extract the most recent price from a SunSirs prodetail HTML page.

    Follows the same table-scanning approach as the existing VSF connector.
    Returns the price from the most recent date row, or None if not found.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise ImportError(
            "beautifulsoup4 required for SunSirs parsing: pip install beautifulsoup4"
        ) from exc

    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")
    if not tables:
        return None

    records: list[tuple[str, float]] = []  # (date_str, price)

    for table in tables:
        for row in table.find_all("tr"):
            cells = row.find_all(["td", "th"])
            if not cells:
                continue
            texts = [c.get_text(" ", strip=True) for c in cells]

            # Find a date cell
            date_str: Optional[str] = None
            for t in texts:
                m = _DATE_RE.search(t)
                if m:
                    date_str = m.group(0)
                    break
            if not date_str:
                continue

            # Find a plausible price in the remaining cells (skip date cell)
            for t in texts:
                if _DATE_RE.search(t):
                    continue
                normalized = t.replace(",", "")
                m = _NUM_RE.search(normalized)
                if not m:
                    continue
                try:
                    val = float(m.group(0).replace(",", ""))
                except ValueError:
                    continue
                lo, hi = price_bounds
                if lo <= val <= hi:
                    records.append((date_str, val))
                    break  # first valid price in this row is enough

    if not records:
        return None

    # Sort by date and return the most recent price
    records.sort(key=lambda x: x[0])
    return records[-1][1]


def fetch_psf_price(
    psf_commodity_id: int = _PSF_DEFAULT_ID,
    timeout: int = 20,
) -> MarketRecord:
    """Fetch China PSF (Polyester Staple Fiber) spot price from SunSirs (RMB/ton).

    Args:
        psf_commodity_id: SunSirs commodity ID for PSF.
                          Default=839 — verify at https://www.sunsirs.com/uk/
                          by searching "Polyester Staple Fiber" if results are empty.
        timeout:          HTTP timeout in seconds.

    Returns:
        MarketRecord with metric_value in RMB/ton.
    """
    url = f"https://www.sunsirs.com/uk/prodetail-{psf_commodity_id}.html"

    try:
        html  = _fetch_html(url, timeout=timeout)
        price = _parse_sunsirs_prodetail(html, _PSF_PRICE_BOUNDS)

        if price is None:
            return _failed(
                _PSF_METRIC, _PSF_UNIT,
                f"SunSirs (commodity_id={psf_commodity_id})",
                f"Page fetched but no PSF price rows parsed from {url}. "
                "The commodity_id may be incorrect — confirm at sunsirs.com/uk/",
            )

        logger.info("PSF [SunSirs commodity_id=%d]: %.0f RMB/ton", psf_commodity_id, price)

        return MarketRecord(
            metric_name=_PSF_METRIC,
            metric_value=round(price, 2),
            unit=_PSF_UNIT,
            source=f"SunSirs (commodity_id={psf_commodity_id})",
            retrieval_timestamp=_now(),
            status="OK",
            error=None,
        )

    except Exception as exc:
        return _failed(
            _PSF_METRIC, _PSF_UNIT,
            f"SunSirs (commodity_id={psf_commodity_id})",
            f"Fetch failed: {exc}",
        )


# ---------------------------------------------------------------------------
# 5. build_market_inputs_dataframe()
# ---------------------------------------------------------------------------

def build_market_inputs_dataframe(
    timeout: int = 15,
    psf_commodity_id: int = _PSF_DEFAULT_ID,
) -> pd.DataFrame:
    """Fetch all market inputs and return a normalised DataFrame.

    Calls all four connectors. Any individual failure produces a FAILED row
    rather than raising an exception — the dataframe is always returned.

    Args:
        timeout:          HTTP timeout (seconds) passed to each connector.
        psf_commodity_id: SunSirs commodity ID for PSF (see fetch_psf_price()).

    Returns:
        DataFrame with columns:
            metric_name, metric_value, unit, source,
            retrieval_timestamp, status, error

    Example output:
        metric_name            metric_value  unit     source                    status
        ICE Cotton No. 2       0.7820        USD/lb   Yahoo Finance (CT=F)      OK
        USD/PKR Exchange Rate  279.50        PKR      open.er-api.com           OK
        SBP Policy Rate        11.00         %        SBP (MPC decisions)       OK
        PSF Price (China)      8450.00       RMB/ton  SunSirs (commodity_id=839) OK
    """
    logger.info("=" * 60)
    logger.info("Fetching market inputs (%s)", _now())
    logger.info("=" * 60)

    records = [
        fetch_cotton_price(timeout=timeout),
        fetch_fx_rate(timeout=timeout),
        fetch_sbp_rate(timeout=timeout),
        fetch_psf_price(psf_commodity_id=psf_commodity_id, timeout=timeout),
    ]

    df = pd.DataFrame([asdict(r) for r in records])

    ok_count   = (df["status"] == "OK").sum()
    fail_count = (df["status"] == "FAILED").sum()
    logger.info("Completed: %d OK, %d FAILED", ok_count, fail_count)

    if fail_count > 0:
        failed_metrics = df.loc[df["status"] != "OK", "metric_name"].tolist()
        logger.warning("Failed metrics: %s", failed_metrics)

    return df


# ---------------------------------------------------------------------------
# Save output to Excel
# ---------------------------------------------------------------------------

def run(
    output_path: Optional[str | Path] = None,
    timeout: int = 15,
    psf_commodity_id: int = _PSF_DEFAULT_ID,
) -> pd.DataFrame:
    """Fetch all market inputs and optionally save to Excel.

    Args:
        output_path:      Optional path to write .xlsx output.
        timeout:          HTTP timeout in seconds.
        psf_commodity_id: SunSirs PSF commodity ID.

    Returns:
        Normalised market inputs DataFrame.
    """
    df = build_market_inputs_dataframe(timeout=timeout, psf_commodity_id=psf_commodity_id)

    if output_path:
        output_path = Path(output_path)
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Market Inputs", index=False)
        logger.info("Saved → %s", output_path)

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch market inputs for procurement intelligence")
    parser.add_argument("--output",   default=None, help="Optional output .xlsx path")
    parser.add_argument("--timeout",  default=15,   type=int, help="HTTP timeout in seconds (default: 15)")
    parser.add_argument("--psf-id",   default=_PSF_DEFAULT_ID, type=int,
                        help=f"SunSirs PSF commodity ID (default: {_PSF_DEFAULT_ID})",
                        dest="psf_id")
    args = parser.parse_args()

    result_df = run(
        output_path=args.output,
        timeout=args.timeout,
        psf_commodity_id=args.psf_id,
    )

    print("\n" + "=" * 72)
    print("MARKET INPUTS SUMMARY")
    print("=" * 72)
    display_cols = ["metric_name", "metric_value", "unit", "source", "status"]
    print(result_df[display_cols].to_string(index=False))
    print("=" * 72)

    failed = result_df[result_df["status"] != "OK"]
    if not failed.empty:
        print("\nFAILED METRICS — error details:")
        for _, row in failed.iterrows():
            print(f"  [{row['metric_name']}] {row['error']}")

    sys.exit(0 if result_df["status"].eq("OK").all() else 1)
