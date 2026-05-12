"""
Ingest real and estimated country-wise cotton prices into Supabase
cotton_country_prices table.

Source classification:
  LIVE      — FRED/ICE-No2 (USA), CEPEA/ESALQ (Brazil if accessible)
  ESTIMATED — ICE/Brazil-basis, ICE/Pakistan-domestic, ICE/Turkey-import
  REGIONAL  — regional/EastAfrica, regional/WestAfrica-CIV,
               regional/EastAfrica-SDN, regional/WestAfrica

Tier A (priority — updated on every run):
  USA          FRED PCOTTINDUSDM (ICE No.2)           LIVE
  Brazil       CEPEA/Esalq → ICE fallback              LIVE / ESTIMATED
  Pakistan     ICE x 0.88 domestic quality discount    ESTIMATED
  Turkey       ICE + $0.04/lb import parity            ESTIMATED

Tier B (secondary — updated on every run):
  Tanzania     ICE x 0.80 East Africa farm-gate        REGIONAL
  Ivory Coast  ICE x 0.82 West Africa CFA-zone         REGIONAL
  Sudan        ICE x 0.78 East Africa (high uncertainty) REGIONAL
  West Africa  ICE x 0.82 West Africa aggregate        REGIONAL

Usage:
  python scripts/ingest_country_sources.py
  python scripts/ingest_country_sources.py --dry-run
  python scripts/ingest_country_sources.py --countries USA Brazil Pakistan
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import requests

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def _get_credentials() -> tuple[str | None, str | None]:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        secrets_path = BASE_DIR / ".streamlit" / "secrets.toml"
        if secrets_path.exists():
            try:
                try:
                    import tomllib
                    with open(secrets_path, "rb") as f:
                        s = tomllib.load(f)
                except ImportError:
                    import tomli
                    with open(secrets_path, "rb") as f:
                        s = tomli.load(f)
                url = url or s.get("SUPABASE_URL")
                key = key or s.get("SUPABASE_SERVICE_ROLE_KEY")
            except Exception:
                pass
    return (str(url).rstrip("/") if url else None,
            str(key).strip() if key else None)


def _get_fx_rates() -> dict[str, float]:
    """Fetch USD exchange rates for PKR and BRL. Returns fallbacks on failure."""
    fallbacks = {"PKR": 278.5, "BRL": 5.10}
    try:
        r = requests.get("https://open.er-api.com/v6/latest/USD", timeout=8)
        rates = r.json().get("rates", {})
        return {
            "PKR": float(rates.get("PKR", fallbacks["PKR"])),
            "BRL": float(rates.get("BRL", fallbacks["BRL"])),
        }
    except Exception:
        logger.warning("FX fetch failed; using fallbacks: %s", fallbacks)
        return fallbacks


def _upsert_to_supabase(
    url: str,
    key: str,
    records: list[dict],
    dry_run: bool = False,
) -> int:
    """Upsert records into cotton_country_prices. Returns count sent."""
    if not records:
        return 0

    if dry_run:
        logger.info("  [dry-run] Would upsert %d records", len(records))
        for r in records[:3]:
            logger.info("    %s", r)
        return 0

    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates",
    }

    batch_size = 200
    total_sent = 0
    for i in range(0, len(records), batch_size):
        batch = records[i: i + batch_size]
        r = requests.post(
            f"{url}/rest/v1/cotton_country_prices",
            headers=headers,
            params={"on_conflict": "date,country,source"},
            json=batch,
            timeout=30,
        )
        if r.status_code in (200, 201):
            total_sent += len(batch)
        else:
            logger.error(
                "  Upsert batch %d failed (%d): %s",
                i // batch_size + 1, r.status_code, r.text[:200],
            )

    return total_sent


def _delete_sample_rows(url: str, key: str, country: str) -> None:
    """Remove legacy source='sample' rows for a country now covered by a real/estimated source."""
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    r = requests.delete(
        f"{url}/rest/v1/cotton_country_prices",
        headers=headers,
        params={"country": f"eq.{country}", "source": "eq.sample"},
        timeout=15,
    )
    if not r.ok:
        logger.warning("  Sample row cleanup for %s failed (%d): %s",
                       country, r.status_code, r.text[:100])


# ── Per-country ingest functions ───────────────────────────────────────────────

def ingest_usa(pkr_rate: float, url: str, key: str, dry_run: bool) -> str:
    logger.info("Ingesting USA (FRED PCOTTINDUSDM)...")
    from country_sources.usa_cotton import fetch_usa_cotton_monthly
    try:
        df = fetch_usa_cotton_monthly(start_date="2015-01-01", pkr_rate=pkr_rate)
    except RuntimeError as exc:
        return f"FAIL: {exc}"

    logger.info("  %d rows (%s to %s)", len(df), df["date"].min(), df["date"].max())
    records = df.to_dict(orient="records")
    sent = _upsert_to_supabase(url, key, records, dry_run)
    if not dry_run:
        _delete_sample_rows(url, key, "USA")
    return f"dry-run: {len(records)} rows ready" if dry_run else f"OK: {sent}/{len(records)} rows"


def ingest_brazil(
    pkr_rate: float, brl_rate: float, ice_series,
    url: str, key: str, dry_run: bool,
) -> str:
    logger.info("Ingesting Brazil (CEPEA/Esalq -> ICE fallback)...")
    from country_sources.brazil import fetch_brazil_cotton_monthly
    try:
        df = fetch_brazil_cotton_monthly(
            start_date="2015-01-01",
            pkr_rate=pkr_rate,
            brl_rate=brl_rate,
            ice_series=ice_series,
        )
    except RuntimeError as exc:
        return f"FAIL: {exc}"

    source_used = df["source"].iloc[0] if not df.empty else "unknown"
    logger.info("  %d rows via %s (%s to %s)",
                len(df), source_used, df["date"].min(), df["date"].max())
    records = df.to_dict(orient="records")
    sent = _upsert_to_supabase(url, key, records, dry_run)
    if not dry_run:
        _delete_sample_rows(url, key, "Brazil")
    label = f"dry-run: {len(records)} rows ready" if dry_run else f"OK: {sent}/{len(records)} rows"
    return f"{label} [{source_used}]"


def ingest_pakistan(pkr_rate: float, ice_series, url: str, key: str, dry_run: bool) -> str:
    logger.info("Ingesting Pakistan (ICE domestic estimate)...")
    from country_sources.pakistan import fetch_pakistan_cotton_monthly
    try:
        df = fetch_pakistan_cotton_monthly(
            start_date="2015-01-01",
            pkr_rate=pkr_rate,
            ice_series=ice_series,
        )
    except RuntimeError as exc:
        return f"FAIL: {exc}"

    logger.info("  %d rows (%s to %s)", len(df), df["date"].min(), df["date"].max())
    records = df.to_dict(orient="records")
    sent = _upsert_to_supabase(url, key, records, dry_run)
    if not dry_run:
        _delete_sample_rows(url, key, "Pakistan")
    return f"dry-run: {len(records)} rows ready" if dry_run else f"OK: {sent}/{len(records)} rows"


def ingest_turkey(pkr_rate: float, ice_series, url: str, key: str, dry_run: bool) -> str:
    logger.info("Ingesting Turkey (ICE import parity estimate)...")
    from country_sources.turkey import fetch_turkey_cotton_monthly
    try:
        df = fetch_turkey_cotton_monthly(
            start_date="2015-01-01",
            pkr_rate=pkr_rate,
            ice_series=ice_series,
        )
    except RuntimeError as exc:
        return f"FAIL: {exc}"

    logger.info("  %d rows (%s to %s)", len(df), df["date"].min(), df["date"].max())
    records = df.to_dict(orient="records")
    sent = _upsert_to_supabase(url, key, records, dry_run)
    if not dry_run:
        _delete_sample_rows(url, key, "Turkey")
    return f"dry-run: {len(records)} rows ready" if dry_run else f"OK: {sent}/{len(records)} rows"


def ingest_tanzania(pkr_rate: float, ice_series, url: str, key: str, dry_run: bool) -> str:
    logger.info("Ingesting Tanzania (East Africa regional)...")
    from country_sources.tanzania import fetch_tanzania_cotton_monthly
    try:
        df = fetch_tanzania_cotton_monthly(
            start_date="2015-01-01",
            pkr_rate=pkr_rate,
            ice_series=ice_series,
        )
    except RuntimeError as exc:
        return f"FAIL: {exc}"

    logger.info("  %d rows (%s to %s)", len(df), df["date"].min(), df["date"].max())
    records = df.to_dict(orient="records")
    sent = _upsert_to_supabase(url, key, records, dry_run)
    if not dry_run:
        _delete_sample_rows(url, key, "Tanzania")
    return f"dry-run: {len(records)} rows ready" if dry_run else f"OK: {sent}/{len(records)} rows"


def ingest_ivory_coast(pkr_rate: float, ice_series, url: str, key: str, dry_run: bool) -> str:
    logger.info("Ingesting Ivory Coast (West Africa CFA regional)...")
    from country_sources.ivory_coast import fetch_ivory_coast_cotton_monthly
    try:
        df = fetch_ivory_coast_cotton_monthly(
            start_date="2015-01-01",
            pkr_rate=pkr_rate,
            ice_series=ice_series,
        )
    except RuntimeError as exc:
        return f"FAIL: {exc}"

    logger.info("  %d rows (%s to %s)", len(df), df["date"].min(), df["date"].max())
    records = df.to_dict(orient="records")
    sent = _upsert_to_supabase(url, key, records, dry_run)
    if not dry_run:
        _delete_sample_rows(url, key, "Ivory Coast")
    return f"dry-run: {len(records)} rows ready" if dry_run else f"OK: {sent}/{len(records)} rows"


def ingest_sudan(pkr_rate: float, ice_series, url: str, key: str, dry_run: bool) -> str:
    logger.info("Ingesting Sudan (East Africa regional — high uncertainty)...")
    from country_sources.sudan import fetch_sudan_cotton_monthly
    try:
        df = fetch_sudan_cotton_monthly(
            start_date="2015-01-01",
            pkr_rate=pkr_rate,
            ice_series=ice_series,
        )
    except RuntimeError as exc:
        return f"FAIL: {exc}"

    logger.info("  %d rows (%s to %s)", len(df), df["date"].min(), df["date"].max())
    records = df.to_dict(orient="records")
    sent = _upsert_to_supabase(url, key, records, dry_run)
    if not dry_run:
        _delete_sample_rows(url, key, "Sudan")
    return f"dry-run: {len(records)} rows ready" if dry_run else f"OK: {sent}/{len(records)} rows"


def ingest_west_africa(pkr_rate: float, ice_series, url: str, key: str, dry_run: bool) -> str:
    logger.info("Ingesting West Africa (regional aggregate)...")
    from country_sources.west_africa import fetch_west_africa_cotton_monthly
    try:
        df = fetch_west_africa_cotton_monthly(
            start_date="2015-01-01",
            pkr_rate=pkr_rate,
            ice_series=ice_series,
        )
    except RuntimeError as exc:
        return f"FAIL: {exc}"

    logger.info("  %d rows (%s to %s)", len(df), df["date"].min(), df["date"].max())
    records = df.to_dict(orient="records")
    sent = _upsert_to_supabase(url, key, records, dry_run)
    if not dry_run:
        _delete_sample_rows(url, key, "West Africa")
    return f"dry-run: {len(records)} rows ready" if dry_run else f"OK: {sent}/{len(records)} rows"


# ── Orchestrator ───────────────────────────────────────────────────────────────

# All known country names after migration (for bootstrap reference)
ALL_COUNTRIES = [
    "USA", "Brazil", "Pakistan", "Turkey",
    "Tanzania", "Ivory Coast", "Sudan", "West Africa",
]

# Source IDs classified as LIVE (used by bootstrap + dashboard)
LIVE_SOURCE_IDS = {"FRED/ICE-No2", "CEPEA/ESALQ"}

# Source IDs classified as ESTIMATED (basis-derived, documented methodology)
ESTIMATED_SOURCE_IDS = {
    "ICE/Brazil-basis",
    "ICE/Pakistan-domestic",
    "ICE/Turkey-import",
}

# Source IDs classified as REGIONAL (benchmark estimates, lower precision)
REGIONAL_SOURCE_IDS = {
    "regional/EastAfrica",
    "regional/WestAfrica-CIV",
    "regional/EastAfrica-SDN",
    "regional/WestAfrica",
}


def run_all(dry_run: bool = False, countries: list[str] | None = None) -> None:
    """
    Ingest all country cotton connectors.
    countries: optional filter list (e.g. ["USA", "Brazil"]). None = all.
    """
    url, key = _get_credentials()
    if not url or not key:
        logger.error("Missing Supabase credentials.")
        sys.exit(1)

    fx = _get_fx_rates()
    pkr_rate = fx["PKR"]
    brl_rate = fx["BRL"]
    logger.info("FX: 1 USD = %.2f PKR, %.4f BRL", pkr_rate, brl_rate)

    # Fetch ICE No.2 once — shared by all basis-adjusted connectors
    logger.info("Fetching ICE No.2 series from FRED (shared by all connectors)...")
    from country_sources._common import fetch_ice_no2_series
    try:
        ice_series = fetch_ice_no2_series(start_date="2015-01-01")
        logger.info("  ICE series: %d rows (%s to %s)",
                    len(ice_series), ice_series.index[0].date(), ice_series.index[-1].date())
    except Exception as exc:
        logger.error("ICE No.2 fetch failed: %s — aborting country ingestion", exc)
        sys.exit(1)

    want = set(c.lower() for c in countries) if countries else None

    def _should_run(name: str) -> bool:
        return want is None or name.lower() in want

    # Tier A — priority
    results: dict[str, str] = {}
    if _should_run("usa"):
        results["USA"] = ingest_usa(pkr_rate, url, key, dry_run)
    if _should_run("brazil"):
        results["Brazil"] = ingest_brazil(pkr_rate, brl_rate, ice_series, url, key, dry_run)
    if _should_run("pakistan"):
        results["Pakistan"] = ingest_pakistan(pkr_rate, ice_series, url, key, dry_run)
    if _should_run("turkey"):
        results["Turkey"] = ingest_turkey(pkr_rate, ice_series, url, key, dry_run)

    # Tier B — secondary
    if _should_run("tanzania"):
        results["Tanzania"] = ingest_tanzania(pkr_rate, ice_series, url, key, dry_run)
    if _should_run("ivory coast"):
        results["Ivory Coast"] = ingest_ivory_coast(pkr_rate, ice_series, url, key, dry_run)
    if _should_run("sudan"):
        results["Sudan"] = ingest_sudan(pkr_rate, ice_series, url, key, dry_run)
    if _should_run("west africa"):
        results["West Africa"] = ingest_west_africa(pkr_rate, ice_series, url, key, dry_run)

    logger.info("=== Country Sources Summary ===")
    for country, status in results.items():
        logger.info("  %-15s %s", country + ":", status)

    if any("FAIL" in v for v in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest country-wise cotton price sources")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch and validate without writing to Supabase")
    parser.add_argument("--countries", nargs="+",
                        metavar="COUNTRY",
                        help="Limit to specific countries (e.g. USA Brazil Pakistan)")
    args = parser.parse_args()
    run_all(dry_run=args.dry_run, countries=args.countries)
