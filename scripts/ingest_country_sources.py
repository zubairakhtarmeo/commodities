"""
Ingest real country-wise cotton prices into Supabase cotton_country_prices.

Currently integrated real sources:
  USA  — FRED PCOTTINDUSDM (ICE Cotton No.2, Memphis delivery)

Sample/demo sources (not touched by this script):
  Brazil, Ivory Coast, Pakistan, Sudan, Tanzania, Turkey, West Africa
  These remain in the DB with source='sample'.

Usage:
  python scripts/ingest_country_sources.py
  python scripts/ingest_country_sources.py --dry-run
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


def _get_fx_rate() -> float:
    try:
        r = requests.get("https://open.er-api.com/v6/latest/USD", timeout=8)
        return float(r.json()["rates"]["PKR"])
    except Exception:
        logger.warning("FX fetch failed; using fallback 278.5 PKR/USD")
        return 278.5


def _upsert_to_supabase(
    url: str,
    key: str,
    records: list[dict],
    dry_run: bool = False,
) -> int:
    """Upsert records into cotton_country_prices. Returns count of records sent."""
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
        batch = records[i : i + batch_size]
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


def _delete_sample_rows(url: str, key: str, country: str) -> int:
    """Remove legacy source='sample' rows for a country now covered by a real source."""
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "count=exact",
    }
    r = requests.delete(
        f"{url}/rest/v1/cotton_country_prices",
        headers=headers,
        params={"country": f"eq.{country}", "source": "eq.sample"},
        timeout=15,
    )
    if r.status_code == 204:
        cr = r.headers.get("Content-Range", "")
        # 204 means success; Content-Range not returned on DELETE
        return 0
    if r.ok:
        return 0
    logger.warning("  Sample row cleanup for %s failed (%d): %s", country, r.status_code, r.text[:100])
    return -1


def ingest_usa(pkr_rate: float, url: str, key: str, dry_run: bool) -> str:
    """Fetch USA cotton from FRED and push to Supabase. Returns status string."""
    logger.info("Ingesting USA cotton (FRED PCOTTINDUSDM)...")

    from country_sources.usa_cotton import fetch_usa_cotton_monthly
    try:
        df = fetch_usa_cotton_monthly(start_date="2015-01-01", pkr_rate=pkr_rate)
    except RuntimeError as exc:
        return f"FAIL: {exc}"

    logger.info(
        "  Fetched %d rows  (%s to %s)",
        len(df), df["date"].min(), df["date"].max(),
    )

    records = df.to_dict(orient="records")
    sent = _upsert_to_supabase(url, key, records, dry_run)

    if dry_run:
        return f"dry-run: {len(records)} rows ready"

    # Clean up any legacy sample rows for USA (idempotent)
    _delete_sample_rows(url, key, "USA")

    logger.info("  Upserted %d/%d rows for USA", sent, len(records))
    return f"OK: {sent}/{len(records)} rows"


def run_all(dry_run: bool = False) -> None:
    url, key = _get_credentials()
    if not url or not key:
        logger.error("Missing Supabase credentials.")
        sys.exit(1)

    pkr_rate = _get_fx_rate()
    logger.info("FX: 1 USD = %.2f PKR", pkr_rate)

    results = {"USA": ingest_usa(pkr_rate, url, key, dry_run)}

    logger.info("=== Country Sources Summary ===")
    for country, status in results.items():
        logger.info("  %s: %s", country, status)

    if any("FAIL" in v for v in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest real country cotton sources")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch and validate without writing to Supabase")
    args = parser.parse_args()
    run_all(dry_run=args.dry_run)
