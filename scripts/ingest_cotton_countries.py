"""
Ingest country-wise cotton prices from CSV into Supabase.

CSV format required:
  date,country,price_usd_per_lb,source
  2026-01-01,USA,0.74,supplier_quote
  2026-01-01,Brazil,0.71,supplier_quote

Usage:
  python scripts/ingest_cotton_countries.py
  python scripts/ingest_cotton_countries.py --file path/to/custom.csv

Credentials: set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY as env vars,
or they will be read from .streamlit/secrets.toml automatically.
"""

from __future__ import annotations

import sys
import os
import argparse
import pandas as pd
import requests
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent


def get_supabase_config() -> tuple[str, str]:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_ANON_KEY")

    if not url or not key:
        secrets_path = BASE_DIR / ".streamlit" / "secrets.toml"
        if secrets_path.exists():
            try:
                import tomllib  # Python 3.11+
                with open(secrets_path, "rb") as f:
                    secrets = tomllib.load(f)
            except ImportError:
                import tomli  # pip install tomli
                with open(secrets_path, "rb") as f:
                    secrets = tomli.load(f)
            url = url or secrets.get("SUPABASE_URL")
            key = key or secrets.get("SUPABASE_SERVICE_ROLE_KEY") or secrets.get("SUPABASE_ANON_KEY")

    if not url or not key:
        raise SystemExit("Missing SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY. "
                         "Set as env vars or add to .streamlit/secrets.toml.")
    return str(url).rstrip("/"), str(key).strip()


def get_fx_rate() -> float:
    try:
        r = requests.get("https://open.er-api.com/v6/latest/USD", timeout=10)
        return float(r.json()["rates"]["PKR"])
    except Exception:
        return 278.0


COUNTRY_NORMALIZE: dict[str, str] = {
    "pak": "Pakistan", "pakistan": "Pakistan",
    "pak local": "Pakistan", "pak balochi": "Pakistan",
    "usa": "USA", "us": "USA", "united states": "USA",
    "memphis": "USA", "sjv": "USA", "pima": "USA",
    "brazil": "Brazil", "brazilian": "Brazil",
    "argentina": "Argentina",
    "turkey": "Turkey", "turkish": "Turkey",
    "tanzania": "Tanzania",
    "sudan": "Sudan",
    "ivory coast": "Ivory Coast",
    "west africa": "West Africa", "west african": "West Africa",
    "mexico": "Mexico", "mexican": "Mexico",
    "turkmenistan": "Turkmenistan",
    "tajikistan": "Tajikistan",
    "afghanistan": "Afghanistan", "afghani": "Afghanistan",
    "greece": "Greece", "greek": "Greece",
    "uganda": "Uganda",
    "mozambique": "Mozambique",
    "egypt": "Egypt", "giza": "Egypt",
}


def normalize_country(raw: str) -> str:
    if not raw:
        return "Unknown"
    key = str(raw).lower().strip()
    for k, v in COUNTRY_NORMALIZE.items():
        if k in key:
            return v
    return str(raw).title().strip()


LB_PER_MAUND = 82.2857


def ingest_file(csv_path: str, supabase_url: str, supabase_key: str) -> int:
    print(f"Reading: {csv_path}")
    df = pd.read_csv(csv_path)

    required = {"date", "country", "price_usd_per_lb"}
    missing = required - set(df.columns)
    if missing:
        print(f"❌ Missing columns: {missing}")
        print(f"   Found columns: {list(df.columns)}")
        sys.exit(1)

    fx = get_fx_rate()
    print(f"FX rate: {fx:.2f} PKR/USD")

    records = []
    for _, row in df.iterrows():
        country = normalize_country(str(row["country"]))
        price_usd = float(row["price_usd_per_lb"])
        price_pkr = price_usd * fx * LB_PER_MAUND
        records.append({
            "date": str(pd.to_datetime(row["date"]).date()),
            "country": country,
            "price_usd_per_lb": round(price_usd, 6),
            "price_pkr_per_maund": round(price_pkr, 2),
            "source": str(row.get("source", "manual")),
        })

    print(f"Upserting {len(records)} records...")

    headers = {
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates",
    }

    batch_size = 100
    success = 0
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        r = requests.post(
            f"{supabase_url}/rest/v1/cotton_country_prices",
            headers=headers,
            json=batch,
            timeout=30,
        )
        if r.status_code in (200, 201):
            success += len(batch)
        else:
            print(f"❌ Batch {i // batch_size + 1} failed ({r.status_code}): {r.text[:200]}")

    print(f"✅ Inserted {success}/{len(records)} records")
    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest country cotton prices into Supabase")
    parser.add_argument(
        "--file",
        default=str(BASE_DIR / "data" / "raw" / "cotton" / "cotton_country_prices.csv"),
        help="Path to input CSV",
    )
    args = parser.parse_args()

    url, key = get_supabase_config()
    ingest_file(args.file, url, key)
