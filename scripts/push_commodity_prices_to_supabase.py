"""Upload commodity CSV history into Supabase table `commodity_prices`.

This is the easiest way to fix "No data available" on Streamlit Cloud.

1) Create the table (see docs/supabase_streamlit_cloud.md)
2) Set env vars:
   $env:SUPABASE_URL = "https://..."
   $env:SUPABASE_SERVICE_ROLE_KEY = "..."
3) Run:
    python scripts/push_commodity_prices_to_supabase.py

It will read the standard monthly CSVs from data/raw/* and upsert them.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import pandas as pd

RAW_DATA_DIR = Path("data/raw")

DEFAULT_ASSETS = [
    ("cotton/cotton_usd_monthly", "USD/lb"),
    ("cotton/cotton_pkr_monthly", "PKR/maund"),
    ("polyester/polyester_usd_monthly", "USD/ton"),
    ("polyester/polyester_pkr_monthly", "PKR/ton"),
    ("viscose/viscose_usd_monthly", "USD/ton"),
    ("viscose/viscose_pkr_monthly", "PKR/kg"),
    ("energy/natural_gas_usd_monthly_clean", "USD/MMBTU"),
    ("energy/natural_gas_pkr_monthly_clean", "PKR/MMBTU"),
    ("energy/crude_oil_brent_usd_monthly_clean", "USD/barrel"),
    ("energy/crude_oil_brent_pkr_monthly_clean", "PKR/barrel"),
]


def detect_timestamp_col(df: pd.DataFrame) -> str:
    for c in ["timestamp", "date", "datetime", "time"]:
        if c in df.columns:
            return c
    return df.columns[0]


def detect_value_col(df: pd.DataFrame) -> str:
    for c in ["value", "price_usd", "price_pkr", "price", "close"]:
        if c in df.columns:
            return c
    # fallback: first numeric col
    for c in df.columns:
        if str(c).lower() in ["timestamp", "date", "datetime", "time", "index"]:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    raise ValueError("No numeric value column found")


def asset_path_from_csv_path(csv_path: Path) -> str:
    rel = csv_path.relative_to(RAW_DATA_DIR)
    # Remove one or more .csv suffixes (handles accidental .csv.csv)
    name = rel.as_posix()
    while name.lower().endswith(".csv"):
        name = name[:-4]
    return name


def discover_csv_assets() -> list[Path]:
    # Skip backups and non-data folders
    all_csvs = [p for p in RAW_DATA_DIR.rglob("*.csv") if p.is_file()]
    filtered = []
    for p in all_csvs:
        parts = [x.lower() for x in p.parts]
        if "backups" in parts:
            continue
        filtered.append(p)
    return sorted(filtered)


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload data/raw CSV history into Supabase commodity_prices")
    parser.add_argument("--url", dest="url", default=os.environ.get("SUPABASE_URL"))
    parser.add_argument(
        "--key",
        dest="key",
        default=os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_ANON_KEY"),
        help="Prefer SUPABASE_SERVICE_ROLE_KEY",
    )
    parser.add_argument("--dry-run", action="store_true", help="List CSVs that would be uploaded")
    parser.add_argument("--limit", type=int, default=0, help="Only upload first N CSV files (0 = all)")
    args = parser.parse_args()

    url = args.url
    key = args.key
    if not url or not key:
        raise SystemExit("Missing SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_ANON_KEY).")

    from supabase import create_client

    client = create_client(url, key)

    csvs = discover_csv_assets()
    if args.limit and args.limit > 0:
        csvs = csvs[: int(args.limit)]

    if args.dry_run:
        for p in csvs:
            print(asset_path_from_csv_path(p))
        print(f"count: {len(csvs)}")
        return 0

    total_rows = 0
    for csv_path in csvs:
        asset_path = asset_path_from_csv_path(csv_path)
        df = pd.read_csv(csv_path)
        ts_col = detect_timestamp_col(df)
        val_col = detect_value_col(df)

        out = pd.DataFrame({
            "asset_path": asset_path,
            "timestamp": pd.to_datetime(df[ts_col], errors="coerce"),
            "value": pd.to_numeric(df[val_col], errors="coerce"),
            "currency": None,
            "source": "csv_import",
        }).dropna(subset=["timestamp", "value"])

        # Upsert in batches to avoid request limits
        rows = out.to_dict(orient="records")
        batch_size = 500
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            client.table("commodity_prices").upsert(batch).execute()

        total_rows += len(rows)
        print(f"ok: {asset_path} ({len(rows)} rows)")

    print(f"done. total rows upserted: {total_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
