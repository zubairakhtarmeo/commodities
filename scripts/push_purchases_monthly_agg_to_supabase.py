"""Upload purchases_monthly_agg.csv into Supabase table `purchases_monthly_agg`.

This enables the Streamlit Cloud app to render the Quarterly Purchasing Forecast
without requiring file uploads (code-only deployment).

Prereqs
- Create the table (see docs/supabase_streamlit_cloud.md)
- Set env vars:
  $env:SUPABASE_URL = "https://..."
  $env:SUPABASE_SERVICE_ROLE_KEY = "..."

Run
  python scripts/push_purchases_monthly_agg_to_supabase.py

By default it reads:
  data/processed/purchases_clean/purchases_monthly_agg.csv
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import requests

DEFAULT_CSV = Path("data") / "processed" / "purchases_clean" / "purchases_monthly_agg.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload purchases_monthly_agg.csv into Supabase purchases_monthly_agg")
    parser.add_argument("--csv", dest="csv_path", default=str(DEFAULT_CSV))
    parser.add_argument("--url", dest="url", default=os.environ.get("SUPABASE_URL"))
    parser.add_argument(
        "--key",
        dest="key",
        default=os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_ANON_KEY"),
        help="Prefer SUPABASE_SERVICE_ROLE_KEY",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate file + show row count without uploading")
    args = parser.parse_args()

    url = args.url
    key = args.key
    if not url or not key:
        raise SystemExit("Missing SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_ANON_KEY).")

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise SystemExit("CSV is empty; nothing to upload.")

    required = {"commodity", "month"}
    missing = [c for c in sorted(required) if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    # Normalize types
    df = df.copy()
    df["month"] = pd.to_datetime(df["month"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["month", "commodity"])

    # Keep only known columns for upload (safe + stable)
    keep_cols = [
        c
        for c in [
            "operating_unit",
            "commodity",
            "month",
            "lines",
            "total_qty",
            "total_qty_kg",
            "total_amount",
            "avg_unit_price",
        ]
        if c in df.columns
    ]
    out = df[keep_cols].copy()

    # Convert numeric columns
    for c in ["lines", "total_qty", "total_qty_kg", "total_amount", "avg_unit_price"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.where(pd.notnull(out), None)

    if args.dry_run:
        print(f"ok: {csv_path} ({len(out)} rows ready)")
        print(f"columns: {list(out.columns)}")
        return 0

    base_url = str(url).rstrip("/")
    key = str(key).strip()

    def headers(extra: dict | None = None) -> dict:
        h = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        if extra:
            h.update(extra)
        return h

    # Upsert in batches
    rows = out.to_dict(orient="records")
    batch_size = 500
    total = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        resp = requests.post(
            f"{base_url}/rest/v1/purchases_monthly_agg",
            headers=headers({"Prefer": "resolution=merge-duplicates,return=minimal"}),
            params={"on_conflict": "commodity,month,operating_unit"},
            json=batch,
            timeout=60,
        )
        if not resp.ok:
            raise SystemExit(f"Supabase upsert failed: HTTP {resp.status_code} {resp.text}")
        total += len(batch)

    print(f"done. rows upserted: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
