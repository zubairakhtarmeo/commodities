"""Push one prediction record into Supabase.

Usage (PowerShell):
  $env:SUPABASE_URL="https://..."; $env:SUPABASE_SERVICE_ROLE_KEY="...";
  python scripts/push_prediction_record.py --asset BTCUSD --as-of 2026-01-29 --target 2026-01-30 --pred 89390 --actual 89784 --unit "USD"

This script is intended for scheduled jobs (Task Scheduler / cron / GitHub Actions).
"""

from __future__ import annotations

import argparse
import os
from datetime import date
import requests


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", required=True)
    parser.add_argument("--as-of", dest="as_of_date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--target", dest="target_date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--pred", dest="predicted_value", required=True, type=float)
    parser.add_argument("--actual", dest="actual_value", required=False, type=float)
    parser.add_argument("--unit", default="")
    parser.add_argument("--model", dest="model_name", default="default")
    parser.add_argument("--frequency", default="daily")
    parser.add_argument("--horizon", default="1d")
    args = parser.parse_args()

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_ANON_KEY")
    if not url or not key:
        raise SystemExit("Missing SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_ANON_KEY)")

    base_url = str(url).rstrip("/")
    key = str(key).strip()

    row = {
        "asset": args.asset,
        "as_of_date": args.as_of_date,
        "target_date": args.target_date,
        "predicted_value": float(args.predicted_value),
        "actual_value": float(args.actual_value) if args.actual_value is not None else None,
        "unit": args.unit,
        "model_name": args.model_name,
        "frequency": args.frequency,
        "horizon": args.horizon,
    }

    resp = requests.post(
        f"{base_url}/rest/v1/prediction_records",
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates,return=minimal",
        },
        json=[row],
        timeout=60,
    )
    if not resp.ok:
        raise SystemExit(f"Supabase upsert failed: HTTP {resp.status_code} {resp.text}")
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
