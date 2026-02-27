"""Bulk-seed Supabase `prediction_records` from local artifact CSVs.

Why this exists
- Streamlit Cloud can render the AI Predictions page from Supabase.
- This repo already contains model validation artifacts like:
    artifacts/EURUSD/predictions_linear_ridge_h1.csv
  with columns: asof,target_time,y_true,y_pred
- This script uploads those rows into Supabase so the dashboard shows charts.

Usage (PowerShell)
  $env:SUPABASE_URL="https://..."; $env:SUPABASE_SERVICE_ROLE_KEY="...";
  python scripts/push_predictions_from_artifacts_to_supabase.py

Notes
- Requires the table `prediction_records` (see docs/supabase_streamlit_cloud.md).
- Uses on_conflict: asset,as_of_date,target_date,model_name,horizon
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


def _infer_model_and_horizon(pred_file: Path) -> tuple[str, str, str]:
    """Infer (model_name, horizon, frequency) from a filename.

    Example: predictions_linear_ridge_h1.csv -> ("linear_ridge", "30d", "monthly")
    """

    name = pred_file.name
    model_name = "default"
    if name.startswith("predictions_"):
        mid = name[len("predictions_") :]
        mid = mid.replace(".csv", "")
        # split on _h if present
        if "_h" in mid:
            model_name = mid.split("_h", 1)[0] or "default"
        else:
            model_name = mid or "default"

    # Artifacts here are monthly validations; map to a horizon that exists in the UI.
    horizon = "30d"
    frequency = "monthly"
    return model_name, horizon, frequency


def _iter_prediction_files(artifacts_dir: Path) -> Iterable[Path]:
    for p in artifacts_dir.rglob("predictions_*.csv"):
        if p.is_file():
            yield p


def _postgrest_headers(key: str) -> dict:
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates,return=minimal",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts", help="Artifacts directory (default: artifacts)")
    parser.add_argument("--dry-run", action="store_true", help="Parse files but do not upload")
    parser.add_argument("--chunk", type=int, default=5000, help="Rows per request (default: 5000)")
    args = parser.parse_args()

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_ANON_KEY")
    if not url or not key:
        raise SystemExit("Missing SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_ANON_KEY)")

    base_url = str(url).rstrip("/")
    key = str(key).strip()

    artifacts_dir = Path(args.artifacts)
    if not artifacts_dir.exists():
        raise SystemExit(f"Artifacts directory not found: {artifacts_dir}")

    files = list(_iter_prediction_files(artifacts_dir))
    if not files:
        raise SystemExit(f"No prediction artifact files found under: {artifacts_dir}")

    total_uploaded = 0
    for pred_file in files:
        asset = pred_file.parent.name
        model_name, horizon, frequency = _infer_model_and_horizon(pred_file)

        df = pd.read_csv(pred_file)
        expected_cols = {"asof", "target_time", "y_pred"}
        if not expected_cols.issubset(set(df.columns)):
            print(f"skip {pred_file}: missing required columns {sorted(expected_cols)}")
            continue

        df["asof"] = pd.to_datetime(df["asof"], errors="coerce")
        df["target_time"] = pd.to_datetime(df["target_time"], errors="coerce")
        df = df.dropna(subset=["asof", "target_time"])

        df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")
        if "y_true" in df.columns:
            df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")

        df = df.dropna(subset=["y_pred"])
        if df.empty:
            print(f"skip {pred_file}: no rows")
            continue

        rows = []
        for _, r in df.iterrows():
            rows.append(
                {
                    "asset": str(asset),
                    "as_of_date": pd.Timestamp(r["asof"]).date().isoformat(),
                    "target_date": pd.Timestamp(r["target_time"]).date().isoformat(),
                    "predicted_value": float(r["y_pred"]),
                    "actual_value": float(r["y_true"]) if ("y_true" in df.columns and pd.notna(r.get("y_true"))) else None,
                    "unit": "",
                    "model_name": str(model_name),
                    "frequency": str(frequency),
                    "horizon": str(horizon),
                }
            )

        if args.dry_run:
            print(f"dry-run {pred_file}: {len(rows)} rows -> asset={asset}, model={model_name}, horizon={horizon}")
            continue

        # Upload in chunks
        endpoint = f"{base_url}/rest/v1/prediction_records"
        params = {"on_conflict": "asset,as_of_date,target_date,model_name,horizon"}
        for i in range(0, len(rows), int(args.chunk)):
            chunk = rows[i : i + int(args.chunk)]
            resp = requests.post(
                endpoint,
                headers=_postgrest_headers(key),
                params=params,
                json=chunk,
                timeout=90,
            )
            if not resp.ok:
                raise SystemExit(f"Upload failed for {pred_file}: HTTP {resp.status_code} {resp.text}")
            total_uploaded += len(chunk)

        print(f"uploaded {pred_file}: {len(rows)} rows")

    print(f"done: uploaded {total_uploaded} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
