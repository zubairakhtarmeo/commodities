"""Ingest messy Oracle Excel purchase exports into clean commodity datasets.

Example:
  python scripts/ingest_oracle_purchases.py --input data/raw/purchases/oracle --pattern "*.xlsx"

Outputs (default):
    data/processed/purchases_clean/purchases_master.csv
    data/processed/purchases_clean/purchases_monthly_agg.csv
    data/processed/purchases_clean/purchases_<commodity>.csv
    data/processed/purchases_clean/ingest_report.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import json
import pandas as pd

# Add repo root to import path (consistent with other scripts)
sys.path.append(str(Path(__file__).parent.parent))

from forecasting.ingestion.oracle_purchases import build_monthly_aggregates, ingest_oracle_purchases_dir
from forecasting.ingestion.oracle_purchases import clean_oracle_purchases_workbook


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Directory containing Oracle .xlsx exports")
    parser.add_argument("--pattern", type=str, default="*.xlsx", help="Glob pattern, e.g. *.xlsx")
    parser.add_argument(
        "--sheet",
        type=str,
        default="auto",
        help="Sheet name or index; use 'auto' to scan all sheets (default)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path("data") / "processed" / "purchases_clean"),
        help="Output directory",
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sheet selection
    sheet_arg = str(args.sheet).strip()
    sheet: object | None
    if sheet_arg.lower() == "auto":
        sheet = None
    elif sheet_arg.isdigit():
        sheet = int(sheet_arg)
    else:
        sheet = sheet_arg

    # Ingest with per-file intelligence when sheet=auto
    report: dict = {"input": str(input_dir), "pattern": args.pattern, "sheet": sheet_arg, "files": []}
    files = sorted(input_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files matched {args.pattern!r} in {str(input_dir)!r}")

    if sheet is None:
        frames = []
        for f in files:
            cleaned, meta = clean_oracle_purchases_workbook(xlsx_path=f)
            report["files"].append(meta)
            if meta.get("status") == "ok" and not cleaned.empty:
                frames.append(cleaned)
        if frames:
            df = pd.concat(frames, ignore_index=True)
        else:
            df = ingest_oracle_purchases_dir(input_dir=input_dir, pattern=args.pattern, sheet_name=None)
    else:
        df = ingest_oracle_purchases_dir(input_dir=input_dir, pattern=args.pattern, sheet_name=sheet)
    master_path = output_dir / "purchases_master.csv"
    df.to_csv(master_path, index=False)

    report.update(
        {
            "rows": int(len(df)),
            "outputs": {
                "master": str(master_path),
            },
        }
    )

    monthly = build_monthly_aggregates(df)
    monthly_path = output_dir / "purchases_monthly_agg.csv"
    monthly.to_csv(monthly_path, index=False)
    report["outputs"].update({"monthly": str(monthly_path)})

    # Per-commodity exports
    for commodity, sub in df.groupby("commodity"):
        safe = (
            str(commodity)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
        )
        sub.to_csv(output_dir / f"purchases_{safe}.csv", index=False)

    report_path = output_dir / "ingest_report.json"
    report["outputs"].update({"report": str(report_path)})
    report["commodities"] = df["commodity"].value_counts().to_dict() if len(df) else {}
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"✓ Wrote {master_path}")
    print(f"✓ Wrote {monthly_path}")
    print(f"✓ Wrote {output_dir / 'purchases_<commodity>.csv'}")
    print(f"✓ Wrote {report_path}")
    print(f"Rows: {len(df):,}")
    print("\nTop commodities:")
    print(df["commodity"].value_counts().head(15).to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
