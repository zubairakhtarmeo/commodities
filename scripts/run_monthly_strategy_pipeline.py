"""
run_monthly_strategy_pipeline.py
---------------------------------
Master monthly automation pipeline for MG Apparel procurement strategy.

Execution flow (8 steps):
    1. Oracle inventory extraction   → MG_STOCK_TILL_DATE_MULTIPLE_UNIT.xlsx
    2. Oracle consumption extraction → MG_TRANSACTION_REGISTER_INV.xlsx
    3. Inventory cleaning            → detail_df, summary_df
    4. Consumption cleaning          → detail_df, summary_df, diagnostics_df
    5. Monthly archive (BEFORE any workbook modification)
    6. Workbook update               → Raw Material + Consumption sheets
    7. Procurement engine            → strategy_df, validation_df
    8. Validation report             → reports/monthly_validation_report.md

Constraints:
    - Market Inputs sheet is NOT updated (requires live market data run separately)
    - Strategy sheet formulas are never touched
    - Archive is created before any workbook modification
    - Pipeline stops immediately on Oracle, archive, workbook, or engine failure

Usage:
    # Full pipeline (previous month, default paths)
    python scripts/run_monthly_strategy_pipeline.py

    # Specify reporting month
    python scripts/run_monthly_strategy_pipeline.py --year 2026 --month 4

    # Skip Oracle extraction (use existing files, e.g. already downloaded)
    python scripts/run_monthly_strategy_pipeline.py --skip-extraction

    # Full example with all paths explicit
    python scripts/run_monthly_strategy_pipeline.py \\
        --oracle-dir   "D:\\path" \\
        --workbook     "data/strategy/Strategies.xlsx" \\
        --archive-dir  "data/strategy/archive" \\
        --mapping-csv  "scripts/commodity_mapping.csv" \\
        --year 2026 --month 4

Notes:
    - inventory_data.py and consumption_data.py must have their base_directory
      set to the same path as --oracle-dir (default: D:\\path).
    - The pipeline derives period_days from calendar.monthrange(year, month)
      and never hardcodes 30.
"""

from __future__ import annotations

import argparse
import calendar
import logging
import subprocess
import sys
import textwrap
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Logging setup — set before any imports that also configure logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPTS_DIR   = Path(__file__).parent
PROJECT_ROOT  = SCRIPTS_DIR.parent

DEFAULT_ORACLE_DIR  = Path(r"D:\path")
DEFAULT_WORKBOOK    = PROJECT_ROOT / "data" / "strategy" / "Strategies.xlsx"
DEFAULT_ARCHIVE_DIR = PROJECT_ROOT / "data" / "strategy" / "archive"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_MAPPING_CSV = SCRIPTS_DIR / "commodity_mapping.csv"
DEFAULT_OVERRIDES_CSV = SCRIPTS_DIR / "item_code_overrides.csv"

INVENTORY_FILENAME  = "MG_STOCK_TILL_DATE_MULTIPLE_UNIT.xlsx"
CONSUMPTION_FILENAME = "MG_TRANSACTION_REGISTER_INV.xlsx"

INVENTORY_SCRIPT  = SCRIPTS_DIR / "inventory_data.py"
CONSUMPTION_SCRIPT = SCRIPTS_DIR / "consumption_data.py"


# ---------------------------------------------------------------------------
# Pipeline result accumulator
# ---------------------------------------------------------------------------

class PipelineResult:
    """Accumulates step results and timing for the final report."""

    def __init__(self, year: int, month: int):
        self.year  = year
        self.month = month
        self.steps: list[dict] = []
        self.start_time = datetime.now()

    def record(self, step: int, name: str, status: str,
               detail: str, affected: str = "") -> None:
        self.steps.append({
            "step": step, "name": name, "status": status,
            "detail": detail, "affected": affected,
            "ts": datetime.now().strftime("%H:%M:%S"),
        })
        icon = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗"}.get(status, "·")
        logger.info("Step %d %-30s [%s] %s", step, name, status, detail[:80])

    def has_failure(self) -> bool:
        return any(s["status"] == "FAIL" for s in self.steps)

    def counts(self) -> tuple[int, int, int]:
        p = sum(1 for s in self.steps if s["status"] == "PASS")
        w = sum(1 for s in self.steps if s["status"] == "WARN")
        f = sum(1 for s in self.steps if s["status"] == "FAIL")
        return p, w, f


# ---------------------------------------------------------------------------
# Step 1 & 2 — Oracle extraction
# ---------------------------------------------------------------------------

def _run_extraction_script(
    script_path: Path,
    output_file:  Path,
    result:       PipelineResult,
    step:         int,
    name:         str,
) -> None:
    """Run a Selenium extraction script via subprocess and verify output.

    Raises:
        SystemExit: if the script fails or the output file is not found.
    """
    logger.info("Step %d: Launching Oracle extraction → %s", step, script_path.name)
    try:
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=False,
            timeout=900,  # 15 min max for Oracle browser automation
        )
        if proc.returncode != 0:
            result.record(step, name, "FAIL",
                          f"Script exited with code {proc.returncode}.")
            raise SystemExit(f"Pipeline stopped: {name} failed (exit {proc.returncode}).")
    except subprocess.TimeoutExpired:
        result.record(step, name, "FAIL",
                      "Script timed out after 15 minutes.")
        raise SystemExit(f"Pipeline stopped: {name} timed out.")
    except FileNotFoundError:
        result.record(step, name, "FAIL",
                      f"Script not found: {script_path}")
        raise SystemExit(f"Pipeline stopped: {script_path} not found.")

    if not output_file.exists():
        result.record(step, name, "FAIL",
                      f"Output file not found after extraction: {output_file}")
        raise SystemExit(f"Pipeline stopped: {output_file.name} not produced.")

    size_kb = output_file.stat().st_size / 1024
    result.record(step, name, "PASS",
                  f"Downloaded {output_file.name} ({size_kb:.0f} KB).",
                  str(output_file))


def step1_extract_inventory(
    oracle_dir: Path,
    result:     PipelineResult,
    skip:       bool,
) -> Path:
    """Step 1: Run inventory_data.py and return path to downloaded file."""
    output = oracle_dir / INVENTORY_FILENAME
    if skip:
        if not output.exists():
            result.record(1, "extract_inventory", "FAIL",
                          f"--skip-extraction set but file missing: {output}")
            raise SystemExit(f"Pipeline stopped: {output.name} not found.")
        size_kb = output.stat().st_size / 1024
        result.record(1, "extract_inventory", "PASS",
                      f"Skipped extraction — using existing file ({size_kb:.0f} KB).",
                      str(output))
    else:
        _run_extraction_script(INVENTORY_SCRIPT, output, result, 1, "extract_inventory")
    return output


def step2_extract_consumption(
    oracle_dir: Path,
    result:     PipelineResult,
    skip:       bool,
) -> Path:
    """Step 2: Run consumption_data.py and return path to downloaded file."""
    output = oracle_dir / CONSUMPTION_FILENAME
    if skip:
        if not output.exists():
            result.record(2, "extract_consumption", "FAIL",
                          f"--skip-extraction set but file missing: {output}")
            raise SystemExit(f"Pipeline stopped: {output.name} not found.")
        size_kb = output.stat().st_size / 1024
        result.record(2, "extract_consumption", "PASS",
                      f"Skipped extraction — using existing file ({size_kb:.0f} KB).",
                      str(output))
    else:
        _run_extraction_script(CONSUMPTION_SCRIPT, output, result, 2, "extract_consumption")
    return output


# ---------------------------------------------------------------------------
# Step 3 — Inventory cleaning
# ---------------------------------------------------------------------------

def step3_clean_inventory(
    inventory_xlsx: Path,
    mapping_csv:    Optional[Path],
    overrides_csv:  Optional[Path],
    result:         PipelineResult,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Step 3: Clean Oracle inventory extract.

    Returns:
        (detail_df, summary_df)
    """
    sys.path.insert(0, str(SCRIPTS_DIR))
    from clean_inventory import clean_inventory as _clean_inv
    from commodity_mapper import CommodityMapper

    mapper = CommodityMapper.from_csv_or_default(mapping_csv)

    try:
        detail_df, summary_df = _clean_inv(inventory_xlsx, mapper=mapper)
    except Exception as exc:
        result.record(3, "clean_inventory", "FAIL", str(exc))
        raise SystemExit(f"Pipeline stopped: clean_inventory failed — {exc}")

    n_rows     = len(detail_df)
    n_unmapped = (detail_df["category"] == "Unmapped").sum() if "category" in detail_df.columns else 0
    n_orgs     = detail_df["org_name"].nunique() if "org_name" in detail_df.columns else 0

    if n_unmapped > 0:
        unmapped_items = (
            detail_df[detail_df["category"] == "Unmapped"]["item_code"].unique().tolist()
        )
        result.record(3, "clean_inventory", "WARN",
                      f"{n_rows} rows, {n_orgs} orgs, {n_unmapped} unmapped item(s).",
                      ", ".join(str(x) for x in unmapped_items[:10]))
    else:
        result.record(3, "clean_inventory", "PASS",
                      f"{n_rows} rows, {n_orgs} orgs, 0 unmapped.")

    return detail_df, summary_df


# ---------------------------------------------------------------------------
# Step 4 — Consumption cleaning
# ---------------------------------------------------------------------------

def step4_clean_consumption(
    consumption_xlsx: Path,
    mapping_csv:      Optional[Path],
    filter_month:     Optional[str],
    result:           PipelineResult,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Step 4: Clean Oracle consumption extract.

    Returns:
        (detail_df, summary_df, diagnostics_df)
    """
    from clean_consumption import clean_consumption as _clean_cons
    from commodity_mapper import CommodityMapper

    mapper = CommodityMapper.from_csv_or_default(mapping_csv)

    try:
        detail_df, summary_df, diagnostics_df = _clean_cons(
            consumption_xlsx, mapper=mapper, filter_month=filter_month
        )
    except Exception as exc:
        result.record(4, "clean_consumption", "FAIL", str(exc))
        raise SystemExit(f"Pipeline stopped: clean_consumption failed — {exc}")

    n_rows     = len(detail_df)
    n_unmapped = (detail_df["category"] == "Unmapped").sum() if "category" in detail_df.columns else 0
    n_orgs     = detail_df["org_name"].nunique() if "org_name" in detail_df.columns else 0

    # Check for positive-qty rows (returns) and flag as WARN if count is unexpected
    pos_rows = 0
    if "primary_qty" in detail_df.columns:
        pos_rows = (detail_df["primary_qty"] > 0).sum()

    msg = f"{n_rows} transactions, {n_orgs} orgs, {n_unmapped} unmapped, {pos_rows} return rows."
    status = "WARN" if n_unmapped > 0 else "PASS"
    affected = ""
    if n_unmapped > 0:
        affected = ", ".join(
            detail_df[detail_df["category"] == "Unmapped"]["item_code"].unique().tolist()[:10]
        )
    result.record(4, "clean_consumption", status, msg, affected)

    return detail_df, summary_df, diagnostics_df


# ---------------------------------------------------------------------------
# Step 5 — Monthly archive (BEFORE workbook modification)
# ---------------------------------------------------------------------------

def step5_create_archive(
    workbook_path: Path,
    archive_dir:   Path,
    year:          int,
    month:         int,
    result:        PipelineResult,
) -> None:
    """Step 5: Create timestamped archive copy BEFORE any workbook changes."""
    from update_strategy_workbook import create_monthly_archive

    try:
        archive_path = create_monthly_archive(workbook_path, archive_dir, year, month)
        result.record(5, "monthly_archive", "PASS",
                      f"Archive created: {archive_path.name}",
                      str(archive_path))
    except FileExistsError as exc:
        result.record(5, "monthly_archive", "WARN",
                      str(exc))
        logger.warning("Archive already exists — skipping (not overwriting).")
    except Exception as exc:
        result.record(5, "monthly_archive", "FAIL", str(exc))
        raise SystemExit(f"Pipeline stopped: archive creation failed — {exc}")


# ---------------------------------------------------------------------------
# Step 6 — Workbook update (Raw Material + Consumption only)
# ---------------------------------------------------------------------------

def step6_update_workbook(
    workbook_path:        Path,
    inventory_detail_df:  pd.DataFrame,
    consumption_detail_df:pd.DataFrame,
    period_label:         str,
    result:               PipelineResult,
) -> None:
    """Step 6: Populate Raw Material and Consumption sheets.

    Market Inputs and Strategy sheet formulas are NOT touched.
    """
    from update_strategy_workbook import (
        load_workbook_for_update,
        update_raw_material_sheet,
        update_consumption_sheet,
        save_workbook,
    )

    sub_report: list[dict] = []
    try:
        wb = load_workbook_for_update(workbook_path)
        update_raw_material_sheet(wb, inventory_detail_df, sub_report)
        update_consumption_sheet(wb, consumption_detail_df, period_label, sub_report)
        save_workbook(wb, workbook_path, sub_report)
    except SystemExit:
        raise
    except Exception as exc:
        result.record(6, "update_workbook", "FAIL", str(exc))
        raise SystemExit(f"Pipeline stopped: workbook update failed — {exc}")

    fails = [r for r in sub_report if r["status"] == "FAIL"]
    warns = [r for r in sub_report if r["status"] == "WARN"]
    if fails:
        detail = "; ".join(r["detail"][:60] for r in fails[:3])
        result.record(6, "update_workbook", "FAIL",
                      f"{len(fails)} FAIL(s) in workbook update: {detail}")
        raise SystemExit("Pipeline stopped: workbook update produced FAIL checks.")
    elif warns:
        detail = "; ".join(r["detail"][:60] for r in warns[:2])
        result.record(6, "update_workbook", "WARN",
                      f"Workbook updated with {len(warns)} warning(s): {detail}")
    else:
        result.record(6, "update_workbook", "PASS",
                      f"Raw Material and Consumption sheets updated successfully.")


# ---------------------------------------------------------------------------
# Step 7 — Procurement engine
# ---------------------------------------------------------------------------

def step7_run_procurement_engine(
    inventory_detail_df:  pd.DataFrame,
    consumption_detail_df:pd.DataFrame,
    period_days:          int,
    result:               PipelineResult,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Step 7: Compute procurement recommendations.

    Converts detail DataFrames to the long-format expected by the engine.

    Returns:
        (strategy_df, engine_validation_df)
    """
    from procurement_engine import run as engine_run, wide_to_long
    from clean_inventory import clean_inventory as _dummy  # ensure path set

    # Aggregate inventory detail → long format (org_name, category, inventory_qty)
    inv_long = (
        inventory_detail_df
        .groupby(["org_name", "category"], as_index=False)
        .agg(inventory_qty=("primary_qty", "sum"))
    )

    # Aggregate consumption detail → long format (org_name, category, net_consumption)
    # Net consumption = -(sum of primary_qty) because Oracle signs: issues < 0, returns > 0
    cons_agg = (
        consumption_detail_df
        .groupby(["org_name", "category"], as_index=False)
        .agg(raw_sum=("primary_qty", "sum"))
    )
    cons_agg["net_consumption"] = -cons_agg["raw_sum"]
    cons_long = cons_agg[["org_name", "category", "net_consumption"]].copy()
    cons_long = cons_long.rename(columns={"category": "category"})

    try:
        strategy_df, engine_validation_df = engine_run(
            inventory_df=inv_long,
            consumption_df=cons_long,
            period_days=period_days,
        )
    except Exception as exc:
        result.record(7, "procurement_engine", "FAIL", str(exc))
        raise SystemExit(f"Pipeline stopped: procurement_engine failed — {exc}")

    n_buy     = (strategy_df["action"] == "BUY").sum()
    n_hold    = (strategy_df["action"] == "HOLD").sum()
    n_monitor = (strategy_df["action"] == "MONITOR").sum()
    engine_fails = (engine_validation_df["status"] == "FAIL").sum()

    status = "FAIL" if engine_fails > 0 else "PASS"
    if engine_fails > 0:
        fail_detail = "; ".join(
            engine_validation_df[engine_validation_df["status"] == "FAIL"]["detail"].tolist()[:2]
        )
        result.record(7, "procurement_engine", "FAIL",
                      f"Engine validation: {engine_fails} FAIL(s). {fail_detail}")
        raise SystemExit("Pipeline stopped: procurement engine produced validation failures.")

    result.record(7, "procurement_engine", "PASS",
                  f"{len(strategy_df)} rows — BUY:{n_buy}  HOLD:{n_hold}  MONITOR:{n_monitor}")

    return strategy_df, engine_validation_df


# ---------------------------------------------------------------------------
# Step 8 — Validation report
# ---------------------------------------------------------------------------

def step8_write_report(
    result:               PipelineResult,
    inventory_detail_df:  pd.DataFrame,
    consumption_detail_df:pd.DataFrame,
    strategy_df:          pd.DataFrame,
    engine_validation_df: pd.DataFrame,
    diagnostics_df:       pd.DataFrame,
    inventory_summary_df: pd.DataFrame,
    consumption_summary_df: pd.DataFrame,
    reports_dir:          Path,
    period_days:          int,
    period_label:         str,
) -> Path:
    """Step 8: Write reports/monthly_validation_report.md."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "monthly_validation_report.md"

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    p, w, f = result.counts()
    month_name = date(result.year, result.month, 1).strftime("%B %Y")

    # Unmapped item codes
    unmapped_inv  = (inventory_detail_df["category"] == "Unmapped").sum() \
        if "category" in inventory_detail_df.columns else 0
    unmapped_cons = (consumption_detail_df["category"] == "Unmapped").sum() \
        if "category" in consumption_detail_df.columns else 0
    unmapped_items = []
    if "category" in inventory_detail_df.columns:
        unmapped_items += inventory_detail_df[
            inventory_detail_df["category"] == "Unmapped"
        ]["item_code"].unique().tolist()
    if "category" in consumption_detail_df.columns:
        unmapped_items += consumption_detail_df[
            consumption_detail_df["category"] == "Unmapped"
        ]["item_code"].unique().tolist()
    unmapped_items = sorted(set(unmapped_items))

    # Org coverage
    inv_orgs  = set(inventory_detail_df["org_name"].unique()) \
        if "org_name" in inventory_detail_df.columns else set()
    cons_orgs = set(consumption_detail_df["org_name"].unique()) \
        if "org_name" in consumption_detail_df.columns else set()
    inv_only  = sorted(inv_orgs - cons_orgs)
    cons_only = sorted(cons_orgs - inv_orgs)

    # Category coverage
    inv_cats  = set(inventory_detail_df["category"].unique()) \
        if "category" in inventory_detail_df.columns else set()
    cons_cats = set(consumption_detail_df["category"].unique()) \
        if "category" in consumption_detail_df.columns else set()

    # Procurement summary
    buy_rows = strategy_df[strategy_df["action"] == "BUY"] if not strategy_df.empty else pd.DataFrame()
    total_procurement_kgs = buy_rows["procurement_qty"].sum() if not buy_rows.empty else 0

    # Build report lines
    lines = [
        f"# Monthly Validation Report — {month_name}",
        f"",
        f"**Generated:** {now}  ",
        f"**Reporting period:** {period_label}  ",
        f"**Period days:** {period_days}  ",
        f"**Overall:** {p} PASS · {w} WARN · {f} FAIL",
        f"",
        f"---",
        f"",
        f"## Pipeline Steps",
        f"",
        f"| Step | Name | Status | Detail |",
        f"|------|------|--------|--------|",
    ]
    for s in result.steps:
        icon = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗"}.get(s["status"], " ")
        lines.append(
            f"| {s['step']} | {s['name']} | {icon} {s['status']} | {s['detail'][:90]} |"
        )

    lines += [
        f"",
        f"---",
        f"",
        f"## Data Summary",
        f"",
        f"### Inventory",
        f"",
        f"- **Rows processed:** {len(inventory_detail_df):,}",
        f"- **Organisations:** {len(inv_orgs)}",
        f"- **Unmapped item codes:** {unmapped_inv}",
        f"",
    ]

    if not inventory_summary_df.empty:
        lines.append("**Inventory by Org (summary):**")
        lines.append("")
        lines.append(inventory_summary_df.to_markdown(index=False))
        lines.append("")

    lines += [
        f"",
        f"### Consumption",
        f"",
        f"- **Transactions processed:** {len(consumption_detail_df):,}",
        f"- **Organisations:** {len(cons_orgs)}",
        f"- **Unmapped item codes:** {unmapped_cons}",
    ]

    # Return rows
    pos_rows = (consumption_detail_df["primary_qty"] > 0).sum() \
        if "primary_qty" in consumption_detail_df.columns else 0
    lines.append(f"- **Return rows (positive qty):** {pos_rows}")
    lines.append("")

    if not consumption_summary_df.empty:
        lines.append("**Net Consumption by Org (summary):**")
        lines.append("")
        lines.append(consumption_summary_df.to_markdown(index=False))
        lines.append("")

    # Transaction diagnostics
    if not diagnostics_df.empty:
        lines.append("**Transaction Type Diagnostics:**")
        lines.append("")
        lines.append(diagnostics_df.to_markdown(index=False))
        lines.append("")

    # Org coverage
    lines += [
        f"---",
        f"",
        f"## Coverage Checks",
        f"",
        f"### Org Alignment",
        f"",
    ]
    if inv_only:
        lines.append(f"⚠ **In inventory only (no consumption):** {', '.join(inv_only)}")
    else:
        lines.append("✓ No orgs appear in inventory only.")
    if cons_only:
        lines.append(f"⚠ **In consumption only (no inventory):** {', '.join(cons_only)}")
    else:
        lines.append("✓ No orgs appear in consumption only.")
    lines.append("")

    lines += [
        f"### Category Coverage",
        f"",
        f"- Inventory categories: `{', '.join(sorted(inv_cats))}`",
        f"- Consumption categories: `{', '.join(sorted(cons_cats))}`",
        f"",
    ]

    if unmapped_items:
        lines.append(f"⚠ **Unmapped item codes ({len(unmapped_items)}):**")
        lines.append("")
        for code in unmapped_items[:20]:
            lines.append(f"  - `{code}`")
        if len(unmapped_items) > 20:
            lines.append(f"  - ... and {len(unmapped_items) - 20} more")
        lines.append("")

    # Procurement recommendations
    lines += [
        f"---",
        f"",
        f"## Procurement Recommendations",
        f"",
    ]
    if not strategy_df.empty:
        n_buy     = (strategy_df["action"] == "BUY").sum()
        n_hold    = (strategy_df["action"] == "HOLD").sum()
        n_monitor = (strategy_df["action"] == "MONITOR").sum()
        lines += [
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total org-commodity pairs | {len(strategy_df)} |",
            f"| BUY (shortfall > 0) | {n_buy} |",
            f"| HOLD (stock adequate) | {n_hold} |",
            f"| MONITOR (no consumption data) | {n_monitor} |",
            f"| Total procurement quantity (Kgs) | {total_procurement_kgs:,.0f} |",
            f"",
        ]

        buy_table = strategy_df[strategy_df["action"] == "BUY"][
            ["org_name", "commodity", "inventory_qty", "monthly_consumption",
             "need_45_days", "shortfall", "days_cover", "confidence"]
        ]
        if not buy_table.empty:
            lines.append("**BUY Recommendations:**")
            lines.append("")
            lines.append(buy_table.to_markdown(index=False))
            lines.append("")

    # Engine validation
    lines += [
        f"---",
        f"",
        f"## Engine Validation Checks",
        f"",
    ]
    if not engine_validation_df.empty:
        lines.append(engine_validation_df.to_markdown(index=False))
    lines.append("")

    # Write markdown report
    report_path.write_text("\n".join(lines), encoding="utf-8")
    result.record(8, "write_report", "PASS",
                  f"Report written: {report_path.relative_to(PROJECT_ROOT)}")

    # Write strategy CSV and metadata JSON for Streamlit dashboard consumption.
    # Both files are written together or neither is — this prevents a stale CSV
    # from surfacing under a new period label when strategy_df is empty.
    import json as _json

    csv_path  = reports_dir / "procurement_strategy.csv"
    meta_path = reports_dir / "procurement_strategy_meta.json"
    meta = {
        "period_label": period_label,
        "year": result.year,
        "month": result.month,
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "period_days": period_days,
        "total_rows": len(strategy_df),
    }

    if not strategy_df.empty:
        strategy_df.to_csv(csv_path, index=False)
        meta_path.write_text(_json.dumps(meta, indent=2), encoding="utf-8")
        result.record(8, "write_strategy_csv", "PASS",
                      f"Strategy CSV: {csv_path.relative_to(PROJECT_ROOT)} ({len(strategy_df)} rows)")
    else:
        result.record(8, "write_strategy_csv", "WARN",
                      "strategy_df is empty — CSV and meta not written to avoid stale dashboard data")

    return report_path


# ---------------------------------------------------------------------------
# Main pipeline orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(
    oracle_dir:       Path,
    workbook_path:    Path,
    archive_dir:      Path,
    reports_dir:      Path,
    mapping_csv:      Optional[Path],
    overrides_csv:    Optional[Path],
    year:             int,
    month:            int,
    skip_extraction:  bool,
) -> PipelineResult:
    """Execute the full 8-step pipeline.

    Args:
        oracle_dir:      Directory where extraction scripts drop Oracle files.
        workbook_path:   Path to Strategies.xlsx.
        archive_dir:     Archive directory for monthly snapshots.
        reports_dir:     Directory for validation report output.
        mapping_csv:     Optional path to commodity_mapping.csv.
        overrides_csv:   Optional path to item_code_overrides.csv.
        year:            Reporting year.
        month:           Reporting month (1–12).
        skip_extraction: If True, skip Steps 1–2 and use existing Oracle files.

    Returns:
        PipelineResult with all step records.
    """
    period_days  = calendar.monthrange(year, month)[1]
    month_label  = date(year, month, 1).strftime("%B %Y")
    first_day    = date(year, month, 1).strftime("%d-%b")
    last_day     = date(year, month, period_days).strftime("%d-%b")
    period_label = f"{month_label} ({first_day} to {last_day})"

    filter_month = f"{year:04d}-{month:02d}"

    result = PipelineResult(year, month)

    logger.info("=" * 60)
    logger.info("MG Apparel Monthly Strategy Pipeline")
    logger.info("Period : %s  (%d days)", period_label, period_days)
    logger.info("Workbook: %s", workbook_path)
    logger.info("=" * 60)

    # ── Step 1: Oracle inventory extraction ───────────────────────────────
    inventory_xlsx = step1_extract_inventory(oracle_dir, result, skip_extraction)

    # ── Step 2: Oracle consumption extraction ─────────────────────────────
    consumption_xlsx = step2_extract_consumption(oracle_dir, result, skip_extraction)

    # ── Step 3: Clean inventory ────────────────────────────────────────────
    inventory_detail_df, inventory_summary_df = step3_clean_inventory(
        inventory_xlsx, mapping_csv, overrides_csv, result
    )

    # ── Step 4: Clean consumption ──────────────────────────────────────────
    consumption_detail_df, consumption_summary_df, diagnostics_df = step4_clean_consumption(
        consumption_xlsx, mapping_csv, filter_month, result
    )

    # ── Step 5: Monthly archive (BEFORE workbook modification) ────────────
    step5_create_archive(workbook_path, archive_dir, year, month, result)

    # ── Step 6: Update workbook ────────────────────────────────────────────
    step6_update_workbook(
        workbook_path,
        inventory_detail_df,
        consumption_detail_df,
        period_label,
        result,
    )

    # ── Step 7: Procurement engine ─────────────────────────────────────────
    strategy_df, engine_validation_df = step7_run_procurement_engine(
        inventory_detail_df,
        consumption_detail_df,
        period_days,
        result,
    )

    # ── Step 8: Validation report ──────────────────────────────────────────
    report_path = step8_write_report(
        result=result,
        inventory_detail_df=inventory_detail_df,
        consumption_detail_df=consumption_detail_df,
        strategy_df=strategy_df,
        engine_validation_df=engine_validation_df,
        diagnostics_df=diagnostics_df,
        inventory_summary_df=inventory_summary_df,
        consumption_summary_df=consumption_summary_df,
        reports_dir=reports_dir,
        period_days=period_days,
        period_label=period_label,
    )

    # ── Summary ────────────────────────────────────────────────────────────
    p, w, f = result.counts()
    elapsed  = (datetime.now() - result.start_time).total_seconds()
    logger.info("=" * 60)
    logger.info("Pipeline complete in %.0fs — PASS:%d  WARN:%d  FAIL:%d", elapsed, p, w, f)
    logger.info("Report: %s", report_path)
    logger.info("=" * 60)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _prev_month() -> tuple[int, int]:
    """Return (year, month) for the previous calendar month."""
    today = date.today()
    if today.month == 1:
        return today.year - 1, 12
    return today.year, today.month - 1


def main() -> None:
    prev_year, prev_month = _prev_month()

    parser = argparse.ArgumentParser(
        description="MG Apparel — Monthly Strategy Pipeline (Phase 3C)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              # Full pipeline — previous month, default paths
              python scripts/run_monthly_strategy_pipeline.py

              # Specific month
              python scripts/run_monthly_strategy_pipeline.py --year 2026 --month 4

              # Skip Oracle extraction (files already downloaded)
              python scripts/run_monthly_strategy_pipeline.py --skip-extraction

              # Full explicit paths
              python scripts/run_monthly_strategy_pipeline.py \\
                  --oracle-dir   "D:\\path" \\
                  --workbook     "data/strategy/Strategies.xlsx" \\
                  --archive-dir  "data/strategy/archive" \\
                  --reports-dir  "reports" \\
                  --year 2026 --month 4
        """),
    )

    parser.add_argument(
        "--oracle-dir",
        default=str(DEFAULT_ORACLE_DIR),
        dest="oracle_dir",
        help=f"Directory where Oracle extraction scripts download files. "
             f"Must match base_directory in inventory_data.py / consumption_data.py. "
             f"Default: {DEFAULT_ORACLE_DIR}",
    )
    parser.add_argument(
        "--workbook",
        default=str(DEFAULT_WORKBOOK),
        help=f"Path to Strategies.xlsx. Default: {DEFAULT_WORKBOOK.relative_to(PROJECT_ROOT)}",
    )
    parser.add_argument(
        "--archive-dir",
        default=str(DEFAULT_ARCHIVE_DIR),
        dest="archive_dir",
        help=f"Archive directory. Default: {DEFAULT_ARCHIVE_DIR.relative_to(PROJECT_ROOT)}",
    )
    parser.add_argument(
        "--reports-dir",
        default=str(DEFAULT_REPORTS_DIR),
        dest="reports_dir",
        help=f"Reports output directory. Default: {DEFAULT_REPORTS_DIR.relative_to(PROJECT_ROOT)}",
    )
    parser.add_argument(
        "--mapping-csv",
        default=str(DEFAULT_MAPPING_CSV) if DEFAULT_MAPPING_CSV.exists() else None,
        dest="mapping_csv",
        help="Path to commodity_mapping.csv (optional).",
    )
    parser.add_argument(
        "--overrides-csv",
        default=str(DEFAULT_OVERRIDES_CSV) if DEFAULT_OVERRIDES_CSV.exists() else None,
        dest="overrides_csv",
        help="Path to item_code_overrides.csv (optional).",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=prev_year,
        help=f"Reporting year. Default: {prev_year} (previous month).",
    )
    parser.add_argument(
        "--month",
        type=int,
        default=prev_month,
        choices=range(1, 13),
        metavar="MONTH",
        help=f"Reporting month (1–12). Default: {prev_month} (previous month).",
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        dest="skip_extraction",
        help="Skip Steps 1–2 (Oracle Selenium). Use existing downloaded files.",
    )

    args = parser.parse_args()

    result = run_pipeline(
        oracle_dir       = Path(args.oracle_dir),
        workbook_path    = Path(args.workbook),
        archive_dir      = Path(args.archive_dir),
        reports_dir      = Path(args.reports_dir),
        mapping_csv      = Path(args.mapping_csv) if args.mapping_csv else None,
        overrides_csv    = Path(args.overrides_csv) if args.overrides_csv else None,
        year             = args.year,
        month            = args.month,
        skip_extraction  = args.skip_extraction,
    )

    p, w, f = result.counts()
    sys.exit(0 if f == 0 else 1)


if __name__ == "__main__":
    main()
