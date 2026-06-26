"""
procurement_orchestrator.py
-----------------------------
PSE-3C — Procurement Strategy Orchestration Layer.

Wires together the three independently-correct, unit-agnostic upstream
modules:

    clean_inventory.py        (Kg, unmodified, untouched by this module)
    origin_classifier.py      (Kg in, Kg out -- "qty_kg", unmodified by this module)

...into the one module that has an absolute unit requirement:

    procurement_strategy_engine.py   (requires TONS -- hardcoded constants
                                       110 t/day, 1,733 / 6,958 t reorder
                                       points, 2,750 t safety stock, all
                                       approved in PSE-2 / PSE-2.7)

THE KG -> TONS CONVERSION HAPPENS EXACTLY ONCE, IN THIS FILE, IN
convert_kg_to_tons() BELOW. No other module in the pipeline performs any
unit conversion. This is the approved outcome of the PSE-3B.5 Unit
Consistency Audit:

    Root cause: Oracle exports both inventory and consumption in Kg
                ("All quantities in Kgs" -- Strategies.xlsx Raw Material
                sheet row 2; "Primary Qty (Kgs)" -- both Raw Material and
                Consumption sheet headers). clean_inventory.py,
                clean_consumption.py, origin_classifier.py, and the
                existing procurement_engine.py are all correctly
                unit-agnostic pass-throughs -- procurement_engine.py never
                breaks because its only constant (MIN_COVER_DAYS=45) is a
                pure day-count, not an absolute mass value.
                procurement_strategy_engine.py (PSE-3B) is the first module
                with absolute ton-denominated constants, so it is the first
                module that requires a conversion to exist anywhere in the
                pipeline.

Per the audit's Option C recommendation: conversion occurs in the
orchestration layer, not inside clean_inventory.py, origin_classifier.py,
or procurement_strategy_engine.py's business logic.

DOES NOT MODIFY (per explicit instruction, carried from PSE-3A through
PSE-3C):
    clean_inventory.py, clean_consumption.py, inventory_data.py,
    consumption_data.py, update_strategy_workbook.py, procurement_engine.py

Usage:
    python procurement_orchestrator.py --workbook "data/strategy/Strategies.xlsx"
    python procurement_orchestrator.py --input "D:/.../MG_STOCK_TILL_DATE_MULTIPLE_UNIT.xlsx"
    python procurement_orchestrator.py --workbook "..." --strict   (exit 2 on sanity warnings)
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from origin_classifier import aggregate_by_origin, classify_dataframe, load_overrides_csv
from procurement_strategy_engine import build_strategy_output_v2
from validate_origin_classification import (
    load_detail_from_oracle_export,
    load_detail_from_workbook,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# THE single conversion factor used anywhere in this pipeline.
KG_PER_TON = 1000.0

# Confirmed PSE-1 business rule -- used only for the sanity guard in Part 3,
# never for the conversion itself.
MAX_STORAGE_CAPACITY_TONS = 45_000.0

# Engineering defaults for the sanity guard (not confirmed business rules --
# same status as WATCH_BUFFER_PCT / MIX_TOLERANCE_PCT_POINTS in PSE-3B).
SANITY_CAPACITY_MULTIPLIER = 3.0          # >3x capacity -> strong unit-mismatch signal
SANITY_DAYS_COVER_CEILING = 1000.0        # >1000 days cover -> implausible
CONVERSION_TOLERANCE_TONS = 0.01          # absolute tolerance for the Part 2 check

WARN = "WARNING"
NOTE = "NOTE"


# ===========================================================================
# PART 1 -- ORCHESTRATION: LOAD -> CLASSIFY -> AGGREGATE -> CONVERT
# ===========================================================================

def load_inventory(
    input_path: Optional[str] = None,
    workbook_path: Optional[str] = None,
    mapping_csv: Optional[str] = None,
) -> pd.DataFrame:
    """Step 1 -- Load inventory detail_df via the existing, unmodified pipeline.

    Exactly one of input_path / workbook_path must be given. Reuses
    validate_origin_classification.py's loaders rather than duplicating
    them -- this orchestrator adds no new Oracle/Excel-reading logic.

    Returns:
        detail_df with columns: item_code, description, org_name,
        primary_qty (Kg, unconverted), subinventory_code, category.
    """
    if not input_path and not workbook_path:
        raise ValueError("Provide either input_path (raw Oracle export) or workbook_path.")
    if input_path:
        return load_detail_from_oracle_export(input_path, mapping_csv)
    return load_detail_from_workbook(workbook_path)


def classify_and_aggregate(
    detail_df: pd.DataFrame,
    overrides: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """Steps 2-3 -- Run origin classification and aggregate by org x origin.

    Returns:
        DataFrame with columns: org_name, origin, qty_kg
        (Kg -- origin_classifier.py never converts units; see its docstring.)
    """
    classified_df = classify_dataframe(detail_df, overrides=overrides)
    origin_summary_kg = aggregate_by_origin(classified_df)
    return origin_summary_kg


def convert_kg_to_tons(origin_summary_kg_df: pd.DataFrame) -> pd.DataFrame:
    """Step 4 -- THE single Kg -> Tons conversion point for this entire pipeline.

    This is the ONLY function, in the ONLY file, in the ONLY pipeline run,
    that divides a quantity by KG_PER_TON. If a unit bug is ever suspected
    again, this function is the one place to check.

    Args:
        origin_summary_kg_df: Output of classify_and_aggregate() --
                               columns org_name, origin, qty_kg.

    Returns:
        Same DataFrame with both columns retained for auditability:
            org_name, origin, qty_kg, tons
        'tons' = qty_kg / 1000.0  -- visible, traceable, never silent.
    """
    required = {"org_name", "origin", "qty_kg"}
    missing = required - set(origin_summary_kg_df.columns)
    if missing:
        raise ValueError(
            f"origin_summary_kg_df is missing required columns: {sorted(missing)}. "
            "Expected output of origin_classifier.aggregate_by_origin()."
        )

    df = origin_summary_kg_df.copy()
    # --- THE CONVERSION ---
    df["tons"] = df["qty_kg"] / KG_PER_TON
    # --- END CONVERSION ---
    return df


# ===========================================================================
# PART 2 -- DATA VALIDATION (conversion correctness)
# ===========================================================================

def validate_conversion(origin_summary_converted_df: pd.DataFrame) -> dict:
    """Verify Kg total / 1000 == Tons total, within tolerance, for the
    overall total and for each origin bucket (LOCAL / IMPORTED / UNKNOWN).

    This validates that convert_kg_to_tons() did exactly what it claims --
    a regression guard against the conversion being accidentally duplicated,
    skipped for a subset of rows, or applied to the wrong column.

    Returns:
        dict with keys:
            checks            -- list of {bucket, kg_total, tons_total,
                                  expected_tons, diff_tons, passed}
            all_passed         -- bool
    """
    checks = []

    def _check(label: str, kg_total: float, tons_total: float) -> dict:
        expected_tons = kg_total / KG_PER_TON
        diff = abs(tons_total - expected_tons)
        passed = diff <= CONVERSION_TOLERANCE_TONS
        return {
            "bucket": label,
            "kg_total": round(kg_total, 4),
            "tons_total": round(tons_total, 4),
            "expected_tons": round(expected_tons, 4),
            "diff_tons": round(diff, 6),
            "passed": passed,
        }

    overall_kg = origin_summary_converted_df["qty_kg"].sum()
    overall_tons = origin_summary_converted_df["tons"].sum()
    checks.append(_check("TOTAL", overall_kg, overall_tons))

    for origin in ("LOCAL", "IMPORTED", "UNKNOWN"):
        subset = origin_summary_converted_df[origin_summary_converted_df["origin"] == origin]
        kg_total = subset["qty_kg"].sum() if not subset.empty else 0.0
        tons_total = subset["tons"].sum() if not subset.empty else 0.0
        checks.append(_check(origin, kg_total, tons_total))

    return {
        "checks": checks,
        "all_passed": all(c["passed"] for c in checks),
    }


# ===========================================================================
# PART 3 -- SANITY GUARDS (defensive, against accidental unit mismatch)
# ===========================================================================

def run_sanity_guards(position: dict) -> list[dict]:
    """Defensive checks against an accidental Kg/Ton mismatch slipping through.

    These guards do NOT perform or repeat the conversion -- they only ask
    "does the converted result look physically plausible?" against the
    confirmed 45,000-ton storage capacity and the 110 t/day consumption
    rate. A result ~1000x too large (the exact signature of a missed Kg->t
    conversion) will be caught here even if validate_conversion() above
    passed (that function only checks internal arithmetic consistency, not
    real-world plausibility).

    Per PSE-3C instructions: warnings are raised (via warnings.warn AND
    returned in the result list) but execution is not silently continued --
    every warning is surfaced in the printed report and the CLI exits with
    a distinct non-zero code when any fire (see --strict).

    Returns:
        list of {level, code, message} dicts, empty if all checks pass.
    """
    warnings_out: list[dict] = []
    total_tons = position["total_inventory_tons"]
    total_days_cover = position["total_days_cover"]

    hard_ceiling = MAX_STORAGE_CAPACITY_TONS * SANITY_CAPACITY_MULTIPLIER
    if total_tons > hard_ceiling:
        msg = (
            f"Total inventory ({total_tons:,.1f} tons) exceeds {SANITY_CAPACITY_MULTIPLIER:.0f}x "
            f"the confirmed {MAX_STORAGE_CAPACITY_TONS:,.0f}-ton storage capacity. "
            "This is the exact signature of an unconverted Kg value reaching the engine. "
            "Verify convert_kg_to_tons() was applied before trusting this output."
        )
        warnings.warn(msg, stacklevel=2)
        warnings_out.append({"level": WARN, "code": "CAPACITY_3X_EXCEEDED", "message": msg})
    elif total_tons > MAX_STORAGE_CAPACITY_TONS:
        msg = (
            f"Total inventory ({total_tons:,.1f} tons) exceeds the {MAX_STORAGE_CAPACITY_TONS:,.0f}-ton "
            "storage capacity ceiling. Confirm this reflects genuine over-capacity stock, "
            "not a unit error."
        )
        warnings_out.append({"level": NOTE, "code": "CAPACITY_EXCEEDED", "message": msg})

    if total_days_cover > SANITY_DAYS_COVER_CEILING:
        msg = (
            f"Total days cover ({total_days_cover:,.1f} days, "
            f"~{total_days_cover/365:.1f} years) exceeds the {SANITY_DAYS_COVER_CEILING:.0f}-day "
            "sanity ceiling at 110 tons/day consumption. Implausible for working inventory -- "
            "likely an unconverted Kg value."
        )
        warnings.warn(msg, stacklevel=2)
        warnings_out.append({"level": WARN, "code": "DAYS_COVER_IMPLAUSIBLE", "message": msg})

    if 0 < total_tons < 1.0:
        msg = (
            f"Total inventory ({total_tons:.4f} tons) is implausibly small. "
            "Possible double-conversion (Kg divided by 1000 twice) or empty/wrong source data."
        )
        warnings.warn(msg, stacklevel=2)
        warnings_out.append({"level": WARN, "code": "INVENTORY_IMPLAUSIBLY_SMALL", "message": msg})

    return warnings_out


# ===========================================================================
# PART 4 -- END-TO-END EXECUTION
# ===========================================================================

def run_orchestration(
    input_path: Optional[str] = None,
    workbook_path: Optional[str] = None,
    mapping_csv: Optional[str] = None,
    overrides_csv: Optional[str] = None,
) -> dict:
    """Full pipeline: Inventory -> Origin Classification -> Conversion ->
    Procurement Strategy Engine.

    Returns:
        dict with keys:
            detail_df, origin_summary_kg, origin_summary_converted,
            conversion_validation, sanity_warnings, strategy_output (dict)
    """
    overrides = load_overrides_csv(overrides_csv) if overrides_csv else None

    # Step 1
    detail_df = load_inventory(input_path, workbook_path, mapping_csv)

    # Steps 2-3
    origin_summary_kg = classify_and_aggregate(detail_df, overrides=overrides)

    # Step 4 -- THE conversion
    origin_summary_converted = convert_kg_to_tons(origin_summary_kg)

    # Part 2 -- validate the conversion arithmetic
    conversion_validation = validate_conversion(origin_summary_converted)

    # Engine input: org_name, origin, tons (drop qty_kg -- engine contract is tons-only)
    engine_input = origin_summary_converted[["org_name", "origin", "tons"]]

    strategy_output = build_strategy_output_v2(engine_input)

    # Part 3 -- sanity guards against the converted position
    position_dict = {
        "total_inventory_tons": strategy_output.total_inventory_tons,
        "total_days_cover": strategy_output.total_days_cover,
    }
    sanity_warnings = run_sanity_guards(position_dict)

    return {
        "detail_df": detail_df,
        "origin_summary_kg": origin_summary_kg,
        "origin_summary_converted": origin_summary_converted,
        "conversion_validation": conversion_validation,
        "sanity_warnings": sanity_warnings,
        "strategy_output": strategy_output,
    }


def print_report(result: dict) -> None:
    so = result["strategy_output"]
    cv = result["conversion_validation"]
    sw = result["sanity_warnings"]

    print("=" * 78)
    print("PSE-3C ORCHESTRATION -- END-TO-END RUN REPORT")
    print("=" * 78)

    print("\n--- PART 2: CONVERSION VALIDATION (Kg total / 1000 == Tons total) ---")
    for c in cv["checks"]:
        status = "PASS" if c["passed"] else "FAIL"
        print(f"  [{status}] {c['bucket']:8s}  kg={c['kg_total']:>14,.1f}  "
              f"tons={c['tons_total']:>10,.4f}  expected={c['expected_tons']:>10,.4f}  "
              f"diff={c['diff_tons']:.6f}")
    print(f"  Overall: {'ALL PASSED' if cv['all_passed'] else 'FAILED -- DO NOT TRUST OUTPUT'}")

    print("\n--- PART 3: SANITY GUARDS ---")
    if not sw:
        print("  No warnings. Inventory position is within plausible bounds.")
    else:
        for w in sw:
            print(f"  [{w['level']}] {w['code']}: {w['message']}")

    print("\n--- PART 4: END-TO-END RESULT (real production data) ---")
    print(f"  1. Local Inventory      : {so.local_inventory_tons:>12,.1f} tons")
    print(f"  2. Imported Inventory   : {so.imported_inventory_tons:>12,.1f} tons")
    print(f"  3. Total Inventory      : {so.total_inventory_tons:>12,.1f} tons "
          f"(includes {so.unknown_inventory_tons:,.1f} unclassified)")
    print(f"  4. Local Mix %          : {so.local_mix_pct:>12.2f}%")
    print(f"  5. Imported Mix %       : {so.imported_mix_pct:>12.2f}%")
    print(f"  6. Action Recommendation: {so.action}")
    print(f"     Reason               : {so.reason}")
    print(f"  7. Reorder Status       : local={so.local_status}  imported={so.imported_status}")
    print(f"     Local days cover     : {so.local_days_cover:.1f} days")
    print(f"     Imported days cover  : {so.imported_days_cover:.1f} days")
    print("=" * 78)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PSE-3C orchestration: Inventory -> Origin -> Kg->Tons -> Strategy Engine"
    )
    parser.add_argument("--input", default=None, help="Raw Oracle export (.xlsx)")
    parser.add_argument("--workbook", default=None, help="Strategies.xlsx (Raw Material sheet)")
    parser.add_argument("--mapping-csv", default=None, dest="mapping_csv")
    parser.add_argument("--overrides", default=None, dest="overrides_csv")
    parser.add_argument("--strict", action="store_true",
                         help="Exit with code 2 if any sanity warning fires")
    args = parser.parse_args()

    result = run_orchestration(
        input_path=args.input,
        workbook_path=args.workbook,
        mapping_csv=args.mapping_csv,
        overrides_csv=args.overrides_csv,
    )
    print_report(result)

    if not result["conversion_validation"]["all_passed"]:
        sys.exit(1)
    if args.strict and result["sanity_warnings"]:
        sys.exit(2)
    sys.exit(0)
