"""
origin_classifier.py
---------------------
Classifies Cotton inventory rows into LOCAL / IMPORTED / UNKNOWN origin
based on the Oracle item Description field.

Runs downstream of clean_inventory.py's detail_df (NOT summary_df --
summary_df has already pivoted away the description column and cannot be
used for origin classification).

Matching rule (deliberately conservative -- "do not guess"):
    - Search the description for the whole word "LOCAL" and the whole word
      "IMPORTED" (case-insensitive).
    - Exactly one of the two found  -> classified accordingly.
    - Both found                    -> AMBIGUOUS, returned as UNKNOWN.
    - Neither found                 -> UNKNOWN.
    - An optional item_code override CSV takes priority over text matching,
      for resolving descriptions that carry no LOCAL/IMPORTED marker at all
      (e.g. "TURKISH TURKISH RAW COTTON REGENAGRI").

Scope: only rows with category == "Cotton" are classified. Fiber, Stretch
Fiber, Viscose, and Cotton Waste are out of scope -- the 45/55 local/
imported mix policy (confirmed PSE-1 business rule) applies to Cotton only.
Out-of-scope rows get origin=None, distinct from UNKNOWN (which means
"Cotton, but could not be classified").

This module does not modify, import internals from, or otherwise alter
clean_inventory.py, clean_consumption.py, or run_forecasts.py. It consumes
clean_inventory.py's public detail_df output only.

Usage:
    from origin_classifier import classify_origin, classify_dataframe

    classify_origin("PAK-LOCAL")                        # -> "LOCAL"
    classify_origin("SUDAN-IMPORTED")                    # -> "IMPORTED"
    classify_origin("TURKISH RAW COTTON REGENAGRI")      # -> "UNKNOWN"

    classified_df = classify_dataframe(detail_df)        # adds 'origin' column
    origin_summary = aggregate_by_origin(classified_df)  # org_name x origin totals

CLI:
    python origin_classifier.py --input "D:/path/MG_STOCK_TILL_DATE_MULTIPLE_UNIT.xlsx"
    python origin_classifier.py --input "..." --overrides scripts/origin_overrides.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOCAL = "LOCAL"
IMPORTED = "IMPORTED"
UNKNOWN = "UNKNOWN"

# Whole-word matches only -- "LOCALITY" or "IMPORTEDLY" must not match.
_LOCAL_RE = re.compile(r"\bLOCAL\b", re.IGNORECASE)
_IMPORTED_RE = re.compile(r"\bIMPORTED\b", re.IGNORECASE)

# Origin classification scope -- confirmed PSE-1 business rule: the 45/55
# local/imported mix target applies to Cotton only.
IN_SCOPE_CATEGORY = "Cotton"

REQUIRED_DETAIL_COLUMNS = {"item_code", "description", "category", "org_name", "primary_qty"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_str(value: object) -> str:
    """Coerce any value to a stripped string; None/NaN -> empty string."""
    if value is None:
        return ""
    try:
        if isinstance(value, float) and math.isnan(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


# ---------------------------------------------------------------------------
# Core classification
# ---------------------------------------------------------------------------

def classify_origin(
    description: object,
    item_code: object = None,
    overrides: Optional[dict[str, str]] = None,
) -> str:
    """Classify a single description string into LOCAL / IMPORTED / UNKNOWN.

    Args:
        description: Oracle item description text.
        item_code:   Optional item code, checked against `overrides` first.
        overrides:   Optional {item_code: origin} dict for manually resolved
                     items whose description carries no LOCAL/IMPORTED text.

    Returns:
        "LOCAL", "IMPORTED", or "UNKNOWN". Never guesses: a description
        containing both markers, or neither, resolves to UNKNOWN.
    """
    if overrides and item_code is not None:
        code = _clean_str(item_code).upper()
        if code in overrides:
            return overrides[code]

    desc = _clean_str(description).upper()
    has_local = bool(_LOCAL_RE.search(desc))
    has_imported = bool(_IMPORTED_RE.search(desc))

    if has_local and has_imported:
        return UNKNOWN  # ambiguous -- do not guess
    if has_local:
        return LOCAL
    if has_imported:
        return IMPORTED
    return UNKNOWN


def load_overrides_csv(csv_path: str | Path) -> dict[str, str]:
    """Load an item_code -> origin override mapping from CSV.

    CSV format:
        item_code,origin
        COT-100048,IMPORTED

    Rows with an origin value outside {LOCAL, IMPORTED, UNKNOWN} are
    rejected (logged, not raised). A missing file returns an empty dict --
    overrides are optional, never required for the classifier to run.
    """
    path = Path(csv_path)
    if not path.exists():
        return {}

    overrides: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            code = _clean_str(row.get("item_code")).upper()
            origin = _clean_str(row.get("origin")).upper()
            if not code or origin not in (LOCAL, IMPORTED, UNKNOWN):
                continue
            overrides[code] = origin
    return overrides


def classify_dataframe(
    detail_df: pd.DataFrame,
    overrides: Optional[dict[str, str]] = None,
    scope_category: str = IN_SCOPE_CATEGORY,
) -> pd.DataFrame:
    """Classify origin for every row of an inventory detail_df.

    Args:
        detail_df: Output of clean_inventory.run() -- detail_df. Must contain
                   columns: item_code, description, category, org_name,
                   primary_qty. Do NOT pass summary_df (no description column).
        overrides: Optional item_code -> origin override dict
                   (see load_overrides_csv).
        scope_category: Category that origin classification applies to.
                        Rows outside this category get origin=None -- "out
                        of scope", distinct from UNKNOWN ("in scope, but
                        unclassified").

    Returns:
        Copy of detail_df with a new 'origin' column:
            "LOCAL" | "IMPORTED" | "UNKNOWN" | None (out of scope)
    """
    missing = REQUIRED_DETAIL_COLUMNS - set(detail_df.columns)
    if missing:
        raise ValueError(
            f"detail_df is missing required columns: {sorted(missing)}. "
            "origin_classifier must run on clean_inventory.py's detail_df "
            "output, not summary_df (which has no description column)."
        )

    df = detail_df.copy()

    def _row_origin(row) -> Optional[str]:
        if row["category"] != scope_category:
            return None
        return classify_origin(row["description"], row["item_code"], overrides)

    df["origin"] = df.apply(_row_origin, axis=1)
    return df

 
def aggregate_by_origin(
    classified_df: pd.DataFrame,
    qty_col: str = "primary_qty",
) -> pd.DataFrame:
    """Aggregate classified inventory into org_name x origin totals.

    Only rows with a non-null 'origin' (in-scope Cotton rows) are included.
    Feeds the procurement strategy engine's inventory position calculation
    (PSE-3B/3C) -- but NOT directly: the output here is in whatever unit
    `qty_col` carries, which for Oracle-sourced detail_df is Kg (see
    PSE-3B.5 Unit Consistency Audit). The Kg -> tons conversion is
    deliberately NOT performed in this module -- it happens exactly once,
    visibly, in the orchestration layer (procurement_orchestrator.py),
    immediately before this output is handed to procurement_strategy_engine.

    The output column is named 'qty_kg', not 'tons', specifically so that no
    downstream consumer can mistake this for already-converted data -- a
    column literally named 'tons' that silently contained Kg was the root
    cause identified in the PSE-3B.5 audit.

    Returns:
        DataFrame with columns: org_name, origin, qty_kg
    """
    if "origin" not in classified_df.columns:
        raise ValueError(
            "classified_df has no 'origin' column. Run classify_dataframe() first."
        )
    scoped = classified_df[classified_df["origin"].notna()].copy()
    agg = (
        scoped
        .groupby(["org_name", "origin"], as_index=False)
        .agg(qty_kg=(qty_col, "sum"))
    )
    return agg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify Oracle cotton inventory into LOCAL / IMPORTED / UNKNOWN"
    )
    parser.add_argument("--input", required=True,
                         help="Path to MG_STOCK_TILL_DATE_MULTIPLE_UNIT.xlsx")
    parser.add_argument("--overrides", default=None,
                         help="Optional item_code->origin override CSV")
    parser.add_argument("--mapping-csv", default=None, dest="mapping_csv",
                         help="Optional commodity_mapping.csv passed through to clean_inventory")
    args = parser.parse_args()

    from clean_inventory import run as clean_inventory_run

    detail_df, _summary_df = clean_inventory_run(args.input, mapping_csv=args.mapping_csv)
    overrides = load_overrides_csv(args.overrides) if args.overrides else None

    classified = classify_dataframe(detail_df, overrides=overrides)
    cotton = classified[classified["category"] == IN_SCOPE_CATEGORY]

    totals = cotton.groupby("origin")["primary_qty"].sum()
    total_tons = cotton["primary_qty"].sum()

    print("\n=== ORIGIN CLASSIFICATION SUMMARY ===")
    for origin in (LOCAL, IMPORTED, UNKNOWN):
        tons = totals.get(origin, 0.0)
        pct = (tons / total_tons * 100) if total_tons else 0.0
        print(f"  {origin:10s}: {tons:>14,.1f} tons  ({pct:5.1f}%)")
    print(f"  {'TOTAL':10s}: {total_tons:>14,.1f} tons")

    coverage = (
        (totals.get(LOCAL, 0.0) + totals.get(IMPORTED, 0.0)) / total_tons * 100
        if total_tons else 0.0
    )
    print(f"\nClassification coverage: {coverage:.2f}%")
