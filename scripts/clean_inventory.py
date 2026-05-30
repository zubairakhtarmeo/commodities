"""
clean_inventory.py
------------------
Cleans and aggregates Oracle inventory extract from MG_STOCK_TILL_DATE_MULTIPLE_UNIT.

Pipeline:
    1. Load Excel → auto-detect header row (tolerant of Oracle title rows)
    2. Retain five columns: Item Code, Description, Org Name, Primary Qty,
       SubInventory Code
    3. Filter rows: SubInventory Code IN ['RM-COTTON', 'RM-FIBER']
    4. Classify each row into a material category via CommodityMapper
    5. Return two outputs:
       - detail_df:  cleaned row-level dataframe
       - summary_df: Org Name × Category totals + Total Inventory column
                     (matches Raw Material sheet structure)

Mapping source priority:
    1. --mapping-csv argument (external commodity_mapping.csv)
    2. Built-in DEFAULT_PREFIX_RULES  (fallback, no CSV required)
    3. Description regex             (last resort)

Usage:
    from clean_inventory import run

    detail, summary = run("D:/path/MG_STOCK_TILL_DATE_MULTIPLE_UNIT.xlsx")
    detail, summary = run(
        "D:/path/MG_STOCK_TILL_DATE_MULTIPLE_UNIT.xlsx",
        mapping_csv="scripts/commodity_mapping.csv",
    )
    print(summary)

CLI:
    python clean_inventory.py --input   "D:/path/MG_STOCK_TILL_DATE_MULTIPLE_UNIT.xlsx"
                              --output  "D:/path/inventory_summary.xlsx"
                              --mapping-csv "scripts/commodity_mapping.csv"
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from commodity_mapper import CommodityMapper, UNMAPPED


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Subinventory codes to keep (uppercase); extend this list to include new RM codes
KEEP_SUBINVENTORY: list[str] = ["RM-COTTON", "RM-FIBER"]

# Candidate column name variants per logical field (lowercased snake_case).
# Oracle BI Publisher may label the same column differently across report versions.
COLUMN_CANDIDATES: dict[str, list[str]] = {
    "item_code":         ["item_code", "item", "item_number", "itemcode", "item_no"],
    "description":       ["description", "item_description", "item_desc", "desc"],
    "org_name":          ["org_name", "organization_name", "organization", "org", "unit"],
    "primary_qty":       ["primary_qty", "primary_quantity", "on_hand_qty", "qty",
                          "quantity", "total_qty", "bal_qty", "balance_qty"],
    "subinventory_code": ["subinventory_code", "subinventory", "sub_inventory",
                          "sub_inv", "subinv", "subinventory_name"],
}

OUTPUT_COLS = ["item_code", "description", "org_name", "primary_qty",
               "subinventory_code", "category"]


# ---------------------------------------------------------------------------
# Header detection
# ---------------------------------------------------------------------------

def _snake(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")


def _score_header_row(raw: pd.DataFrame, row_idx: int) -> int:
    """Score a row by how many expected column keywords it contains."""
    row_text = " | ".join(_snake(v) for v in raw.iloc[row_idx].tolist())
    score = 0
    for candidates in COLUMN_CANDIDATES.values():
        if any(c in row_text for c in candidates):
            score += 1
    return score


def _detect_header_row(raw: pd.DataFrame, max_scan: int = 60) -> int:
    """Return the 0-based index of the best candidate header row."""
    best_row, best_score = 0, 0
    for i in range(min(max_scan, len(raw))):
        score = _score_header_row(raw, i)
        if score > best_score:
            best_score, best_row = score, i
    if best_score < 2:
        raise ValueError(
            f"Could not detect a header row (best score={best_score}). "
            "Verify the file is MG_STOCK_TILL_DATE_MULTIPLE_UNIT.xlsx and "
            "at least 2 expected columns are present in the first 60 rows."
        )
    return best_row


# ---------------------------------------------------------------------------
# Column resolution
# ---------------------------------------------------------------------------

def _resolve_columns(df: pd.DataFrame) -> dict[str, Optional[str]]:
    """Map each logical field name to the actual DataFrame column name (or None)."""
    normalised = {_snake(c): c for c in df.columns}
    resolved: dict[str, Optional[str]] = {}
    for field, candidates in COLUMN_CANDIDATES.items():
        resolved[field] = next((normalised[c] for c in candidates if c in normalised), None)
    return resolved


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def load_raw(xlsx_path: Path) -> pd.DataFrame:
    """Read the Excel file with no assumed header."""
    return pd.read_excel(xlsx_path, sheet_name=0, header=None, engine="openpyxl")


def clean_inventory(
    xlsx_path: Path,
    mapper: Optional[CommodityMapper] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full cleaning pipeline for the inventory extract.

    Args:
        xlsx_path: Path to MG_STOCK_TILL_DATE_MULTIPLE_UNIT.xlsx
        mapper:    CommodityMapper instance. Uses built-in rules when None.

    Returns:
        detail_df:  Row-level cleaned dataframe (OUTPUT_COLS).
        summary_df: Wide pivot — rows = Org Name, columns = Category + Total Inventory.
    """
    if mapper is None:
        mapper = CommodityMapper.default()

    xlsx_path = Path(xlsx_path)
    raw = load_raw(xlsx_path)

    header_row = _detect_header_row(raw)
    header = [_snake(v) for v in raw.iloc[header_row].tolist()]

    df = raw.iloc[header_row + 1:].copy()
    df.columns = header
    df = df.dropna(how="all")
    df = df.loc[:, ~df.columns.str.fullmatch(r"unnamed(_\d+)?", case=False, na=False)]

    col_map = _resolve_columns(df)

    detail = pd.DataFrame()

    def _get(field: str) -> pd.Series:
        col = col_map[field]
        return df[col] if col else pd.Series([""] * len(df), index=df.index)

    detail["item_code"]         = _get("item_code").astype(str).str.strip()
    detail["description"]       = _get("description").astype(str).str.strip()
    detail["org_name"]          = _get("org_name").astype(str).str.strip()
    detail["primary_qty"]       = pd.to_numeric(_get("primary_qty"), errors="coerce")
    detail["subinventory_code"] = _get("subinventory_code").astype(str).str.strip().str.upper()

    detail = detail.dropna(subset=["primary_qty"])
    detail = detail[detail["item_code"].str.len() > 0]

    # Keep only raw-material subinventory codes
    keep_codes = [c.upper() for c in KEEP_SUBINVENTORY]
    detail = detail[detail["subinventory_code"].isin(keep_codes)].copy()

    if detail.empty:
        empty_detail = pd.DataFrame(columns=OUTPUT_COLS)
        empty_summary = pd.DataFrame(columns=["org_name", "Total Inventory"])
        return empty_detail, empty_summary

    # Classify using the provided (or default) mapper
    detail["category"] = detail.apply(
        lambda row: mapper.classify(row["item_code"], row["description"]),
        axis=1,
    )

    detail = detail[OUTPUT_COLS].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Summary: Org Name × Category pivot + Total Inventory column
    # ------------------------------------------------------------------
    agg = (
        detail
        .groupby(["org_name", "category"], as_index=False, sort=True)
        .agg(total_qty=("primary_qty", "sum"))
    )

    summary = agg.pivot_table(
        index="org_name",
        columns="category",
        values="total_qty",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    summary.columns.name = None

    # Add Total Inventory = row-wise sum of all category columns
    category_cols = [c for c in summary.columns if c != "org_name"]
    summary["Total Inventory"] = summary[category_cols].sum(axis=1)

    # Reorder: org_name | [categories sorted] | Total Inventory
    summary = summary[["org_name"] + sorted(category_cols) + ["Total Inventory"]]

    return detail, summary


# ---------------------------------------------------------------------------
# Save outputs to Excel
# ---------------------------------------------------------------------------

def run(
    input_path: str | Path,
    output_path: Optional[str | Path] = None,
    mapping_csv: Optional[str | Path] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Entry point: clean inventory and optionally write Excel output.

    Args:
        input_path:  Path to MG_STOCK_TILL_DATE_MULTIPLE_UNIT.xlsx
        output_path: Optional path to write the summary Excel workbook.
        mapping_csv: Optional path to commodity_mapping.csv.
                     Falls back to built-in rules when None or file missing.

    Returns:
        (detail_df, summary_df)
    """
    mapper = CommodityMapper.from_csv_or_default(mapping_csv)
    detail, summary = clean_inventory(Path(input_path), mapper=mapper)

    if output_path:
        output_path = Path(output_path)
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            detail.to_excel(writer, sheet_name="Detail", index=False)
            summary.to_excel(writer, sheet_name="Summary", index=False)
        print(f"Wrote inventory output → {output_path}")

    return detail, summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean Oracle inventory extract")
    parser.add_argument("--input",       required=True, help="Path to MG_STOCK_TILL_DATE_MULTIPLE_UNIT.xlsx")
    parser.add_argument("--output",      default=None,  help="Optional output Excel path")
    parser.add_argument("--mapping-csv", default=None,  help="Optional path to commodity_mapping.csv",
                        dest="mapping_csv")
    args = parser.parse_args()

    detail_df, summary_df = run(args.input, args.output, args.mapping_csv)

    print(f"\nMapper source : {'CSV' if args.mapping_csv else 'built-in rules'}")
    print(f"Detail rows   : {len(detail_df)}")
    print(f"Summary shape : {summary_df.shape}")
    print("\nSummary:")
    print(summary_df.to_string(index=False))
