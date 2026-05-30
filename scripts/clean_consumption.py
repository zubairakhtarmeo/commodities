"""
clean_consumption.py
--------------------
Cleans and aggregates Oracle transaction register extract from
MG_TRANSACTION_REGISTER_INV.

Pipeline:
    1. Load Excel → auto-detect header row
    2. Retain eight columns: Item Code, Item Desc, Org Name, Lot Number,
       Subinventory, Txn Type, Primary Qty, Date
    3. Classify each row into a material category via CommodityMapper
    4. Compute consumption_qty = ABS(primary_qty)
       (Oracle records inventory issues as negative quantities)
    5. Return three outputs:
       - detail_df:     cleaned row-level dataframe with year_month
       - summary_df:    Org Name × Category + Total Consumed (no year_month)
       - diagnostics_df: positive/negative qty counts per Transaction Type
                         — confirms Oracle sign convention before go-live

Mapping source priority:
    1. --mapping-csv argument (external commodity_mapping.csv)
    2. Built-in DEFAULT_PREFIX_RULES  (fallback)
    3. Description regex             (last resort)

Usage:
    from clean_consumption import run

    detail, summary, diagnostics = run("D:/path/MG_TRANSACTION_REGISTER_INV.xlsx")

    # With CSV mappings and month filter:
    detail, summary, diagnostics = run(
        "D:/path/MG_TRANSACTION_REGISTER_INV.xlsx",
        mapping_csv="scripts/commodity_mapping.csv",
        filter_month="2025-04",
    )
    print(summary)
    print(diagnostics)

CLI:
    python clean_consumption.py --input  "D:/path/MG_TRANSACTION_REGISTER_INV.xlsx"
                                --output "D:/path/consumption_summary.xlsx"
                                --mapping-csv "scripts/commodity_mapping.csv"
                                --month  "2025-04"
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

COLUMN_CANDIDATES: dict[str, list[str]] = {
    "item_code":    ["item_code", "item", "item_number", "itemcode", "item_no"],
    "item_desc":    ["item_desc", "item_description", "description", "item_name",
                     "material_description"],
    "org_name":     ["org_name", "organization_name", "organization", "org",
                     "operating_unit", "unit"],
    "lot_number":   ["lot_number", "lot_no", "lot", "batch_no", "batch_number"],
    "subinventory": ["subinventory", "subinventory_code", "sub_inv", "subinv",
                     "subinventory_name"],
    "txn_type":     ["txn_type", "transaction_type", "txn_type_name",
                     "transaction_type_name", "transaction"],
    "primary_qty":  ["primary_qty", "primary_quantity", "quantity", "qty",
                     "transaction_quantity", "txn_qty"],
    "date":         ["date", "transaction_date", "txn_date", "accounting_date",
                     "gl_date"],
}

# Subinventory codes that hold actual raw material stock.
# Only issues (negative qty) from these locations represent real consumption.
RM_SUBINVENTORY_CODES: list[str] = ["RM-COTTON", "RM-FIBER"]

# Oracle appends a function-suffix to every org name in the transaction register
# (e.g. "MSM - Spinning U1 - Raw Material").  Strip these so names match the
# inventory report (which uses "MSM - Spinning U1") and the procurement engine.
ORG_SUFFIXES_TO_STRIP: list[str] = [
    " - Raw Material",
    " - Manufacturing",
    " - General Store",
    " - Trading",
    " - Knitting",
]

OUTPUT_COLS = [
    "item_code", "item_desc", "org_name", "lot_number",
    "subinventory", "txn_type", "primary_qty", "consumption_qty",
    "date", "year_month", "category",
]


# ---------------------------------------------------------------------------
# Org name normalisation
# ---------------------------------------------------------------------------

def _strip_org_suffix(org: object) -> str:
    """Remove Oracle function-suffixes from org names (e.g. ' - Raw Material')."""
    s = str(org).strip() if org is not None else ""
    for suffix in ORG_SUFFIXES_TO_STRIP:
        if s.endswith(suffix):
            return s[: -len(suffix)].strip()
    return s


# ---------------------------------------------------------------------------
# Header detection
# ---------------------------------------------------------------------------

def _snake(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")


def _score_header_row(raw: pd.DataFrame, row_idx: int) -> int:
    row_text = " | ".join(_snake(v) for v in raw.iloc[row_idx].tolist())
    score = 0
    for candidates in COLUMN_CANDIDATES.values():
        if any(c in row_text for c in candidates):
            score += 1
    return score


def _detect_header_row(raw: pd.DataFrame, max_scan: int = 60) -> int:
    best_row, best_score = 0, 0
    for i in range(min(max_scan, len(raw))):
        score = _score_header_row(raw, i)
        if score > best_score:
            best_score, best_row = score, i
    if best_score < 2:
        raise ValueError(
            f"Could not detect a header row (best score={best_score}). "
            "Verify the file is MG_TRANSACTION_REGISTER_INV.xlsx and "
            "at least 2 expected columns are present in the first 60 rows."
        )
    return best_row


# ---------------------------------------------------------------------------
# Column resolution
# ---------------------------------------------------------------------------

def _resolve_columns(df: pd.DataFrame) -> dict[str, Optional[str]]:
    normalised = {_snake(c): c for c in df.columns}
    resolved: dict[str, Optional[str]] = {}
    for field, candidates in COLUMN_CANDIDATES.items():
        resolved[field] = next((normalised[c] for c in candidates if c in normalised), None)
    return resolved


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def txn_diagnostics(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Build a transaction-type sign-convention diagnostic report.

    Groups raw transactions by Transaction Type and counts how many rows have
    positive vs negative primary_qty.  This lets you confirm before go-live
    that Oracle is issuing stock as negatives and receiving as positives.

    Args:
        detail_df: Output of clean_consumption() detail dataframe.

    Returns:
        DataFrame with columns:
            txn_type           — Oracle transaction type label
            positive_qty_count — rows where primary_qty > 0
            negative_qty_count — rows where primary_qty < 0
            zero_qty_count     — rows where primary_qty == 0
            total_rows         — total rows for this type
    """
    if detail_df.empty or "primary_qty" not in detail_df.columns:
        return pd.DataFrame(columns=[
            "txn_type", "positive_qty_count", "negative_qty_count",
            "zero_qty_count", "total_rows",
        ])

    df = detail_df.copy()
    df["_sign"] = pd.cut(
        df["primary_qty"],
        bins=[-float("inf"), -1e-9, 1e-9, float("inf")],
        labels=["negative", "zero", "positive"],
    )

    diag = (
        df.groupby(["txn_type", "_sign"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    diag.columns.name = None

    # Ensure all three sign columns exist even if no rows of that sign exist
    for col in ["negative", "zero", "positive"]:
        if col not in diag.columns:
            diag[col] = 0

    diag = diag.rename(columns={
        "positive": "positive_qty_count",
        "negative": "negative_qty_count",
        "zero":     "zero_qty_count",
    })
    diag["total_rows"] = (
        diag["positive_qty_count"]
        + diag["negative_qty_count"]
        + diag["zero_qty_count"]
    )
    diag = diag[["txn_type", "positive_qty_count", "negative_qty_count",
                 "zero_qty_count", "total_rows"]]

    return diag.sort_values("total_rows", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def load_raw(xlsx_path: Path) -> pd.DataFrame:
    return pd.read_excel(xlsx_path, sheet_name=0, header=None, engine="openpyxl")


def clean_consumption(
    xlsx_path: Path,
    mapper: Optional[CommodityMapper] = None,
    filter_month: Optional[str] = None,
    filter_rm_only: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Full cleaning pipeline for the transaction register extract.

    Args:
        xlsx_path:      Path to MG_TRANSACTION_REGISTER_INV.xlsx
        mapper:         CommodityMapper instance. Uses built-in rules when None.
        filter_month:   Optional "YYYY-MM" string to restrict to one month.
                        When None, all months present in the file are processed.
        filter_rm_only: When True (default), restrict to RM-COTTON / RM-FIBER
                        subinventory **issue** rows (primary_qty < 0).  These are
                        the only rows that represent actual raw-material consumption.
                        Set False to return all transactions (e.g. for diagnostics).

    Returns:
        detail_df:      Row-level dataframe (OUTPUT_COLS).
        summary_df:     Wide pivot — rows = Org Name,
                        columns = [categories] + Total Consumed.
        diagnostics_df: Transaction-type sign diagnostic report (built before
                        any RM filter so it reflects the full sign picture).
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

    detail["item_code"]    = _get("item_code").astype(str).str.strip()
    detail["item_desc"]    = _get("item_desc").astype(str).str.strip()
    # Strip Oracle function-suffix so org names match inventory (e.g. "MSM - Spinning U1")
    detail["org_name"]     = _get("org_name").astype(str).str.strip().apply(_strip_org_suffix)
    detail["lot_number"]   = _get("lot_number").astype(str).str.strip()
    detail["subinventory"] = _get("subinventory").astype(str).str.strip()
    detail["txn_type"]     = _get("txn_type").astype(str).str.strip()
    detail["primary_qty"]  = pd.to_numeric(_get("primary_qty"), errors="coerce")

    # Oracle exports dates in various formats; dayfirst handles DD-MON-YY
    detail["date"] = pd.to_datetime(_get("date"), errors="coerce", dayfirst=True)

    detail = detail.dropna(subset=["primary_qty"])
    detail = detail[detail["item_code"].str.len() > 0]

    # Build diagnostics BEFORE filtering — must see raw signs across all txn types
    diagnostics = txn_diagnostics(detail)

    if filter_rm_only:
        # Keep only issues (negative qty) from RM subinventories.
        # The Oracle transaction register includes both sides of every transfer:
        # the issuing side (negative) and the receiving side (positive).  Only the
        # issuing side from RM-COTTON / RM-FIBER represents actual consumption.
        rm_codes = [c.upper() for c in RM_SUBINVENTORY_CODES]
        detail = detail[
            detail["subinventory"].str.upper().isin(rm_codes) &
            (detail["primary_qty"] < 0)
        ].copy()

    # Consumption qty: Oracle issues stock as negative → ABS normalises to positive.
    detail["consumption_qty"] = detail["primary_qty"].abs()

    # Year-month period for granular analysis
    detail["year_month"] = detail["date"].dt.to_period("M")

    if filter_month:
        target_period = pd.Period(filter_month, freq="M")
        detail = detail[detail["year_month"] == target_period].copy()

    if detail.empty:
        empty_detail = pd.DataFrame(columns=OUTPUT_COLS)
        empty_summary = pd.DataFrame(columns=["org_name", "Total Consumed"])
        return empty_detail, empty_summary, diagnostics

    detail["category"] = detail.apply(
        lambda row: mapper.classify(row["item_code"], row["item_desc"]),
        axis=1,
    )

    # Serialise period to string before returning
    detail["year_month"] = detail["year_month"].astype(str)
    detail = detail[OUTPUT_COLS].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Summary: Org Name × Category (all months collapsed) + Total Consumed
    # ------------------------------------------------------------------
    agg = (
        detail
        .groupby(["org_name", "category"], as_index=False, sort=True)
        .agg(total_consumption=("consumption_qty", "sum"))
    )

    summary = agg.pivot_table(
        index="org_name",
        columns="category",
        values="total_consumption",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    summary.columns.name = None

    category_cols = [c for c in summary.columns if c != "org_name"]
    summary["Total Consumed"] = summary[category_cols].sum(axis=1)

    # Reorder: org_name | [categories sorted] | Total Consumed
    summary = summary[["org_name"] + sorted(category_cols) + ["Total Consumed"]]

    return detail, summary, diagnostics


# ---------------------------------------------------------------------------
# Cross-period helper (for trend sheets)
# ---------------------------------------------------------------------------

def monthly_totals(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Total consumption per month × category across all orgs.

    Useful for trend charts and the monthly Consumption sheet.
    Returns columns: year_month, category, total_consumption.
    """
    if detail_df.empty:
        return pd.DataFrame(columns=["year_month", "category", "total_consumption"])
    return (
        detail_df
        .groupby(["year_month", "category"], as_index=False, sort=True)
        .agg(total_consumption=("consumption_qty", "sum"))
    )


# ---------------------------------------------------------------------------
# Save outputs to Excel
# ---------------------------------------------------------------------------

def run(
    input_path: str | Path,
    output_path: Optional[str | Path] = None,
    mapping_csv: Optional[str | Path] = None,
    filter_month: Optional[str] = None,
    filter_rm_only: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Entry point: clean consumption and optionally write Excel output.

    Args:
        input_path:     Path to MG_TRANSACTION_REGISTER_INV.xlsx
        output_path:    Optional path to write the output Excel workbook.
        mapping_csv:    Optional path to commodity_mapping.csv.
        filter_month:   Optional "YYYY-MM" to restrict to one month.
        filter_rm_only: Restrict to RM subinventory issues only (default True).

    Returns:
        (detail_df, summary_df, diagnostics_df)
    """
    mapper = CommodityMapper.from_csv_or_default(mapping_csv)
    detail, summary, diagnostics = clean_consumption(
        Path(input_path), mapper=mapper, filter_month=filter_month,
        filter_rm_only=filter_rm_only,
    )

    if output_path:
        output_path = Path(output_path)
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            detail.to_excel(writer, sheet_name="Detail", index=False)
            summary.to_excel(writer, sheet_name="Summary", index=False)
            monthly_totals(detail).to_excel(writer, sheet_name="Monthly Totals", index=False)
            diagnostics.to_excel(writer, sheet_name="Diagnostics", index=False)
        print(f"Wrote consumption output → {output_path}")

    return detail, summary, diagnostics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean Oracle consumption extract")
    parser.add_argument("--input",       required=True, help="Path to MG_TRANSACTION_REGISTER_INV.xlsx")
    parser.add_argument("--output",      default=None,  help="Optional output Excel path")
    parser.add_argument("--mapping-csv", default=None,  help="Optional path to commodity_mapping.csv",
                        dest="mapping_csv")
    parser.add_argument("--month",       default=None,  help='Optional month filter e.g. "2025-04"')
    args = parser.parse_args()

    detail_df, summary_df, diag_df = run(
        args.input, args.output, args.mapping_csv, args.month
    )

    print(f"\nMapper source : {'CSV' if args.mapping_csv else 'built-in rules'}")
    print(f"Detail rows   : {len(detail_df)}")
    print(f"Summary shape : {summary_df.shape}")
    print("\nSummary:")
    print(summary_df.to_string(index=False))
    print("\nDiagnostics (sign convention by transaction type):")
    print(diag_df.to_string(index=False))
