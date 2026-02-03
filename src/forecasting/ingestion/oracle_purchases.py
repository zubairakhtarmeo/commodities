"""Oracle purchasing history ingestion + cleaning.

Goal: take messy Oracle-export Excel files (title rows, filters, Unnamed cols)
and produce a clean, standardized purchases dataset usable for procurement sizing.

This module is intentionally defensive:
- Auto-detects the header row by scanning for known column keywords
- Normalizes column names and coerces types (dates, numbers)
- Classifies each line into a commodity/material bucket using description rules

No business-sensitive assumptions are hard-coded beyond generic keyword mapping.
Tune mapping rules via `MaterialClassifier` overrides.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


def _norm_text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _snake(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _to_number(series: pd.Series) -> pd.Series:
    # handles commas, parentheses negatives, stray spaces
    def _coerce(x: object) -> float:
        t = _norm_text(x)
        if not t:
            return float("nan")
        t = t.replace(",", "")
        # (123) style negatives
        if t.startswith("(") and t.endswith(")"):
            t = "-" + t[1:-1]
        try:
            return float(t)
        except Exception:
            return float("nan")

    return series.map(_coerce)


def _header_tokens() -> list[list[str]]:
    # Canonical header tokens and common Oracle-ish variants.
    # Keep broad; the scoring threshold keeps it safe.
    return [
        ["grn date", "receipt date", "grn_dt", "date"],
        ["description", "description of material", "item description", "material description"],
        ["uom", "unit of measure"],
        ["receipt qty", "receipt", "deliver qty", "po qty", "bill qty", "quantity", "qty", "received qty"],
        ["rate", "rate rs", "unit price", "price"],
        ["amount", "value", "line amount", "value pkr", "amount pkr"],
        ["supplier", "vendor", "party", "party name"],
        ["operating unit", "org", "organization", "business unit", "unit"],
        ["po", "po no", "po number", "purchase order"],
    ]


def score_header_row(raw: pd.DataFrame, row_index: int) -> int:
    row = raw.iloc[row_index]
    joined = " | ".join(_norm_text(v).lower() for v in row.tolist())
    if not joined.strip():
        return 0
    score = 0
    for variants in _header_tokens():
        if any(v in joined for v in variants):
            score += 1
    return score


def detect_header_row(raw: pd.DataFrame, *, max_scan_rows: int = 80) -> Optional[tuple[int, int]]:
    """Return (header_row_index, score) for a raw Excel dataframe (header=None)."""

    scan_n = min(max_scan_rows, len(raw))
    best_row: Optional[int] = None
    best_score = 0

    for i in range(scan_n):
        score = score_header_row(raw, i)
        if score <= 0:
            continue

        if score > best_score:
            best_score = score
            best_row = i

    # Require at least 3 strong hits to avoid picking a title row.
    if best_score >= 3:
        return best_row, best_score
    return None


def _read_sheet_raw(xlsx: pd.ExcelFile, sheet_name: object) -> pd.DataFrame:
    return pd.read_excel(xlsx, sheet_name=sheet_name, header=None, engine="openpyxl")


def clean_oracle_purchases_workbook(
    *,
    xlsx_path: Path,
    classifier: Optional[MaterialClassifier] = None,
    max_scan_rows: int = 80,
) -> tuple[pd.DataFrame, dict]:
    """Load and clean the best sheet from an Oracle export workbook.

    Scans all sheets and picks the (sheet, header_row) with the highest header score.
    If multiple are tied, prefers the one yielding more usable rows.
    """

    classifier = classifier or MaterialClassifier()
    meta: dict = {"file": xlsx_path.name, "status": "ok"}

    try:
        xlsx = pd.ExcelFile(xlsx_path, engine="openpyxl")
    except Exception as e:
        meta.update({"status": "error", "error": f"Failed to open workbook: {e}"})
        return pd.DataFrame(), meta

    best: Optional[tuple[int, int, object, pd.DataFrame]] = None
    # (score, usable_rows, sheet, raw)
    for sheet in xlsx.sheet_names:
        try:
            raw = _read_sheet_raw(xlsx, sheet)
        except Exception:
            continue

        if raw is None or raw.empty:
            continue

        detected = detect_header_row(raw, max_scan_rows=max_scan_rows)
        if detected is None:
            continue
        header_row, score = detected

        # Estimate usable rows below header row
        body = raw.iloc[header_row + 1 :]
        usable_rows = int(body.dropna(axis=0, how="all").shape[0])
        if usable_rows <= 0:
            continue

        candidate = (score, usable_rows, sheet, raw)
        if best is None:
            best = candidate
        else:
            if (candidate[0], candidate[1]) > (best[0], best[1]):
                best = candidate

    if best is None:
        meta.update({"status": "skipped", "reason": "No sheet with detectable header + rows"})
        return pd.DataFrame(), meta

    score, usable_rows, sheet, raw = best
    header_row, _ = detect_header_row(raw, max_scan_rows=max_scan_rows) or (None, None)
    if header_row is None:
        meta.update({"status": "skipped", "reason": "Header vanished on re-check"})
        return pd.DataFrame(), meta

    df = _clean_from_raw(raw=raw, header_row=header_row, sheet_name=sheet, xlsx_path=xlsx_path, classifier=classifier)
    meta.update(
        {
            "sheet": sheet,
            "header_row": int(header_row),
            "header_score": int(score),
            "estimated_rows": int(usable_rows),
            "rows": int(len(df)),
        }
    )
    if len(df) <= 0:
        meta.update({"status": "skipped", "reason": "No valid rows after cleaning"})
    return df, meta


def _clean_from_raw(
    *,
    raw: pd.DataFrame,
    header_row: int,
    sheet_name: object,
    xlsx_path: Path,
    classifier: MaterialClassifier,
) -> pd.DataFrame:
    header = raw.iloc[header_row].tolist()
    df = raw.iloc[header_row + 1 :].copy()
    df.columns = _standardize_columns(header)

    # Drop fully empty rows and 'unnamed' columns
    df = df.dropna(axis=0, how="all")
    df = df.loc[:, ~df.columns.str.fullmatch(r"unnamed(_\d+)?", case=False, na=False)]

    # Standardize common columns
    # Keep raw extras too; just add normalized aliases used downstream.
    col_date = _pick_col(df, ["grn_date", "receipt_date", "date"])
    col_desc = _pick_col(df, ["description_of_material", "description", "material_description", "item_description"])
    col_uom = _pick_col(df, ["uom", "unit_of_measure"])
    col_qty = _pick_col(
        df,
        [
            "receipt_qty",
            "received_qty",
            "receipt",
            "deliver_qty",
            "po_qty",
            "bill_qty",
            "sec_qty",
            "qty",
            "quantity",
        ],
    )
    col_rate = _pick_col(df, ["rate", "rate_rs", "unit_price", "price"])
    col_amount = _pick_col(df, ["amount", "value", "line_amount", "value_pkr", "amount_pkr"])
    col_supplier = _pick_col(df, ["supplier", "vendor", "party", "party_name", "supplier_name", "vendor_name"])
    col_unit = _pick_col(df, ["operating_unit", "business_unit", "organization", "org", "unit"])
    col_po = _pick_col(df, ["po_no", "po", "po_number", "purchase_order"])

    out = pd.DataFrame()
    # Oracle exports in Pakistan commonly use day-month-year; using dayfirst avoids noisy warnings.
    out["grn_date"] = pd.to_datetime(df[col_date], errors="coerce", dayfirst=True) if col_date else pd.NaT
    out["description"] = df[col_desc].map(_norm_text) if col_desc else ""
    out["uom"] = df[col_uom].map(_norm_text) if col_uom else ""

    out["receipt_qty"] = _to_number(df[col_qty]) if col_qty else float("nan")
    out["rate"] = _to_number(df[col_rate]) if col_rate else float("nan")
    out["amount"] = _to_number(df[col_amount]) if col_amount else float("nan")

    # Normalize quantity to kilograms when possible (for consistent sizing)
    out["receipt_qty_kg"] = out["receipt_qty"]
    try:
        u = out["uom"].astype(str).str.strip().str.lower()
        is_lb = u.isin(["lb", "lbs", "pound", "pounds"]) | u.str.contains(r"\blb\b", regex=True)
        is_kg = u.isin(["kg", "kgs", "kilogram", "kilograms"]) | u.str.contains(r"\bkg\b", regex=True)
        out.loc[is_lb & out["receipt_qty"].notna(), "receipt_qty_kg"] = out.loc[is_lb & out["receipt_qty"].notna(), "receipt_qty"] * 0.45359237
        # If unit is unknown (bags, bales, etc), set to NaN to avoid mixing units.
        out.loc[~(is_lb | is_kg), "receipt_qty_kg"] = float("nan")
    except Exception:
        pass

    out["supplier"] = df[col_supplier].map(_norm_text) if col_supplier else ""
    out["operating_unit"] = df[col_unit].map(_norm_text) if col_unit else ""
    out["po_number"] = df[col_po].map(_norm_text) if col_po else ""

    out["commodity"] = out["description"].map(classifier.classify)

    # Derived metrics
    out["unit_price"] = out["amount"] / out["receipt_qty"]
    out.loc[~out["receipt_qty"].notna(), "unit_price"] = float("nan")
    out["unit_price"] = out["unit_price"].replace([float("inf"), float("-inf")], float("nan"))

    out["source_file"] = xlsx_path.name
    out["source_sheet"] = str(sheet_name)
    out["source_header_row"] = int(header_row)

    # Drop obviously invalid rows (no date and no description and no qty)
    out = out.dropna(axis=0, how="all", subset=["grn_date", "receipt_qty", "amount"])

    return out


@dataclass
class MaterialClassifier:
    """Classify a purchase line to a commodity bucket from the description."""

    # Exact/substring overrides win first.
    overrides: dict[str, str] = field(default_factory=dict)

    # Regex rules (ordered). Keep these high-level; refine as you see real data.
    rules: list[tuple[str, str]] = field(
        default_factory=lambda: [
            (r"\bvisco(se)?\b|\brayon\b|\bvsf\b|\bvisc\b", "Viscose"),
            (r"\bpoly(ester)?\b|\bpsf\b|\bfdy\b|\bdty\b|\bpoy\b|\bpet\b|\bfiber\b|\bfibre\b", "Polyester"),
            (r"\bcot(ton)?\b|\braw\s*cot\b|\bcotn\b", "Cotton"),
            # Energy buckets are included for completeness; many Oracle lines will be non-energy.
            (r"\bnatural\s*gas\b|\blng\b|\brlng\b|\bmm\s*btu\b|\bmmbtu\b", "Natural Gas"),
            (r"\bcrude\b|\bbrent\b|\bfuel\s*oil\b|\bhsfo\b|\blsfo\b", "Crude Oil"),
        ]
    )

    def classify(self, description: object) -> str:
        d = _norm_text(description)
        if not d:
            return "Unmapped"

        d_upper = d.upper()
        for needle, mapped in self.overrides.items():
            if needle.upper() in d_upper:
                return mapped

        d_lower = d.lower()
        for pattern, mapped in self.rules:
            if re.search(pattern, d_lower):
                return mapped

        return "Other"


def _standardize_columns(columns: Iterable[object]) -> list[str]:
    out: list[str] = []
    for c in columns:
        s = _snake(_norm_text(c))
        out.append(s if s else "unnamed")
    return out


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def clean_oracle_purchases_sheet(
    *,
    xlsx_path: Path,
    sheet_name: object = 0,
    classifier: Optional[MaterialClassifier] = None,
) -> pd.DataFrame:
    """Load and clean a single Excel sheet from an Oracle export."""

    classifier = classifier or MaterialClassifier()

    raw = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None, engine="openpyxl")
    detected = detect_header_row(raw)
    if detected is None:
        raise ValueError(f"Could not detect header row in {xlsx_path.name} (sheet={sheet_name!r})")

    header_row, _score = detected

    return _clean_from_raw(
        raw=raw,
        header_row=header_row,
        sheet_name=sheet_name,
        xlsx_path=xlsx_path,
        classifier=classifier,
    )


def ingest_oracle_purchases_dir(
    *,
    input_dir: Path,
    pattern: str = "*.xlsx",
    sheet_name: object = 0,
    classifier: Optional[MaterialClassifier] = None,
) -> pd.DataFrame:
    """Ingest all matching Excel files in a directory into a single clean dataframe."""

    input_dir = Path(input_dir)
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched {pattern!r} in {str(input_dir)!r}")

    frames: list[pd.DataFrame] = []
    for f in files:
        # If user explicitly passes a sheet name/index, honor it.
        if sheet_name is not None:
            try:
                frames.append(clean_oracle_purchases_sheet(xlsx_path=f, sheet_name=sheet_name, classifier=classifier))
            except Exception:
                continue
        else:
            cleaned, meta = clean_oracle_purchases_workbook(xlsx_path=f, classifier=classifier)
            if meta.get("status") == "ok" and not cleaned.empty:
                frames.append(cleaned)

    if not frames:
        return pd.DataFrame(
            columns=[
                "grn_date",
                "description",
                "uom",
                "receipt_qty",
                "rate",
                "amount",
                "supplier",
                "operating_unit",
                "po_number",
                "commodity",
                "unit_price",
                "source_file",
                "source_sheet",
                "source_header_row",
            ]
        )

    df = pd.concat(frames, ignore_index=True)

    # Basic hygiene
    df["commodity"] = df["commodity"].fillna("Unmapped")
    df = df.sort_values(["grn_date", "source_file"], na_position="last").reset_index(drop=True)

    return df


def build_monthly_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate purchases to monthly totals per operating unit and commodity."""

    out = df.copy()
    out["month"] = pd.to_datetime(out["grn_date"], errors="coerce").dt.to_period("M").dt.to_timestamp()

    g = out.groupby(["operating_unit", "commodity", "month"], dropna=False)

    def _sum_min1(s: pd.Series) -> float:
        return float(pd.to_numeric(s, errors="coerce").sum(min_count=1))

    agg = (
        g.agg(
            lines=("description", "count"),
            total_qty=("receipt_qty", _sum_min1),
            total_qty_kg=("receipt_qty_kg", _sum_min1),
            total_amount=("amount", _sum_min1),
        )
        .reset_index()
    )

    agg["avg_unit_price"] = agg["total_amount"] / agg["total_qty"]
    agg.loc[~(agg["total_qty"] > 0), "avg_unit_price"] = float("nan")
    agg["avg_unit_price"] = agg["avg_unit_price"].replace([float("inf"), float("-inf")], float("nan"))
    return agg.sort_values(["commodity", "operating_unit", "month"]).reset_index(drop=True)
