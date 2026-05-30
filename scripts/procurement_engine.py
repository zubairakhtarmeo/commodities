"""
procurement_engine.py
---------------------
Procurement decision engine for raw material purchasing.

Combines clean inventory and clean consumption data to produce per-organisation,
per-commodity procurement recommendations based on the 45-day stock policy.

Scope (Phase 3A):
    Reproduce the inventory and consumption logic from strategies.xlsx only.
    No financial trading models, no market pricing, no hedging.

Approved business rules:
    1. Fiber and Stretch Fiber are independent commodities. Never combined.
    2. Net consumption = issues − returns (do NOT use ABS(primary_qty)).
    3. Missing consumption → MONITOR action, LOW confidence.
    4. Cotton Waste → inventory reported only, MONITOR action.
    5. 45-day minimum stock policy (board-approved).

Action values:
    BUY     — shortfall > 0 (stock below 45-day requirement)
    HOLD    — shortfall = 0 (stock meets or exceeds 45-day requirement)
    MONITOR — consumption data absent for this org-commodity

Inputs:
    inventory_df   — org_name, category, inventory_qty
    consumption_df — org_name, category, net_consumption  (monthly net)
    period_days    — actual calendar days in the reporting month
                     derive with: calendar.monthrange(year, month)[1]

Primary output:
    strategy_df — one row per org × commodity

Secondary output:
    validation_report_df — data quality checks

Usage:
    from procurement_engine import run

    strategy_df, validation_df = run(
        inventory_df=inv_long,
        consumption_df=cons_long,
        period_days=calendar.monthrange(2026, 4)[1],
    )

    # Wide-format inputs (from clean_inventory / clean_consumption summaries):
    strategy_df, validation_df = run(
        inventory_df=wide_to_long(inv_wide, value_name="inventory_qty"),
        consumption_df=wide_to_long(cons_wide, value_name="net_consumption"),
        period_days=30,
    )

CLI:
    python procurement_engine.py \\
        --inventory  "path/inventory_summary.xlsx" \\
        --consumption "path/consumption_summary.xlsx" \\
        --period-days 30 \\
        --output     "path/strategy_output.xlsx"
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Commodity constants
# ---------------------------------------------------------------------------

COMMODITY_COTTON        = "Cotton"
COMMODITY_FIBER         = "Fiber"
COMMODITY_STRETCH_FIBER = "Stretch Fiber"
COMMODITY_COTTON_WASTE  = "Cotton Waste"

_ALL_KNOWN_COMMODITIES = frozenset([
    COMMODITY_COTTON,
    COMMODITY_FIBER,
    COMMODITY_STRETCH_FIBER,
    COMMODITY_COTTON_WASTE,
])

# ---------------------------------------------------------------------------
# Policy constant  (board-approved — change requires formal approval)
# ---------------------------------------------------------------------------

MIN_COVER_DAYS: int = 45   # Board-approved minimum stock policy in days

# ---------------------------------------------------------------------------
# Input normalisation helpers
# ---------------------------------------------------------------------------

def _snake(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")


def _normalise_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy().rename(columns={c: _snake(c) for c in df.columns})


def wide_to_long(
    wide_df: pd.DataFrame,
    id_col: str = "org_name",
    value_name: str = "value",
    exclude_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Melt a wide org × category DataFrame into long format.

    Converts:
        org_name | Cotton | Fiber | Stretch Fiber | Cotton Waste | Total ...
    To:
        org_name | category | <value_name>

    Columns whose snake_case name starts with "total" are always excluded.
    Pass exclude_cols for any additional columns to skip.

    Args:
        wide_df:      Wide-format summary from clean_inventory or clean_consumption.
        id_col:       Row identifier column (default "org_name").
        value_name:   Name of the melted value column (default "value").
        exclude_cols: Additional column names to exclude from melting.
    """
    df = _normalise_cols(wide_df)
    id_col_n = _snake(id_col)

    skip = {id_col_n}
    if exclude_cols:
        skip.update(_snake(c) for c in exclude_cols)
    skip.update(c for c in df.columns if c.startswith("total"))

    category_cols = [c for c in df.columns if c not in skip]

    long = df.melt(
        id_vars=id_col_n,
        value_vars=category_cols,
        var_name="category",
        value_name=value_name,
    )

    # Restore human-readable category names from snake_case
    _restore = {_snake(c): c for c in [
        "Cotton", "Fiber", "Stretch Fiber", "Cotton Waste",
        "Fiber Waste", "Chemicals", "Packaging",
    ]}
    long["category"] = long["category"].map(
        lambda x: _restore.get(x, x.replace("_", " ").title())
    )
    return long


# ---------------------------------------------------------------------------
# Core calculation helpers
# ---------------------------------------------------------------------------

def _action(
    shortfall:       float,
    net_consumption: float,
    has_consumption: bool,
    commodity:       str,
) -> str:
    """Return the procurement action label.

    Rules (in priority order):
        1. Cotton Waste          → always MONITOR (inventory-visibility only)
        2. No consumption pair   → MONITOR
        3. Net consumption <= 0  → MONITOR
        4. Shortfall > 0         → BUY
        5. Shortfall = 0         → HOLD

    Action and Confidence are fully independent.
    """
    if commodity == COMMODITY_COTTON_WASTE:
        return "MONITOR"
    if not has_consumption or net_consumption <= 0:
        return "MONITOR"
    return "BUY" if shortfall > 0 else "HOLD"


def _confidence(
    has_inventory:   bool,
    has_consumption: bool,
    net_consumption: float,
) -> str:
    """Return a data-quality confidence score (independent of action).

    HIGH   — inventory and consumption both present, net_consumption > 0
    MEDIUM — inventory present but consumption absent/zero
    LOW    — consumption data entirely absent or net_consumption <= 0
    """
    if not has_consumption or net_consumption <= 0:
        return "LOW"
    if not has_inventory:
        return "MEDIUM"
    return "HIGH"


# ---------------------------------------------------------------------------
# Strategy computation
# ---------------------------------------------------------------------------

def compute_strategy(
    inventory_df:   pd.DataFrame,
    consumption_df: pd.DataFrame,
    period_days:    int,
) -> pd.DataFrame:
    """Compute procurement decisions for all org × commodity combinations.

    Pure function — no I/O, no side effects.

    Args:
        inventory_df:   Long-format: org_name, category, inventory_qty.
        consumption_df: Long-format: org_name, category, net_consumption.
                        Only present when the org had transactions for that
                        commodity. Absent rows = missing consumption.
        period_days:    Actual calendar days in the reporting month.
                        Pass calendar.monthrange(year, month)[1].

    Returns:
        DataFrame with columns:
            org_name, commodity, inventory_qty, monthly_consumption,
            daily_consumption, need_45_days, days_cover, shortfall,
            procurement_qty, action, confidence

    Raises:
        ValueError: if required columns are missing or period_days is invalid.
    """
    if period_days is None:
        period_days = 30
    if not (1 <= period_days <= 31):
        raise ValueError(f"period_days must be 1–31; got {period_days}")

    inv  = _normalise_cols(inventory_df)
    cons = _normalise_cols(consumption_df)

    for df, label, cols in [
        (inv,  "inventory_df",   ["org_name", "category", "inventory_qty"]),
        (cons, "consumption_df", ["org_name", "category", "net_consumption"]),
    ]:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"{label} missing columns: {missing}. Got: {list(df.columns)}")

    # Build the full org × commodity universe from both inputs combined
    inv_pairs  = set(zip(inv["org_name"],  inv["category"]))
    cons_pairs = set(zip(cons["org_name"], cons["category"]))
    all_pairs  = sorted(
        (org, cat) for org, cat in (inv_pairs | cons_pairs)
        if cat in _ALL_KNOWN_COMMODITIES
    )

    if not all_pairs:
        logger.warning("No recognised commodity categories in inputs.")
        return pd.DataFrame()

    inv_lookup  = inv.set_index(["org_name", "category"])["inventory_qty"].to_dict()
    cons_lookup = cons.set_index(["org_name", "category"])["net_consumption"].to_dict()

    rows: list[dict] = []

    for org, commodity in all_pairs:
        inv_qty  = float(inv_lookup.get((org, commodity), 0.0))
        net_cons = float(cons_lookup.get((org, commodity), 0.0))

        # Pair-level check: MONITOR when this specific org+commodity pair is
        # absent from consumption, or when the org is present but this commodity
        # has no transactions (net_consumption <= 0 catches that case too).
        has_consumption = (org, commodity) in cons_pairs

        # daily_rate is only meaningful when actual positive consumption exists.
        daily_rate = net_cons / period_days if (has_consumption and net_cons > 0) else 0.0
        need_45d   = daily_rate * MIN_COVER_DAYS
        shortfall  = max(0.0, need_45d - inv_qty)
        days_cover = (inv_qty / daily_rate) if daily_rate > 0 else None

        rows.append({
            "org_name":            org,
            "commodity":           commodity,
            "inventory_qty":       round(inv_qty,    3),
            "monthly_consumption": round(net_cons,   3),
            "daily_consumption":   round(daily_rate, 4),
            "need_45_days":        round(need_45d,   3),
            "days_cover":          round(days_cover, 1) if days_cover is not None else None,
            "shortfall":           round(shortfall,  3),
            "procurement_qty":     round(shortfall,  3),
            "action":              _action(shortfall, net_cons, has_consumption, commodity),
            "confidence":          _confidence(inv_qty > 0, has_consumption, net_cons),
        })

    strategy_df = pd.DataFrame(rows)

    # Sort: commodity order (Cotton → Fiber → Stretch Fiber → Cotton Waste),
    # then by shortfall descending within each commodity block.
    commodity_order = {
        COMMODITY_COTTON: 0, COMMODITY_FIBER: 1,
        COMMODITY_STRETCH_FIBER: 2, COMMODITY_COTTON_WASTE: 3,
    }
    strategy_df["_ord"] = strategy_df["commodity"].map(commodity_order).fillna(9)
    strategy_df = (
        strategy_df
        .sort_values(["_ord", "shortfall"], ascending=[True, False])
        .drop(columns=["_ord"])
        .reset_index(drop=True)
    )

    return strategy_df


# ---------------------------------------------------------------------------
# Aggregate totals
# ---------------------------------------------------------------------------

def compute_totals(strategy_df: pd.DataFrame) -> pd.DataFrame:
    """Per-commodity totals across all organisations.

    Returns one row per commodity with:
        commodity, total_inventory_qty, total_monthly_consumption,
        total_shortfall, total_procurement_qty
    """
    if strategy_df.empty:
        return pd.DataFrame()

    cols = ["inventory_qty", "monthly_consumption", "shortfall", "procurement_qty"]
    present = [c for c in cols if c in strategy_df.columns]

    totals = (
        strategy_df
        .groupby("commodity", sort=False)[present]
        .sum(min_count=1)
        .reset_index()
    )
    totals.columns = ["commodity"] + [f"total_{c}" for c in present]
    return totals


# ---------------------------------------------------------------------------
# Validation layer
# ---------------------------------------------------------------------------

_CHECK_INVENTORY   = "inventory"
_CHECK_CONSUMPTION = "consumption"
_CHECK_PROCUREMENT = "procurement"

_PASS = "PASS"
_WARN = "WARN"
_FAIL = "FAIL"


def _vrow(category: str, check_name: str, status: str,
          detail: str, affected: str = "") -> dict:
    return {
        "check_category": category,
        "check_name":     check_name,
        "status":         status,
        "detail":         detail,
        "affected":       affected,
    }


def validate_inputs(
    inventory_df:   pd.DataFrame,
    consumption_df: pd.DataFrame,
    period_days:    int,
) -> pd.DataFrame:
    """Run data quality and business rule checks.

    Never raises — all issues are captured as rows in the returned DataFrame.

    Returns:
        DataFrame with columns:
            check_category — "inventory" | "consumption" | "procurement"
            check_name     — short identifier
            status         — "PASS" | "WARN" | "FAIL"
            detail         — description of the finding
            affected       — org / commodity names involved (if any)
    """
    checks: list[dict] = []
    inv  = _normalise_cols(inventory_df)
    cons = _normalise_cols(consumption_df)

    # ── Inventory ────────────────────────────────────────────────────────────

    if "inventory_qty" not in inv.columns:
        checks.append(_vrow(_CHECK_INVENTORY, "column_presence", _FAIL,
                            "inventory_df missing 'inventory_qty' column."))
    else:
        neg = inv[inv["inventory_qty"] < 0]
        if neg.empty:
            checks.append(_vrow(_CHECK_INVENTORY, "no_negative_inventory", _PASS,
                                "All inventory quantities are non-negative."))
        else:
            affected = ", ".join(
                f"{r['org_name']} / {r['category']}"
                for _, r in neg.head(5).iterrows()
            )
            checks.append(_vrow(_CHECK_INVENTORY, "no_negative_inventory", _FAIL,
                                f"{len(neg)} row(s) have negative inventory quantity. "
                                "This indicates an Oracle data issue.",
                                affected))

    if "category" in inv.columns:
        unknown = set(inv["category"].unique()) - _ALL_KNOWN_COMMODITIES
        if not unknown:
            checks.append(_vrow(_CHECK_INVENTORY, "commodity_mapping", _PASS,
                                "All inventory categories match known commodities."))
        else:
            checks.append(_vrow(_CHECK_INVENTORY, "commodity_mapping", _WARN,
                                f"{len(unknown)} unrecognised category value(s) in inventory — "
                                "rows will be excluded from strategy.",
                                ", ".join(sorted(unknown))))

    if "category" in inv.columns and "category" in cons.columns:
        inv_cats  = set(inv["category"].unique())
        cons_cats = set(cons["category"].unique()) - {COMMODITY_COTTON_WASTE}
        cons_only = cons_cats - inv_cats
        if not cons_only:
            checks.append(_vrow(_CHECK_INVENTORY, "category_universe", _PASS,
                                "All consumption categories exist in inventory category universe."))
        else:
            checks.append(_vrow(_CHECK_INVENTORY, "category_universe", _WARN,
                                f"{len(cons_only)} category/categories in consumption have no "
                                "matching inventory entry — inventory_qty will default to 0.",
                                ", ".join(sorted(cons_only))))

    if "org_name" in inv.columns and "category" in inv.columns:
        dup_inv = inv[inv.duplicated(subset=["org_name", "category"], keep=False)]
        if dup_inv.empty:
            checks.append(_vrow(_CHECK_INVENTORY, "no_duplicate_rows", _PASS,
                                "No duplicate org + commodity rows in inventory."))
        else:
            affected = ", ".join(
                f"{r['org_name']} / {r['category']}"
                for _, r in dup_inv.drop_duplicates(subset=["org_name", "category"]).head(5).iterrows()
            )
            checks.append(_vrow(_CHECK_INVENTORY, "no_duplicate_rows", _FAIL,
                                f"{len(dup_inv)} rows are duplicates on org_name + category in inventory. "
                                "Aggregation required before passing to engine.",
                                affected))

    if "org_name" in inv.columns and "org_name" in cons.columns:
        inv_orgs  = set(inv["org_name"].unique())
        cons_orgs = set(cons["org_name"].unique())
        only_inv  = inv_orgs - cons_orgs
        only_cons = cons_orgs - inv_orgs
        if only_inv:
            checks.append(_vrow(_CHECK_INVENTORY, "org_coverage", _WARN,
                                f"{len(only_inv)} org(s) in inventory but not in consumption — "
                                "will be flagged MONITOR.",
                                ", ".join(sorted(only_inv))))
        if only_cons:
            checks.append(_vrow(_CHECK_INVENTORY, "org_coverage", _WARN,
                                f"{len(only_cons)} org(s) in consumption but not in inventory — "
                                "inventory_qty will default to 0.",
                                ", ".join(sorted(only_cons))))
        if not only_inv and not only_cons:
            checks.append(_vrow(_CHECK_INVENTORY, "org_coverage", _PASS,
                                "All orgs present in both inventory and consumption."))

    # ── Consumption ──────────────────────────────────────────────────────────

    if "org_name" in cons.columns and "category" in cons.columns:
        dup_cons = cons[cons.duplicated(subset=["org_name", "category"], keep=False)]
        if dup_cons.empty:
            checks.append(_vrow(_CHECK_CONSUMPTION, "no_duplicate_rows", _PASS,
                                "No duplicate org + commodity rows in consumption."))
        else:
            affected = ", ".join(
                f"{r['org_name']} / {r['category']}"
                for _, r in dup_cons.drop_duplicates(subset=["org_name", "category"]).head(5).iterrows()
            )
            checks.append(_vrow(_CHECK_CONSUMPTION, "no_duplicate_rows", _FAIL,
                                f"{len(dup_cons)} rows are duplicates on org_name + category in consumption. "
                                "Aggregation required before passing to engine.",
                                affected))

    if "net_consumption" not in cons.columns:
        checks.append(_vrow(_CHECK_CONSUMPTION, "column_presence", _FAIL,
                            "consumption_df missing 'net_consumption' column."))
    else:
        neg_cons = cons[cons["net_consumption"] < 0]
        if neg_cons.empty:
            checks.append(_vrow(_CHECK_CONSUMPTION, "no_negative_net_consumption", _PASS,
                                "No negative net consumption values."))
        else:
            affected = ", ".join(
                f"{r['org_name']} / {r['category']}"
                for _, r in neg_cons.head(5).iterrows()
            )
            checks.append(_vrow(_CHECK_CONSUMPTION, "no_negative_net_consumption", _WARN,
                                f"{len(neg_cons)} org-commodity pair(s) have negative net "
                                "consumption (returns exceed issues). "
                                "Daily rate will be set to 0; action will be MONITOR.",
                                affected))

        if len(cons) >= 3:
            daily = cons["net_consumption"] / max(period_days, 1)
            mu, sigma = daily.mean(), daily.std()
            if sigma > 0:
                outliers = cons[daily > mu + 3 * sigma]
                if not outliers.empty:
                    affected = ", ".join(
                        f"{r['org_name']} / {r['category']}"
                        for _, r in outliers.head(5).iterrows()
                    )
                    checks.append(_vrow(_CHECK_CONSUMPTION, "abnormal_consumption", _WARN,
                                        f"{len(outliers)} org-commodity pair(s) exceed "
                                        "mean + 3 std dev. Verify Oracle extract.",
                                        affected))
                else:
                    checks.append(_vrow(_CHECK_CONSUMPTION, "abnormal_consumption", _PASS,
                                        "No statistical outliers in consumption values."))

    if 28 <= period_days <= 31:
        checks.append(_vrow(_CHECK_CONSUMPTION, "period_days", _PASS,
                            f"period_days = {period_days} (valid calendar month length)."))
    else:
        checks.append(_vrow(_CHECK_CONSUMPTION, "period_days", _FAIL,
                            f"period_days = {period_days} is outside 28–31. "
                            "Verify the reporting period."))

    # ── Procurement output ───────────────────────────────────────────────────

    try:
        strategy_df = compute_strategy(inventory_df, consumption_df, period_days)
    except Exception as exc:
        checks.append(_vrow(_CHECK_PROCUREMENT, "engine_error", _FAIL,
                            f"compute_strategy raised an error: {exc}"))
        return pd.DataFrame(checks)

    if not strategy_df.empty:
        neg_short = strategy_df[strategy_df["shortfall"] < 0]
        if neg_short.empty:
            checks.append(_vrow(_CHECK_PROCUREMENT, "no_negative_shortfall", _PASS,
                                "All shortfall values are non-negative."))
        else:
            checks.append(_vrow(_CHECK_PROCUREMENT, "no_negative_shortfall", _FAIL,
                                f"{len(neg_short)} row(s) have negative shortfall. "
                                "Logic error — review compute_strategy."))

        neg_pq = strategy_df[strategy_df["procurement_qty"] < 0]
        if neg_pq.empty:
            checks.append(_vrow(_CHECK_PROCUREMENT, "no_negative_procurement_qty", _PASS,
                                "All procurement_qty values are non-negative."))
        else:
            checks.append(_vrow(_CHECK_PROCUREMENT, "no_negative_procurement_qty", _FAIL,
                                f"{len(neg_pq)} row(s) have negative procurement_qty. "
                                "Logic error — review compute_strategy."))

        invalid_cover = strategy_df[
            strategy_df["days_cover"].notna() & (strategy_df["days_cover"] < 0)
        ]
        if invalid_cover.empty:
            checks.append(_vrow(_CHECK_PROCUREMENT, "valid_days_cover", _PASS,
                                "All days_cover values are non-negative."))
        else:
            checks.append(_vrow(_CHECK_PROCUREMENT, "valid_days_cover", _FAIL,
                                f"{len(invalid_cover)} row(s) have negative days_cover.",
                                ", ".join(invalid_cover["org_name"].head(3).tolist())))

        buy_rows = strategy_df[strategy_df["action"] == "BUY"]
        if not buy_rows.empty:
            critical = buy_rows[buy_rows["days_cover"].notna() & (buy_rows["days_cover"] < 1)]
            if not critical.empty:
                affected = ", ".join(
                    f"{r['org_name']} / {r['commodity']}"
                    for _, r in critical.iterrows()
                )
                checks.append(_vrow(_CHECK_PROCUREMENT, "critical_stock_alert", _FAIL,
                                    f"{len(critical)} org-commodity pair(s) have less than "
                                    "1 day of stock cover. Immediate procurement required.",
                                    affected))

        monitor_rows = strategy_df[strategy_df["action"] == "MONITOR"]
        if not monitor_rows.empty:
            affected = ", ".join(
                f"{r['org_name']} / {r['commodity']}"
                for _, r in monitor_rows.iterrows()
            )
            checks.append(_vrow(_CHECK_PROCUREMENT, "monitor_orgs", _WARN,
                                f"{len(monitor_rows)} org-commodity pair(s) have no "
                                "consumption data and cannot be evaluated.",
                                affected))

        unmapped = strategy_df[strategy_df["commodity"] == "Unmapped"]
        if not unmapped.empty:
            checks.append(_vrow(_CHECK_PROCUREMENT, "unmapped_categories", _WARN,
                                f"{len(unmapped)} row(s) have Unmapped commodity. "
                                "Check commodity_mapper configuration."))

    return pd.DataFrame(checks)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(
    inventory_df:   pd.DataFrame,
    consumption_df: pd.DataFrame,
    period_days:    Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Main entry point for the procurement engine.

    Runs validation then computes the full strategy. Always returns both
    outputs — callers inspect the validation report and decide whether to act.

    Args:
        inventory_df:   Long-format: org_name, category, inventory_qty.
                        Use wide_to_long() to convert clean_inventory summary.
        consumption_df: Long-format: org_name, category, net_consumption.
                        Use wide_to_long() to convert clean_consumption summary.
        period_days:    Actual calendar days in the reporting month.
                        Pass calendar.monthrange(year, month)[1].
                        Defaults to 30 with a warning when None.

    Returns:
        (strategy_df, validation_report_df)
    """
    if period_days is None:
        logger.warning(
            "period_days not specified — defaulting to 30. "
            "Pass calendar.monthrange(year, month)[1] for monthly accuracy."
        )
        period_days = 30

    validation_report = validate_inputs(inventory_df, consumption_df, period_days)

    fail_count = (validation_report["status"] == _FAIL).sum()
    warn_count = (validation_report["status"] == _WARN).sum()
    if fail_count > 0:
        logger.warning("Validation: %d FAIL, %d WARN — proceeding with calculations.",
                       fail_count, warn_count)
    elif warn_count > 0:
        logger.info("Validation: %d WARN, 0 FAIL — proceeding.", warn_count)
    else:
        logger.info("Validation: all checks PASS.")

    strategy_df = compute_strategy(inventory_df, consumption_df, period_days)

    return strategy_df, validation_report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_long(path: Path, value_col: str) -> pd.DataFrame:
    """Load long-format or wide-format DataFrame from Excel or CSV.

    Wide-format summaries are automatically melted via wide_to_long().
    """
    path = Path(path)
    df = (pd.read_excel(path, engine="openpyxl")
          if path.suffix.lower() in (".xlsx", ".xls")
          else pd.read_csv(path))

    df_n = _normalise_cols(df)
    if value_col in df_n.columns:
        return df_n
    return wide_to_long(df, value_name=value_col)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Procurement Engine — inventory and consumption decision logic"
    )
    parser.add_argument("--inventory",   required=True,
                        help="Inventory summary (xlsx or csv)")
    parser.add_argument("--consumption", required=True,
                        help="Consumption summary (xlsx or csv)")
    parser.add_argument("--period-days", type=int, default=None, dest="period_days",
                        help="Calendar days in the reporting month "
                             "(default: auto-detect or 30)")
    parser.add_argument("--output",      default=None,
                        help="Optional output Excel path")
    args = parser.parse_args()

    inv_df  = _load_long(args.inventory,   "inventory_qty")
    cons_df = _load_long(args.consumption, "net_consumption")

    strategy, validation = run(inv_df, cons_df, period_days=args.period_days)

    print("\n=== STRATEGY OUTPUT ===")
    print(strategy.to_string(index=False))

    print("\n=== TOTALS ===")
    print(compute_totals(strategy).to_string(index=False))

    print("\n=== VALIDATION REPORT ===")
    print(validation.to_string(index=False))

    if args.output:
        out = Path(args.output)
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            strategy.to_excel(writer,   sheet_name="Strategy",   index=False)
            compute_totals(strategy).to_excel(writer, sheet_name="Totals", index=False)
            validation.to_excel(writer, sheet_name="Validation", index=False)
        print(f"\nWrote → {out}")
