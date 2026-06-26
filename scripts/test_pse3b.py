"""
test_pse3b.py
--------------
Validation harness for procurement_strategy_engine.py (PSE-3B).

Builds synthetic origin_summary_df inputs (the same shape produced by
origin_classifier.aggregate_by_origin() in PSE-3A: columns org_name, origin,
tons) for five required scenarios plus three bonus edge cases, runs
build_strategy_output_v2() against each, prints full output, and asserts
the expected action/status to catch regressions.

Does not modify clean_inventory.py, clean_consumption.py, run_forecasts.py,
or origin_classifier.py. Imports procurement_strategy_engine only.

Run:
    python test_pse3b.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from procurement_strategy_engine import (
    ACTION_BUY_IMPORTED,
    ACTION_BUY_LOCAL,
    ACTION_BUY_MIXED,
    ACTION_CRITICAL,
    ACTION_HOLD,
    ACTION_WATCH,
    IMPORTED_ROP_TONS,
    LOCAL_ROP_TONS,
    STATUS_CRITICAL,
    STATUS_REORDER,
    STATUS_SAFE,
    STATUS_WATCH,
    build_strategy_output_v2,
)


def _make_origin_df(local_tons: float, imported_tons: float, unknown_tons: float = 0.0) -> pd.DataFrame:
    """Build a minimal origin_summary_df with a single org for test purposes."""
    rows = []
    if local_tons:
        rows.append({"org_name": "TEST-ORG", "origin": "LOCAL", "tons": local_tons})
    if imported_tons:
        rows.append({"org_name": "TEST-ORG", "origin": "IMPORTED", "tons": imported_tons})
    if unknown_tons:
        rows.append({"org_name": "TEST-ORG", "origin": "UNKNOWN", "tons": unknown_tons})
    return pd.DataFrame(rows, columns=["org_name", "origin", "tons"])


def _print_result(name: str, local_tons: float, imported_tons: float, result, unknown_tons: float = 0.0) -> None:
    d = result.to_dict()
    print("=" * 78)
    print(f"SCENARIO: {name}")
    print(f"  Input -> local={local_tons:,.1f}t  imported={imported_tons:,.1f}t"
          + (f"  unknown={unknown_tons:,.1f}t" if unknown_tons else ""))
    print("-" * 78)
    print(f"  total_inventory_tons      : {d['total_inventory_tons']:,.1f}")
    print(f"  local_inventory_tons      : {d['local_inventory_tons']:,.1f}")
    print(f"  imported_inventory_tons   : {d['imported_inventory_tons']:,.1f}")
    print(f"  local_mix_pct / imported  : {d['local_mix_pct']:.1f}% / {d['imported_mix_pct']:.1f}%")
    print(f"  local_days_cover          : {d['local_days_cover']:.1f} days")
    print(f"  imported_days_cover       : {d['imported_days_cover']:.1f} days")
    print(f"  total_days_cover          : {d['total_days_cover']:.1f} days")
    print(f"  local_status / imported   : {d['local_status']} / {d['imported_status']}")
    print(f"  local_gap_tons            : {d['local_gap_tons']:,.1f}")
    print(f"  imported_gap_tons         : {d['imported_gap_tons']:,.1f}")
    print(f"  rebalance_required        : {d['rebalance_required']}")
    print(f"  ACTION                    : {d['action']}   (rule {d['rule_fired']})")
    print(f"  reason                    : {d['reason']}")
    print()


def run_tests() -> bool:
    all_passed = True

    def check(condition: bool, label: str) -> None:
        nonlocal all_passed
        status = "PASS" if condition else "FAIL"
        if not condition:
            all_passed = False
        print(f"  [{status}] {label}")

    # -----------------------------------------------------------------
    # Scenario 1 -- Low inventory (both sources below safety stock floor)
    # -----------------------------------------------------------------
    local, imported = 800.0, 1000.0
    df = _make_origin_df(local, imported)
    r1 = build_strategy_output_v2(df)
    _print_result("1. Low inventory (CRITICAL)", local, imported, r1)
    check(r1.local_status == STATUS_CRITICAL, "local_status == CRITICAL")
    check(r1.imported_status == STATUS_CRITICAL, "imported_status == CRITICAL")
    check(r1.action == ACTION_CRITICAL, "action == CRITICAL")

    # -----------------------------------------------------------------
    # Scenario 2 -- High inventory, balanced mix (both SAFE, HOLD)
    # -----------------------------------------------------------------
    local, imported = 20_000.0, 24_444.0  # ~45/55 split of ~44,444 tons
    df = _make_origin_df(local, imported)
    r2 = build_strategy_output_v2(df)
    _print_result("2. High inventory, balanced mix (HOLD)", local, imported, r2)
    check(r2.local_status == STATUS_SAFE, "local_status == SAFE")
    check(r2.imported_status == STATUS_SAFE, "imported_status == SAFE")
    check(r2.rebalance_required is False, "rebalance_required == False")
    check(r2.action == ACTION_HOLD, "action == HOLD")

    # -----------------------------------------------------------------
    # Scenario 3 -- Local shortage (local at REORDER, imported SAFE)
    # -----------------------------------------------------------------
    local, imported = 1_500.0, 15_000.0
    df = _make_origin_df(local, imported)
    r3 = build_strategy_output_v2(df)
    _print_result("3. Local shortage (BUY_LOCAL)", local, imported, r3)
    check(r3.local_status == STATUS_REORDER, "local_status == REORDER")
    check(r3.imported_status == STATUS_SAFE, "imported_status == SAFE")
    check(r3.action == ACTION_BUY_LOCAL, "action == BUY_LOCAL")

    # -----------------------------------------------------------------
    # Scenario 4 -- Imported shortage (imported at REORDER, local SAFE)
    # -----------------------------------------------------------------
    local, imported = 10_000.0, 6_000.0
    df = _make_origin_df(local, imported)
    r4 = build_strategy_output_v2(df)
    _print_result("4. Imported shortage (BUY_IMPORTED)", local, imported, r4)
    check(r4.local_status == STATUS_SAFE, "local_status == SAFE")
    check(r4.imported_status == STATUS_REORDER, "imported_status == REORDER")
    check(r4.action == ACTION_BUY_IMPORTED, "action == BUY_IMPORTED")

    # -----------------------------------------------------------------
    # Scenario 5 -- Balanced inventory (both SAFE, mix within tolerance)
    # -----------------------------------------------------------------
    local, imported = 9_000.0, 11_000.0  # 45%/55% exactly, both well above ROP
    df = _make_origin_df(local, imported)
    r5 = build_strategy_output_v2(df)
    _print_result("5. Balanced inventory (HOLD)", local, imported, r5)
    check(r5.local_status == STATUS_SAFE, "local_status == SAFE")
    check(r5.imported_status == STATUS_SAFE, "imported_status == SAFE")
    check(abs(r5.local_mix_pct - 45.0) < 0.01, "local_mix_pct == 45.0%")
    check(r5.rebalance_required is False, "rebalance_required == False")
    check(r5.action == ACTION_HOLD, "action == HOLD")

    # -----------------------------------------------------------------
    # BONUS Scenario 6 -- Both sources REORDER simultaneously (BUY_MIXED)
    # -----------------------------------------------------------------
    local, imported = 1_700.0, 6_900.0  # both just under their ROP
    df = _make_origin_df(local, imported)
    r6 = build_strategy_output_v2(df)
    _print_result("6. [Bonus] Both at reorder point (BUY_MIXED)", local, imported, r6)
    check(r6.local_status == STATUS_REORDER, "local_status == REORDER")
    check(r6.imported_status == STATUS_REORDER, "imported_status == REORDER")
    check(r6.action == ACTION_BUY_MIXED, "action == BUY_MIXED")

    # -----------------------------------------------------------------
    # BONUS Scenario 7 -- WATCH band (just above ROP, within buffer)
    # -----------------------------------------------------------------
    local, imported = LOCAL_ROP_TONS * 1.05, 20_000.0  # 5% above local ROP -> WATCH
    df = _make_origin_df(local, imported)
    r7 = build_strategy_output_v2(df)
    _print_result("7. [Bonus] Local in WATCH band", local, imported, r7)
    check(r7.local_status == STATUS_WATCH, "local_status == WATCH")
    check(r7.action == ACTION_WATCH, "action == WATCH")

    # -----------------------------------------------------------------
    # BONUS Scenario 8 -- Mix rebalance trigger (both SAFE, mix skewed)
    # -----------------------------------------------------------------
    local, imported = 5_000.0, 25_000.0  # 16.7% / 83.3% -- way outside 45/55 +-10pts
    df = _make_origin_df(local, imported)
    r8 = build_strategy_output_v2(df)
    _print_result("8. [Bonus] Mix skewed toward imported (BUY_LOCAL via mix rule)", local, imported, r8)
    check(r8.local_status == STATUS_SAFE, "local_status == SAFE")
    check(r8.imported_status == STATUS_SAFE, "imported_status == SAFE")
    check(r8.rebalance_required is True, "rebalance_required == True")
    check(r8.overweight_imported is True, "overweight_imported == True")
    check(r8.action == ACTION_BUY_LOCAL, "action == BUY_LOCAL (mix correction, rule 6)")

    # -----------------------------------------------------------------
    # BONUS Scenario 9 -- UNKNOWN tons present (physical total includes them,
    #                     mix/cover-by-source excludes them)
    # -----------------------------------------------------------------
    local, imported, unknown = 9_000.0, 11_000.0, 5_000.0
    df = _make_origin_df(local, imported, unknown)
    r9 = build_strategy_output_v2(df)
    _print_result("9. [Bonus] UNKNOWN tons present", local, imported, r9, unknown_tons=unknown)
    check(r9.unknown_inventory_tons == unknown, "unknown_inventory_tons captured correctly")
    check(r9.total_inventory_tons == local + imported + unknown,
          "total_inventory_tons includes UNKNOWN (physical reality)")
    check(r9.classified_inventory_tons == local + imported,
          "classified_inventory_tons excludes UNKNOWN (mix basis)")
    check(abs(r9.local_mix_pct - 45.0) < 0.01,
          "local_mix_pct computed on classified tons only, unaffected by UNKNOWN")

    print("=" * 78)
    print("ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED")
    print("=" * 78)
    return all_passed


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
