"""
test_strategy_engine.py
-----------------------
Tests for procurement_strategy_engine.py (PSE-3B).
Covers: approved constants, status classifier, inventory position, reorder triggers.
"""
import math
import pytest
import pandas as pd

from procurement_strategy_engine import (
    DAILY_CONSUMPTION_TOTAL,
    DAILY_CONSUMPTION_LOCAL,
    DAILY_CONSUMPTION_IMPORTED,
    LOCAL_MIX_TARGET,
    IMPORTED_MIX_TARGET,
    MIN_STOCK_DAYS,
    LOCAL_LEAD_TIME_DAYS,
    IMPORTED_LEAD_TIME_DAYS,
    SAFETY_STOCK_LOCAL_TONS,
    SAFETY_STOCK_IMPORTED_TONS,
    SAFETY_STOCK_TOTAL_TONS,
    LOCAL_ROP_TONS,
    IMPORTED_ROP_TONS,
    WATCH_BUFFER_PCT,
    STATUS_CRITICAL, STATUS_REORDER, STATUS_WATCH, STATUS_SAFE,
    _classify_supply_status,
    compute_inventory_position,
    compute_reorder_triggers,
)


# ===========================================================================
# Business-rule constants (PSE-2 / PSE-2.7 approved — must never change silently)
# ===========================================================================

class TestApprovedConstants:
    def test_daily_consumption_total(self):
        assert DAILY_CONSUMPTION_TOTAL == 110.0

    def test_mix_targets_sum_to_one(self):
        assert abs(LOCAL_MIX_TARGET + IMPORTED_MIX_TARGET - 1.0) < 1e-9

    def test_local_mix_target(self):
        assert LOCAL_MIX_TARGET == 0.45

    def test_imported_mix_target(self):
        assert IMPORTED_MIX_TARGET == 0.55

    def test_daily_consumption_local(self):
        assert abs(DAILY_CONSUMPTION_LOCAL - 49.5) < 1e-6

    def test_daily_consumption_imported(self):
        assert abs(DAILY_CONSUMPTION_IMPORTED - 60.5) < 1e-6

    def test_daily_parts_sum_to_total(self):
        assert abs(DAILY_CONSUMPTION_LOCAL + DAILY_CONSUMPTION_IMPORTED - DAILY_CONSUMPTION_TOTAL) < 1e-9

    def test_min_stock_days(self):
        assert MIN_STOCK_DAYS == 25

    def test_local_lead_time(self):
        assert LOCAL_LEAD_TIME_DAYS == 10

    def test_imported_lead_time(self):
        assert IMPORTED_LEAD_TIME_DAYS == 90

    def test_safety_stock_local(self):
        expected = MIN_STOCK_DAYS * DAILY_CONSUMPTION_LOCAL   # 1237.5
        assert abs(SAFETY_STOCK_LOCAL_TONS - expected) < 1e-6

    def test_safety_stock_imported(self):
        expected = MIN_STOCK_DAYS * DAILY_CONSUMPTION_IMPORTED  # 1512.5
        assert abs(SAFETY_STOCK_IMPORTED_TONS - expected) < 1e-6

    def test_safety_stock_total(self):
        expected = MIN_STOCK_DAYS * DAILY_CONSUMPTION_TOTAL  # 2750.0
        assert abs(SAFETY_STOCK_TOTAL_TONS - expected) < 1e-6

    def test_local_rop_approved_value(self):
        """PSE-2.7 approved: LOCAL ROP = 1,733 tons."""
        assert LOCAL_ROP_TONS == 1733

    def test_imported_rop_approved_value(self):
        """PSE-2.7 approved: IMPORTED ROP = 6,958 tons."""
        assert IMPORTED_ROP_TONS == 6958

    def test_local_rop_formula(self):
        """ROP = (MIN_STOCK_DAYS + lead_time) × daily_rate, rounded half-up."""
        raw = (MIN_STOCK_DAYS + LOCAL_LEAD_TIME_DAYS) * DAILY_CONSUMPTION_LOCAL
        assert raw == pytest.approx(1732.5, abs=0.01)
        assert LOCAL_ROP_TONS == math.floor(raw + 0.5)   # round-half-up → 1733

    def test_imported_rop_formula(self):
        raw = (MIN_STOCK_DAYS + IMPORTED_LEAD_TIME_DAYS) * DAILY_CONSUMPTION_IMPORTED
        assert raw == pytest.approx(6957.5, abs=0.01)
        assert IMPORTED_ROP_TONS == math.floor(raw + 0.5)  # → 6958


# ===========================================================================
# _classify_supply_status — four status bands
# ===========================================================================

class TestClassifySupplyStatus:
    SS = SAFETY_STOCK_LOCAL_TONS   # 1237.5
    ROP = LOCAL_ROP_TONS           # 1733
    WATCH_CEILING = ROP * (1 + WATCH_BUFFER_PCT)  # 1733 * 1.15 = 1992.95

    def test_at_zero_is_critical(self):
        assert _classify_supply_status(0.0, self.SS, self.ROP) == STATUS_CRITICAL

    def test_at_safety_stock_is_critical(self):
        assert _classify_supply_status(self.SS, self.SS, self.ROP) == STATUS_CRITICAL

    def test_just_above_safety_stock_is_reorder(self):
        assert _classify_supply_status(self.SS + 0.1, self.SS, self.ROP) == STATUS_REORDER

    def test_at_rop_is_reorder(self):
        assert _classify_supply_status(self.ROP, self.SS, self.ROP) == STATUS_REORDER

    def test_just_above_rop_is_watch(self):
        assert _classify_supply_status(self.ROP + 0.1, self.SS, self.ROP) == STATUS_WATCH

    def test_at_watch_ceiling_is_watch(self):
        assert _classify_supply_status(self.WATCH_CEILING, self.SS, self.ROP) == STATUS_WATCH

    def test_just_above_watch_ceiling_is_safe(self):
        assert _classify_supply_status(self.WATCH_CEILING + 0.1, self.SS, self.ROP) == STATUS_SAFE

    def test_large_inventory_is_safe(self):
        assert _classify_supply_status(20_000.0, self.SS, self.ROP) == STATUS_SAFE

    def test_production_local_800t_is_critical(self):
        """Real production scenario: local = 800t → CRITICAL."""
        assert _classify_supply_status(800.0, SAFETY_STOCK_LOCAL_TONS, LOCAL_ROP_TONS) == STATUS_CRITICAL

    def test_production_imported_2000t_is_reorder(self):
        """Real production scenario: imported = 2000t → REORDER (< 6958 ROP)."""
        assert _classify_supply_status(2000.0, SAFETY_STOCK_IMPORTED_TONS, IMPORTED_ROP_TONS) == STATUS_REORDER

    def test_imported_1900_watch(self):
        """1900t local is just above LOCAL_ROP but within WATCH band."""
        assert _classify_supply_status(1900.0, SAFETY_STOCK_LOCAL_TONS, LOCAL_ROP_TONS) == STATUS_WATCH

    def test_imported_8000_watch(self):
        """8000t imported is above IMPORTED_ROP (6958) but below WATCH ceiling (6958×1.15=8001.7)."""
        assert _classify_supply_status(8000.0, SAFETY_STOCK_IMPORTED_TONS, IMPORTED_ROP_TONS) == STATUS_WATCH

    def test_imported_8100_safe(self):
        """8100t imported exceeds WATCH ceiling (8001.7) → SAFE."""
        assert _classify_supply_status(8100.0, SAFETY_STOCK_IMPORTED_TONS, IMPORTED_ROP_TONS) == STATUS_SAFE


# ===========================================================================
# compute_inventory_position
# ===========================================================================

class TestComputeInventoryPosition:
    def _df(self, local, imported, unknown=0.0):
        rows = []
        if local:
            rows.append({"org_name": "A", "origin": "LOCAL",    "tons": local})
        if imported:
            rows.append({"org_name": "A", "origin": "IMPORTED", "tons": imported})
        if unknown:
            rows.append({"org_name": "A", "origin": "UNKNOWN",  "tons": unknown})
        return pd.DataFrame(rows)

    def test_basic_mix(self):
        pos = compute_inventory_position(self._df(4000, 8000))
        assert pos["local_inventory_tons"] == pytest.approx(4000.0)
        assert pos["imported_inventory_tons"] == pytest.approx(8000.0)
        assert pos["total_inventory_tons"] == pytest.approx(12000.0)
        assert pos["local_mix_pct"] == pytest.approx(4000/12000*100, abs=0.01)

    def test_empty_dataframe_returns_zeros(self):
        pos = compute_inventory_position(pd.DataFrame())
        for key in ("local_inventory_tons", "imported_inventory_tons", "total_inventory_tons"):
            assert pos[key] == 0.0

    def test_none_dataframe_returns_zeros(self):
        pos = compute_inventory_position(None)
        assert pos["total_inventory_tons"] == 0.0

    def test_unknown_included_in_total_not_mix(self):
        pos = compute_inventory_position(self._df(4000, 8000, unknown=500))
        assert pos["total_inventory_tons"] == pytest.approx(12500.0)
        assert pos["unknown_inventory_tons"] == pytest.approx(500.0)
        assert pos["local_mix_pct"] == pytest.approx(4000/12000*100, abs=0.01)

    def test_days_cover_formula(self):
        pos = compute_inventory_position(self._df(4000, 8000))
        assert pos["local_days_cover"] == pytest.approx(4000 / DAILY_CONSUMPTION_LOCAL, abs=0.01)
        assert pos["imported_days_cover"] == pytest.approx(8000 / DAILY_CONSUMPTION_IMPORTED, abs=0.01)
        assert pos["total_days_cover"] == pytest.approx(12000 / DAILY_CONSUMPTION_TOTAL, abs=0.01)

    def test_mix_pcts_sum_to_100(self):
        pos = compute_inventory_position(self._df(4000, 8000))
        assert abs(pos["local_mix_pct"] + pos["imported_mix_pct"] - 100.0) < 0.01


# ===========================================================================
# compute_reorder_triggers
# ===========================================================================

class TestComputeReorderTriggers:
    def test_local_critical_imported_watch(self):
        """800t local → CRITICAL; 8000t imported is WATCH (below 8001.7 ceiling)."""
        t = compute_reorder_triggers(800.0, 8000.0)
        assert t["local_status"]    == STATUS_CRITICAL
        assert t["imported_status"] == STATUS_WATCH

    def test_imported_8100_gives_safe(self):
        t = compute_reorder_triggers(5000.0, 8100.0)
        assert t["imported_status"] == STATUS_SAFE

    def test_rop_constants_returned(self):
        t = compute_reorder_triggers(5000.0, 5000.0)
        assert t["local_reorder_trigger"]    == LOCAL_ROP_TONS
        assert t["imported_reorder_trigger"] == IMPORTED_ROP_TONS

    def test_safety_stock_returned(self):
        t = compute_reorder_triggers(5000.0, 5000.0)
        assert t["local_safety_stock_tons"]    == SAFETY_STOCK_LOCAL_TONS
        assert t["imported_safety_stock_tons"] == SAFETY_STOCK_IMPORTED_TONS
