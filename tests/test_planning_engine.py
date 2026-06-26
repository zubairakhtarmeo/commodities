"""
test_planning_engine.py
-----------------------
Tests for procurement_planning_engine.py (PSE-3D).
Covers: requirement sizing, timing, price signals, run_pse3d integration.
"""
from __future__ import annotations

from datetime import date

import pytest

from procurement_strategy_engine import (
    LOCAL_ROP_TONS, IMPORTED_ROP_TONS,
    SAFETY_STOCK_LOCAL_TONS, SAFETY_STOCK_IMPORTED_TONS,
    STATUS_CRITICAL, STATUS_REORDER, STATUS_WATCH, STATUS_SAFE,
)
from procurement_planning_engine import (
    PRICE_RISING, PRICE_FALLING, PRICE_NEUTRAL,
    LB_PER_METRIC_TON,
    compute_procurement_requirement,
    compute_purchase_timing,
    compute_price_signal,
    run_pse3d,
)

TODAY = date(2026, 6, 27)
CURRENT_PRICE = 0.78
PKR_RATE = 278.5


# ===========================================================================
# compute_procurement_requirement
# ===========================================================================

class TestComputeProcurementRequirement:
    def test_above_rop_no_deficit(self):
        req = compute_procurement_requirement(
            local_inventory_tons=5000.0,
            imported_inventory_tons=10000.0,
        )
        assert req["deficit_local_tons"] == 0.0
        assert req["deficit_imported_tons"] == 0.0

    def test_below_rop_has_deficit(self):
        req = compute_procurement_requirement(
            local_inventory_tons=800.0,
            imported_inventory_tons=2000.0,
        )
        assert req["deficit_local_tons"] == pytest.approx(LOCAL_ROP_TONS - 800.0, abs=0.01)
        assert req["deficit_imported_tons"] == pytest.approx(IMPORTED_ROP_TONS - 2000.0, abs=0.01)

    def test_capacity_constraint_caps_total(self):
        """Total purchase must not exceed headroom when near capacity ceiling."""
        req = compute_procurement_requirement(
            local_inventory_tons=800.0,
            imported_inventory_tons=44000.0,
            total_inventory_tons=44800.0,
            max_storage_capacity_tons=45_000.0,
        )
        assert req["capacity_constrained"] is True
        headroom = 45_000.0 - 44800.0
        assert req["local_qty_required"] + req["imported_qty_required"] <= headroom + 0.01

    def test_no_capacity_constraint_when_below_ceiling(self):
        req = compute_procurement_requirement(
            local_inventory_tons=800.0,
            imported_inventory_tons=2000.0,
        )
        assert req["capacity_constrained"] is False

    def test_deficit_priority_over_mix_correction(self):
        """Security rule: structural deficit always >= mix correction quantity."""
        req = compute_procurement_requirement(
            local_inventory_tons=800.0,
            imported_inventory_tons=8000.0,
        )
        assert req["local_qty_required"] >= req["mix_correction_local_tons"]

    def test_zero_inventory_full_deficit(self):
        req = compute_procurement_requirement(
            local_inventory_tons=0.0,
            imported_inventory_tons=0.0,
        )
        assert req["deficit_local_tons"] == pytest.approx(LOCAL_ROP_TONS, abs=0.01)
        assert req["deficit_imported_tons"] == pytest.approx(IMPORTED_ROP_TONS, abs=0.01)

    def test_future_requirement_positive_when_stock_low(self):
        req = compute_procurement_requirement(
            local_inventory_tons=800.0,
            imported_inventory_tons=2000.0,
        )
        assert req["future_requirement"] >= 0.0

    def test_non_negative_all_outputs(self):
        req = compute_procurement_requirement(
            local_inventory_tons=20000.0,
            imported_inventory_tons=15000.0,
        )
        for key, val in req.items():
            if isinstance(val, float):
                assert val >= 0.0, f"{key} should be non-negative"

    def test_45000_ton_capacity_constant(self):
        """PSE-1 business rule: storage ceiling is 45,000 tons."""
        req = compute_procurement_requirement(
            local_inventory_tons=0.0,
            imported_inventory_tons=0.0,
            max_storage_capacity_tons=45_000.0,
        )
        headroom = 45_000.0 - 0.0
        assert req["local_qty_required"] + req["imported_qty_required"] <= headroom + 0.01


# ===========================================================================
# compute_purchase_timing
# ===========================================================================

class TestComputePurchaseTiming:
    def test_critical_order_today(self):
        t = compute_purchase_timing(
            inventory_tons=800.0,
            status=STATUS_CRITICAL,
            reorder_point_tons=LOCAL_ROP_TONS,
            daily_consumption_rate=49.5,
            today=TODAY,
        )
        assert t["recommended_order_date"] == TODAY.isoformat()
        # deferral_window_days == 0 because CRITICAL stock is already at or below SS
        assert t["deferral_window_days"] == 0

    def test_reorder_order_today(self):
        t = compute_purchase_timing(
            inventory_tons=1500.0,
            status=STATUS_REORDER,
            reorder_point_tons=LOCAL_ROP_TONS,
            daily_consumption_rate=49.5,
            today=TODAY,
        )
        assert t["recommended_order_date"] == TODAY.isoformat()

    def test_safe_rising_order_date_not_later_than_neutral(self):
        t_rising = compute_purchase_timing(
            inventory_tons=4000.0,
            status=STATUS_SAFE,
            reorder_point_tons=LOCAL_ROP_TONS,
            daily_consumption_rate=49.5,
            price_signal=PRICE_RISING,
            today=TODAY,
        )
        t_neutral = compute_purchase_timing(
            inventory_tons=4000.0,
            status=STATUS_SAFE,
            reorder_point_tons=LOCAL_ROP_TONS,
            daily_consumption_rate=49.5,
            price_signal=PRICE_NEUTRAL,
            today=TODAY,
        )
        from datetime import date as _date
        rec_rising  = _date.fromisoformat(t_rising["recommended_order_date"])
        rec_neutral = _date.fromisoformat(t_neutral["recommended_order_date"])
        # Rising price signal can only pull order earlier, never later
        assert rec_rising <= rec_neutral

    def test_order_date_never_after_latest_safe(self):
        for status in (STATUS_CRITICAL, STATUS_REORDER, STATUS_WATCH, STATUS_SAFE):
            t = compute_purchase_timing(
                inventory_tons=2000.0,
                status=status,
                reorder_point_tons=LOCAL_ROP_TONS,
                daily_consumption_rate=49.5,
                today=TODAY,
            )
            from datetime import date as _date
            rec = _date.fromisoformat(t["recommended_order_date"])
            safe = _date.fromisoformat(t["latest_safe_order_date"])
            assert rec <= safe, f"status={status}: recommended after latest safe"


# ===========================================================================
# compute_price_signal
# ===========================================================================

class TestComputePriceSignal:
    """compute_price_signal returns a dict with keys: signal, price_delta_pct, confidence, spread_pct."""

    def test_rising_signal(self):
        r = compute_price_signal(0.78, 0.97, None)
        assert r["signal"] == PRICE_RISING

    def test_falling_signal(self):
        r = compute_price_signal(0.78, 0.60, None)
        assert r["signal"] == PRICE_FALLING

    def test_neutral_within_threshold(self):
        r = compute_price_signal(0.78, 0.785, None)
        assert r["signal"] == PRICE_NEUTRAL

    def test_delta_pct_positive_when_rising(self):
        r = compute_price_signal(0.78, 0.97, None)
        assert r["price_delta_pct"] > 0   # forecast above current → positive delta

    def test_delta_pct_negative_when_falling(self):
        r = compute_price_signal(0.78, 0.60, None)
        assert r["price_delta_pct"] < 0

    def test_zero_current_price_no_crash(self):
        r = compute_price_signal(0.0, 0.97, None)
        assert r["signal"] in (PRICE_RISING, PRICE_NEUTRAL, PRICE_FALLING, None)

    def test_returns_dict_with_required_keys(self):
        r = compute_price_signal(0.78, 0.97, None)
        assert isinstance(r, dict)
        for key in ("signal", "price_delta_pct", "confidence"):
            assert key in r


# ===========================================================================
# run_pse3d (integration of planning pipeline)
# ===========================================================================

class TestRunPse3d:
    def test_critical_local_returns_plan(self):
        plan = run_pse3d(
            local_inventory_tons=800.0,
            imported_inventory_tons=8000.0,
            local_status=STATUS_CRITICAL,
            imported_status=STATUS_SAFE,
            total_inventory_tons=8800.0,
            current_price_usd_per_lb=CURRENT_PRICE,
            forecast_h1_usd_per_lb=0.97,
            forecast_h3_usd_per_lb=1.06,
            pkr_rate=PKR_RATE,
            today=TODAY,
        )
        assert plan is not None
        assert plan.should_buy_now is True

    def test_plan_has_required_keys(self):
        plan = run_pse3d(
            local_inventory_tons=4000.0,
            imported_inventory_tons=8000.0,
            local_status=STATUS_SAFE,
            imported_status=STATUS_SAFE,
            total_inventory_tons=12000.0,
            current_price_usd_per_lb=CURRENT_PRICE,
            forecast_h1_usd_per_lb=0.78,
            forecast_h3_usd_per_lb=0.78,
            pkr_rate=PKR_RATE,
            today=TODAY,
        )
        d = plan.to_dict()
        for key in ("run_date", "should_buy_now", "primary_action",
                    "local_recommendation", "imported_recommendation",
                    "requirement", "risk_level"):
            assert key in d, f"Missing key: {key}"

    def test_recommendation_has_order_date(self):
        plan = run_pse3d(
            local_inventory_tons=800.0,
            imported_inventory_tons=2000.0,
            local_status=STATUS_CRITICAL,
            imported_status=STATUS_REORDER,
            total_inventory_tons=2800.0,
            current_price_usd_per_lb=CURRENT_PRICE,
            forecast_h1_usd_per_lb=0.78,
            forecast_h3_usd_per_lb=0.78,
            pkr_rate=PKR_RATE,
            today=TODAY,
        )
        assert plan.local_recommendation.get("latest_safe_order_date") is not None
        assert plan.imported_recommendation.get("latest_safe_order_date") is not None

    def test_non_negative_quantities(self):
        plan = run_pse3d(
            local_inventory_tons=800.0,
            imported_inventory_tons=2000.0,
            local_status=STATUS_CRITICAL,
            imported_status=STATUS_REORDER,
            total_inventory_tons=2800.0,
            current_price_usd_per_lb=CURRENT_PRICE,
            forecast_h1_usd_per_lb=0.78,
            forecast_h3_usd_per_lb=0.78,
            pkr_rate=PKR_RATE,
            today=TODAY,
        )
        assert plan.requirement["local_qty_required"] >= 0.0
        assert plan.requirement["imported_qty_required"] >= 0.0
