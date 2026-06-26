"""
test_calendar_engine.py
-----------------------
Tests for procurement_calendar_engine.py (PSE-4A).
Covers: event structure, ordering, horizon coverage, trigger types.
"""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from procurement_strategy_engine import (
    LOCAL_ROP_TONS, IMPORTED_ROP_TONS,
    SAFETY_STOCK_LOCAL_TONS, SAFETY_STOCK_IMPORTED_TONS,
    STATUS_CRITICAL, STATUS_REORDER, STATUS_WATCH, STATUS_SAFE,
)
from procurement_calendar_engine import (
    compute_procurement_calendar,
    TRIGGER_IMMEDIATE,
    TRIGGER_PROJECTED_REORDER,
    VIEW_HORIZONS_DAYS,
    DEFAULT_HORIZON_DAYS,
)

TODAY = date(2026, 6, 27)


# ===========================================================================
# Helpers
# ===========================================================================

def _cal(local_inv, imp_inv, local_status, imp_status, horizon=DEFAULT_HORIZON_DAYS):
    return compute_procurement_calendar(
        local_inventory_tons=local_inv,
        imported_inventory_tons=imp_inv,
        local_status=local_status,
        imported_status=imp_status,
        total_inventory_tons=local_inv + imp_inv,
        today=TODAY,
        horizon_days=horizon,
    )


# ===========================================================================
# Output structure
# ===========================================================================

class TestCalendarStructure:
    def test_returns_required_keys(self):
        cal = _cal(800.0, 8000.0, STATUS_CRITICAL, STATUS_SAFE)
        for key in ("events", "view_30_days", "view_90_days", "view_180_days"):
            assert key in cal

    def test_events_is_list(self):
        cal = _cal(800.0, 8000.0, STATUS_CRITICAL, STATUS_SAFE)
        assert isinstance(cal["events"], list)

    def test_each_event_has_required_fields(self):
        cal = _cal(800.0, 8000.0, STATUS_CRITICAL, STATUS_SAFE)
        # Field is 'expected_arrival_date', not 'arrival_date'
        required = {"order_date", "expected_arrival_date", "quantity_tons", "source", "trigger"}
        for ev in cal["events"]:
            for field in required:
                assert field in ev, f"Event missing field: {field}"

    def test_view_keys_are_lists(self):
        cal = _cal(800.0, 8000.0, STATUS_CRITICAL, STATUS_SAFE)
        for key in ("view_30_days", "view_90_days", "view_180_days"):
            assert isinstance(cal[key], list)


# ===========================================================================
# Event ordering and horizon compliance
# ===========================================================================

class TestEventOrdering:
    def test_events_ordered_by_order_date(self):
        cal = _cal(800.0, 2000.0, STATUS_CRITICAL, STATUS_REORDER)
        dates = [ev["order_date"] for ev in cal["events"]]
        assert dates == sorted(dates), "Events must be ordered by order_date ascending"

    def test_no_event_beyond_horizon(self):
        horizon = 90
        cal = _cal(800.0, 8000.0, STATUS_CRITICAL, STATUS_SAFE, horizon=horizon)
        cutoff = TODAY + timedelta(days=horizon)
        for ev in cal["events"]:
            ev_date = date.fromisoformat(ev["order_date"])
            assert ev_date <= cutoff, f"Event {ev['order_date']} beyond horizon {cutoff}"

    def test_immediate_event_on_today_for_critical(self):
        cal = _cal(800.0, 8000.0, STATUS_CRITICAL, STATUS_SAFE)
        immediate = [ev for ev in cal["events"] if ev["trigger"] == TRIGGER_IMMEDIATE]
        assert len(immediate) >= 1
        assert any(ev["order_date"] == TODAY.isoformat() for ev in immediate)


# ===========================================================================
# Quantities
# ===========================================================================

class TestCalendarQuantities:
    def test_immediate_quantity_non_negative(self):
        cal = _cal(800.0, 2000.0, STATUS_CRITICAL, STATUS_REORDER)
        for ev in cal["events"]:
            assert ev["quantity_tons"] >= 0.0

    def test_local_immediate_covers_deficit(self):
        local_inv = 800.0
        cal = _cal(local_inv, 8000.0, STATUS_CRITICAL, STATUS_SAFE)
        local_immediate = sum(
            ev["quantity_tons"]
            for ev in cal["events"]
            if ev["source"] == "LOCAL" and ev["trigger"] == TRIGGER_IMMEDIATE
        )
        deficit = max(0, LOCAL_ROP_TONS - local_inv)
        assert local_immediate >= deficit - 0.01

    def test_source_labels_valid(self):
        cal = _cal(800.0, 2000.0, STATUS_CRITICAL, STATUS_REORDER)
        for ev in cal["events"]:
            assert ev["source"] in ("LOCAL", "IMPORTED"), f"Unknown source: {ev['source']}"

    def test_arrival_after_order_date(self):
        cal = _cal(800.0, 2000.0, STATUS_CRITICAL, STATUS_REORDER)
        for ev in cal["events"]:
            order   = date.fromisoformat(ev["order_date"])
            arrival = date.fromisoformat(ev["expected_arrival_date"])
            assert arrival >= order, f"Arrival before order date for {ev}"


# ===========================================================================
# Horizon-based view counts
# ===========================================================================

class TestHorizonViews:
    def test_30d_view_subset_of_90d_view(self):
        cal = _cal(800.0, 2000.0, STATUS_CRITICAL, STATUS_REORDER, horizon=180)
        events_30 = {e["order_date"] for e in cal["view_30_days"]}
        events_90 = {e["order_date"] for e in cal["view_90_days"]}
        # All events visible within 30 days must also appear in the 90-day view
        assert events_30.issubset(events_90), (
            f"30-day events not a subset of 90-day: {events_30 - events_90}"
        )

    def test_view_30d_max_date(self):
        cal = _cal(800.0, 2000.0, STATUS_CRITICAL, STATUS_REORDER, horizon=180)
        cutoff = TODAY + timedelta(days=30)
        for ev in cal["view_30_days"]:
            assert date.fromisoformat(ev["order_date"]) <= cutoff
