"""Economic guardrails for decision layer outputs.

Guardrails ensure business-safe outputs without weakening analytical logic:
1. Price floors: prevent negative/impossible prices
2. Scenario caps: limit extreme % moves to realistic ranges
3. Nonlinear dampening: reduce impact of tail shocks
4. Business labeling: flag extreme scenarios for review

Design:
- Applied AFTER elasticity computation, BEFORE business decisions
- Transparent: flags when applied
- Configurable: per-asset tolerance levels
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class ExtremeFlag(Enum):
    """Severity level for extreme scenarios."""
    NORMAL = "NORMAL"
    ELEVATED = "ELEVATED"  # Unusual but plausible
    EXTREME = "EXTREME"  # Tail risk; needs review
    IMPLAUSIBLE = "IMPLAUSIBLE"  # Outside reasonable bounds


@dataclass
class GuardrailConfig:
    """Configuration for guardrails."""
    asset_id: str
    price_floor: float  # minimum reasonable price (e.g., 0.10 for cotton = $0.10/cwt)
    max_move_pct: float = 25.0  # maximum single-scenario move (%)
    dampening_threshold_pct: float = 15.0  # apply dampening above this move
    dampening_factor: float = 0.5  # nonlinear dampening (0 = full, 1 = no dampening)
    warning_threshold_pct: float = 20.0  # flag as ELEVATED above this
    extreme_threshold_pct: float = 30.0  # flag as EXTREME above this

    @classmethod
    def for_cotton(cls) -> "GuardrailConfig":
        """Preset guardrails for cotton commodity."""
        return cls(
            asset_id="COTTON",
            price_floor=1.0,  # USD/cwt minimum
            max_move_pct=35.0,  # cotton can be volatile
            dampening_threshold_pct=20.0,
            dampening_factor=0.6,
            warning_threshold_pct=15.0,
            extreme_threshold_pct=25.0,
        )

    @classmethod
    def for_eurusd(cls) -> "GuardrailConfig":
        """Preset guardrails for FX."""
        return cls(
            asset_id="EURUSD",
            price_floor=0.5,  # EUR/USD minimum
            max_move_pct=10.0,  # FX less volatile
            dampening_threshold_pct=7.0,
            dampening_factor=0.7,
            warning_threshold_pct=5.0,
            extreme_threshold_pct=8.0,
        )


@dataclass
class GuardrailResult:
    """Output of guardrail application."""
    original_price: float
    guardrailed_price: float
    price_change_pct: float  # original change %
    guardrailed_pct: float  # after guardrails
    dampening_applied: bool
    floor_applied: bool
    extreme_flag: ExtremeFlag
    adjustments: list[str] = field(default_factory=list)


class GuardrailEngine:
    """Apply economic guardrails to scenario outputs."""

    def __init__(self, config: GuardrailConfig):
        self.config = config

    def apply_guardrails(
        self,
        base_price: float,
        scenario_price: float,
        scenario_name: str = "scenario",
    ) -> GuardrailResult:
        """Apply guardrails to a scenario price outcome.

        Args:
            base_price: ML forecast baseline
            scenario_price: scenario result (may be unrealistic)
            scenario_name: for logging

        Returns:
            GuardrailResult with guardrailed price and flags.
        """
        adjustments = []
        guardrailed_price = scenario_price
        dampening_applied = False
        floor_applied = False

        # Compute original % change
        price_change_pct = (scenario_price - base_price) / base_price * 100.0 if base_price != 0 else 0.0

        # 1. Check for extreme moves; apply nonlinear dampening if needed
        abs_move = abs(price_change_pct)
        if abs_move > self.config.dampening_threshold_pct:
            # Apply dampening: reduce impact of tail moves
            dampened_move = self._nonlinear_dampening(price_change_pct)
            guardrailed_price = base_price * (1.0 + dampened_move / 100.0)
            dampening_applied = True
            adjustments.append(
                f"Nonlinear dampening applied: {price_change_pct:.1f}% → {dampened_move:.1f}%"
            )

        # 2. Hard cap on max move
        if abs_move > self.config.max_move_pct:
            capped_move = (
                self.config.max_move_pct if price_change_pct > 0 else -self.config.max_move_pct
            )
            guardrailed_price = base_price * (1.0 + capped_move / 100.0)
            adjustments.append(
                f"Max move cap applied: {price_change_pct:.1f}% → {capped_move:.1f}%"
            )

        # 3. Price floor: prevent negative or implausibly low prices
        if guardrailed_price < self.config.price_floor:
            guardrailed_price = self.config.price_floor
            floor_applied = True
            adjustments.append(
                f"Price floor applied: ${guardrailed_price:.4f} (minimum realistic price)"
            )

        # 4. Classify extremity
        final_move_pct = (guardrailed_price - base_price) / base_price * 100.0 if base_price != 0 else 0.0
        abs_final_move = abs(final_move_pct)

        if abs_final_move > self.config.extreme_threshold_pct:
            extreme_flag = ExtremeFlag.EXTREME
        elif abs_final_move > self.config.warning_threshold_pct:
            extreme_flag = ExtremeFlag.ELEVATED
        else:
            extreme_flag = ExtremeFlag.NORMAL

        # If floor was applied and move is huge, mark as implausible
        if floor_applied and abs(price_change_pct) > 50.0:
            extreme_flag = ExtremeFlag.IMPLAUSIBLE

        return GuardrailResult(
            original_price=scenario_price,
            guardrailed_price=guardrailed_price,
            price_change_pct=price_change_pct,
            guardrailed_pct=final_move_pct,
            dampening_applied=dampening_applied,
            floor_applied=floor_applied,
            extreme_flag=extreme_flag,
            adjustments=adjustments,
        )

    def _nonlinear_dampening(self, price_change_pct: float) -> float:
        """Apply nonlinear dampening to large shocks.

        Uses a smooth dampening curve: for moves beyond threshold,
        dampen increasingly as move gets more extreme.

        Formula: dampened = sign(x) * threshold + (x - threshold) * dampening_factor
        """
        threshold = self.config.dampening_threshold_pct
        factor = self.config.dampening_factor
        sign = 1.0 if price_change_pct >= 0 else -1.0
        abs_move = abs(price_change_pct)

        if abs_move <= threshold:
            return price_change_pct

        # Beyond threshold, dampen the excess
        excess = abs_move - threshold
        dampened_excess = excess * factor
        dampened_move = threshold + dampened_excess

        return sign * dampened_move

    def batch_guardrails(self, base_price: float, scenario_results: dict) -> dict:
        """Apply guardrails to multiple scenario prices.

        Args:
            base_price: baseline forecast
            scenario_results: {scenario_name: price}

        Returns:
            {scenario_name: GuardrailResult}
        """
        return {
            name: self.apply_guardrails(base_price, price, name)
            for name, price in scenario_results.items()
        }

    def summarize_guardrails(self, results: dict) -> str:
        """Generate human-readable summary of guardrail applications."""
        lines = [f"\n{self.config.asset_id} Guardrail Summary:"]
        lines.append("-" * 60)

        normal_count = sum(1 for r in results.values() if r.extreme_flag == ExtremeFlag.NORMAL)
        elevated_count = sum(1 for r in results.values() if r.extreme_flag == ExtremeFlag.ELEVATED)
        extreme_count = sum(1 for r in results.values() if r.extreme_flag == ExtremeFlag.EXTREME)
        implausible_count = sum(
            1 for r in results.values() if r.extreme_flag == ExtremeFlag.IMPLAUSIBLE
        )

        lines.append(f"Normal scenarios:      {normal_count}")
        lines.append(f"Elevated risk:        {elevated_count}")
        lines.append(f"Extreme scenarios:    {extreme_count}")
        lines.append(f"Implausible:          {implausible_count}")

        any_dampened = sum(1 for r in results.values() if r.dampening_applied)
        any_floored = sum(1 for r in results.values() if r.floor_applied)

        if any_dampened:
            lines.append(f"\nNonlinear dampening applied to {any_dampened} scenarios")
        if any_floored:
            lines.append(f"Price floor applied to {any_floored} scenarios")

        return "\n".join(lines)
