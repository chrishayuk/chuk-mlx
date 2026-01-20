"""
Percentage Calculator Expert.

Handles percentage calculations: discounts, taxes, tips, changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..extraction.gsm8k_extractor import ProblemSpec


@dataclass
class PercentageResult:
    """Result from percentage calculation."""
    final_value: float | None
    percentage_amount: float | None
    steps: list[str]


class PercentageExpert:
    """
    Expert for percentage problems.

    Patterns:
    - Discount: final = base × (1 - percent/100)
    - Tax/Tip: final = base × (1 + percent/100)
    - Percent change: change = (new - old) / old × 100
    """

    def solve(self, spec: ProblemSpec) -> PercentageResult:
        """
        Solve percentage problem.
        """
        base = spec.metadata.get("base_value", 100)
        percent = spec.metadata.get("percentage", 0)
        operation = spec.metadata.get("operation", "discount")

        steps = []
        steps.append(f"Base value: ${base:.2f}")
        steps.append(f"Percentage: {percent}%")

        if operation in ("discount", "off", "decrease"):
            # Subtract percentage
            amount = base * (percent / 100)
            final = base - amount
            steps.append(f"Discount amount: ${base:.2f} × {percent}% = ${amount:.2f}")
            steps.append(f"Final price: ${base:.2f} - ${amount:.2f} = ${final:.2f}")

        elif operation in ("tax", "tip", "increase"):
            # Add percentage
            amount = base * (percent / 100)
            final = base + amount
            steps.append(f"{operation.title()} amount: ${base:.2f} × {percent}% = ${amount:.2f}")
            steps.append(f"Final total: ${base:.2f} + ${amount:.2f} = ${final:.2f}")

        elif operation == "find_percent":
            # Find percentage change
            new_val = spec.metadata.get("new_value", base)
            change = ((new_val - base) / base) * 100
            final = change
            amount = change
            steps.append(f"Change: ({new_val} - {base}) / {base} × 100 = {change:.1f}%")

        else:
            final = base * (1 - percent / 100)
            amount = base * (percent / 100)
            steps.append(f"Result: ${final:.2f}")

        return PercentageResult(
            final_value=final,
            percentage_amount=amount,
            steps=steps,
        )

    def solve_from_spec(self, spec: ProblemSpec) -> float | None:
        """Solve and return numeric answer."""
        result = self.solve(spec)
        return result.final_value
