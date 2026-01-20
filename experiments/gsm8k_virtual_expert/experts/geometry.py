"""
Geometry Expert.

Handles area, perimeter, volume calculations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ..extraction.gsm8k_extractor import ProblemSpec


@dataclass
class GeometryResult:
    """Result from geometry calculation."""
    value: float | None
    unit: str
    formula: str
    steps: list[str]


class GeometryExpert:
    """
    Expert for geometry problems.

    Applies standard formulas for common shapes.
    """

    FORMULAS = {
        "rectangle": {
            "area": lambda l, w: l * w,
            "perimeter": lambda l, w: 2 * (l + w),
        },
        "square": {
            "area": lambda s, _: s * s,
            "perimeter": lambda s, _: 4 * s,
        },
        "circle": {
            "area": lambda r, _: math.pi * r * r,
            "circumference": lambda r, _: 2 * math.pi * r,
        },
        "triangle": {
            "area": lambda b, h: 0.5 * b * h,
            "perimeter": lambda a, b, c=None: a + b + (c if c else a),
        },
        "cube": {
            "volume": lambda s, _: s ** 3,
            "surface_area": lambda s, _: 6 * s * s,
        },
    }

    def solve(self, spec: ProblemSpec) -> GeometryResult:
        """
        Solve geometry problem using appropriate formula.
        """
        shape = spec.metadata.get("shape", "rectangle")
        target = spec.target or "area"
        steps = []

        # Get dimensions
        length = spec.metadata.get("length") or spec.metadata.get("dimension", 1)
        width = spec.metadata.get("width", length)

        steps.append(f"Shape: {shape}")
        steps.append(f"Dimensions: {length} × {width}" if width != length else f"Side: {length}")

        # Get formula
        if shape in self.FORMULAS and target in self.FORMULAS[shape]:
            formula_fn = self.FORMULAS[shape][target]
            value = formula_fn(length, width)

            formula_str = self._get_formula_string(shape, target)
            steps.append(f"Formula: {formula_str}")
            steps.append(f"Result: {value:.2f}")

            # Determine unit
            unit = self._get_unit(target)

            return GeometryResult(
                value=value,
                unit=unit,
                formula=formula_str,
                steps=steps,
            )

        steps.append(f"Unknown formula for {shape} {target}")
        return GeometryResult(value=None, unit="", formula="", steps=steps)

    def _get_formula_string(self, shape: str, target: str) -> str:
        """Get human-readable formula string."""
        formulas = {
            ("rectangle", "area"): "A = l × w",
            ("rectangle", "perimeter"): "P = 2(l + w)",
            ("square", "area"): "A = s²",
            ("square", "perimeter"): "P = 4s",
            ("circle", "area"): "A = πr²",
            ("circle", "circumference"): "C = 2πr",
            ("triangle", "area"): "A = ½bh",
            ("cube", "volume"): "V = s³",
        }
        return formulas.get((shape, target), "?")

    def _get_unit(self, target: str) -> str:
        """Get unit for target property."""
        if target in ("area", "surface_area"):
            return "sq units"
        elif target == "volume":
            return "cubic units"
        else:
            return "units"

    def solve_from_spec(self, spec: ProblemSpec) -> float | None:
        """Solve and return numeric answer."""
        result = self.solve(spec)
        return result.value
