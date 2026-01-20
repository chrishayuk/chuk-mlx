"""
Comparison Expert.

Handles problems comparing quantities and finding differences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..extraction.gsm8k_extractor import ProblemSpec


@dataclass
class ComparisonResult:
    """Result from comparison calculation."""
    values: dict[str, float]
    difference: float | None
    answer: float | None
    steps: list[str]


class ComparisonExpert:
    """
    Expert for comparison problems.

    Pattern: "X has N times as many as Y. Y has M. How many more does X have?"
    """

    def solve(self, spec: ProblemSpec) -> ComparisonResult:
        """
        Solve comparison problem.
        """
        steps = []
        values = {}

        # Build values from entities
        for entity in spec.entities:
            if entity.value is not None:
                values[entity.name] = entity.value
                steps.append(f"{entity.name} has {entity.value}")

        # Apply relationships
        relationship = spec.metadata.get("relationship")
        factor = spec.metadata.get("factor", 1)

        if relationship == "multiply":
            # Find entity with known value and entity that needs to be computed
            known_entity = None
            unknown_entity = None

            for entity in spec.entities:
                if entity.value is not None:
                    known_entity = entity.name

            # The entity NOT in values is the unknown one
            for entity in spec.entities:
                if entity.name not in values:
                    unknown_entity = entity.name

            if known_entity and unknown_entity:
                # The unknown entity has factor times the known
                unknown_val = values[known_entity] * factor
                values[unknown_entity] = unknown_val
                steps.append(f"{unknown_entity} = {factor} × {values[known_entity]} = {unknown_val}")
            elif len(spec.entities) >= 2 and factor > 1:
                # Both entities might have been extracted - check metadata
                # In "Tom has 3x as many as Jane. Jane has 5", Tom = 3 * Jane
                entity_names = [e.name for e in spec.entities]
                if len(values) == 1:
                    known = list(values.keys())[0]
                    unknown = [n for n in entity_names if n != known][0] if len(entity_names) > 1 else None
                    if unknown:
                        unknown_val = values[known] * factor
                        values[unknown] = unknown_val
                        steps.append(f"{unknown} = {factor} × {values[known]} = {unknown_val}")

        # Calculate difference
        difference = None
        if len(values) >= 2:
            sorted_vals = sorted(values.values(), reverse=True)
            difference = sorted_vals[0] - sorted_vals[1]
            steps.append(f"Difference = {sorted_vals[0]} - {sorted_vals[1]} = {difference}")

        # Determine answer
        answer = None
        if spec.target == "difference":
            answer = difference
        elif spec.target and spec.target in values:
            answer = values[spec.target]
        elif difference is not None:
            answer = difference

        return ComparisonResult(
            values=values,
            difference=difference,
            answer=answer,
            steps=steps,
        )

    def solve_from_spec(self, spec: ProblemSpec) -> float | None:
        """Solve and return numeric answer."""
        result = self.solve(spec)
        return result.answer
