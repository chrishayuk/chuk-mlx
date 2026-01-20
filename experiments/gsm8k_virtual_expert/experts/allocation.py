"""
Allocation Expert using CSP solver.

Handles distribution problems with constraints.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..extraction.gsm8k_extractor import ProblemSpec


@dataclass
class AllocationResult:
    """Result from allocation solving."""
    allocations: dict[str, float]
    steps: list[str]
    answer: float | None


class AllocationExpert:
    """
    Expert for allocation/distribution problems.

    Uses OR-Tools CP-SAT for constraint satisfaction.

    Example:
        "Split $100 among Alice, Bob, Carol where Alice gets twice what Bob gets"
        Variables: alice, bob, carol
        Constraints:
          - alice + bob + carol = 100
          - alice = 2 * bob
    """

    def __init__(self):
        self._ortools_available = self._check_ortools()

    def _check_ortools(self) -> bool:
        try:
            from ortools.sat.python import cp_model
            return True
        except Exception:
            return False

    def solve(self, spec: ProblemSpec) -> AllocationResult:
        """
        Solve allocation problem using CSP.
        """
        if not self._ortools_available:
            return self._solve_algebraic(spec)

        from ortools.sat.python import cp_model

        total = int(spec.metadata.get("total", 100))
        entities = [e.name for e in spec.entities]

        # Check for "equally" or "each" - indicates equal split
        # This handles cases like "Split $120 equally among 4 friends"
        problem_lower = spec.raw_text.lower() if spec.raw_text else ""
        if "equally" in problem_lower or ("each" in problem_lower and "friend" in problem_lower):
            n_match = re.search(r'(\d+)\s*(?:friends?|people|persons?|children|kids)', problem_lower)
            if n_match:
                n = int(n_match.group(1))
                per_entity = total / n
                allocations = {f"person_{i+1}": per_entity for i in range(n)}
                return AllocationResult(
                    allocations=allocations,
                    steps=[f"Equal split: {total} / {n} = {per_entity} each"],
                    answer=per_entity,
                )

        if len(entities) < 2:
            return AllocationResult(
                allocations={},
                steps=["Not enough entities to allocate"],
                answer=None,
            )

        steps = []
        model = cp_model.CpModel()

        # Create variables (0 to total for each entity)
        vars = {}
        for entity in entities:
            vars[entity] = model.NewIntVar(0, total, entity)

        steps.append(f"Variables: {list(vars.keys())}")

        # Sum constraint
        model.Add(sum(vars.values()) == total)
        steps.append(f"Constraint: sum = {total}")

        # Apply extracted constraints
        for constraint in spec.constraints:
            if constraint["type"] == "ratio":
                e1, e2, factor = constraint["entity1"], constraint["entity2"], constraint["factor"]
                if e1 in vars and e2 in vars:
                    model.Add(vars[e1] == int(factor) * vars[e2])
                    steps.append(f"Constraint: {e1} = {factor} × {e2}")

            elif constraint["type"] == "difference":
                e1, e2, diff = constraint["entity1"], constraint["entity2"], constraint["difference"]
                if e1 in vars and e2 in vars:
                    model.Add(vars[e1] == vars[e2] + diff)
                    steps.append(f"Constraint: {e1} = {e2} + {diff}")

        # Solve
        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            allocations = {name: solver.Value(var) for name, var in vars.items()}
            steps.append(f"Solution: {allocations}")

            # Determine answer based on target or first entity
            answer = None
            if spec.target and spec.target in allocations:
                answer = float(allocations[spec.target])
            elif allocations:
                answer = float(list(allocations.values())[0])

            return AllocationResult(
                allocations=allocations,
                steps=steps,
                answer=answer,
            )
        else:
            steps.append("No solution found")
            return AllocationResult(
                allocations={},
                steps=steps,
                answer=None,
            )

    def _solve_algebraic(self, spec: ProblemSpec) -> AllocationResult:
        """
        Fallback algebraic solution when OR-Tools unavailable.
        """
        total = spec.metadata.get("total", 100)
        entities = [e.name for e in spec.entities]
        steps = ["Using algebraic method (OR-Tools not available)"]

        # Simple two-entity case with ratio
        if len(entities) == 2 and spec.constraints:
            for c in spec.constraints:
                if c["type"] == "ratio":
                    factor = c["factor"]
                    # e1 = factor * e2, e1 + e2 = total
                    # factor * e2 + e2 = total
                    # e2 * (factor + 1) = total
                    e2_val = total / (factor + 1)
                    e1_val = factor * e2_val

                    allocations = {
                        c["entity1"]: e1_val,
                        c["entity2"]: e2_val,
                    }
                    steps.append(f"Solved: {allocations}")

                    return AllocationResult(
                        allocations=allocations,
                        steps=steps,
                        answer=e1_val if spec.target == c["entity1"] else e2_val,
                    )

        # Check for "equally" or "each" - indicates equal split
        problem_lower = spec.raw_text.lower() if spec.raw_text else ""
        if "equally" in problem_lower or "each" in problem_lower:
            # Find the number of entities from problem text
            n_match = re.search(r'(\d+)\s*(?:friends?|people|person|children|kids)', problem_lower)
            if n_match:
                n = int(n_match.group(1))
                per_entity = total / n
                allocations = {f"person_{i+1}": per_entity for i in range(n)}
                steps.append(f"Equal split: {total} / {n} = {per_entity} each")
                return AllocationResult(
                    allocations=allocations,
                    steps=steps,
                    answer=per_entity,
                )

        # Equal split fallback
        if entities:
            per_entity = total / len(entities)
            allocations = {e: per_entity for e in entities}
            steps.append(f"Equal split: {per_entity} each")

            return AllocationResult(
                allocations=allocations,
                steps=steps,
                answer=per_entity,
            )

        return AllocationResult(allocations={}, steps=steps, answer=None)

    def solve_from_spec(self, spec: ProblemSpec) -> float | None:
        """Solve and return numeric answer."""
        result = self.solve(spec)
        return result.answer
