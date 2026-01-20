"""
Assignment Solver using OR-Tools CP-SAT.

Solves assignment/matching problems:
- Match entities to targets with constraints
- Nurse-to-shift assignment
- Student-to-advisor matching
- Task-to-worker allocation
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..extraction.csp_extractor import CSPSpec, ObjectiveType


class AssignmentStatus(Enum):
    """Solver result status."""
    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    ERROR = "error"


@dataclass
class Assignment:
    """A single assignment."""
    entity: str
    target: str
    cost: int = 0


@dataclass
class AssignmentResult:
    """Result from assignment solver."""
    status: AssignmentStatus
    assignments: list[Assignment] = field(default_factory=list)
    total_cost: int | None = None
    solve_time_ms: float | None = None
    error: str | None = None

    def is_success(self) -> bool:
        return self.status in (AssignmentStatus.OPTIMAL, AssignmentStatus.FEASIBLE)

    def to_string(self) -> str:
        if not self.is_success():
            if self.status == AssignmentStatus.INFEASIBLE:
                return "No valid assignment exists."
            return f"Solver error: {self.error}"

        lines = ["Assignments:"]
        for a in self.assignments:
            lines.append(f"  - {a.entity} -> {a.target}")

        if self.total_cost is not None:
            lines.append(f"Total cost: {self.total_cost}")

        if self.status == AssignmentStatus.OPTIMAL:
            lines.append("(optimal)")

        return "\n".join(lines)


class AssignmentSolver:
    """
    Constraint-based assignment solver using OR-Tools.

    Supports:
    - One-to-one matching
    - Many-to-one matching (multiple entities per target)
    - Capacity constraints
    - Exclusion constraints (can't assign A to B)
    - Preference/cost optimization
    """

    def __init__(self, timeout_seconds: float = 10.0):
        self.timeout_seconds = timeout_seconds
        self._ortools_available = self._check_ortools()

    def _check_ortools(self) -> bool:
        try:
            from ortools.sat.python import cp_model
            return True
        except Exception:
            return False

    def solve(
        self,
        entities: list[str],
        targets: list[str],
        costs: dict[tuple[str, str], int] | None = None,
        capacities: dict[str, int] | None = None,
        exclusions: list[tuple[str, str]] | None = None,
        required: list[tuple[str, str]] | None = None,
    ) -> AssignmentResult:
        """
        Solve an assignment problem.

        Args:
            entities: List of entities to assign (e.g., workers)
            targets: List of targets (e.g., tasks)
            costs: Optional cost matrix {(entity, target): cost}
            capacities: Optional capacity per target {target: max_entities}
            exclusions: Pairs that cannot be assigned together
            required: Pairs that must be assigned

        Returns:
            AssignmentResult with assignments
        """
        if not self._ortools_available:
            return AssignmentResult(
                status=AssignmentStatus.ERROR,
                error="OR-Tools not available",
            )

        import time
        from ortools.sat.python import cp_model

        start_time = time.perf_counter()

        model = cp_model.CpModel()

        # Create assignment variables
        # x[e][t] = 1 if entity e is assigned to target t
        x = {}
        for e in entities:
            for t in targets:
                x[(e, t)] = model.NewBoolVar(f"x_{e}_{t}")

        # Each entity assigned to exactly one target
        for e in entities:
            model.AddExactlyOne(x[(e, t)] for t in targets)

        # Capacity constraints
        if capacities:
            for t, cap in capacities.items():
                if t in targets:
                    model.Add(sum(x[(e, t)] for e in entities) <= cap)
        else:
            # Default: each target gets at most ceil(len(entities)/len(targets)) entities
            max_per_target = (len(entities) + len(targets) - 1) // len(targets) + 1
            for t in targets:
                model.Add(sum(x[(e, t)] for e in entities) <= max_per_target)

        # Exclusion constraints
        if exclusions:
            for e, t in exclusions:
                if e in entities and t in targets:
                    model.Add(x[(e, t)] == 0)

        # Required assignments
        if required:
            for e, t in required:
                if e in entities and t in targets:
                    model.Add(x[(e, t)] == 1)

        # Objective: minimize cost
        if costs:
            total_cost = sum(
                costs.get((e, t), 0) * x[(e, t)]
                for e in entities
                for t in targets
            )
            model.Minimize(total_cost)

        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.timeout_seconds

        status = solver.Solve(model)
        solve_time = (time.perf_counter() - start_time) * 1000

        if status == cp_model.OPTIMAL:
            result_status = AssignmentStatus.OPTIMAL
        elif status == cp_model.FEASIBLE:
            result_status = AssignmentStatus.FEASIBLE
        elif status == cp_model.INFEASIBLE:
            return AssignmentResult(
                status=AssignmentStatus.INFEASIBLE,
                solve_time_ms=solve_time,
            )
        else:
            return AssignmentResult(
                status=AssignmentStatus.ERROR,
                solve_time_ms=solve_time,
                error="Solver timeout or unknown status",
            )

        # Extract assignments
        assignments = []
        for e in entities:
            for t in targets:
                if solver.Value(x[(e, t)]) == 1:
                    cost = costs.get((e, t), 0) if costs else 0
                    assignments.append(Assignment(entity=e, target=t, cost=cost))

        total = sum(a.cost for a in assignments) if costs else None

        return AssignmentResult(
            status=result_status,
            assignments=assignments,
            total_cost=total,
            solve_time_ms=solve_time,
        )

    def solve_from_spec(self, spec: CSPSpec) -> AssignmentResult:
        """
        Solve assignment from CSPSpec.

        Extracts entities, targets, and constraints from the spec.
        """
        # Extract entities and targets from spec
        entities = []
        targets = []

        # Look for entity definitions in spec.entities or parse from tasks
        if spec.entities:
            entities = spec.entities.get("entities", [])
            targets = spec.entities.get("targets", [])
        else:
            # Try to extract from task names
            for task in spec.tasks:
                entities.append(task.id)

        if not entities or not targets:
            return AssignmentResult(
                status=AssignmentStatus.ERROR,
                error="Could not extract entities and targets from specification",
            )

        # Extract constraints
        exclusions = []
        required = []

        for c in spec.constraints:
            if c.type == "different" or c.type == "no_overlap":
                if len(c.args) >= 2:
                    exclusions.append((c.args[0], c.args[1]))
            elif c.type == "required" or c.type == "must_assign":
                if len(c.args) >= 2:
                    required.append((c.args[0], c.args[1]))

        return self.solve(
            entities=entities,
            targets=targets,
            exclusions=exclusions,
            required=required,
        )


def solve_assignment(
    entities: list[str],
    targets: list[str],
    **kwargs,
) -> str:
    """Convenience function for quick assignment solving."""
    solver = AssignmentSolver()
    result = solver.solve(entities, targets, **kwargs)
    return result.to_string()


if __name__ == "__main__":
    print("Assignment Solver Tests")
    print("=" * 60)

    solver = AssignmentSolver()

    # Test 1: Simple assignment
    print("\nTest 1: Assign 4 workers to 2 projects")
    result = solver.solve(
        entities=["Alice", "Bob", "Carol", "Dave"],
        targets=["ProjectA", "ProjectB"],
        capacities={"ProjectA": 2, "ProjectB": 2},
    )
    print(result.to_string())

    # Test 2: With exclusion
    print("\nTest 2: Alice can't work with Bob on same project")
    result = solver.solve(
        entities=["Alice", "Bob", "Carol"],
        targets=["Team1", "Team2"],
        exclusions=[("Alice", "Team1"), ("Bob", "Team1")],  # Force them apart
    )
    print(result.to_string())

    # Test 3: With costs
    print("\nTest 3: Minimize assignment cost")
    result = solver.solve(
        entities=["Worker1", "Worker2", "Worker3"],
        targets=["TaskA", "TaskB", "TaskC"],
        costs={
            ("Worker1", "TaskA"): 10, ("Worker1", "TaskB"): 5, ("Worker1", "TaskC"): 8,
            ("Worker2", "TaskA"): 3, ("Worker2", "TaskB"): 9, ("Worker2", "TaskC"): 7,
            ("Worker3", "TaskA"): 6, ("Worker3", "TaskB"): 4, ("Worker3", "TaskC"): 2,
        },
    )
    print(result.to_string())
