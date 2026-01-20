"""
Packing Solver using OR-Tools CP-SAT.

Solves packing problems:
- Bin packing (minimize bins)
- Knapsack (maximize value)
- Container loading
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..extraction.csp_extractor import CSPSpec, ObjectiveType


class PackingStatus(Enum):
    """Solver result status."""
    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    ERROR = "error"


@dataclass
class Bin:
    """A bin with assigned items."""
    bin_id: int
    items: list[str]
    used_capacity: int
    total_capacity: int


@dataclass
class PackingResult:
    """Result from packing solver."""
    status: PackingStatus
    bins: list[Bin] = field(default_factory=list)
    total_bins: int | None = None
    total_value: int | None = None  # For knapsack
    solve_time_ms: float | None = None
    error: str | None = None

    def is_success(self) -> bool:
        return self.status in (PackingStatus.OPTIMAL, PackingStatus.FEASIBLE)

    def to_string(self) -> str:
        if not self.is_success():
            if self.status == PackingStatus.INFEASIBLE:
                return "No valid packing exists."
            return f"Solver error: {self.error}"

        lines = []

        if self.total_value is not None:
            # Knapsack result
            lines.append("Selected items:")
            for b in self.bins:
                for item in b.items:
                    lines.append(f"  - {item}")
            lines.append(f"Total value: {self.total_value}")
            lines.append(f"Capacity used: {self.bins[0].used_capacity}/{self.bins[0].total_capacity}")
        else:
            # Bin packing result
            for b in self.bins:
                items_str = ", ".join(b.items) if b.items else "(empty)"
                lines.append(f"Bin {b.bin_id}: [{items_str}] ({b.used_capacity}/{b.total_capacity})")

            lines.append(f"Total bins used: {self.total_bins}")

        if self.status == PackingStatus.OPTIMAL:
            lines.append("(optimal)")

        return "\n".join(lines)


class PackingSolver:
    """
    Packing solver using OR-Tools CP-SAT.

    Supports:
    - Bin packing (minimize number of bins)
    - Knapsack (maximize value under capacity)
    - Multi-bin packing with different capacities
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

    def solve_bin_packing(
        self,
        items: dict[str, int],  # {item_name: size}
        bin_capacity: int,
        max_bins: int | None = None,
    ) -> PackingResult:
        """
        Solve bin packing problem - minimize number of bins.

        Args:
            items: Dictionary of {item_name: item_size}
            bin_capacity: Capacity of each bin
            max_bins: Maximum bins allowed (defaults to len(items))

        Returns:
            PackingResult with bin assignments
        """
        if not self._ortools_available:
            return PackingResult(
                status=PackingStatus.ERROR,
                error="OR-Tools not available",
            )

        import time
        from ortools.sat.python import cp_model

        start_time = time.perf_counter()

        item_names = list(items.keys())
        item_sizes = list(items.values())
        n_items = len(item_names)

        # Check feasibility
        if any(s > bin_capacity for s in item_sizes):
            return PackingResult(
                status=PackingStatus.INFEASIBLE,
                solve_time_ms=(time.perf_counter() - start_time) * 1000,
                error="Some items exceed bin capacity",
            )

        # Upper bound on bins needed
        if max_bins is None:
            max_bins = n_items

        model = cp_model.CpModel()

        # Variables
        # x[i][b] = 1 if item i is in bin b
        x = {}
        for i in range(n_items):
            for b in range(max_bins):
                x[(i, b)] = model.NewBoolVar(f"x_{i}_{b}")

        # y[b] = 1 if bin b is used
        y = [model.NewBoolVar(f"y_{b}") for b in range(max_bins)]

        # Each item in exactly one bin
        for i in range(n_items):
            model.AddExactlyOne(x[(i, b)] for b in range(max_bins))

        # Bin capacity constraints
        for b in range(max_bins):
            model.Add(
                sum(item_sizes[i] * x[(i, b)] for i in range(n_items)) <= bin_capacity
            )

        # Link y to x: if any item in bin b, then y[b] = 1
        for b in range(max_bins):
            for i in range(n_items):
                model.Add(y[b] >= x[(i, b)])

        # Symmetry breaking: use bins in order
        for b in range(max_bins - 1):
            model.Add(y[b] >= y[b + 1])

        # Objective: minimize bins used
        model.Minimize(sum(y))

        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.timeout_seconds

        status = solver.Solve(model)
        solve_time = (time.perf_counter() - start_time) * 1000

        if status == cp_model.OPTIMAL:
            result_status = PackingStatus.OPTIMAL
        elif status == cp_model.FEASIBLE:
            result_status = PackingStatus.FEASIBLE
        elif status == cp_model.INFEASIBLE:
            return PackingResult(
                status=PackingStatus.INFEASIBLE,
                solve_time_ms=solve_time,
            )
        else:
            return PackingResult(
                status=PackingStatus.ERROR,
                solve_time_ms=solve_time,
                error="Solver timeout",
            )

        # Extract solution
        bins = []
        total_bins = 0

        for b in range(max_bins):
            if solver.Value(y[b]) == 1:
                total_bins += 1
                bin_items = []
                used = 0
                for i in range(n_items):
                    if solver.Value(x[(i, b)]) == 1:
                        bin_items.append(item_names[i])
                        used += item_sizes[i]
                bins.append(Bin(
                    bin_id=b + 1,
                    items=bin_items,
                    used_capacity=used,
                    total_capacity=bin_capacity,
                ))

        return PackingResult(
            status=result_status,
            bins=bins,
            total_bins=total_bins,
            solve_time_ms=solve_time,
        )

    def solve_knapsack(
        self,
        items: dict[str, tuple[int, int]],  # {item_name: (value, weight)}
        capacity: int,
    ) -> PackingResult:
        """
        Solve knapsack problem - maximize value under capacity.

        Args:
            items: Dictionary of {item_name: (value, weight)}
            capacity: Knapsack capacity

        Returns:
            PackingResult with selected items
        """
        if not self._ortools_available:
            return PackingResult(
                status=PackingStatus.ERROR,
                error="OR-Tools not available",
            )

        import time
        from ortools.sat.python import cp_model

        start_time = time.perf_counter()

        item_names = list(items.keys())
        values = [items[n][0] for n in item_names]
        weights = [items[n][1] for n in item_names]
        n = len(item_names)

        model = cp_model.CpModel()

        # x[i] = 1 if item i is selected
        x = [model.NewBoolVar(f"x_{i}") for i in range(n)]

        # Capacity constraint
        model.Add(sum(weights[i] * x[i] for i in range(n)) <= capacity)

        # Objective: maximize value
        model.Maximize(sum(values[i] * x[i] for i in range(n)))

        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.timeout_seconds

        status = solver.Solve(model)
        solve_time = (time.perf_counter() - start_time) * 1000

        if status == cp_model.OPTIMAL:
            result_status = PackingStatus.OPTIMAL
        elif status == cp_model.FEASIBLE:
            result_status = PackingStatus.FEASIBLE
        else:
            return PackingResult(
                status=PackingStatus.INFEASIBLE,
                solve_time_ms=solve_time,
            )

        # Extract solution
        selected = []
        total_value = 0
        total_weight = 0

        for i in range(n):
            if solver.Value(x[i]) == 1:
                selected.append(item_names[i])
                total_value += values[i]
                total_weight += weights[i]

        return PackingResult(
            status=result_status,
            bins=[Bin(bin_id=1, items=selected, used_capacity=total_weight, total_capacity=capacity)],
            total_value=total_value,
            solve_time_ms=solve_time,
        )

    def solve_from_spec(self, spec: CSPSpec) -> PackingResult:
        """Solve packing from CSPSpec."""
        # Try to determine packing type from spec
        if spec.objective == ObjectiveType.MAXIMIZE_VALUE:
            # Knapsack
            items = {}
            capacity = 100  # Default

            for task in spec.tasks:
                # Assume duration is weight, priority is value
                items[task.id] = (task.priority or 1, task.duration)

            if spec.entities and "capacity" in spec.entities:
                capacity = spec.entities["capacity"]

            return self.solve_knapsack(items, capacity)
        else:
            # Bin packing
            items = {t.id: t.duration for t in spec.tasks}
            capacity = 100  # Default

            if spec.entities and "bin_capacity" in spec.entities:
                capacity = spec.entities["bin_capacity"]

            return self.solve_bin_packing(items, capacity)


def solve_bin_packing(items: dict[str, int], bin_capacity: int) -> str:
    """Convenience function for bin packing."""
    solver = PackingSolver()
    result = solver.solve_bin_packing(items, bin_capacity)
    return result.to_string()


def solve_knapsack(items: dict[str, tuple[int, int]], capacity: int) -> str:
    """Convenience function for knapsack."""
    solver = PackingSolver()
    result = solver.solve_knapsack(items, capacity)
    return result.to_string()


if __name__ == "__main__":
    print("Packing Solver Tests")
    print("=" * 60)

    solver = PackingSolver()

    # Test 1: Bin packing
    print("\nTest 1: Pack items into bins of capacity 10")
    result = solver.solve_bin_packing(
        items={"A": 3, "B": 5, "C": 2, "D": 7, "E": 4},
        bin_capacity=10,
    )
    print(result.to_string())

    # Test 2: Knapsack
    print("\nTest 2: Knapsack - maximize value (capacity=15)")
    result = solver.solve_knapsack(
        items={
            "Gold": (100, 10),     # value=100, weight=10
            "Silver": (70, 8),    # value=70, weight=8
            "Bronze": (50, 5),    # value=50, weight=5
            "Iron": (30, 3),      # value=30, weight=3
            "Lead": (20, 12),     # value=20, weight=12
        },
        capacity=15,
    )
    print(result.to_string())

    # Test 3: Larger bin packing
    print("\nTest 3: Pack 8 items into bins of capacity 12")
    result = solver.solve_bin_packing(
        items={"a": 8, "b": 4, "c": 2, "d": 6, "e": 3, "f": 5, "g": 7, "h": 1},
        bin_capacity=12,
    )
    print(result.to_string())
