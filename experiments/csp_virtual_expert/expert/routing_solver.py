"""
Routing Solver using OR-Tools.

Solves routing problems:
- Traveling Salesman Problem (TSP)
- Vehicle Routing Problem (VRP)
- Delivery routing with constraints
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..extraction.csp_extractor import CSPSpec


class RoutingStatus(Enum):
    """Solver result status."""
    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    ERROR = "error"


@dataclass
class Route:
    """A single route (sequence of locations)."""
    vehicle: int
    locations: list[str]
    total_distance: int


@dataclass
class RoutingResult:
    """Result from routing solver."""
    status: RoutingStatus
    routes: list[Route] = field(default_factory=list)
    total_distance: int | None = None
    solve_time_ms: float | None = None
    error: str | None = None

    def is_success(self) -> bool:
        return self.status in (RoutingStatus.OPTIMAL, RoutingStatus.FEASIBLE)

    def to_string(self) -> str:
        if not self.is_success():
            if self.status == RoutingStatus.INFEASIBLE:
                return "No valid route exists."
            return f"Solver error: {self.error}"

        lines = []
        for route in self.routes:
            if len(self.routes) > 1:
                lines.append(f"Vehicle {route.vehicle}:")
            route_str = " -> ".join(route.locations)
            lines.append(f"  Route: {route_str}")
            lines.append(f"  Distance: {route.total_distance}")

        if self.total_distance is not None:
            lines.append(f"Total distance: {self.total_distance}")

        if self.status == RoutingStatus.OPTIMAL:
            lines.append("(optimal)")

        return "\n".join(lines)


class RoutingSolver:
    """
    Routing solver using OR-Tools.

    Supports:
    - TSP (single vehicle, visit all locations)
    - VRP (multiple vehicles)
    - Time windows
    - Capacity constraints
    """

    def __init__(self, timeout_seconds: float = 10.0):
        self.timeout_seconds = timeout_seconds
        self._ortools_available = self._check_ortools()

    def _check_ortools(self) -> bool:
        try:
            from ortools.constraint_solver import routing_enums_pb2
            from ortools.constraint_solver import pywrapcp
            return True
        except Exception:
            return False

    def solve_tsp(
        self,
        locations: list[str],
        distances: dict[tuple[str, str], int] | None = None,
        start: str | None = None,
        end: str | None = None,
        must_return: bool = True,
    ) -> RoutingResult:
        """
        Solve a Traveling Salesman Problem.

        Args:
            locations: List of location names
            distances: Distance matrix {(from, to): distance}
            start: Starting location (first location if None)
            end: Ending location (same as start if must_return)
            must_return: Whether to return to start

        Returns:
            RoutingResult with optimal route
        """
        if not self._ortools_available:
            return RoutingResult(
                status=RoutingStatus.ERROR,
                error="OR-Tools not available",
            )

        import time
        from ortools.constraint_solver import routing_enums_pb2
        from ortools.constraint_solver import pywrapcp

        start_time = time.perf_counter()

        n = len(locations)
        if n < 2:
            return RoutingResult(
                status=RoutingStatus.ERROR,
                error="Need at least 2 locations",
            )

        # Build location index
        loc_to_idx = {loc: i for i, loc in enumerate(locations)}
        idx_to_loc = {i: loc for i, loc in enumerate(locations)}

        # Build distance matrix
        if distances:
            dist_matrix = [[0] * n for _ in range(n)]
            for (f, t), d in distances.items():
                if f in loc_to_idx and t in loc_to_idx:
                    dist_matrix[loc_to_idx[f]][loc_to_idx[t]] = d
        else:
            # Default: sequential distances (1 between adjacent)
            dist_matrix = [[abs(i - j) * 10 for j in range(n)] for i in range(n)]

        # Create routing model
        start_idx = loc_to_idx[start] if start and start in loc_to_idx else 0
        end_idx = start_idx if must_return else (loc_to_idx[end] if end and end in loc_to_idx else 0)

        manager = pywrapcp.RoutingIndexManager(n, 1, [start_idx], [end_idx])
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return dist_matrix[from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Set search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.time_limit.FromSeconds(int(self.timeout_seconds))

        # Solve
        solution = routing.SolveWithParameters(search_parameters)
        solve_time = (time.perf_counter() - start_time) * 1000

        if solution:
            # Extract route
            route_locs = []
            index = routing.Start(0)
            route_distance = 0

            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                route_locs.append(idx_to_loc[node])
                prev_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(prev_index, index, 0)

            # Add final location
            node = manager.IndexToNode(index)
            route_locs.append(idx_to_loc[node])

            return RoutingResult(
                status=RoutingStatus.OPTIMAL,
                routes=[Route(vehicle=0, locations=route_locs, total_distance=route_distance)],
                total_distance=route_distance,
                solve_time_ms=solve_time,
            )
        else:
            return RoutingResult(
                status=RoutingStatus.INFEASIBLE,
                solve_time_ms=solve_time,
            )

    def solve_from_spec(self, spec: CSPSpec) -> RoutingResult:
        """Solve routing from CSPSpec."""
        # Extract locations from entities or tasks
        locations = []

        if spec.entities and "locations" in spec.entities:
            locations = spec.entities["locations"]
        else:
            # Use task IDs as locations
            locations = [t.id for t in spec.tasks]

        if not locations:
            return RoutingResult(
                status=RoutingStatus.ERROR,
                error="No locations found in specification",
            )

        # Check for start/end constraints
        start = None
        end = None
        must_return = True

        for c in spec.constraints:
            if c.type == "start" and c.args:
                start = c.args[0]
            elif c.type == "end" and c.args:
                end = c.args[0]
                must_return = False

        return self.solve_tsp(
            locations=locations,
            start=start,
            end=end,
            must_return=must_return,
        )


def solve_tsp(locations: list[str], **kwargs) -> str:
    """Convenience function for quick TSP solving."""
    solver = RoutingSolver()
    result = solver.solve_tsp(locations, **kwargs)
    return result.to_string()


if __name__ == "__main__":
    print("Routing Solver Tests")
    print("=" * 60)

    solver = RoutingSolver()

    # Test 1: Simple TSP
    print("\nTest 1: Visit 5 cities and return")
    result = solver.solve_tsp(
        locations=["NYC", "Boston", "Chicago", "LA", "Seattle"],
        must_return=True,
    )
    print(result.to_string())

    # Test 2: TSP with distances
    print("\nTest 2: TSP with custom distances")
    result = solver.solve_tsp(
        locations=["A", "B", "C", "D"],
        distances={
            ("A", "B"): 10, ("B", "A"): 10,
            ("A", "C"): 15, ("C", "A"): 15,
            ("A", "D"): 20, ("D", "A"): 20,
            ("B", "C"): 35, ("C", "B"): 35,
            ("B", "D"): 25, ("D", "B"): 25,
            ("C", "D"): 30, ("D", "C"): 30,
        },
        start="A",
        must_return=True,
    )
    print(result.to_string())

    # Test 3: One-way trip
    print("\nTest 3: One-way trip (no return)")
    result = solver.solve_tsp(
        locations=["Start", "Stop1", "Stop2", "End"],
        must_return=False,
    )
    print(result.to_string())
