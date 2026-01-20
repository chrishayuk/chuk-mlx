"""
Scheduling Solver using OR-Tools CP-SAT.

Wraps the OR-Tools constraint programming solver for scheduling problems.
Takes a CSPSpec and returns an optimal or feasible schedule.

Install OR-Tools: pip install ortools
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# Import CSP types
from ..extraction.csp_extractor import CSPSpec, Task, Constraint, ObjectiveType


class SolverStatus(Enum):
    """Solver result status."""

    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class TaskSchedule:
    """Scheduled time for a single task."""

    task_id: str
    start: int  # Start time in minutes
    end: int  # End time in minutes
    duration: int  # Duration in minutes

    def format_time(self, minutes: int, base_hour: int = 9) -> str:
        """Format minutes as HH:MM."""
        total_mins = base_hour * 60 + minutes
        hours = total_mins // 60
        mins = total_mins % 60
        return f"{hours}:{mins:02d}"

    def to_string(self, base_hour: int = 9) -> str:
        """Format as 'TaskID: HH:MM - HH:MM'."""
        return f"{self.task_id}: {self.format_time(self.start, base_hour)} - {self.format_time(self.end, base_hour)}"


@dataclass
class SchedulingResult:
    """Result from the scheduling solver."""

    status: SolverStatus
    schedules: list[TaskSchedule] = field(default_factory=list)
    objective_value: int | None = None
    solve_time_ms: float | None = None
    error: str | None = None
    conflicts: list[str] = field(default_factory=list)  # For infeasible cases

    def is_success(self) -> bool:
        return self.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)

    def to_string(self, base_hour: int = 9) -> str:
        """Format result as human-readable string."""
        if not self.is_success():
            if self.status == SolverStatus.INFEASIBLE:
                lines = ["No solution exists."]
                if self.conflicts:
                    lines.append("Conflicting constraints:")
                    for conflict in self.conflicts:
                        lines.append(f"  - {conflict}")
                return "\n".join(lines)
            elif self.status == SolverStatus.TIMEOUT:
                return "Solver timed out before finding a solution."
            else:
                return f"Solver error: {self.error}"

        lines = ["Schedule:"]
        for schedule in sorted(self.schedules, key=lambda s: s.start):
            lines.append(f"  - {schedule.to_string(base_hour)}")

        if self.objective_value is not None:
            lines.append(f"Total time: {self.objective_value} minutes")
            if self.status == SolverStatus.OPTIMAL:
                lines.append("(optimal)")

        return "\n".join(lines)


class SchedulingSolver:
    """
    Constraint-based scheduling solver using OR-Tools CP-SAT.

    Supports:
    - Task duration constraints
    - No-overlap constraints
    - Before/after ordering constraints
    - Fixed time constraints
    - Time window constraints
    - Minimize makespan objective
    """

    def __init__(self, timeout_seconds: float = 10.0):
        self.timeout_seconds = timeout_seconds
        self._ortools_available = self._check_ortools()

    def _check_ortools(self) -> bool:
        """Check if OR-Tools is available and working."""
        try:
            from ortools.sat.python import cp_model
            return True
        except (ImportError, Exception):
            # Catch any import error including protobuf version mismatches
            return False

    @property
    def is_available(self) -> bool:
        """Check if solver is available."""
        return self._ortools_available

    def solve(self, spec: CSPSpec) -> SchedulingResult:
        """
        Solve a scheduling CSP.

        Args:
            spec: The CSP specification

        Returns:
            SchedulingResult with the solution or error
        """
        if not self._ortools_available:
            return SchedulingResult(
                status=SolverStatus.ERROR,
                error="OR-Tools not available. Run: pip install ortools (check protobuf compatibility)",
            )

        try:
            from ortools.sat.python import cp_model
        except Exception as e:
            return SchedulingResult(
                status=SolverStatus.ERROR,
                error=f"OR-Tools import failed: {e}",
            )

        import time

        start_time = time.perf_counter()

        # Create the model
        model = cp_model.CpModel()

        # Determine horizon (latest possible end time)
        if spec.window:
            horizon = spec.window[1] - spec.window[0]
            base_time = spec.window[0]
        else:
            # Default 8-hour window starting at 9am
            horizon = 8 * 60  # 480 minutes
            base_time = 9 * 60  # 9:00 AM

        # Pre-check: detect infeasibility before building model
        total_duration = sum(t.duration for t in spec.tasks)
        if total_duration > horizon:
            return SchedulingResult(
                status=SolverStatus.INFEASIBLE,
                solve_time_ms=(time.perf_counter() - start_time) * 1000,
                conflicts=[
                    f"Total task duration ({total_duration} min) exceeds "
                    f"time window ({horizon} min)"
                ],
            )

        # Check if any individual task exceeds window
        for task in spec.tasks:
            if task.duration > horizon:
                return SchedulingResult(
                    status=SolverStatus.INFEASIBLE,
                    solve_time_ms=(time.perf_counter() - start_time) * 1000,
                    conflicts=[
                        f"Task '{task.id}' duration ({task.duration} min) exceeds "
                        f"time window ({horizon} min)"
                    ],
                )

        # Create variables for each task
        task_vars: dict[str, dict[str, Any]] = {}

        for task in spec.tasks:
            start_var = model.NewIntVar(0, horizon - task.duration, f"start_{task.id}")
            end_var = model.NewIntVar(task.duration, horizon, f"end_{task.id}")
            interval_var = model.NewIntervalVar(
                start_var, task.duration, end_var, f"interval_{task.id}"
            )

            task_vars[task.id] = {
                "start": start_var,
                "end": end_var,
                "interval": interval_var,
                "duration": task.duration,
            }

            # Duration constraint
            model.Add(end_var == start_var + task.duration)

            # Fixed start time if specified
            if task.fixed_start is not None:
                fixed_offset = task.fixed_start - base_time
                if 0 <= fixed_offset <= horizon - task.duration:
                    model.Add(start_var == fixed_offset)

        # Add constraints
        for constraint in spec.constraints:
            if constraint.type == "no_overlap":
                # No overlap between specified tasks
                if len(constraint.args) == 2:
                    task_a, task_b = constraint.args
                    if task_a in task_vars and task_b in task_vars:
                        model.AddNoOverlap(
                            [task_vars[task_a]["interval"], task_vars[task_b]["interval"]]
                        )
                elif len(constraint.args) > 2:
                    # Multiple tasks can't overlap
                    intervals = [
                        task_vars[tid]["interval"]
                        for tid in constraint.args
                        if tid in task_vars
                    ]
                    if len(intervals) > 1:
                        model.AddNoOverlap(intervals)

            elif constraint.type == "before":
                # First task must end before second starts
                if len(constraint.args) >= 2:
                    first, second = constraint.args[0], constraint.args[1]
                    if first in task_vars and second in task_vars:
                        model.Add(
                            task_vars[first]["end"] <= task_vars[second]["start"]
                        )

            elif constraint.type == "after":
                # First task must start after second ends
                if len(constraint.args) >= 2:
                    first, second = constraint.args[0], constraint.args[1]
                    if first in task_vars and second in task_vars:
                        model.Add(
                            task_vars[first]["start"] >= task_vars[second]["end"]
                        )

            elif constraint.type == "fixed_time":
                # Task starts at fixed time
                if constraint.args and constraint.args[0] in task_vars:
                    task_id = constraint.args[0]
                    fixed_time = constraint.params.get("time", 0)
                    fixed_offset = fixed_time - base_time
                    if 0 <= fixed_offset <= horizon - task_vars[task_id]["duration"]:
                        model.Add(task_vars[task_id]["start"] == fixed_offset)

        # Add no-overlap for all tasks if this is a scheduling problem
        # (default behavior - one resource)
        if spec.csp_type.value == "scheduling" and len(spec.tasks) > 1:
            all_intervals = [task_vars[t.id]["interval"] for t in spec.tasks]
            model.AddNoOverlap(all_intervals)

        # Objective
        if spec.objective == ObjectiveType.MINIMIZE_MAKESPAN:
            makespan = model.NewIntVar(0, horizon, "makespan")
            for task in spec.tasks:
                model.Add(makespan >= task_vars[task.id]["end"])
            model.Minimize(makespan)

        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.timeout_seconds

        status = solver.Solve(model)
        solve_time = (time.perf_counter() - start_time) * 1000

        # Process results
        if status == cp_model.OPTIMAL:
            result_status = SolverStatus.OPTIMAL
        elif status == cp_model.FEASIBLE:
            result_status = SolverStatus.FEASIBLE
        elif status == cp_model.INFEASIBLE:
            return SchedulingResult(
                status=SolverStatus.INFEASIBLE,
                solve_time_ms=solve_time,
                conflicts=self._analyze_conflicts(spec),
            )
        elif status == cp_model.MODEL_INVALID:
            return SchedulingResult(
                status=SolverStatus.INFEASIBLE,
                solve_time_ms=solve_time,
                conflicts=["Model constraints are invalid or contradictory"],
            )
        else:
            return SchedulingResult(
                status=SolverStatus.TIMEOUT,
                solve_time_ms=solve_time,
            )

        # Extract solution
        schedules = []
        for task in spec.tasks:
            start = solver.Value(task_vars[task.id]["start"])
            end = solver.Value(task_vars[task.id]["end"])
            schedules.append(
                TaskSchedule(
                    task_id=task.id,
                    start=start,
                    end=end,
                    duration=task.duration,
                )
            )

        objective_value = None
        if spec.objective == ObjectiveType.MINIMIZE_MAKESPAN:
            objective_value = solver.Value(makespan)

        return SchedulingResult(
            status=result_status,
            schedules=schedules,
            objective_value=objective_value,
            solve_time_ms=solve_time,
        )

    def _analyze_conflicts(self, spec: CSPSpec) -> list[str]:
        """
        Analyze an infeasible specification to identify likely conflicts.

        This is a heuristic analysis - true IIS computation is complex.
        """
        conflicts = []

        # Check total duration vs window
        total_duration = sum(t.duration for t in spec.tasks)
        if spec.window:
            window_size = spec.window[1] - spec.window[0]
            if total_duration > window_size:
                conflicts.append(
                    f"Total task duration ({total_duration} min) exceeds "
                    f"time window ({window_size} min)"
                )

        # Check fixed time conflicts
        fixed_tasks = [t for t in spec.tasks if t.fixed_start is not None]
        for i, t1 in enumerate(fixed_tasks):
            for t2 in fixed_tasks[i + 1 :]:
                t1_end = t1.fixed_start + t1.duration
                t2_end = t2.fixed_start + t2.duration
                if not (t1_end <= t2.fixed_start or t2_end <= t1.fixed_start):
                    conflicts.append(
                        f"Fixed times for {t1.id} and {t2.id} overlap"
                    )

        # Check circular dependencies
        before_edges = {}
        for c in spec.constraints:
            if c.type == "before" and len(c.args) >= 2:
                before_edges.setdefault(c.args[0], []).append(c.args[1])
            elif c.type == "after" and len(c.args) >= 2:
                before_edges.setdefault(c.args[1], []).append(c.args[0])

        # Simple cycle detection
        visited = set()
        path = set()

        def has_cycle(node: str) -> bool:
            if node in path:
                return True
            if node in visited:
                return False
            visited.add(node)
            path.add(node)
            for neighbor in before_edges.get(node, []):
                if has_cycle(neighbor):
                    return True
            path.remove(node)
            return False

        for task in spec.tasks:
            if has_cycle(task.id):
                conflicts.append("Circular ordering constraints detected")
                break

        if not conflicts:
            conflicts.append("Unable to determine specific conflict")

        return conflicts

    def format_solution(self, result: SchedulingResult, spec: CSPSpec) -> str:
        """
        Format solver result as natural language.

        Args:
            result: The solver result
            spec: Original specification (for context)

        Returns:
            Human-readable solution string
        """
        base_hour = 9  # Default 9 AM start
        if spec.window:
            base_hour = spec.window[0] // 60

        return result.to_string(base_hour)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def solve_scheduling(spec: CSPSpec, timeout: float = 10.0) -> SchedulingResult:
    """
    Convenience function to solve a scheduling problem.

    Args:
        spec: CSP specification
        timeout: Solver timeout in seconds

    Returns:
        SchedulingResult
    """
    solver = SchedulingSolver(timeout_seconds=timeout)
    return solver.solve(spec)


def quick_schedule(
    tasks: list[tuple[str, int]],
    constraints: list[tuple[str, ...]] | None = None,
    window: tuple[int, int] | None = None,
) -> str:
    """
    Quick scheduling helper for simple cases.

    Args:
        tasks: List of (name, duration_minutes) tuples
        constraints: Optional list of constraint tuples like ("no_overlap", "A", "B")
        window: Optional (start_minutes, end_minutes) tuple

    Returns:
        Formatted schedule string

    Example:
        >>> quick_schedule([("Alice", 120), ("Bob", 60)])
        'Schedule:\\n  - Alice: 9:00 - 11:00\\n  - Bob: 11:00 - 12:00'
    """
    from ..extraction.csp_extractor import CSPType

    spec = CSPSpec(
        csp_type=CSPType.SCHEDULING,
        tasks=[Task(id=name, duration=dur) for name, dur in tasks],
        window=window,
        objective=ObjectiveType.MINIMIZE_MAKESPAN,
    )

    if constraints:
        for c in constraints:
            if c[0] == "no_overlap":
                spec.constraints.append(Constraint.no_overlap(*c[1:]))
            elif c[0] == "before":
                spec.constraints.append(Constraint.before(c[1], c[2]))
            elif c[0] == "after":
                spec.constraints.append(Constraint.after(c[1], c[2]))

    solver = SchedulingSolver()
    result = solver.solve(spec)
    return solver.format_solution(result, spec)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Scheduling Solver Tests")
    print("=" * 60)

    # Check OR-Tools
    try:
        from ortools.sat.python import cp_model

        print("OR-Tools available")
    except ImportError:
        print("OR-Tools NOT available - install with: pip install ortools")
        sys.exit(1)

    from ..extraction.csp_extractor import CSPType

    # Test 1: Simple scheduling
    print("\nTest 1: Simple scheduling (3 tasks)")
    spec1 = CSPSpec(
        csp_type=CSPType.SCHEDULING,
        tasks=[
            Task(id="Alice", duration=120),
            Task(id="Bob", duration=60),
            Task(id="Carol", duration=90),
        ],
        window=(9 * 60, 17 * 60),  # 9am to 5pm
        objective=ObjectiveType.MINIMIZE_MAKESPAN,
    )

    solver = SchedulingSolver()
    result1 = solver.solve(spec1)
    print(solver.format_solution(result1, spec1))

    # Test 2: With ordering constraint
    print("\nTest 2: With ordering (Carol before Alice)")
    spec2 = CSPSpec(
        csp_type=CSPType.SCHEDULING,
        tasks=[
            Task(id="Alice", duration=120),
            Task(id="Bob", duration=60),
            Task(id="Carol", duration=90),
        ],
        constraints=[Constraint.before("Carol", "Alice")],
        window=(9 * 60, 17 * 60),
        objective=ObjectiveType.MINIMIZE_MAKESPAN,
    )

    result2 = solver.solve(spec2)
    print(solver.format_solution(result2, spec2))

    # Test 3: With fixed time
    print("\nTest 3: Fixed time (dentist at 2pm)")
    spec3 = CSPSpec(
        csp_type=CSPType.SCHEDULING,
        tasks=[
            Task(id="gym", duration=60),
            Task(id="lunch", duration=90),
            Task(id="dentist", duration=60, fixed_start=14 * 60),  # 2pm
        ],
        window=(9 * 60, 17 * 60),
        objective=ObjectiveType.MINIMIZE_MAKESPAN,
    )

    result3 = solver.solve(spec3)
    print(solver.format_solution(result3, spec3))

    # Test 4: Infeasible
    print("\nTest 4: Infeasible (too many tasks)")
    spec4 = CSPSpec(
        csp_type=CSPType.SCHEDULING,
        tasks=[
            Task(id="A", duration=180),
            Task(id="B", duration=180),
            Task(id="C", duration=180),
        ],
        window=(9 * 60, 12 * 60),  # Only 3 hours
        objective=ObjectiveType.MINIMIZE_MAKESPAN,
    )

    result4 = solver.solve(spec4)
    print(solver.format_solution(result4, spec4))

    # Test 5: Quick schedule helper
    print("\nTest 5: Quick schedule helper")
    print(quick_schedule([("Meeting", 60), ("Coding", 120), ("Review", 30)]))
