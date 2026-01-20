"""
CSP Virtual Expert Plugin.

Detects constraint satisfaction problems in model output and routes them
to exact solvers. Supports scheduling, assignment, routing, packing, and
coloring problems.

This plugin follows the Virtual Expert architecture:
1. can_handle() - Fast pattern matching for CSP triggers
2. execute() - Extract CSP spec from CoT, solve, format result
3. get_calibration_prompts() - Training data for router calibration
"""

from __future__ import annotations

import re
from typing import Any

# Import base class - handle both integrated and standalone modes
try:
    from chuk_lazarus.inference.virtual_experts.base import VirtualExpertPlugin
except ImportError:
    from abc import ABC, abstractmethod

    class VirtualExpertPlugin(ABC):
        """Standalone base class for testing without lazarus."""

        name: str = "base"
        description: str = "Base virtual expert"
        priority: int = 0

        @abstractmethod
        def can_handle(self, prompt: str) -> bool:
            pass

        @abstractmethod
        def execute(self, prompt: str) -> str | None:
            pass

        @abstractmethod
        def get_calibration_prompts(self) -> tuple[list[str], list[str]]:
            pass


# Import CSP components
from ..extraction.csp_extractor import (
    CSPSpec,
    CSPType,
    extract_csp_spec,
    detect_csp_type,
)
from .scheduling_solver import SchedulingSolver, SchedulingResult, SolverStatus


class CSPVirtualExpertPlugin(VirtualExpertPlugin):
    """
    Virtual expert for constraint satisfaction problems.

    Detects when model output contains a structured CSP specification,
    extracts the problem, solves it with an exact solver, and returns
    the optimal solution.

    Trigger Patterns:
    - "SOLVE:" after constraint specification (Format A)
    - "Solution:" after constraint listing (Format C)
    - solve_csp() function call (Format B)
    - JSON with "tasks" and "constraints" (Format D)

    Supported Problem Types:
    - Scheduling: Task scheduling with time windows and dependencies
    - Assignment: Matching/allocation problems (future)
    - Routing: TSP/VRP problems (future)
    - Packing: Bin packing, knapsack (future)
    - Coloring: Graph coloring, register allocation (future)

    Example:
        >>> plugin = CSPVirtualExpertPlugin()
        >>> cot_output = '''
        ... TASKS: [Alice:2hr, Bob:1hr]
        ... CONSTRAINTS: [no_overlap(Alice, Bob)]
        ... OBJECTIVE: minimize_makespan
        ... SOLVE:
        ... '''
        >>> plugin.execute(cot_output)
        'Schedule:\\n  - Alice: 9:00 - 11:00\\n  - Bob: 11:00 - 12:00'
    """

    name = "csp"
    description = "Solves constraint satisfaction problems exactly"
    priority = 9  # Between compiler (8) and math (10)

    # Trigger patterns that indicate CSP specification is complete
    TRIGGER_PATTERNS = [
        r"\bSOLVE:\s*$",  # Format A: SOLVE: at end
        r"\bSolution:\s*$",  # Format C: Solution: at end
        r"solve_csp\s*\(",  # Format B: function call
        r"\"tasks\":\s*\[",  # Format D: JSON with tasks
        r"Finding (?:optimal )?schedule",  # Natural triggers
        r"Applying constraint solver",
        r"Let me solve this (?:systematically|exactly)",
    ]

    # Keywords that suggest CSP content
    CSP_KEYWORDS = [
        "schedule",
        "assign",
        "allocate",
        "route",
        "pack",
        "constraint",
        "no_overlap",
        "minimize",
        "maximize",
        "feasible",
        "optimal",
    ]

    def __init__(
        self,
        timeout_seconds: float = 10.0,
        confidence_threshold: float = 0.7,
    ):
        """
        Initialize the CSP expert.

        Args:
            timeout_seconds: Solver timeout
            confidence_threshold: Minimum confidence to trigger
        """
        self.timeout = timeout_seconds
        self.confidence_threshold = confidence_threshold
        self.scheduling_solver = SchedulingSolver(timeout_seconds=timeout_seconds)

    def can_handle(self, prompt: str) -> bool:
        """
        Check if prompt contains a triggerable CSP specification.

        Fast pre-filter using pattern matching. The router's learned
        direction in activation space provides additional signal.

        Args:
            prompt: The model output (CoT) to check

        Returns:
            True if this looks like a CSP ready to solve
        """
        # Check for explicit trigger patterns
        for pattern in self.TRIGGER_PATTERNS:
            if re.search(pattern, prompt, re.IGNORECASE | re.MULTILINE):
                return True

        # Check for CSP keyword density
        prompt_lower = prompt.lower()
        keyword_count = sum(1 for kw in self.CSP_KEYWORDS if kw in prompt_lower)

        # If high keyword density, check for structure
        if keyword_count >= 3:
            # Look for task-like patterns
            has_tasks = bool(
                re.search(r"\w+\s*[:(-]\s*\d+\s*(?:hr|hour|min)", prompt, re.IGNORECASE)
            )
            # Look for constraint patterns
            has_constraints = bool(
                re.search(r"(?:no_overlap|before|after|can't|cannot)", prompt, re.IGNORECASE)
            )
            return has_tasks and has_constraints

        return False

    def execute(self, prompt: str) -> str | None:
        """
        Extract CSP from prompt, solve, and return formatted result.

        Args:
            prompt: The model output containing CSP specification

        Returns:
            Formatted solution string, or None if extraction/solving fails
        """
        # Extract CSP specification
        spec = extract_csp_spec(prompt)
        if spec is None:
            # Try to build spec from looser patterns
            spec = self._extract_loose(prompt)

        if spec is None or not spec.is_valid():
            return None

        # Detect CSP type if not already set
        if spec.csp_type == CSPType.GENERIC:
            spec.csp_type = detect_csp_type(prompt)

        # Route to appropriate solver
        if spec.csp_type == CSPType.SCHEDULING:
            return self._solve_scheduling(spec)
        elif spec.csp_type == CSPType.ASSIGNMENT:
            return self._solve_assignment(spec)
        elif spec.csp_type == CSPType.ROUTING:
            return self._solve_routing(spec)
        elif spec.csp_type == CSPType.PACKING:
            return self._solve_packing(spec)
        elif spec.csp_type == CSPType.COLORING:
            return self._solve_coloring(spec)
        else:
            # Try scheduling as default
            return self._solve_scheduling(spec)

    def get_calibration_prompts(self) -> tuple[list[str], list[str]]:
        """
        Return calibration prompts for router training.

        These prompts help the MoE router learn when to activate
        the CSP virtual expert based on hidden state patterns.

        Returns:
            (positive_prompts, negative_prompts) tuple
        """
        from ..data.prompts import get_calibration_prompts

        return get_calibration_prompts()

    # =========================================================================
    # SOLVER DISPATCH
    # =========================================================================

    def _solve_scheduling(self, spec: CSPSpec) -> str:
        """Solve a scheduling problem."""
        result = self.scheduling_solver.solve(spec)
        return self.scheduling_solver.format_solution(result, spec)

    def _solve_assignment(self, spec: CSPSpec) -> str:
        """Solve an assignment problem."""
        from .assignment_solver import AssignmentSolver

        solver = AssignmentSolver(timeout_seconds=self.timeout)
        result = solver.solve_from_spec(spec)
        return result.to_string()

    def _solve_routing(self, spec: CSPSpec) -> str:
        """Solve a routing problem (TSP/VRP)."""
        from .routing_solver import RoutingSolver

        solver = RoutingSolver(timeout_seconds=self.timeout)
        result = solver.solve_from_spec(spec)
        return result.to_string()

    def _solve_packing(self, spec: CSPSpec) -> str:
        """Solve a packing/bin-packing problem."""
        from .packing_solver import PackingSolver

        solver = PackingSolver(timeout_seconds=self.timeout)
        result = solver.solve_from_spec(spec)
        return result.to_string()

    def _solve_coloring(self, spec: CSPSpec) -> str:
        """Solve a graph coloring problem."""
        # Coloring can be modeled as assignment with conflict constraints
        # For now, use scheduling with no-overlap as a simple approximation
        return self._solve_scheduling(spec)

    # =========================================================================
    # LOOSE EXTRACTION (Fallback)
    # =========================================================================

    def _extract_loose(self, text: str) -> CSPSpec | None:
        """
        Attempt loose extraction when structured formats fail.

        Looks for:
        - Names followed by durations anywhere in text
        - Natural language constraints
        - Time-related patterns
        """
        from ..extraction.csp_extractor import (
            Task,
            Constraint,
            ObjectiveType,
            parse_duration,
            parse_time,
        )

        spec = CSPSpec(raw_text=text)

        # Find task patterns: "Name (duration)" or "Name: duration" or "Name - duration"
        task_patterns = [
            r"(\b[A-Z][a-z]+\b)\s*\((\d+\.?\d*\s*(?:hr|hour|min|h|m)?)\)",
            r"(\b[A-Z][a-z]+\b)\s*:\s*(\d+\.?\d*\s*(?:hr|hour|min|h|m)?)",
            r"(\b[A-Z][a-z]+\b)\s*-\s*(\d+\.?\d*\s*(?:hr|hour|min|h|m)?)",
            r"(\b(?:gym|lunch|dinner|meeting|dentist|doctor)\b)\s*\((\d+\.?\d*\s*(?:hr|hour|min|h|m)?)\)",
        ]

        found_tasks = set()
        for pattern in task_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                name, duration = match.groups()
                name = name.capitalize()
                if name not in found_tasks:
                    found_tasks.add(name)
                    spec.tasks.append(Task(id=name, duration=parse_duration(duration)))

        # Find fixed time patterns: "X at 2pm" or "X fixed at 14:00"
        for match in re.finditer(
            r"(\b[A-Z][a-z]+\b)\s+(?:at|fixed at)\s+([\d:]+\s*(?:am|pm)?)",
            text,
            re.IGNORECASE,
        ):
            name, time_str = match.groups()
            name = name.capitalize()
            # Update task if it exists
            for task in spec.tasks:
                if task.id == name:
                    task.fixed_start = parse_time(time_str)

        # Find ordering constraints
        for match in re.finditer(
            r"(\b[A-Z][a-z]+\b)\s+before\s+(\b[A-Z][a-z]+\b)",
            text,
            re.IGNORECASE,
        ):
            spec.constraints.append(
                Constraint.before(match.group(1).capitalize(), match.group(2).capitalize())
            )

        # Find no-overlap constraints
        for match in re.finditer(
            r"(\b[A-Z][a-z]+\b)\s+(?:and\s+)?(\b[A-Z][a-z]+\b)\s+(?:can't|cannot|can not)\s+overlap",
            text,
            re.IGNORECASE,
        ):
            spec.constraints.append(
                Constraint.no_overlap(match.group(1).capitalize(), match.group(2).capitalize())
            )

        # Set objective
        if "minimize" in text.lower() and "time" in text.lower():
            spec.objective = ObjectiveType.MINIMIZE_MAKESPAN

        # Set type
        spec.csp_type = detect_csp_type(text)

        return spec if spec.is_valid() else None


# =============================================================================
# TESTING
# =============================================================================


def test_csp_plugin():
    """Run basic tests on the CSP plugin."""
    print("CSP Virtual Expert Plugin Tests")
    print("=" * 60)

    plugin = CSPVirtualExpertPlugin()

    # Test 1: Format A (declarative blocks)
    test1 = """
    Let me structure this scheduling problem:

    TASKS: [Alice:2hr, Bob:1hr, Carol:1.5hr]
    WINDOW: [9:00, 12:00]
    CONSTRAINTS: [no_overlap(Alice, Bob)]
    OBJECTIVE: minimize_makespan
    SOLVE:
    """

    print("\nTest 1: Format A (declarative blocks)")
    print(f"Can handle: {plugin.can_handle(test1)}")
    result = plugin.execute(test1)
    print(f"Result:\n{result}")

    # Test 2: Format C (natural structured)
    test2 = """
    Let me formalize the constraints:
    - Tasks: Alice (2hr), Bob (1hr)
    - Time window: 9am to 12pm
    - Constraint: Alice and Bob cannot overlap
    - Goal: Minimize total time
    Solution:
    """

    print("\nTest 2: Format C (natural structured)")
    print(f"Can handle: {plugin.can_handle(test2)}")
    result = plugin.execute(test2)
    print(f"Result:\n{result}")

    # Test 3: Loose extraction
    test3 = """
    Schedule my day: gym (1hr), lunch meeting (1.5hr), dentist at 2pm (1hr).
    The dentist appointment is fixed. Minimize total time.
    Finding optimal schedule...
    """

    print("\nTest 3: Loose extraction with fixed time")
    print(f"Can handle: {plugin.can_handle(test3)}")
    result = plugin.execute(test3)
    print(f"Result:\n{result}")

    # Test 4: Non-CSP should not trigger
    test4 = "What is 127 * 89?"

    print("\nTest 4: Non-CSP (should not handle)")
    print(f"Can handle: {plugin.can_handle(test4)}")

    # Test 5: Calibration prompts
    print("\nTest 5: Calibration prompts")
    positive, negative = plugin.get_calibration_prompts()
    print(f"Positive examples: {len(positive)}")
    print(f"Negative examples: {len(negative)}")
    print(f"Sample positive: {positive[0][:60]}...")
    print(f"Sample negative: {negative[0][:60]}...")


if __name__ == "__main__":
    test_csp_plugin()
