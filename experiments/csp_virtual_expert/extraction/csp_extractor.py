"""
CSP Specification Extractor.

Parses Chain-of-Thought output into structured CSPSpec objects that can be
passed to constraint solvers. Supports multiple input formats:

- Format A: Declarative blocks (TASKS, CONSTRAINTS, OBJECTIVE, SOLVE)
- Format B: Functional call style (solve_csp(...))
- Format C: Natural structured (bullet points)
- Format D: JSON structured

The extractor attempts each format in order and returns the first successful parse.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CSPType(Enum):
    """Types of constraint satisfaction problems."""

    SCHEDULING = "scheduling"
    ASSIGNMENT = "assignment"
    ROUTING = "routing"
    PACKING = "packing"
    COLORING = "coloring"
    GENERIC = "generic"


class ObjectiveType(Enum):
    """Optimization objectives."""

    MINIMIZE_MAKESPAN = "minimize_makespan"
    MINIMIZE_COST = "minimize_cost"
    MINIMIZE_DISTANCE = "minimize_distance"
    MAXIMIZE_VALUE = "maximize_value"
    MINIMIZE_BINS = "minimize_bins"
    MINIMIZE_COLORS = "minimize_colors"
    FEASIBILITY = "feasibility"  # Just find any solution


@dataclass
class Task:
    """A task/activity in a scheduling problem."""

    id: str
    duration: int  # Duration in minutes
    fixed_start: int | None = None  # Fixed start time if any
    earliest_start: int | None = None
    latest_end: int | None = None
    priority: int = 0

    def __post_init__(self):
        # Convert duration to int if needed
        if isinstance(self.duration, float):
            self.duration = int(self.duration)


@dataclass
class Constraint:
    """A constraint in a CSP."""

    type: str  # no_overlap, before, after, same_time, different, etc.
    args: list[str]  # Entity IDs involved
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def no_overlap(cls, *entities: str) -> "Constraint":
        return cls(type="no_overlap", args=list(entities))

    @classmethod
    def before(cls, first: str, second: str) -> "Constraint":
        return cls(type="before", args=[first, second])

    @classmethod
    def after(cls, first: str, second: str) -> "Constraint":
        return cls(type="after", args=[first, second])

    @classmethod
    def fixed_time(cls, entity: str, time: int) -> "Constraint":
        return cls(type="fixed_time", args=[entity], params={"time": time})

    @classmethod
    def different(cls, *entities: str) -> "Constraint":
        return cls(type="different", args=list(entities))


@dataclass
class CSPSpec:
    """
    Complete specification of a constraint satisfaction problem.

    This is the intermediate representation between natural language
    and solver-specific formats.
    """

    csp_type: CSPType = CSPType.GENERIC
    tasks: list[Task] = field(default_factory=list)
    constraints: list[Constraint] = field(default_factory=list)
    objective: ObjectiveType = ObjectiveType.FEASIBILITY
    window: tuple[int, int] | None = None  # (start_minutes, end_minutes)
    entities: dict[str, Any] = field(default_factory=dict)  # For non-scheduling CSPs
    raw_text: str = ""  # Original text for debugging

    def is_valid(self) -> bool:
        """Check if spec has minimum required information."""
        has_tasks = len(self.tasks) > 0
        has_entities = len(self.entities) > 0
        return has_tasks or has_entities


# =============================================================================
# PARSING UTILITIES
# =============================================================================


def parse_time(time_str: str) -> int:
    """
    Parse time string to minutes since midnight.

    Supports formats: "9:00", "9am", "14:30", "2pm", "2:30pm", "noon"
    """
    time_str = time_str.strip().lower()

    # Handle special cases
    if time_str == "noon":
        return 12 * 60
    if time_str == "midnight":
        return 0

    # Try HH:MM format
    match = re.match(r"(\d{1,2}):(\d{2})\s*(am|pm)?", time_str)
    if match:
        hours, mins, period = match.groups()
        hours, mins = int(hours), int(mins)
        if period == "pm" and hours != 12:
            hours += 12
        elif period == "am" and hours == 12:
            hours = 0
        return hours * 60 + mins

    # Try Hpm/Ham format
    match = re.match(r"(\d{1,2})\s*(am|pm)", time_str)
    if match:
        hours, period = match.groups()
        hours = int(hours)
        if period == "pm" and hours != 12:
            hours += 12
        elif period == "am" and hours == 12:
            hours = 0
        return hours * 60

    # Try just hours
    match = re.match(r"(\d{1,2})", time_str)
    if match:
        hours = int(match.group(1))
        # Assume 24-hour if > 12, else assume AM for morning, PM for afternoon
        if hours <= 6:
            hours += 12  # Assume PM for small hours
        return hours * 60

    return 0


def parse_duration(duration_str: str) -> int:
    """
    Parse duration string to minutes.

    Supports formats: "2hr", "2 hours", "30min", "1.5hr", "90 minutes"
    """
    duration_str = duration_str.strip().lower()

    # Try hours
    match = re.match(r"([\d.]+)\s*(hr|hour|hours|h)", duration_str)
    if match:
        return int(float(match.group(1)) * 60)

    # Try minutes
    match = re.match(r"([\d.]+)\s*(min|minute|minutes|m)", duration_str)
    if match:
        return int(float(match.group(1)))

    # Try days
    match = re.match(r"([\d.]+)\s*(day|days|d)", duration_str)
    if match:
        return int(float(match.group(1)) * 24 * 60)

    # Try just a number (assume minutes if small, hours if reasonable)
    match = re.match(r"([\d.]+)", duration_str)
    if match:
        val = float(match.group(1))
        if val <= 10:
            return int(val * 60)  # Assume hours
        return int(val)  # Assume minutes

    return 60  # Default 1 hour


def parse_tasks(tasks_str: str) -> list[Task]:
    """
    Parse task definitions from various formats.

    Supports:
    - "Alice:2hr, Bob:1hr"
    - "Alice (2hr), Bob (1hr)"
    - "Task A - 2 hours"
    - "[Alice:2hr, Bob:1hr]"
    """
    tasks = []

    # Try colon format: "Name:duration"
    for match in re.finditer(r"(\w+)\s*:\s*([\d.]+\s*(?:hr|hour|min|h|m)?)", tasks_str):
        name, duration = match.groups()
        tasks.append(Task(id=name, duration=parse_duration(duration)))

    if tasks:
        return tasks

    # Try parentheses format: "Name (duration)"
    for match in re.finditer(r"(\w+)\s*\(([\d.]+\s*(?:hr|hour|min|h|m)?)\)", tasks_str):
        name, duration = match.groups()
        tasks.append(Task(id=name, duration=parse_duration(duration)))

    if tasks:
        return tasks

    # Try dash format: "Name - duration"
    for match in re.finditer(r"(\w+)\s*-\s*([\d.]+\s*(?:hr|hour|min|h|m)?)", tasks_str):
        name, duration = match.groups()
        tasks.append(Task(id=name, duration=parse_duration(duration)))

    return tasks


def parse_constraints(constraints_str: str) -> list[Constraint]:
    """
    Parse constraint definitions.

    Supports:
    - "no_overlap(Alice, Bob)"
    - "Alice before Bob"
    - "Alice can't overlap with Bob"
    - "Alice and Bob cannot be scheduled together"
    """
    constraints = []

    # Function-style: no_overlap(A, B)
    for match in re.finditer(r"no_overlap\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)", constraints_str, re.IGNORECASE):
        constraints.append(Constraint.no_overlap(match.group(1), match.group(2)))

    # Function-style: before(A, B)
    for match in re.finditer(r"before\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)", constraints_str, re.IGNORECASE):
        constraints.append(Constraint.before(match.group(1), match.group(2)))

    # Natural: "A before B"
    for match in re.finditer(r"(\w+)\s+before\s+(\w+)", constraints_str, re.IGNORECASE):
        constraints.append(Constraint.before(match.group(1), match.group(2)))

    # Natural: "A after B"
    for match in re.finditer(r"(\w+)\s+after\s+(\w+)", constraints_str, re.IGNORECASE):
        constraints.append(Constraint.after(match.group(1), match.group(2)))

    # Natural: "A can't overlap with B" or "A cannot overlap B"
    for match in re.finditer(r"(\w+)\s+(?:can't|cannot|can not)\s+overlap\s+(?:with\s+)?(\w+)", constraints_str, re.IGNORECASE):
        constraints.append(Constraint.no_overlap(match.group(1), match.group(2)))

    # Natural: "A and B can't be together"
    for match in re.finditer(r"(\w+)\s+and\s+(\w+)\s+(?:can't|cannot|can not)", constraints_str, re.IGNORECASE):
        constraints.append(Constraint.no_overlap(match.group(1), match.group(2)))

    # Fixed time: "X at 2pm" or "X fixed at 14:00"
    for match in re.finditer(r"(\w+)\s+(?:at|fixed at)\s+([\d:]+\s*(?:am|pm)?)", constraints_str, re.IGNORECASE):
        entity, time_str = match.groups()
        constraints.append(Constraint.fixed_time(entity, parse_time(time_str)))

    return constraints


def parse_window(window_str: str) -> tuple[int, int] | None:
    """
    Parse time window specification.

    Supports:
    - "[9:00, 12:00]"
    - "9am to 5pm"
    - "9:00-17:00"
    """
    # Bracket format: [start, end]
    match = re.search(r"\[\s*([\d:]+\s*(?:am|pm)?)\s*,\s*([\d:]+\s*(?:am|pm)?)\s*\]", window_str, re.IGNORECASE)
    if match:
        return parse_time(match.group(1)), parse_time(match.group(2))

    # Natural: "X to Y" or "X - Y" or "X, Y"
    match = re.search(r"([\d:]+\s*(?:am|pm)?)\s*(?:to|,|-)\s*([\d:]+\s*(?:am|pm)?)", window_str, re.IGNORECASE)
    if match:
        return parse_time(match.group(1)), parse_time(match.group(2))

    return None


def parse_objective(objective_str: str) -> ObjectiveType:
    """Parse objective from string."""
    objective_str = objective_str.strip().lower()

    if "makespan" in objective_str:
        return ObjectiveType.MINIMIZE_MAKESPAN
    if "cost" in objective_str:
        return ObjectiveType.MINIMIZE_COST
    if "distance" in objective_str or "travel" in objective_str:
        return ObjectiveType.MINIMIZE_DISTANCE
    if "value" in objective_str or "maximize" in objective_str:
        return ObjectiveType.MAXIMIZE_VALUE
    if "bin" in objective_str or "container" in objective_str:
        return ObjectiveType.MINIMIZE_BINS
    if "color" in objective_str or "slot" in objective_str:
        return ObjectiveType.MINIMIZE_COLORS

    return ObjectiveType.FEASIBILITY


# =============================================================================
# FORMAT-SPECIFIC EXTRACTORS
# =============================================================================


def try_extract_format_a(text: str) -> CSPSpec | None:
    """
    Try to extract CSP from Format A (declarative blocks).

    Format:
        TASKS: [Alice:2hr, Bob:1hr]
        WINDOW: [9:00, 12:00]
        CONSTRAINTS: [no_overlap(Alice, Bob)]
        OBJECTIVE: minimize_makespan
        SOLVE:
    """
    spec = CSPSpec(raw_text=text)

    # Extract TASKS block
    tasks_match = re.search(r"TASKS:\s*\[(.*?)\]", text, re.DOTALL | re.IGNORECASE)
    if tasks_match:
        spec.tasks = parse_tasks(tasks_match.group(1))

    # Extract CONSTRAINTS block
    constraints_match = re.search(r"CONSTRAINTS:\s*\[(.*?)\]", text, re.DOTALL | re.IGNORECASE)
    if constraints_match:
        spec.constraints = parse_constraints(constraints_match.group(1))

    # Extract WINDOW
    window_match = re.search(r"WINDOW:\s*\[(.*?)\]", text, re.IGNORECASE)
    if window_match:
        spec.window = parse_window(window_match.group(1))

    # Extract OBJECTIVE
    objective_match = re.search(r"OBJECTIVE:\s*(\w+)", text, re.IGNORECASE)
    if objective_match:
        spec.objective = parse_objective(objective_match.group(1))

    # Check for SOLVE: trigger
    if "SOLVE:" in text.upper() and spec.is_valid():
        spec.csp_type = CSPType.SCHEDULING
        return spec

    return None


def try_extract_format_b(text: str) -> CSPSpec | None:
    """
    Try to extract CSP from Format B (functional call).

    Format:
        solve_csp(
            tasks=[("Alice", 2), ("Bob", 1)],
            constraints=["no_overlap(Alice, Bob)"],
            objective="minimize_makespan"
        )
    """
    match = re.search(r"solve_csp\s*\((.*?)\)", text, re.DOTALL | re.IGNORECASE)
    if not match:
        return None

    spec = CSPSpec(raw_text=text)
    content = match.group(1)

    # Extract tasks
    tasks_match = re.search(r"tasks\s*=\s*\[(.*?)\]", content, re.DOTALL)
    if tasks_match:
        # Parse tuples: ("Alice", 2)
        for tuple_match in re.finditer(r'\(\s*["\'](\w+)["\']\s*,\s*([\d.]+)\s*\)', tasks_match.group(1)):
            name, duration = tuple_match.groups()
            spec.tasks.append(Task(id=name, duration=int(float(duration) * 60)))

    # Extract constraints
    constraints_match = re.search(r"constraints\s*=\s*\[(.*?)\]", content, re.DOTALL)
    if constraints_match:
        spec.constraints = parse_constraints(constraints_match.group(1))

    # Extract objective
    objective_match = re.search(r"objective\s*=\s*[\"'](\w+)[\"']", content)
    if objective_match:
        spec.objective = parse_objective(objective_match.group(1))

    if spec.is_valid():
        spec.csp_type = CSPType.SCHEDULING
        return spec

    return None


def try_extract_format_c(text: str) -> CSPSpec | None:
    """
    Try to extract CSP from Format C (natural structured).

    Format:
        Let me formalize the constraints:
        - Tasks: Alice (2hr), Bob (1hr)
        - Time window: 9am to 12pm
        - Constraint: Alice and Bob cannot overlap
        - Goal: Minimize total time
        Solution:
    """
    spec = CSPSpec(raw_text=text)

    # Look for "Tasks:" line
    tasks_match = re.search(r"Tasks?:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if tasks_match:
        spec.tasks = parse_tasks(tasks_match.group(1))

    # Look for constraints in various forms
    for pattern in [
        r"Constraints?:\s*(.+?)(?:\n|$)",
        r"Rules?:\s*(.+?)(?:\n|$)",
        r"-\s*Constraint:\s*(.+?)(?:\n|$)",
    ]:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            spec.constraints.extend(parse_constraints(match.group(1)))

    # Look for window
    window_match = re.search(r"(?:Time\s+)?[Ww]indow:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if window_match:
        spec.window = parse_window(window_match.group(1))

    # Look for objective/goal
    objective_match = re.search(r"(?:Goal|Objective):\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if objective_match:
        spec.objective = parse_objective(objective_match.group(1))

    # Check for Solution: trigger
    if re.search(r"\bSolution:\s*$", text, re.MULTILINE | re.IGNORECASE) and spec.is_valid():
        spec.csp_type = CSPType.SCHEDULING
        return spec

    return None


def try_extract_format_d(text: str) -> CSPSpec | None:
    """
    Try to extract CSP from Format D (JSON).

    Format:
        {"tasks": [...], "constraints": [...], "objective": "..."}
    """
    # Find JSON block
    json_match = re.search(r"\{[^{}]*\"tasks\"[^{}]*\}", text, re.DOTALL)
    if not json_match:
        return None

    try:
        data = json.loads(json_match.group(0))
    except json.JSONDecodeError:
        return None

    spec = CSPSpec(raw_text=text)

    # Parse tasks
    if "tasks" in data:
        for task in data["tasks"]:
            if isinstance(task, dict):
                spec.tasks.append(
                    Task(
                        id=task.get("id", task.get("name", "task")),
                        duration=parse_duration(str(task.get("duration", 60))),
                    )
                )
            elif isinstance(task, (list, tuple)) and len(task) >= 2:
                spec.tasks.append(Task(id=str(task[0]), duration=parse_duration(str(task[1]))))

    # Parse constraints
    if "constraints" in data:
        for constraint in data["constraints"]:
            if isinstance(constraint, str):
                spec.constraints.extend(parse_constraints(constraint))

    # Parse objective
    if "objective" in data:
        spec.objective = parse_objective(str(data["objective"]))

    if spec.is_valid():
        spec.csp_type = CSPType.SCHEDULING
        return spec

    return None


# =============================================================================
# MAIN EXTRACTION FUNCTION
# =============================================================================


def extract_csp_spec(text: str) -> CSPSpec | None:
    """
    Extract CSP specification from Chain-of-Thought output.

    Tries multiple formats in order of preference:
    1. Format A (declarative blocks) - cleanest, most parseable
    2. Format D (JSON) - structured
    3. Format B (function call) - code-like
    4. Format C (natural structured) - most flexible

    Args:
        text: The CoT output text to parse

    Returns:
        CSPSpec if extraction succeeds, None otherwise
    """
    # Try each format in order
    extractors = [
        ("format_a", try_extract_format_a),
        ("format_d", try_extract_format_d),
        ("format_b", try_extract_format_b),
        ("format_c", try_extract_format_c),
    ]

    for name, extractor in extractors:
        spec = extractor(text)
        if spec is not None:
            return spec

    return None


def detect_csp_type(text: str) -> CSPType:
    """
    Detect the type of CSP from natural language description.

    Used for routing to the appropriate solver.
    """
    text_lower = text.lower()

    # Scheduling keywords
    scheduling_keywords = ["schedule", "meeting", "appointment", "shift", "timetable", "calendar", "slot"]
    if any(kw in text_lower for kw in scheduling_keywords):
        return CSPType.SCHEDULING

    # Assignment keywords
    assignment_keywords = ["assign", "allocate", "match", "pair", "team", "distribute"]
    if any(kw in text_lower for kw in assignment_keywords):
        return CSPType.ASSIGNMENT

    # Routing keywords
    routing_keywords = ["route", "visit", "tour", "travel", "delivery", "path", "tsp", "vrp"]
    if any(kw in text_lower for kw in routing_keywords):
        return CSPType.ROUTING

    # Packing keywords
    packing_keywords = ["pack", "bin", "knapsack", "load", "container", "fit", "capacity"]
    if any(kw in text_lower for kw in packing_keywords):
        return CSPType.PACKING

    # Coloring keywords
    coloring_keywords = ["color", "frequency", "register", "conflict", "adjacent", "graph"]
    if any(kw in text_lower for kw in coloring_keywords):
        return CSPType.COLORING

    return CSPType.GENERIC


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("CSP Extractor Tests")
    print("=" * 60)

    # Test Format A
    format_a_text = """
    Let me structure this problem:

    TASKS: [Alice:2hr, Bob:1hr, Carol:1.5hr]
    WINDOW: [9:00, 12:00]
    CONSTRAINTS: [no_overlap(Alice, Bob), Carol before Alice]
    OBJECTIVE: minimize_makespan
    SOLVE:
    """

    print("\nFormat A Test:")
    spec = extract_csp_spec(format_a_text)
    if spec:
        print(f"  Tasks: {[(t.id, t.duration) for t in spec.tasks]}")
        print(f"  Constraints: {[(c.type, c.args) for c in spec.constraints]}")
        print(f"  Window: {spec.window}")
        print(f"  Objective: {spec.objective}")
    else:
        print("  FAILED")

    # Test Format C
    format_c_text = """
    Let me formalize the constraints:
    - Tasks: Alice (2hr), Bob (1hr)
    - Time window: 9am to 12pm
    - Constraint: Alice and Bob cannot overlap
    - Goal: Minimize total time
    Solution:
    """

    print("\nFormat C Test:")
    spec = extract_csp_spec(format_c_text)
    if spec:
        print(f"  Tasks: {[(t.id, t.duration) for t in spec.tasks]}")
        print(f"  Constraints: {[(c.type, c.args) for c in spec.constraints]}")
        print(f"  Window: {spec.window}")
    else:
        print("  FAILED")

    # Test parsing functions
    print("\nParsing Tests:")
    print(f"  parse_time('2pm') = {parse_time('2pm')} (expected: 840)")
    print(f"  parse_time('14:30') = {parse_time('14:30')} (expected: 870)")
    print(f"  parse_duration('2hr') = {parse_duration('2hr')} (expected: 120)")
    print(f"  parse_duration('90min') = {parse_duration('90min')} (expected: 90)")

    # Test type detection
    print("\nType Detection Tests:")
    tests = [
        ("Schedule my meetings", CSPType.SCHEDULING),
        ("Assign developers to projects", CSPType.ASSIGNMENT),
        ("Plan delivery route", CSPType.ROUTING),
        ("Pack items into boxes", CSPType.PACKING),
        ("Color the map regions", CSPType.COLORING),
    ]
    for text, expected in tests:
        detected = detect_csp_type(text)
        status = "PASS" if detected == expected else "FAIL"
        print(f"  [{status}] '{text}' -> {detected.value}")
