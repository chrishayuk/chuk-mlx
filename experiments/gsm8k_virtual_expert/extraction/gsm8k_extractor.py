"""
GSM-8K Problem Extractor.

Parses natural language math problems into structured specifications
for routing to appropriate expert solvers.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ProblemType(Enum):
    """GSM-8K problem type categories."""
    ARITHMETIC_CHAIN = "arithmetic_chain"
    RATE_RATIO = "rate_ratio"
    ALLOCATION = "allocation"
    COMPARISON = "comparison"
    SCHEDULING_TIME = "scheduling_time"
    GEOMETRY = "geometry"
    MULTI_CONSTRAINT = "multi_constraint"
    PERCENTAGE = "percentage"
    UNKNOWN = "unknown"


@dataclass
class Entity:
    """An entity in the problem (person, object)."""
    name: str
    value: float | None = None
    unit: str | None = None


@dataclass
class Operation:
    """An operation on entities."""
    type: str  # add, subtract, multiply, divide
    source: str | None = None
    target: str | None = None
    amount: float | None = None
    unit: str | None = None


@dataclass
class ProblemSpec:
    """Extracted problem specification."""
    raw_text: str
    problem_type: ProblemType = ProblemType.UNKNOWN
    entities: list[Entity] = field(default_factory=list)
    operations: list[Operation] = field(default_factory=list)
    constraints: list[dict] = field(default_factory=list)
    target: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "problem_type": self.problem_type.value,
            "entities": [{"name": e.name, "value": e.value, "unit": e.unit} for e in self.entities],
            "operations": [{"type": o.type, "source": o.source, "target": o.target,
                          "amount": o.amount, "unit": o.unit} for o in self.operations],
            "constraints": self.constraints,
            "target": self.target,
            "metadata": self.metadata,
        }


class GSM8KExtractor:
    """
    Extract structured information from GSM-8K problems.

    Supports multiple problem types with specialized extraction logic.
    """

    # Operation verb mappings
    ADD_VERBS = {"buys", "gets", "receives", "finds", "earns", "wins", "collects",
                 "picks", "adds", "gained", "bought", "got", "received", "found",
                 "earned", "won", "collected", "picked", "added"}

    SUBTRACT_VERBS = {"gives", "loses", "spends", "sells", "uses", "eats", "removes",
                      "takes", "gave", "lost", "spent", "sold", "used", "ate",
                      "removed", "took", "away"}

    def __init__(self):
        self._load_patterns()

    def _load_patterns(self):
        """Load extraction patterns from config."""
        patterns_path = Path(__file__).parent.parent / "data" / "problem_types.json"
        if patterns_path.exists():
            with open(patterns_path) as f:
                self.patterns = json.load(f)
        else:
            self.patterns = {"problem_types": {}, "operation_verbs": {}, "quantity_patterns": []}

    def classify(self, problem: str) -> ProblemType:
        """
        Classify problem type using pattern matching.

        Note: For production, use L4 probe instead.
        """
        problem_lower = problem.lower()

        scores = {}
        for type_name, type_info in self.patterns.get("problem_types", {}).items():
            score = 0
            for pattern in type_info.get("patterns", []):
                if re.search(pattern, problem_lower):
                    score += 1
            scores[type_name] = score

        if not scores or max(scores.values()) == 0:
            return ProblemType.UNKNOWN

        best_type = max(scores, key=scores.get)
        return ProblemType(best_type)

    def extract(self, problem: str) -> ProblemSpec:
        """
        Extract structured specification from problem.
        """
        spec = ProblemSpec(raw_text=problem)
        spec.problem_type = self.classify(problem)

        # Route to specialized extractor
        if spec.problem_type == ProblemType.ARITHMETIC_CHAIN:
            self._extract_arithmetic_chain(problem, spec)
        elif spec.problem_type == ProblemType.RATE_RATIO:
            self._extract_rate_ratio(problem, spec)
        elif spec.problem_type == ProblemType.ALLOCATION:
            self._extract_allocation(problem, spec)
        elif spec.problem_type == ProblemType.COMPARISON:
            self._extract_comparison(problem, spec)
        elif spec.problem_type == ProblemType.SCHEDULING_TIME:
            self._extract_time(problem, spec)
        elif spec.problem_type == ProblemType.GEOMETRY:
            self._extract_geometry(problem, spec)
        elif spec.problem_type == ProblemType.PERCENTAGE:
            self._extract_percentage(problem, spec)
        elif spec.problem_type == ProblemType.MULTI_CONSTRAINT:
            self._extract_multi_constraint(problem, spec)
        else:
            # Fallback: try arithmetic chain
            self._extract_arithmetic_chain(problem, spec)

        return spec

    def _extract_arithmetic_chain(self, problem: str, spec: ProblemSpec):
        """Extract entities and sequential operations."""
        sentences = re.split(r'[.!?]', problem)

        for sentence in sentences:
            # Find entities with initial values
            # Pattern: "Name has/had N items"
            for match in re.finditer(
                r'(\b[A-Z][a-z]+\b)\s+(?:has|had|have|starts? with)\s+(\d+(?:\.\d+)?)\s*(\w+)?',
                sentence
            ):
                name, value, unit = match.groups()
                spec.entities.append(Entity(name=name, value=float(value), unit=unit))

            # Find operations - handle "gives N to" pattern specifically
            for match in re.finditer(
                r'(?:(\b[A-Z][a-z]+\b)|(?:She|He|They))\s*(buys?|gets?|receives?|finds?|gives?|loses?|spends?|sells?|eats?|takes?)\s+(\d+(?:\.\d+)?)\s*(?:(\w+)\s+)?(?:more\s+)?(?:to\s+)?(\b[A-Z][a-z]+\b)?',
                sentence, re.IGNORECASE
            ):
                subject, verb, amount, unit, target = match.groups()
                verb_lower = verb.lower()

                if any(v in verb_lower for v in self.ADD_VERBS):
                    op_type = "add"
                elif any(v in verb_lower for v in self.SUBTRACT_VERBS):
                    op_type = "subtract"
                else:
                    op_type = "unknown"

                spec.operations.append(Operation(
                    type=op_type,
                    source=subject,
                    target=target,
                    amount=float(amount) if amount else None,
                    unit=unit,
                ))

        # Find target (usually "how many" question)
        if re.search(r'how many.*does (\w+)', problem, re.IGNORECASE):
            match = re.search(r'how many.*does (\w+)', problem, re.IGNORECASE)
            spec.target = match.group(1)

    def _extract_rate_ratio(self, problem: str, spec: ProblemSpec):
        """Extract rate/ratio problem structure."""
        # Work rate: "N workers in T hours"
        match = re.search(
            r'(\d+)\s*workers?\s*.*?(\d+(?:\.\d+)?)\s*(hours?|days?|minutes?)',
            problem, re.IGNORECASE
        )
        if match:
            workers, time, unit = match.groups()
            spec.metadata["rate_type"] = "work_rate"
            spec.metadata["workers"] = int(workers)
            spec.metadata["time"] = float(time)
            spec.metadata["time_unit"] = unit

            # Find target workers
            target_match = re.search(
                r'(?:how long|how many).*?(\d+)\s*workers?',
                problem, re.IGNORECASE
            )
            if target_match:
                spec.metadata["target_workers"] = int(target_match.group(1))
            return

        # Speed: "N mph/km per hour"
        match = re.search(
            r'(\d+(?:\.\d+)?)\s*(?:mph|miles? per hour|km/?h|kilometers? per hour)',
            problem, re.IGNORECASE
        )
        if match:
            spec.metadata["rate_type"] = "speed"
            spec.metadata["speed"] = float(match.group(1))

            # Find time or distance
            time_match = re.search(r'(\d+(?:\.\d+)?)\s*(hours?|minutes?)', problem, re.IGNORECASE)
            if time_match:
                spec.metadata["time"] = float(time_match.group(1))
                spec.metadata["time_unit"] = time_match.group(2)

    def _extract_allocation(self, problem: str, spec: ProblemSpec):
        """Extract allocation problem structure."""
        # Find total amount
        total_match = re.search(r'\$?(\d+(?:\.\d+)?)', problem)
        if total_match:
            spec.metadata["total"] = float(total_match.group(1))

        # Find named entities
        names = re.findall(r'\b([A-Z][a-z]+)\b', problem)
        # Filter common words
        common = {"The", "How", "What", "If", "She", "He", "They", "It", "This", "That"}
        names = [n for n in names if n not in common]

        for name in set(names):
            spec.entities.append(Entity(name=name))

        # Find ratio constraints
        # "X gets twice what Y gets"
        for match in re.finditer(
            r'(\b[A-Z][a-z]+\b)\s+gets?\s+(twice|half|three times|(\d+) times)\s+(?:what|as much as)\s+(\b[A-Z][a-z]+\b)',
            problem, re.IGNORECASE
        ):
            entity1, multiplier, _, entity2 = match.groups()
            if "twice" in multiplier.lower():
                factor = 2
            elif "half" in multiplier.lower():
                factor = 0.5
            elif "three" in multiplier.lower():
                factor = 3
            else:
                factor = int(re.search(r'\d+', multiplier).group())

            spec.constraints.append({
                "type": "ratio",
                "entity1": entity1,
                "entity2": entity2,
                "factor": factor,
            })

        # "X gets N more than Y"
        for match in re.finditer(
            r'(\b[A-Z][a-z]+\b)\s+gets?\s+(\d+)\s+more than\s+(\b[A-Z][a-z]+\b)',
            problem, re.IGNORECASE
        ):
            entity1, diff, entity2 = match.groups()
            spec.constraints.append({
                "type": "difference",
                "entity1": entity1,
                "entity2": entity2,
                "difference": int(diff),
            })

    def _extract_comparison(self, problem: str, spec: ProblemSpec):
        """Extract comparison problem structure."""
        # Find multiplier relationship: "X has N times as many as Y"
        match = re.search(
            r'(\b[A-Z][a-z]+\b)\s+has\s+(\d+)\s*(?:times|x)\s+(?:as many|as much)\s+(?:\w+\s+)?as\s+(\b[A-Z][a-z]+\b)',
            problem, re.IGNORECASE
        )
        if match:
            entity1, factor, entity2 = match.groups()
            spec.entities.append(Entity(name=entity1))
            spec.entities.append(Entity(name=entity2))
            spec.metadata["relationship"] = "multiply"
            spec.metadata["factor"] = int(factor)
            spec.metadata["multiplier_entity"] = entity1  # This entity has N times the other

        # Find known value - but don't match the "N times" pattern
        # Look for "Name has N" where N is NOT followed by "times"
        for match in re.finditer(
            r'(\b[A-Z][a-z]+\b)\s+has\s+(\d+)(?!\s*(?:times|x))',
            problem, re.IGNORECASE
        ):
            name, value = match.groups()
            # Add entity if not already present
            found = False
            for entity in spec.entities:
                if entity.name == name:
                    entity.value = float(value)
                    found = True
                    break
            if not found:
                spec.entities.append(Entity(name=name, value=float(value)))

        # Check for "how many more"
        if re.search(r'how many more', problem, re.IGNORECASE):
            spec.target = "difference"

    def _extract_time(self, problem: str, spec: ProblemSpec):
        """Extract time calculation structure."""
        # Find start time
        match = re.search(r'(\d{1,2}):?(\d{2})?\s*(am|pm|AM|PM)', problem)
        if match:
            hour, minute, period = match.groups()
            hour = int(hour)
            minute = int(minute) if minute else 0
            if period.lower() == "pm" and hour != 12:
                hour += 12
            elif period.lower() == "am" and hour == 12:
                hour = 0
            spec.metadata["start_hour"] = hour
            spec.metadata["start_minute"] = minute

        # Find duration operations - handle "N hours and M minutes" format
        match = re.search(
            r'(\d+(?:\.\d+)?)\s*hours?\s+(?:and\s+)?(\d+(?:\.\d+)?)\s*(?:minutes?|mins?)',
            problem, re.IGNORECASE
        )
        if match:
            hours, minutes = match.groups()
            spec.operations.append(Operation(
                type="add_time",
                amount=float(hours),
                unit="hours",
            ))
            spec.operations.append(Operation(
                type="add_time",
                amount=float(minutes),
                unit="minutes",
            ))
        else:
            # Single duration
            for match in re.finditer(
                r'(?:travels?|takes?|lasts?|waits?|stops?)\s+(?:for\s+)?(\d+(?:\.\d+)?)\s*(hours?|minutes?|mins?)',
                problem, re.IGNORECASE
            ):
                duration, unit = match.groups()
                spec.operations.append(Operation(
                    type="add_time",
                    amount=float(duration),
                    unit="hours" if "hour" in unit.lower() else "minutes",
                ))

    def _extract_geometry(self, problem: str, spec: ProblemSpec):
        """Extract geometry problem structure."""
        problem_lower = problem.lower()

        # Identify shape
        shapes = ["rectangle", "square", "triangle", "circle", "cube", "cylinder"]
        for shape in shapes:
            if shape in problem_lower:
                spec.metadata["shape"] = shape
                break

        # Extract dimensions
        # "N by M" or "N x M" (with optional units)
        match = re.search(r'(\d+(?:\.\d+)?)\s*(?:m|cm|ft|in)?\s*(?:by|x|×)\s*(\d+(?:\.\d+)?)', problem, re.IGNORECASE)
        if match:
            spec.metadata["length"] = float(match.group(1))
            spec.metadata["width"] = float(match.group(2))

        # "Nm by Mm" format (units attached)
        if "length" not in spec.metadata:
            match = re.search(r'(\d+(?:\.\d+)?)(?:m|cm|ft)?\s+(?:long|by)\s+.*?(\d+(?:\.\d+)?)', problem, re.IGNORECASE)
            if match:
                spec.metadata["length"] = float(match.group(1))
                spec.metadata["width"] = float(match.group(2))

        # "length of N and width of M" format
        if "length" not in spec.metadata:
            len_match = re.search(r'length\s+(?:of\s+)?(\d+(?:\.\d+)?)', problem, re.IGNORECASE)
            wid_match = re.search(r'width\s+(?:of\s+)?(\d+(?:\.\d+)?)', problem, re.IGNORECASE)
            if len_match and wid_match:
                spec.metadata["length"] = float(len_match.group(1))
                spec.metadata["width"] = float(wid_match.group(1))

        # "side N" or "radius N"
        match = re.search(r'(?:side|radius|diameter)\s+(?:of\s+)?(?:length\s+)?(\d+(?:\.\d+)?)', problem, re.IGNORECASE)
        if match:
            spec.metadata["dimension"] = float(match.group(1))

        # Target property
        if "area" in problem_lower:
            spec.target = "area"
        elif "perimeter" in problem_lower:
            spec.target = "perimeter"
        elif "volume" in problem_lower:
            spec.target = "volume"
        elif "circumference" in problem_lower:
            spec.target = "circumference"

    def _extract_percentage(self, problem: str, spec: ProblemSpec):
        """Extract percentage problem structure."""
        # Find base value
        match = re.search(r'\$(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s*dollars?', problem, re.IGNORECASE)
        if match:
            spec.metadata["base_value"] = float(match.group(1) or match.group(2))

        # Find percentage
        match = re.search(r'(\d+(?:\.\d+)?)\s*%', problem)
        if match:
            spec.metadata["percentage"] = float(match.group(1))

        # Determine operation type
        problem_lower = problem.lower()
        if "off" in problem_lower or "discount" in problem_lower:
            spec.metadata["operation"] = "discount"
        elif "tax" in problem_lower:
            spec.metadata["operation"] = "tax"
        elif "tip" in problem_lower:
            spec.metadata["operation"] = "tip"
        elif "increase" in problem_lower:
            spec.metadata["operation"] = "increase"
        elif "decrease" in problem_lower:
            spec.metadata["operation"] = "decrease"

    def _extract_multi_constraint(self, problem: str, spec: ProblemSpec):
        """Extract multi-constraint equation problem."""
        # Find equations
        # "sum of X and Y is N"
        match = re.search(
            r'sum\s+(?:of\s+)?(?:two\s+)?(?:numbers?)?\s*is\s*(\d+)',
            problem, re.IGNORECASE
        )
        if match:
            spec.constraints.append({
                "type": "sum",
                "value": int(match.group(1)),
            })

        # "difference is N"
        match = re.search(
            r'(?:their\s+)?difference\s+is\s*(\d+)',
            problem, re.IGNORECASE
        )
        if match:
            spec.constraints.append({
                "type": "difference",
                "value": int(match.group(1)),
            })

        # "doubled is N more than tripled"
        match = re.search(
            r'(doubled|tripled|halved)\s+is\s+(\d+)\s+more than\s+(doubled|tripled|halved)',
            problem, re.IGNORECASE
        )
        if match:
            spec.metadata["equation_type"] = "linear"


def extract_problem(problem: str) -> ProblemSpec:
    """Convenience function for extracting problem structure."""
    extractor = GSM8KExtractor()
    return extractor.extract(problem)


if __name__ == "__main__":
    print("GSM-8K Extractor Tests")
    print("=" * 60)

    extractor = GSM8KExtractor()

    tests = [
        ("Jenny has 5 apples. She buys 3 more. Then gives 2 away. How many does she have?",
         ProblemType.ARITHMETIC_CHAIN),
        ("If 3 workers paint a house in 6 hours, how long for 2 workers?",
         ProblemType.RATE_RATIO),
        ("Split $100 among Alice, Bob, Carol where Alice gets twice what Bob gets.",
         ProblemType.ALLOCATION),
        ("Tom has 3 times as many marbles as Jane. Jane has 5. How many more does Tom have?",
         ProblemType.COMPARISON),
        ("Train leaves at 9:00am, travels 3 hours, stops for 30 minutes. What time does it arrive?",
         ProblemType.SCHEDULING_TIME),
        ("A rectangle is 5m by 3m. What is its area?",
         ProblemType.GEOMETRY),
        ("A shirt costs $40. It's 25% off. What's the final price?",
         ProblemType.PERCENTAGE),
    ]

    for problem, expected_type in tests:
        spec = extractor.extract(problem)
        status = "PASS" if spec.problem_type == expected_type else "FAIL"
        print(f"\n{status}: {problem[:50]}...")
        print(f"  Type: {spec.problem_type.value} (expected: {expected_type.value})")
        print(f"  Entities: {[e.name for e in spec.entities]}")
        print(f"  Operations: {len(spec.operations)}")
        print(f"  Metadata: {spec.metadata}")
