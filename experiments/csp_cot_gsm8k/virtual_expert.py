"""
CSP-CoT as a Virtual Expert.

Integrates the verifiable trace solver with the virtual expert framework,
enabling LLM-based dispatch to the CSP-CoT solver for GSM-8K problems.
"""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any, ClassVar

from pydantic import Field

# Virtual expert imports
from chuk_virtual_expert.expert import VirtualExpert
from chuk_virtual_expert.models import VirtualExpertAction

# CSP-CoT imports
from .schema.problem import (
    ProblemSpec,
    ProblemType,
    Entity,
    Operation,
    OperationType,
    Query,
    Constraint,
)
from .schema.verifier import TraceVerifier, VerificationStatus
from .generators import generate_trace


class MathOperation:
    """Operations supported by the math word problem expert."""
    SOLVE = "solve"


class MathWordProblemExpert(VirtualExpert):
    """
    Virtual expert for solving GSM-8K style math word problems.

    Uses CSP-CoT: verifiable solver traces instead of English CoT.

    Key features:
    - 100% verifiable traces (every step can be replayed)
    - 100% error detection (invalid traces are caught)
    - Exact computation (no neural arithmetic errors)

    Operations:
        - solve: Parse and solve a word problem with verifiable trace
    """

    name: ClassVar[str] = "math_word_problem"
    description: ClassVar[str] = "Solve math word problems with verifiable traces (CSP-CoT)"
    version: ClassVar[str] = "1.0.0"
    priority: ClassVar[int] = 10  # High priority for math problems

    # File paths
    cot_examples_file: ClassVar[str] = "cot_examples.json"
    schema_file: ClassVar[str] = "schema.json"

    def get_operations(self) -> list[str]:
        """Return list of available operations."""
        return [MathOperation.SOLVE]

    def execute_operation(
        self,
        operation: str,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a math word problem operation."""
        if operation == MathOperation.SOLVE:
            return self.solve(**parameters)
        else:
            return {
                "success": False,
                "error": f"Unknown operation: {operation}",
            }

    def solve(
        self,
        problem_type: str,
        entities: list[dict],
        operations: list[dict] | None = None,
        constraints: list[dict] | None = None,
        query: dict | None = None,
        raw_text: str = "",
    ) -> dict[str, Any]:
        """
        Solve a math word problem with a verifiable trace.

        Args:
            problem_type: Type of problem (entity_tracking, arithmetic_chain, etc.)
            entities: List of entity definitions
            operations: List of operations to perform
            constraints: List of constraints (for allocation problems)
            query: Query specification
            raw_text: Original problem text

        Returns:
            Structured result with answer, trace, and verification status
        """
        try:
            # Build ProblemSpec from parameters
            spec = self._build_spec(
                problem_type=problem_type,
                entities=entities,
                operations=operations or [],
                constraints=constraints or [],
                query=query,
                raw_text=raw_text,
            )

            if not spec.is_valid():
                return {
                    "success": False,
                    "error": "Invalid problem specification",
                    "problem_type": problem_type,
                }

            # Generate trace
            trace = generate_trace(spec)
            if trace is None:
                return {
                    "success": False,
                    "error": f"No generator for problem type: {problem_type}",
                    "problem_type": problem_type,
                }

            # Verify trace
            verifier = TraceVerifier()
            verification = verifier.verify(trace)

            # Build result
            return {
                "success": True,
                "answer": float(trace.answer) if trace.answer is not None else None,
                "verified": verification.status == VerificationStatus.VALID,
                "verification_status": verification.status.value,
                "problem_type": problem_type,
                "trace_steps": len(trace.steps),
                "trace": [
                    {
                        "action": step.action.value,
                        "params": {k: float(v) if isinstance(v, Decimal) else v
                                   for k, v in step.params.items()},
                    }
                    for step in trace.steps
                ],
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "problem_type": problem_type,
            }

    def _build_spec(
        self,
        problem_type: str,
        entities: list[dict],
        operations: list[dict],
        constraints: list[dict],
        query: dict | None,
        raw_text: str,
    ) -> ProblemSpec:
        """Build a ProblemSpec from dict parameters with flexible parsing."""
        # Parse problem type
        try:
            pt = ProblemType(problem_type)
        except ValueError:
            pt = ProblemType.UNKNOWN

        # Build entity lookup for resolving references
        entity_values = {}
        parsed_entities = []
        for e in entities:
            name = e.get("name", "unknown")
            initial = self._parse_numeric(e.get("initial_value"))
            entity_values[name] = initial
            parsed_entities.append(
                Entity(
                    name=name,
                    attribute=e.get("attribute"),
                    initial_value=initial,
                    unit=e.get("unit"),
                )
            )

        # Parse operations with flexible field handling
        parsed_operations = []
        for o in operations:
            op_type_str = o.get("type", "add")
            try:
                op_type = OperationType(op_type_str)
            except ValueError:
                op_type = OperationType.ADD

            # Flexible amount/value handling
            amount = self._resolve_value(o, "amount", entity_values)
            if amount is None:
                amount = self._resolve_value(o, "value", entity_values)

            # Flexible factor handling
            factor = self._resolve_value(o, "factor", entity_values)
            if factor is None:
                factor = self._resolve_value(o, "multiplier", entity_values)

            parsed_operations.append(
                Operation(
                    type=op_type,
                    target=o.get("target", ""),
                    source=o.get("source") or o.get("from"),
                    amount=amount,
                    factor=factor,
                )
            )

        # Parse constraints
        parsed_constraints = []
        for c in constraints:
            parsed_constraints.append(
                Constraint(
                    type=c.get("type", ""),
                    entities=c.get("entities", []),
                    factor=self._parse_numeric(c.get("factor")),
                    value=self._parse_numeric(c.get("value")),
                )
            )

        # Parse query with flexible target handling
        parsed_query = None
        if query:
            target = query.get("target", "")
            # If target is "result" or similar, use the last entity
            if target in ("result", "answer", "final") and parsed_entities:
                target = parsed_entities[0].name
            parsed_query = Query(
                target=target,
                question=query.get("question", "value"),
                compare_a=query.get("compare_a"),
                compare_b=query.get("compare_b"),
            )

        return ProblemSpec(
            problem_type=pt,
            entities=parsed_entities,
            operations=parsed_operations,
            constraints=parsed_constraints,
            query=parsed_query,
            raw_text=raw_text,
        )

    def _parse_numeric(self, value: Any) -> Decimal | None:
        """Parse a value to Decimal, handling various formats."""
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return Decimal(str(value))
        if isinstance(value, str):
            # Try to parse as number
            try:
                # Remove currency symbols and commas
                cleaned = value.replace("$", "").replace(",", "").strip()
                return Decimal(cleaned)
            except:
                return None
        return None

    def _resolve_value(
        self,
        obj: dict,
        key: str,
        entity_values: dict[str, Decimal | None],
    ) -> Decimal | None:
        """
        Resolve a value that might be a number or entity reference.

        Handles cases like:
        - {"amount": 3} -> Decimal(3)
        - {"value": "eggs_eaten"} -> lookup entity_values["eggs_eaten"]
        - {"amount": "3"} -> Decimal(3)
        """
        value = obj.get(key)
        if value is None:
            return None

        # If it's already numeric, parse it
        if isinstance(value, (int, float)):
            return Decimal(str(value))

        if isinstance(value, str):
            # First try to parse as number
            try:
                cleaned = value.replace("$", "").replace(",", "").strip()
                return Decimal(cleaned)
            except:
                pass

            # Try to resolve as entity reference
            if value in entity_values and entity_values[value] is not None:
                return entity_values[value]

        return None


# CoT examples for few-shot extraction
# These examples are carefully designed to show the EXACT schema format
COT_EXAMPLES = [
    {
        "query": "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and bakes 4 into muffins. She sells the rest at $2 each. How much does she make?",
        "action": {
            "expert": "math_word_problem",
            "operation": "solve",
            "parameters": {
                "problem_type": "arithmetic_chain",
                "entities": [{"name": "eggs", "initial_value": 16}],
                "operations": [
                    {"type": "subtract", "target": "eggs", "amount": 3},
                    {"type": "subtract", "target": "eggs", "amount": 4},
                    {"type": "multiply", "target": "eggs", "factor": 2}
                ],
                "query": {"target": "eggs"}
            },
            "confidence": 0.95,
            "reasoning": "Arithmetic chain: 16 eggs - 3 eaten - 4 baked = 9 remaining, then 9 * $2 = $18"
        },
    },
    {
        "query": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts total?",
        "action": {
            "expert": "math_word_problem",
            "operation": "solve",
            "parameters": {
                "problem_type": "arithmetic_chain",
                "entities": [{"name": "total", "initial_value": 2}],
                "operations": [
                    {"type": "add", "target": "total", "amount": 1}
                ],
                "query": {"target": "total"}
            },
            "confidence": 0.95,
            "reasoning": "2 blue + 1 white (half of 2) = 3 total bolts"
        },
    },
    {
        "query": "Jenny has 5 apples. She gives 2 to Bob. How many does Jenny have?",
        "action": {
            "expert": "math_word_problem",
            "operation": "solve",
            "parameters": {
                "problem_type": "entity_tracking",
                "entities": [
                    {"name": "jenny", "initial_value": 5},
                    {"name": "bob", "initial_value": 0}
                ],
                "operations": [
                    {"type": "transfer", "source": "jenny", "target": "bob", "amount": 2}
                ],
                "query": {"target": "jenny"}
            },
            "confidence": 0.95,
            "reasoning": "Entity tracking: Jenny starts with 5, transfers 2 to Bob, has 3 left"
        },
    },
    {
        "query": "James runs 3 sprints 3 times a week. Each sprint is 60 meters. How many meters does he run per week?",
        "action": {
            "expert": "math_word_problem",
            "operation": "solve",
            "parameters": {
                "problem_type": "arithmetic_chain",
                "entities": [{"name": "meters", "initial_value": 3}],
                "operations": [
                    {"type": "multiply", "target": "meters", "factor": 3},
                    {"type": "multiply", "target": "meters", "factor": 60}
                ],
                "query": {"target": "meters"}
            },
            "confidence": 0.95,
            "reasoning": "3 sprints * 3 times/week * 60 meters = 540 meters"
        },
    },
    {
        "query": "Tom has 15 marbles. Jane has 5 marbles. How many more does Tom have than Jane?",
        "action": {
            "expert": "math_word_problem",
            "operation": "solve",
            "parameters": {
                "problem_type": "comparison",
                "entities": [
                    {"name": "tom", "initial_value": 15},
                    {"name": "jane", "initial_value": 5}
                ],
                "operations": [],
                "query": {"target": "difference", "question": "compare", "compare_a": "tom", "compare_b": "jane"}
            },
            "confidence": 0.95,
            "reasoning": "Comparison: Tom (15) - Jane (5) = 10 more"
        },
    },
    {
        "query": "What is the capital of France?",
        "action": {
            "expert": "none",
            "operation": "passthrough",
            "parameters": {},
            "confidence": 0.99,
            "reasoning": "Not a math word problem"
        },
    },
]


# Schema documentation for the prompt
SCHEMA_DOCS = """
## CRITICAL: Parameter Schema

Use ONLY these exact field names with NUMERIC values (not entity names):

### Operations format:
- subtract: {"type": "subtract", "target": "<entity>", "amount": <NUMBER>}
- add: {"type": "add", "target": "<entity>", "amount": <NUMBER>}
- multiply: {"type": "multiply", "target": "<entity>", "factor": <NUMBER>}
- divide: {"type": "divide", "target": "<entity>", "factor": <NUMBER>}
- transfer: {"type": "transfer", "source": "<entity>", "target": "<entity>", "amount": <NUMBER>}

### Key rules:
1. "amount" and "factor" must be NUMBERS, not entity names
2. Use "amount" for add/subtract, "factor" for multiply/divide
3. Chain operations on a SINGLE entity when computing step-by-step
4. The query target should match the entity you want the final value of
"""


def get_cot_examples() -> list[dict]:
    """Get CoT examples for few-shot prompting."""
    return COT_EXAMPLES


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("MathWordProblemExpert Test")
    print("=" * 60)

    expert = MathWordProblemExpert()

    # Test direct execution
    action = VirtualExpertAction(
        expert="math_word_problem",
        operation="solve",
        parameters={
            "problem_type": "entity_tracking",
            "entities": [
                {"name": "jenny", "initial_value": 5},
                {"name": "bob", "initial_value": 0},
            ],
            "operations": [
                {"type": "transfer", "source": "jenny", "target": "bob", "amount": 2}
            ],
            "query": {"target": "jenny"},
            "raw_text": "Jenny has 5 apples. She gives 2 to Bob. How many does Jenny have?",
        },
    )

    result = expert.execute(action)
    print(f"\nTest: Jenny's apples")
    print(f"  Success: {result.success}")
    print(f"  Data: {result.data}")

    # Test arithmetic chain
    action2 = VirtualExpertAction(
        expert="math_word_problem",
        operation="solve",
        parameters={
            "problem_type": "arithmetic_chain",
            "entities": [{"name": "eggs", "initial_value": 16}],
            "operations": [
                {"type": "subtract", "target": "eggs", "amount": 3},
                {"type": "subtract", "target": "eggs", "amount": 4},
                {"type": "multiply", "target": "eggs", "factor": 2},
            ],
            "query": {"target": "eggs"},
        },
    )

    result2 = expert.execute(action2)
    print(f"\nTest: Janet's eggs (16 - 3 - 4) * 2 = 18")
    print(f"  Success: {result2.success}")
    print(f"  Answer: {result2.data.get('answer') if result2.data else None}")
    print(f"  Verified: {result2.data.get('verified') if result2.data else None}")
