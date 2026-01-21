"""
Comparison Trace Generator.

Handles problems asking "how many more/less" between entities.

Examples:
- "Tom has 15, Jane has 5. How many more does Tom have than Jane?"
- "Alice has 3 times as many as Bob. Bob has 5. How many more does Alice have?"
"""

from __future__ import annotations

from decimal import Decimal

from .base import TraceGenerator
from ..schema.trace import Trace, TraceBuilder
from ..schema.problem import ProblemSpec, ProblemType, OperationType


class ComparisonTraceGenerator(TraceGenerator):
    """
    Generator for comparison problems.

    These problems involve:
    - Two or more entities with values
    - Computing the difference or ratio between them
    - A query for how many more/less one has than another
    """

    @property
    def supported_types(self) -> list[ProblemType]:
        return [ProblemType.COMPARISON]

    def generate(self, spec: ProblemSpec) -> Trace:
        """
        Generate trace by initializing entities and computing difference.
        """
        builder = TraceBuilder(problem_type="comparison")

        # Step 1: Initialize all entities
        for entity in spec.entities:
            if entity.initial_value is not None:
                builder.init(entity.name, entity.initial_value)

        # Step 2: Apply any operations (e.g., "Tom has 3 times as many as Jane")
        for op in spec.operations:
            if op.type == OperationType.MULTIPLY:
                if op.factor is not None:
                    builder.multiply(op.target, op.factor)
            elif op.type == OperationType.ADD:
                if op.amount is not None:
                    builder.add(op.target, op.amount)

        # Step 3: Compute comparison if query is a comparison
        if spec.query and spec.query.question == "compare":
            entity_a = spec.query.compare_a
            entity_b = spec.query.compare_b
            if entity_a and entity_b:
                builder.compare(entity_a, entity_b, "difference")
                builder.query("difference")
            else:
                # Fallback to querying the target directly
                builder.query(spec.query.target)
        elif spec.query:
            builder.query(spec.query.target)

        return builder.build()


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    from ..schema.problem import Entity, Operation, Query
    from ..schema.verifier import verify_trace

    print("Comparison Trace Generator Tests")
    print("=" * 60)

    generator = ComparisonTraceGenerator()

    # Test 1: Simple comparison
    print("\nTest 1: Tom has 15, Jane has 5. How many more does Tom have?")
    spec1 = ProblemSpec(
        problem_type=ProblemType.COMPARISON,
        entities=[
            Entity(name="tom", initial_value=Decimal(15)),
            Entity(name="jane", initial_value=Decimal(5)),
        ],
        operations=[],
        query=Query(target="difference", question="compare", compare_a="tom", compare_b="jane"),
    )

    trace1 = generator.generate(spec1)
    result1 = verify_trace(trace1)

    print(f"Trace:\n{trace1.to_yaml_str()}")
    print(f"Valid: {result1.is_valid}")
    print(f"Answer: {trace1.answer} (expected: 10)")

    # Test 2: With multiplication
    print("\nTest 2: Tom has 3x Jane's. Jane has 5. How many more does Tom have?")
    spec2 = ProblemSpec(
        problem_type=ProblemType.COMPARISON,
        entities=[
            Entity(name="jane", initial_value=Decimal(5)),
            Entity(name="tom", initial_value=Decimal(5)),  # Start same, then multiply
        ],
        operations=[
            Operation(type=OperationType.MULTIPLY, target="tom", factor=Decimal(3)),
        ],
        query=Query(target="difference", question="compare", compare_a="tom", compare_b="jane"),
    )

    trace2 = generator.generate(spec2)
    result2 = verify_trace(trace2)

    print(f"Trace:\n{trace2.to_yaml_str()}")
    print(f"Valid: {result2.is_valid}")
    print(f"Answer: {trace2.answer} (expected: 10)")

    # Test 3: How many fewer
    print("\nTest 3: Alice has 8, Bob has 12. How many fewer does Alice have?")
    spec3 = ProblemSpec(
        problem_type=ProblemType.COMPARISON,
        entities=[
            Entity(name="alice", initial_value=Decimal(8)),
            Entity(name="bob", initial_value=Decimal(12)),
        ],
        operations=[],
        query=Query(target="difference", question="compare", compare_a="alice", compare_b="bob"),
    )

    trace3 = generator.generate(spec3)
    result3 = verify_trace(trace3)

    print(f"Trace:\n{trace3.to_yaml_str()}")
    print(f"Valid: {result3.is_valid}")
    print(f"Answer: {trace3.answer} (expected: -4, meaning 4 fewer)")
