"""
Entity Tracking Trace Generator.

Handles problems where entities have values that change through operations.

Examples:
- "Jenny has 5 apples. She gives 2 to Bob. How many does Jenny have?"
- "Tom has 10 marbles. He loses 3 and finds 2. How many does he have?"
"""

from __future__ import annotations

from decimal import Decimal

from .base import TraceGenerator
from ..schema.trace import Trace, TraceBuilder
from ..schema.problem import ProblemSpec, ProblemType, OperationType


class EntityTraceGenerator(TraceGenerator):
    """
    Generator for entity tracking problems.

    These problems involve:
    - One or more entities with initial values
    - Operations that modify entity values (add, subtract, transfer)
    - A query for a specific entity's final value
    """

    @property
    def supported_types(self) -> list[ProblemType]:
        return [ProblemType.ENTITY_TRACKING]

    def generate(self, spec: ProblemSpec) -> Trace:
        """
        Generate trace by initializing entities and applying operations.
        """
        builder = TraceBuilder(problem_type="entity_tracking")

        # Step 1: Initialize all entities
        for entity in spec.entities:
            if entity.initial_value is not None:
                builder.init(entity.name, entity.initial_value)
            else:
                builder.init(entity.name, Decimal(0))

        # Step 2: Apply operations in sequence
        for op in spec.operations:
            if op.type == OperationType.ADD:
                if op.amount is not None:
                    builder.add(op.target, op.amount)

            elif op.type == OperationType.SUBTRACT:
                if op.amount is not None:
                    builder.subtract(op.target, op.amount)

            elif op.type == OperationType.TRANSFER:
                if op.source and op.amount is not None:
                    builder.transfer(op.source, op.target, op.amount)

            elif op.type == OperationType.MULTIPLY:
                if op.factor is not None:
                    builder.multiply(op.target, op.factor)

            elif op.type == OperationType.DIVIDE:
                if op.factor is not None:
                    builder.divide(op.target, op.factor)

        # Step 3: Query the target
        if spec.query:
            builder.query(spec.query.target)

        return builder.build()


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    from ..schema.problem import Entity, Operation, Query
    from ..schema.verifier import verify_trace

    print("Entity Trace Generator Tests")
    print("=" * 60)

    generator = EntityTraceGenerator()

    # Test 1: Simple transfer
    print("\nTest 1: Jenny gives 2 apples to Bob")
    spec1 = ProblemSpec(
        problem_type=ProblemType.ENTITY_TRACKING,
        entities=[
            Entity(name="jenny", initial_value=Decimal(5)),
            Entity(name="bob", initial_value=Decimal(0)),
        ],
        operations=[
            Operation(type=OperationType.TRANSFER, source="jenny", target="bob", amount=Decimal(2)),
        ],
        query=Query(target="jenny"),
    )

    trace1 = generator.generate(spec1)
    result1 = verify_trace(trace1)

    print(f"Trace:\n{trace1.to_yaml_str()}")
    print(f"Valid: {result1.is_valid}")
    print(f"Answer: {trace1.answer}")

    # Test 2: Multiple operations
    print("\nTest 2: Sam loses 3, finds 5")
    spec2 = ProblemSpec(
        problem_type=ProblemType.ENTITY_TRACKING,
        entities=[
            Entity(name="sam", initial_value=Decimal(10)),
        ],
        operations=[
            Operation(type=OperationType.SUBTRACT, target="sam", amount=Decimal(3)),
            Operation(type=OperationType.ADD, target="sam", amount=Decimal(5)),
        ],
        query=Query(target="sam"),
    )

    trace2 = generator.generate(spec2)
    result2 = verify_trace(trace2)

    print(f"Trace:\n{trace2.to_yaml_str()}")
    print(f"Valid: {result2.is_valid}")
    print(f"Answer: {trace2.answer}")

    # Test 3: Multiple transfers
    print("\nTest 3: Lisa gives 5 to Ann and 3 to Bob")
    spec3 = ProblemSpec(
        problem_type=ProblemType.ENTITY_TRACKING,
        entities=[
            Entity(name="lisa", initial_value=Decimal(20)),
            Entity(name="ann", initial_value=Decimal(0)),
            Entity(name="bob", initial_value=Decimal(0)),
        ],
        operations=[
            Operation(type=OperationType.TRANSFER, source="lisa", target="ann", amount=Decimal(5)),
            Operation(type=OperationType.TRANSFER, source="lisa", target="bob", amount=Decimal(3)),
        ],
        query=Query(target="lisa"),
    )

    trace3 = generator.generate(spec3)
    result3 = verify_trace(trace3)

    print(f"Trace:\n{trace3.to_yaml_str()}")
    print(f"Valid: {result3.is_valid}")
    print(f"Answer: {trace3.answer}")
