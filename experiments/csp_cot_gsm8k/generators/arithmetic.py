"""
Arithmetic Chain Trace Generator.

Handles problems with sequential arithmetic operations on a single value.

Examples:
- "Start with 10, add 5, multiply by 2"
- "A pizza has 8 slices. 3 people each eat 2 slices. How many left?"
- "6 bags with 4 oranges each. How many oranges total?"
"""

from __future__ import annotations

from decimal import Decimal

from .base import TraceGenerator
from ..schema.trace import Trace, TraceBuilder
from ..schema.problem import ProblemSpec, ProblemType, OperationType


class ArithmeticTraceGenerator(TraceGenerator):
    """
    Generator for arithmetic chain problems.

    These problems involve:
    - A starting value (explicit or implicit)
    - A sequence of arithmetic operations
    - A query for the final result
    """

    @property
    def supported_types(self) -> list[ProblemType]:
        return [ProblemType.ARITHMETIC_CHAIN, ProblemType.PERCENTAGE]

    def generate(self, spec: ProblemSpec) -> Trace:
        """
        Generate trace by applying operations in sequence.
        """
        builder = TraceBuilder(problem_type=spec.problem_type.value)

        # Determine the main variable name
        if spec.entities:
            main_var = spec.entities[0].name
            initial = spec.entities[0].initial_value or Decimal(0)
        else:
            main_var = "result"
            initial = Decimal(0)

        # Initialize
        builder.init(main_var, initial)

        # Apply operations
        for op in spec.operations:
            target = op.target or main_var

            if op.type == OperationType.ADD:
                if op.amount is not None:
                    builder.add(target, op.amount)

            elif op.type == OperationType.SUBTRACT:
                if op.amount is not None:
                    builder.subtract(target, op.amount)

            elif op.type == OperationType.MULTIPLY:
                if op.factor is not None:
                    builder.multiply(target, op.factor)
                elif op.amount is not None:
                    # Support "multiply by amount" syntax
                    builder.multiply(target, op.amount)

            elif op.type == OperationType.DIVIDE:
                if op.factor is not None:
                    builder.divide(target, op.factor)
                elif op.amount is not None:
                    builder.divide(target, op.amount)

        # Query
        query_target = spec.query.target if spec.query else main_var
        builder.query(query_target)

        return builder.build()


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    from ..schema.problem import Entity, Operation, Query
    from ..schema.verifier import verify_trace

    print("Arithmetic Chain Trace Generator Tests")
    print("=" * 60)

    generator = ArithmeticTraceGenerator()

    # Test 1: Simple chain
    print("\nTest 1: Start with 10, add 5, multiply by 2")
    spec1 = ProblemSpec(
        problem_type=ProblemType.ARITHMETIC_CHAIN,
        entities=[
            Entity(name="x", initial_value=Decimal(10)),
        ],
        operations=[
            Operation(type=OperationType.ADD, target="x", amount=Decimal(5)),
            Operation(type=OperationType.MULTIPLY, target="x", factor=Decimal(2)),
        ],
        query=Query(target="x"),
    )

    trace1 = generator.generate(spec1)
    result1 = verify_trace(trace1)

    print(f"Trace:\n{trace1.to_yaml_str()}")
    print(f"Valid: {result1.is_valid}")
    print(f"Answer: {trace1.answer} (expected: 30)")

    # Test 2: Pizza slices problem
    print("\nTest 2: 8 slices, 3 people eat 2 each")
    # 8 - (3 * 2) = 8 - 6 = 2
    spec2 = ProblemSpec(
        problem_type=ProblemType.ARITHMETIC_CHAIN,
        entities=[
            Entity(name="slices", initial_value=Decimal(8)),
            Entity(name="eaten", initial_value=Decimal(3)),  # 3 people
        ],
        operations=[
            Operation(type=OperationType.MULTIPLY, target="eaten", factor=Decimal(2)),  # each eat 2
            # Now we need to subtract eaten from slices
            # This is a limitation - we need a way to reference another entity's value
        ],
        query=Query(target="slices"),
    )

    # For complex operations, we can model it differently:
    # slices = 8, then subtract 6 (3*2 computed separately)
    spec2_simple = ProblemSpec(
        problem_type=ProblemType.ARITHMETIC_CHAIN,
        entities=[
            Entity(name="slices", initial_value=Decimal(8)),
        ],
        operations=[
            Operation(type=OperationType.SUBTRACT, target="slices", amount=Decimal(6)),  # 3*2=6
        ],
        query=Query(target="slices"),
    )

    trace2 = generator.generate(spec2_simple)
    result2 = verify_trace(trace2)

    print(f"Trace:\n{trace2.to_yaml_str()}")
    print(f"Valid: {result2.is_valid}")
    print(f"Answer: {trace2.answer} (expected: 2)")

    # Test 3: Percentage (25% off $40)
    print("\nTest 3: $40 with 25% off")
    spec3 = ProblemSpec(
        problem_type=ProblemType.PERCENTAGE,
        entities=[
            Entity(name="price", initial_value=Decimal(40)),
        ],
        operations=[
            Operation(type=OperationType.MULTIPLY, target="price", factor=Decimal("0.75")),
        ],
        query=Query(target="price"),
    )

    trace3 = generator.generate(spec3)
    result3 = verify_trace(trace3)

    print(f"Trace:\n{trace3.to_yaml_str()}")
    print(f"Valid: {result3.is_valid}")
    print(f"Answer: {trace3.answer} (expected: 30)")

    # Test 4: Multiplication (6 bags * 4 oranges)
    print("\nTest 4: 6 bags with 4 oranges each")
    spec4 = ProblemSpec(
        problem_type=ProblemType.ARITHMETIC_CHAIN,
        entities=[
            Entity(name="oranges", initial_value=Decimal(6)),
        ],
        operations=[
            Operation(type=OperationType.MULTIPLY, target="oranges", factor=Decimal(4)),
        ],
        query=Query(target="oranges"),
    )

    trace4 = generator.generate(spec4)
    result4 = verify_trace(trace4)

    print(f"Trace:\n{trace4.to_yaml_str()}")
    print(f"Valid: {result4.is_valid}")
    print(f"Answer: {trace4.answer} (expected: 24)")
