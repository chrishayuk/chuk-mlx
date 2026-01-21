"""
Generator Router.

Routes problem specs to the appropriate trace generator.
"""

from __future__ import annotations

from ..schema.trace import Trace
from ..schema.problem import ProblemSpec, ProblemType
from .base import TraceGenerator
from .entity import EntityTraceGenerator
from .arithmetic import ArithmeticTraceGenerator
from .comparison import ComparisonTraceGenerator
from .allocation import AllocationTraceGenerator


# Registry of generators
GENERATORS: list[TraceGenerator] = [
    EntityTraceGenerator(),
    ArithmeticTraceGenerator(),
    ComparisonTraceGenerator(),
    AllocationTraceGenerator(),
]


def route_to_generator(spec: ProblemSpec) -> TraceGenerator | None:
    """
    Find the appropriate generator for a problem spec.

    Args:
        spec: The problem specification

    Returns:
        The generator that can handle this spec, or None
    """
    for generator in GENERATORS:
        if generator.can_handle(spec):
            return generator
    return None


def generate_trace(spec: ProblemSpec) -> Trace | None:
    """
    Generate a trace for a problem spec.

    Convenience function that routes and generates in one call.

    Args:
        spec: The problem specification

    Returns:
        A verifiable trace, or None if no generator can handle the spec
    """
    generator = route_to_generator(spec)
    if generator is None:
        return None
    return generator.generate(spec)


# =============================================================================
# TYPE TO GENERATOR MAPPING
# =============================================================================


def get_generator_for_type(problem_type: ProblemType) -> TraceGenerator | None:
    """Get generator by problem type."""
    for generator in GENERATORS:
        if problem_type in generator.supported_types:
            return generator
    return None


def supported_problem_types() -> list[ProblemType]:
    """Return all problem types that have generators."""
    types = []
    for generator in GENERATORS:
        types.extend(generator.supported_types)
    return list(set(types))


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    from ..schema.problem import Entity, Operation, Query, Constraint, OperationType
    from ..schema.verifier import verify_trace
    from decimal import Decimal

    print("Generator Router Tests")
    print("=" * 60)

    print(f"\nSupported types: {[t.value for t in supported_problem_types()]}")

    # Test routing
    test_specs = [
        ProblemSpec(
            problem_type=ProblemType.ENTITY_TRACKING,
            entities=[Entity(name="jenny", initial_value=Decimal(5))],
            operations=[Operation(type=OperationType.SUBTRACT, target="jenny", amount=Decimal(2))],
            query=Query(target="jenny"),
        ),
        ProblemSpec(
            problem_type=ProblemType.ARITHMETIC_CHAIN,
            entities=[Entity(name="x", initial_value=Decimal(10))],
            operations=[Operation(type=OperationType.ADD, target="x", amount=Decimal(5))],
            query=Query(target="x"),
        ),
        ProblemSpec(
            problem_type=ProblemType.COMPARISON,
            entities=[
                Entity(name="tom", initial_value=Decimal(15)),
                Entity(name="jane", initial_value=Decimal(5)),
            ],
            query=Query(target="difference", question="compare", compare_a="tom", compare_b="jane"),
        ),
        ProblemSpec(
            problem_type=ProblemType.ALLOCATION,
            entities=[Entity(name="alice"), Entity(name="bob")],
            constraints=[
                Constraint(type="sum", entities=["alice", "bob"], value=Decimal(100)),
                Constraint(type="ratio", entities=["alice", "bob"], factor=Decimal(2)),
            ],
            query=Query(target="alice"),
        ),
    ]

    for i, spec in enumerate(test_specs, 1):
        print(f"\nTest {i}: {spec.problem_type.value}")
        generator = route_to_generator(spec)
        print(f"  Generator: {generator.__class__.__name__ if generator else 'None'}")

        trace = generate_trace(spec)
        if trace:
            result = verify_trace(trace)
            print(f"  Answer: {trace.answer}")
            print(f"  Valid: {result.is_valid}")
        else:
            print("  Could not generate trace")

    # Test unknown type
    print("\nTest 5: Unknown type (should return None)")
    unknown_spec = ProblemSpec(problem_type=ProblemType.UNKNOWN, entities=[])
    generator = route_to_generator(unknown_spec)
    print(f"  Generator: {generator}")
