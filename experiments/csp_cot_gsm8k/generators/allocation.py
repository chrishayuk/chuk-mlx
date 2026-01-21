"""
Allocation Trace Generator.

Handles constraint-based allocation problems.

Examples:
- "Split $100 between Alice and Bob. Alice gets twice what Bob gets."
- "Sum of two numbers is 30. One is 4 more than the other."
"""

from __future__ import annotations

from decimal import Decimal

from .base import TraceGenerator
from ..schema.trace import Trace, TraceBuilder
from ..schema.problem import ProblemSpec, ProblemType


class AllocationTraceGenerator(TraceGenerator):
    """
    Generator for allocation problems.

    These problems involve:
    - Multiple entities
    - Constraints (sum, ratio, difference)
    - Need to solve for entity values

    For now, supports simple cases:
    - Two entities with sum constraint and ratio constraint
    - Two entities with sum constraint and difference constraint
    """

    @property
    def supported_types(self) -> list[ProblemType]:
        return [ProblemType.ALLOCATION, ProblemType.RATE_EQUATION]

    def generate(self, spec: ProblemSpec) -> Trace:
        """
        Solve constraints and generate trace.
        """
        builder = TraceBuilder(problem_type="allocation")

        # Extract constraints
        sum_constraint = None
        ratio_constraint = None
        diff_constraint = None

        for c in spec.constraints:
            if c.type == "sum":
                sum_constraint = c
            elif c.type == "ratio":
                ratio_constraint = c
            elif c.type == "difference":
                diff_constraint = c

        # Solve based on constraint types
        if sum_constraint and ratio_constraint:
            # Sum = S, A = k * B
            # A + B = S → k*B + B = S → B = S / (k+1)
            # A = k * B
            total = sum_constraint.value or Decimal(0)
            ratio = ratio_constraint.factor or Decimal(1)

            entities = sum_constraint.entities
            if len(entities) >= 2:
                # Figure out which entity is multiplied
                # ratio_constraint.entities should be [larger, smaller]
                if ratio_constraint.entities and len(ratio_constraint.entities) >= 2:
                    larger = ratio_constraint.entities[0]
                    smaller = ratio_constraint.entities[1]
                else:
                    larger = entities[0]
                    smaller = entities[1]

                smaller_val = total / (ratio + 1)
                larger_val = ratio * smaller_val

                builder.init(smaller, smaller_val)
                builder.init(larger, larger_val)

        elif sum_constraint and diff_constraint:
            # A + B = S, A - B = D
            # A = (S + D) / 2
            # B = (S - D) / 2
            total = sum_constraint.value or Decimal(0)
            diff = diff_constraint.value or Decimal(0)

            entities = sum_constraint.entities
            if len(entities) >= 2:
                entity_a = entities[0]
                entity_b = entities[1]

                val_a = (total + diff) / 2
                val_b = (total - diff) / 2

                builder.init(entity_a, val_a)
                builder.init(entity_b, val_b)

        else:
            # Just initialize entities with their values
            for entity in spec.entities:
                if entity.initial_value is not None:
                    builder.init(entity.name, entity.initial_value)

        # Query
        if spec.query:
            builder.query(spec.query.target)

        return builder.build()


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    from ..schema.problem import Entity, Query, Constraint
    from ..schema.verifier import verify_trace

    print("Allocation Trace Generator Tests")
    print("=" * 60)

    generator = AllocationTraceGenerator()

    # Test 1: Sum and ratio
    print("\nTest 1: Split $100, Alice gets 2x Bob")
    # A + B = 100, A = 2B → 2B + B = 100 → B = 33.33, A = 66.67
    spec1 = ProblemSpec(
        problem_type=ProblemType.ALLOCATION,
        entities=[
            Entity(name="alice"),
            Entity(name="bob"),
        ],
        constraints=[
            Constraint(type="sum", entities=["alice", "bob"], value=Decimal(100)),
            Constraint(type="ratio", entities=["alice", "bob"], factor=Decimal(2)),
        ],
        query=Query(target="alice"),
    )

    trace1 = generator.generate(spec1)
    result1 = verify_trace(trace1)

    print(f"Trace:\n{trace1.to_yaml_str()}")
    print(f"Valid: {result1.is_valid}")
    print(f"Answer: {trace1.answer} (expected: ~66.67)")

    # Test 2: Sum and difference
    print("\nTest 2: Sum is 30, difference is 4")
    # A + B = 30, A - B = 4 → A = 17, B = 13
    spec2 = ProblemSpec(
        problem_type=ProblemType.ALLOCATION,
        entities=[
            Entity(name="x"),
            Entity(name="y"),
        ],
        constraints=[
            Constraint(type="sum", entities=["x", "y"], value=Decimal(30)),
            Constraint(type="difference", entities=["x", "y"], value=Decimal(4)),
        ],
        query=Query(target="x"),
    )

    trace2 = generator.generate(spec2)
    result2 = verify_trace(trace2)

    print(f"Trace:\n{trace2.to_yaml_str()}")
    print(f"Valid: {result2.is_valid}")
    print(f"Answer: {trace2.answer} (expected: 17)")

    # Test 3: Three-way split with ratios
    print("\nTest 3: Split $120, A:B:C = 3:2:1")
    # Total = 120, parts = 3+2+1 = 6
    # A = 120 * 3/6 = 60, B = 120 * 2/6 = 40, C = 120 * 1/6 = 20
    # This requires extended constraint support - for now, pre-compute
    spec3 = ProblemSpec(
        problem_type=ProblemType.ALLOCATION,
        entities=[
            Entity(name="alice", initial_value=Decimal(60)),
            Entity(name="bob", initial_value=Decimal(40)),
            Entity(name="carol", initial_value=Decimal(20)),
        ],
        constraints=[],  # Pre-solved
        query=Query(target="alice"),
    )

    trace3 = generator.generate(spec3)
    result3 = verify_trace(trace3)

    print(f"Trace:\n{trace3.to_yaml_str()}")
    print(f"Valid: {result3.is_valid}")
    print(f"Answer: {trace3.answer} (expected: 60)")
