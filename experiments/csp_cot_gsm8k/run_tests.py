#!/usr/bin/env python3
"""
CSP-CoT Experiment Test Runner.

Validates the full pipeline without requiring an LLM.
"""

from __future__ import annotations

import sys
from pathlib import Path
from decimal import Decimal

# Add experiment to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.csp_cot_gsm8k.schema.trace import TraceBuilder, Trace, Action
from experiments.csp_cot_gsm8k.schema.problem import (
    ProblemSpec,
    ProblemType,
    Entity,
    Operation,
    OperationType,
    Query,
    Constraint,
)
from experiments.csp_cot_gsm8k.schema.verifier import TraceVerifier, verify_trace
from experiments.csp_cot_gsm8k.generators import generate_trace
from experiments.csp_cot_gsm8k.pipeline.executor import CSPCoTExecutor


def test_trace_schema():
    """Test the trace schema and builder."""
    print("\n" + "=" * 60)
    print("TEST: Trace Schema")
    print("=" * 60)

    # Build a trace
    trace = (
        TraceBuilder(problem_type="entity_tracking")
        .init("jenny", 5)
        .init("bob", 0)
        .transfer("jenny", "bob", 2)
        .query("jenny")
        .build()
    )

    print(f"\nTrace for 'Jenny has 5, gives 2 to Bob':")
    print(trace.to_yaml_str())

    # Verify
    result = verify_trace(trace)
    print(f"\nVerification: {result.status.value}")
    print(f"Answer: {trace.answer}")

    assert trace.is_valid(), "Trace should be valid"
    assert trace.answer == Decimal(3), f"Answer should be 3, got {trace.answer}"
    print("PASSED")


def test_generators():
    """Test trace generators for different problem types."""
    print("\n" + "=" * 60)
    print("TEST: Trace Generators")
    print("=" * 60)

    test_cases = [
        # (spec, expected_answer, description)
        (
            ProblemSpec(
                problem_type=ProblemType.ENTITY_TRACKING,
                entities=[
                    Entity(name="sam", initial_value=Decimal(10)),
                ],
                operations=[
                    Operation(type=OperationType.SUBTRACT, target="sam", amount=Decimal(3)),
                    Operation(type=OperationType.ADD, target="sam", amount=Decimal(5)),
                ],
                query=Query(target="sam"),
            ),
            Decimal(12),
            "Sam: 10 - 3 + 5 = 12",
        ),
        (
            ProblemSpec(
                problem_type=ProblemType.ARITHMETIC_CHAIN,
                entities=[
                    Entity(name="total", initial_value=Decimal(6)),
                ],
                operations=[
                    Operation(type=OperationType.MULTIPLY, target="total", factor=Decimal(4)),
                ],
                query=Query(target="total"),
            ),
            Decimal(24),
            "6 * 4 = 24",
        ),
        (
            ProblemSpec(
                problem_type=ProblemType.COMPARISON,
                entities=[
                    Entity(name="tom", initial_value=Decimal(15)),
                    Entity(name="jane", initial_value=Decimal(5)),
                ],
                operations=[],
                query=Query(target="difference", question="compare", compare_a="tom", compare_b="jane"),
            ),
            Decimal(10),
            "Tom 15 - Jane 5 = 10 more",
        ),
        (
            ProblemSpec(
                problem_type=ProblemType.PERCENTAGE,
                entities=[
                    Entity(name="price", initial_value=Decimal(40)),
                ],
                operations=[
                    Operation(type=OperationType.MULTIPLY, target="price", factor=Decimal("0.75")),
                ],
                query=Query(target="price"),
            ),
            Decimal(30),
            "$40 with 25% off = $30",
        ),
    ]

    all_passed = True
    for spec, expected, desc in test_cases:
        trace = generate_trace(spec)
        if trace is None:
            print(f"  FAIL: {desc} - No generator")
            all_passed = False
            continue

        result = verify_trace(trace)
        passed = result.is_valid and trace.answer == expected

        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {desc}")
        print(f"        Answer: {trace.answer} (expected {expected})")
        print(f"        Verified: {result.is_valid}")

        if not passed:
            all_passed = False

    assert all_passed, "Some generator tests failed"
    print("\nAll generator tests PASSED")


def test_verifier():
    """Test trace verification including invalid traces."""
    print("\n" + "=" * 60)
    print("TEST: Trace Verifier")
    print("=" * 60)

    from experiments.csp_cot_gsm8k.schema.trace import Step, State

    # Test 1: Valid trace
    print("\n  Valid trace:")
    valid_trace = TraceBuilder().init("x", 5).add("x", 3).query("x").build()
    result = verify_trace(valid_trace)
    print(f"    Status: {result.status.value}")
    assert result.is_valid, "Valid trace should verify"

    # Test 2: Invalid step (tampered state_after)
    print("\n  Tampered trace (wrong state_after):")
    tampered_trace = Trace(
        steps=[
            Step(
                action=Action.INIT,
                params={"entity": "x", "value": 5},
                state_before=State(),
                state_after=State(values={"x": Decimal(999)}),  # WRONG
            ),
        ],
        answer=Decimal(999),
    )
    result = verify_trace(tampered_trace)
    print(f"    Status: {result.status.value}")
    assert not result.is_valid, "Tampered trace should fail"

    # Test 3: Broken chain (state discontinuity)
    print("\n  Broken chain (state gap):")
    broken_trace = Trace(
        steps=[
            Step(
                action=Action.INIT,
                params={"entity": "x", "value": 5},
                state_before=State(),
                state_after=State(values={"x": Decimal(5)}),
            ),
            Step(
                action=Action.ADD,
                params={"entity": "x", "amount": 3},
                state_before=State(values={"x": Decimal(100)}),  # GAP
                state_after=State(values={"x": Decimal(103)}),
            ),
        ],
        answer=Decimal(103),
    )
    result = verify_trace(broken_trace)
    print(f"    Status: {result.status.value}")
    assert not result.is_valid, "Broken chain should fail"

    print("\nAll verifier tests PASSED")


def test_executor():
    """Test the full executor pipeline."""
    print("\n" + "=" * 60)
    print("TEST: CSP-CoT Executor")
    print("=" * 60)

    executor = CSPCoTExecutor()

    test_specs = [
        ProblemSpec(
            problem_type=ProblemType.ENTITY_TRACKING,
            entities=[
                Entity(name="jenny", initial_value=Decimal(5)),
                Entity(name="bob", initial_value=Decimal(0)),
            ],
            operations=[
                Operation(type=OperationType.TRANSFER, source="jenny", target="bob", amount=Decimal(2)),
            ],
            query=Query(target="jenny"),
            raw_text="Jenny has 5 apples. She gives 2 to Bob.",
        ),
        ProblemSpec(
            problem_type=ProblemType.ARITHMETIC_CHAIN,
            entities=[Entity(name="total", initial_value=Decimal(8))],
            operations=[Operation(type=OperationType.SUBTRACT, target="total", amount=Decimal(6))],
            query=Query(target="total"),
            raw_text="8 slices, 6 eaten. How many left?",
        ),
    ]

    batch_result = executor.execute_batch(test_specs)

    print(f"\n  Total: {batch_result['total']}")
    print(f"  Success: {batch_result['success']}")
    print(f"  Verified: {batch_result['verified']}")

    for i, result in enumerate(batch_result["results"], 1):
        print(f"\n  Problem {i}: {result.problem[:40]}...")
        print(f"    {result.summary()}")

    assert batch_result["success"] == batch_result["total"], "All should succeed"
    assert batch_result["verified"] == batch_result["total"], "All should verify"

    print("\nExecutor tests PASSED")


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("\n" + "=" * 60)
    print("TEST: Edge Cases")
    print("=" * 60)

    # Test 1: Division with decimal result
    print("\n  Division with decimal:")
    trace = TraceBuilder().init("x", 10).divide("x", 3).query("x").build()
    result = verify_trace(trace)
    print(f"    10 / 3 = {trace.answer}")
    print(f"    Verified: {result.is_valid}")
    assert result.is_valid

    # Test 2: Zero values
    print("\n  Zero handling:")
    trace = TraceBuilder().init("x", 0).add("x", 5).query("x").build()
    result = verify_trace(trace)
    print(f"    0 + 5 = {trace.answer}")
    assert result.is_valid and trace.answer == Decimal(5)

    # Test 3: Negative result
    print("\n  Negative result:")
    trace = TraceBuilder().init("x", 5).subtract("x", 10).query("x").build()
    result = verify_trace(trace)
    print(f"    5 - 10 = {trace.answer}")
    assert result.is_valid and trace.answer == Decimal(-5)

    # Test 4: Large numbers
    print("\n  Large numbers:")
    trace = TraceBuilder().init("x", 1000000).multiply("x", 1000).query("x").build()
    result = verify_trace(trace)
    print(f"    1,000,000 * 1,000 = {trace.answer}")
    assert result.is_valid and trace.answer == Decimal(1000000000)

    print("\nEdge case tests PASSED")


def main():
    """Run all tests."""
    print("=" * 60)
    print("CSP-CoT GSM-8K Experiment - Test Suite")
    print("=" * 60)

    try:
        test_trace_schema()
        test_generators()
        test_verifier()
        test_executor()
        test_edge_cases()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)
        print("""
The CSP-CoT experiment is ready. Key components:

1. schema/trace.py     - Verifiable trace format
2. schema/problem.py   - Structured problem spec (LLM extracts this)
3. schema/verifier.py  - Replay verification
4. generators/         - Type-specific trace generators
5. pipeline/parser.py  - Few-shot LLM extraction
6. pipeline/executor.py - End-to-end pipeline

Next steps:
1. Load a model and test LLM-based parsing
2. Run on GSM-8K samples
3. Compare against English CoT baseline
""")
        return 0

    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
