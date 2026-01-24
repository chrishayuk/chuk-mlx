"""
Trace verifier - executes symbolic traces to compute answers.

Uses chuk-virtual-expert TraceVerifier with arithmetic experts for execution.
Provides backward-compatible verify_yaml_output() interface.
"""

from __future__ import annotations

from typing import Any

from chuk_virtual_expert import ExpertRegistry, TraceVerifier
from chuk_virtual_expert_arithmetic import (
    ArithmeticExpert,
    ComparisonExpert,
    EntityTrackExpert,
    PercentageExpert,
    RateEquationExpert,
)


def _build_registry() -> ExpertRegistry:
    """Build registry with all arithmetic experts."""
    registry = ExpertRegistry()
    registry.register(EntityTrackExpert())
    registry.register(ArithmeticExpert())
    registry.register(PercentageExpert())
    registry.register(RateEquationExpert())
    registry.register(ComparisonExpert())
    return registry


# Module-level singleton
_registry = _build_registry()
_verifier = TraceVerifier(_registry)


def verify_yaml_output(yaml_str: str, expected_answer: Any = None, tolerance: float = 0.01) -> dict:
    """
    Verify a complete YAML trace output by executing it.

    Backward-compatible wrapper around TraceVerifier.

    Returns dict with:
        - parsed: bool
        - expert: str or None
        - trace_valid: bool
        - trace_error: str or None
        - computed_answer: the answer computed by executing trace
        - answer_correct: bool (if expected_answer provided)
        - final_state: dict
    """
    result = _verifier.verify(yaml_str, expected_answer, tolerance=tolerance)

    return {
        "parsed": result.parsed,
        "expert": result.expert,
        "trace_valid": result.trace_valid,
        "trace_error": result.trace_error,
        "computed_answer": result.computed_answer,
        "answer_correct": result.answer_correct,
        "final_state": result.final_state,
    }


def execute_trace(trace: list[dict], tolerance: float = 0.01) -> tuple[bool, str, dict, Any]:
    """
    Execute symbolic trace and compute all values.

    Backward-compatible wrapper. Dispatches to the appropriate expert
    based on trace content.

    Returns:
        (valid, error_message, final_state, computed_answer)
    """
    import yaml as _yaml

    # Wrap in YAML format to use the verifier
    # Detect expert from trace operations
    expert_name = _detect_expert(trace)
    yaml_str = _yaml.dump({"expert": expert_name, "trace": trace})

    result = _verifier.execute_yaml(yaml_str)

    error_msg = result.error if not result.success else "valid"
    return result.success, error_msg, result.state, result.answer


def _detect_expert(trace: list[dict]) -> str:
    """Detect which expert should handle a trace based on its operations."""
    ops = set()
    for step in trace:
        for key in step:
            ops.add(key)

    if "transfer" in ops or "consume" in ops or "add" in ops:
        return "entity_track"
    if "percent_off" in ops or "percent_increase" in ops or "percent_of" in ops:
        return "percentage"
    if "compare" in ops:
        return "comparison"
    if "formula" in ops or "given" in ops:
        return "rate_equation"
    return "arithmetic"


# Quick test
if __name__ == "__main__":
    # Test symbolic entity tracking trace (no results)
    print("Test 1: Entity tracking (symbolic)")
    test_trace = [
        {"init": "eggs", "value": 16},
        {"consume": {"entity": "eggs", "amount": 3}},
        {"consume": {"entity": "eggs", "amount": 4}},
        {"compute": {"op": "mul", "args": ["eggs", 2], "var": "revenue"}},
        {"query": "revenue"},
    ]

    valid, error, state, answer = execute_trace(test_trace)
    print(f"Valid: {valid}")
    print(f"State: {state}")
    print(f"Computed answer: {answer}")
    print(f"Expected: 18, Got: {answer}, Correct: {abs(answer - 18) < 0.01}")

    # Test symbolic arithmetic
    print("\nTest 2: Arithmetic (symbolic)")
    test_trace2 = [
        {"init": "price", "value": 100},
        {"init": "tax", "value": 8.5},
        {"init": "shipping", "value": 5},
        {"compute": {"op": "add", "args": ["price", "tax"], "var": "with_tax"}},
        {"compute": {"op": "add", "args": ["with_tax", "shipping"], "var": "total"}},
        {"query": "total"},
    ]

    valid, error, state, answer = execute_trace(test_trace2)
    print(f"Valid: {valid}")
    print(f"Computed answer: {answer}")
    print(f"Expected: 113.5, Got: {answer}")

    # Test percent off
    print("\nTest 3: Percent off (symbolic)")
    test_trace3 = [
        {"init": "price", "value": 100},
        {"init": "discount_rate", "value": 20},
        {"percent_off": {"base": "price", "rate": "discount_rate", "var": "sale_price"}},
        {"query": "sale_price"},
    ]

    valid, error, state, answer = execute_trace(test_trace3)
    print(f"Valid: {valid}")
    print(f"Computed answer: {answer}")
    print(f"Expected: 80, Got: {answer}")

    # Test verify_yaml_output
    print("\nTest 4: verify_yaml_output")
    yaml_str = """
expert: entity_track
trace:
  - {init: x, value: 10}
  - {query: x}
"""
    result = verify_yaml_output(yaml_str, expected_answer=10)
    print(f"Result: {result}")
    assert result["answer_correct"], f"Expected correct, got {result}"
    print("OK")
