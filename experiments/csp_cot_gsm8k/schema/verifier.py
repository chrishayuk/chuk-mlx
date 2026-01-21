"""
Trace Verifier.

Verifies that a trace is valid by replaying each step.
Provides detailed error reporting for invalid traces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from decimal import Decimal

from .trace import Trace, Step, State, Action, apply_action


class VerificationStatus(Enum):
    """Result of trace verification."""

    VALID = "valid"
    INVALID_STEP = "invalid_step"       # A step doesn't verify
    BROKEN_CHAIN = "broken_chain"       # States don't chain correctly
    WRONG_ANSWER = "wrong_answer"       # Final answer doesn't match query
    EMPTY_TRACE = "empty_trace"         # No steps
    MISSING_QUERY = "missing_query"     # No final query step


@dataclass
class StepError:
    """Details about a step verification failure."""

    step_index: int
    step: Step
    expected_state: State
    actual_state: State
    message: str

    def to_dict(self) -> dict:
        return {
            "step_index": self.step_index,
            "action": self.step.action.value,
            "expected": self.expected_state.to_dict(),
            "actual": self.actual_state.to_dict(),
            "message": self.message,
        }


@dataclass
class VerificationResult:
    """
    Complete result of trace verification.

    Includes:
    - Overall status
    - List of any errors found
    - The replayed final state
    - Whether the answer is correct
    """

    status: VerificationStatus
    errors: list[StepError] = field(default_factory=list)
    final_state: State | None = None
    computed_answer: Decimal | None = None
    expected_answer: Decimal | None = None

    @property
    def is_valid(self) -> bool:
        return self.status == VerificationStatus.VALID

    @property
    def answer_correct(self) -> bool:
        if self.computed_answer is None or self.expected_answer is None:
            return False
        return self.computed_answer == self.expected_answer

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "is_valid": self.is_valid,
            "errors": [e.to_dict() for e in self.errors],
            "final_state": self.final_state.to_dict() if self.final_state else None,
            "computed_answer": float(self.computed_answer) if self.computed_answer else None,
            "expected_answer": float(self.expected_answer) if self.expected_answer else None,
            "answer_correct": self.answer_correct,
        }

    def summary(self) -> str:
        """Human-readable summary."""
        if self.is_valid:
            return f"VALID: answer={self.computed_answer}"
        else:
            error_msgs = [e.message for e in self.errors[:3]]
            return f"INVALID ({self.status.value}): {'; '.join(error_msgs)}"


class TraceVerifier:
    """
    Verifies traces by replaying each step.

    The verifier:
    1. Starts with empty state
    2. Applies each action
    3. Checks result matches declared state_after
    4. Checks state continuity between steps
    5. Verifies final answer matches query result
    """

    def verify(self, trace: Trace) -> VerificationResult:
        """
        Verify a trace.

        Args:
            trace: The trace to verify

        Returns:
            VerificationResult with status and any errors
        """
        errors: list[StepError] = []

        # Check for empty trace
        if not trace.steps:
            return VerificationResult(
                status=VerificationStatus.EMPTY_TRACE,
                errors=[],
            )

        # Replay trace
        state = State()

        for i, step in enumerate(trace.steps):
            # Check state continuity (except first step)
            if i > 0:
                prev_step = trace.steps[i - 1]
                if state != step.state_before:
                    errors.append(
                        StepError(
                            step_index=i,
                            step=step,
                            expected_state=state,
                            actual_state=step.state_before,
                            message=f"Step {i}: state_before doesn't match previous state_after",
                        )
                    )

            # Check step verifies (action produces correct result)
            computed_after = apply_action(step.action, step.params, step.state_before)
            if computed_after != step.state_after:
                errors.append(
                    StepError(
                        step_index=i,
                        step=step,
                        expected_state=step.state_after,
                        actual_state=computed_after,
                        message=f"Step {i}: applying {step.action.value} produced wrong state",
                    )
                )

            # Advance state
            state = step.state_after

        # Determine status
        if errors:
            # Check what type of error
            chain_errors = [e for e in errors if "state_before" in e.message]
            step_errors = [e for e in errors if "applying" in e.message]

            if chain_errors and not step_errors:
                status = VerificationStatus.BROKEN_CHAIN
            else:
                status = VerificationStatus.INVALID_STEP

            return VerificationResult(
                status=status,
                errors=errors,
                final_state=state,
            )

        # Check final query
        final_step = trace.steps[-1]
        if final_step.action != Action.QUERY:
            return VerificationResult(
                status=VerificationStatus.MISSING_QUERY,
                errors=[
                    StepError(
                        step_index=len(trace.steps) - 1,
                        step=final_step,
                        expected_state=state,
                        actual_state=state,
                        message="Last step should be QUERY",
                    )
                ],
                final_state=state,
            )

        # Get computed answer
        query_entity = final_step.params.get("entity")
        computed_answer = state.get(query_entity) if query_entity else None

        # Check answer matches
        if trace.answer is not None and computed_answer != trace.answer:
            return VerificationResult(
                status=VerificationStatus.WRONG_ANSWER,
                errors=[
                    StepError(
                        step_index=len(trace.steps) - 1,
                        step=final_step,
                        expected_state=state,
                        actual_state=state,
                        message=f"Answer mismatch: trace says {trace.answer}, computed {computed_answer}",
                    )
                ],
                final_state=state,
                computed_answer=computed_answer,
                expected_answer=trace.answer,
            )

        # All checks passed
        return VerificationResult(
            status=VerificationStatus.VALID,
            errors=[],
            final_state=state,
            computed_answer=computed_answer,
            expected_answer=trace.answer,
        )

    def verify_batch(self, traces: list[Trace]) -> dict[str, Any]:
        """
        Verify a batch of traces and return aggregate statistics.

        Returns:
            Dict with counts and rates for each status
        """
        results = [self.verify(t) for t in traces]

        status_counts = {}
        for r in results:
            status_counts[r.status.value] = status_counts.get(r.status.value, 0) + 1

        valid_count = sum(1 for r in results if r.is_valid)
        correct_count = sum(1 for r in results if r.is_valid and r.answer_correct)

        return {
            "total": len(traces),
            "valid": valid_count,
            "valid_rate": valid_count / len(traces) if traces else 0,
            "correct": correct_count,
            "correct_rate": correct_count / len(traces) if traces else 0,
            "status_counts": status_counts,
            "results": results,
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def verify_trace(trace: Trace) -> VerificationResult:
    """Verify a single trace."""
    return TraceVerifier().verify(trace)


def verify_traces(traces: list[Trace]) -> dict[str, Any]:
    """Verify a batch of traces."""
    return TraceVerifier().verify_batch(traces)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    from .trace import TraceBuilder, State, Step, Action
    from decimal import Decimal

    print("Trace Verifier Tests")
    print("=" * 60)

    verifier = TraceVerifier()

    # Test 1: Valid trace
    print("\nTest 1: Valid trace (should pass)")
    valid_trace = (
        TraceBuilder(problem_type="entity_tracking")
        .init("jenny", 5)
        .init("bob", 0)
        .transfer("jenny", "bob", 2)
        .query("jenny")
        .build()
    )
    result = verifier.verify(valid_trace)
    print(f"  Status: {result.status.value}")
    print(f"  Valid: {result.is_valid}")
    print(f"  Answer: {result.computed_answer}")

    # Test 2: Invalid step (wrong state_after)
    print("\nTest 2: Invalid step (tampered state)")
    invalid_trace = Trace(
        steps=[
            Step(
                action=Action.INIT,
                params={"entity": "x", "value": 5},
                state_before=State(),
                state_after=State(values={"x": Decimal(999)}),  # WRONG!
            ),
            Step(
                action=Action.QUERY,
                params={"entity": "x", "result": 999},
                state_before=State(values={"x": Decimal(999)}),
                state_after=State(values={"x": Decimal(999)}),
            ),
        ],
        answer=Decimal(999),
        problem_type="test",
    )
    result = verifier.verify(invalid_trace)
    print(f"  Status: {result.status.value}")
    print(f"  Valid: {result.is_valid}")
    print(f"  Error: {result.errors[0].message if result.errors else 'none'}")

    # Test 3: Broken chain (state discontinuity)
    print("\nTest 3: Broken chain (state gap)")
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
                state_before=State(values={"x": Decimal(100)}),  # GAP! Should be 5
                state_after=State(values={"x": Decimal(103)}),
            ),
        ],
        answer=Decimal(103),
        problem_type="test",
    )
    result = verifier.verify(broken_trace)
    print(f"  Status: {result.status.value}")
    print(f"  Valid: {result.is_valid}")
    print(f"  Error: {result.errors[0].message if result.errors else 'none'}")

    # Test 4: Wrong answer
    print("\nTest 4: Wrong answer (mismatch)")
    wrong_answer_trace = (
        TraceBuilder(problem_type="test")
        .init("x", 5)
        .add("x", 3)
        .query("x")
        .build()
    )
    # Tamper with the answer
    wrong_answer_trace.answer = Decimal(999)

    result = verifier.verify(wrong_answer_trace)
    print(f"  Status: {result.status.value}")
    print(f"  Valid: {result.is_valid}")
    print(f"  Computed: {result.computed_answer}, Expected: {result.expected_answer}")

    # Test 5: Batch verification
    print("\nTest 5: Batch verification")
    traces = [
        TraceBuilder().init("a", 1).query("a").build(),
        TraceBuilder().init("b", 2).add("b", 3).query("b").build(),
        invalid_trace,
        broken_trace,
    ]
    batch_result = verifier.verify_batch(traces)
    print(f"  Total: {batch_result['total']}")
    print(f"  Valid: {batch_result['valid']} ({batch_result['valid_rate']:.0%})")
    print(f"  Status counts: {batch_result['status_counts']}")
