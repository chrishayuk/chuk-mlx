"""
CSP-CoT Executor.

End-to-end execution: Problem → Parse → Generate Trace → Verify → Answer
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from decimal import Decimal

from ..schema.trace import Trace
from ..schema.problem import ProblemSpec
from ..schema.verifier import TraceVerifier, VerificationResult, VerificationStatus
from ..generators import generate_trace
from .parser import FewShotParser, ManualSpecBuilder


@dataclass
class ExecutionResult:
    """
    Result of executing the CSP-CoT pipeline on a problem.
    """

    # Input
    problem: str
    spec: ProblemSpec | None = None

    # Output
    trace: Trace | None = None
    answer: Decimal | None = None

    # Verification
    verification: VerificationResult | None = None

    # Status
    success: bool = False
    error: str | None = None

    @property
    def is_verified(self) -> bool:
        """Check if answer is verified correct."""
        return (
            self.verification is not None
            and self.verification.status == VerificationStatus.VALID
        )

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "problem": self.problem,
            "spec": self.spec.to_dict() if self.spec else None,
            "trace": self.trace.to_dict() if self.trace else None,
            "answer": float(self.answer) if self.answer is not None else None,
            "verification": self.verification.to_dict() if self.verification else None,
            "success": self.success,
            "is_verified": self.is_verified,
            "error": self.error,
        }

    def summary(self) -> str:
        """Human-readable summary."""
        if not self.success:
            return f"FAILED: {self.error}"
        if self.is_verified:
            return f"VERIFIED: answer={self.answer}"
        else:
            status = self.verification.status.value if self.verification else "unknown"
            return f"UNVERIFIED ({status}): answer={self.answer}"


class CSPCoTExecutor:
    """
    Executes the full CSP-CoT pipeline.

    Pipeline:
    1. Parse problem into ProblemSpec (LLM or manual)
    2. Route to appropriate trace generator
    3. Generate verifiable trace
    4. Verify trace by replay
    5. Return answer with verification status
    """

    def __init__(self, model=None, tokenizer=None):
        """
        Initialize executor.

        Args:
            model: Optional LLM model for parsing
            tokenizer: Optional tokenizer for parsing
        """
        self.parser = FewShotParser(model=model, tokenizer=tokenizer)
        self.verifier = TraceVerifier()

    def execute(self, problem: str) -> ExecutionResult:
        """
        Execute pipeline on a problem.

        Args:
            problem: Natural language math problem

        Returns:
            ExecutionResult with answer and verification
        """
        result = ExecutionResult(problem=problem)

        # Step 1: Parse problem
        spec = self.parser.parse(problem)
        if spec is None:
            result.error = "Failed to parse problem into structured spec"
            return result

        result.spec = spec

        # Step 2: Generate trace
        trace = generate_trace(spec)
        if trace is None:
            result.error = f"No generator for problem type: {spec.problem_type.value}"
            return result

        result.trace = trace
        result.answer = trace.answer

        # Step 3: Verify trace
        verification = self.verifier.verify(trace)
        result.verification = verification

        # Mark success
        result.success = True

        return result

    def execute_from_spec(self, spec: ProblemSpec) -> ExecutionResult:
        """
        Execute pipeline from a pre-parsed spec.

        Useful for testing without LLM parsing.
        """
        result = ExecutionResult(problem=spec.raw_text, spec=spec)

        # Generate trace
        trace = generate_trace(spec)
        if trace is None:
            result.error = f"No generator for problem type: {spec.problem_type.value}"
            return result

        result.trace = trace
        result.answer = trace.answer

        # Verify trace
        verification = self.verifier.verify(trace)
        result.verification = verification

        result.success = True

        return result

    def execute_batch(
        self, problems: list[str] | list[ProblemSpec]
    ) -> dict[str, Any]:
        """
        Execute pipeline on a batch of problems.

        Returns aggregate statistics.
        """
        results = []

        for item in problems:
            if isinstance(item, str):
                result = self.execute(item)
            else:
                result = self.execute_from_spec(item)
            results.append(result)

        # Compute statistics
        total = len(results)
        success = sum(1 for r in results if r.success)
        verified = sum(1 for r in results if r.is_verified)

        return {
            "total": total,
            "success": success,
            "success_rate": success / total if total > 0 else 0,
            "verified": verified,
            "verified_rate": verified / total if total > 0 else 0,
            "results": results,
        }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    from ..schema.problem import Entity, Operation, Query, Constraint, OperationType, ProblemType
    from decimal import Decimal

    print("CSP-CoT Executor Tests")
    print("=" * 60)

    executor = CSPCoTExecutor()

    # Test from pre-built specs (no LLM needed)
    test_specs = [
        # Entity tracking
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
            raw_text="Jenny has 5 apples. She gives 2 to Bob. How many does Jenny have?",
        ),
        # Arithmetic chain
        ProblemSpec(
            problem_type=ProblemType.ARITHMETIC_CHAIN,
            entities=[
                Entity(name="oranges", initial_value=Decimal(6)),
            ],
            operations=[
                Operation(type=OperationType.MULTIPLY, target="oranges", factor=Decimal(4)),
            ],
            query=Query(target="oranges"),
            raw_text="6 bags with 4 oranges each. How many oranges?",
        ),
        # Comparison
        ProblemSpec(
            problem_type=ProblemType.COMPARISON,
            entities=[
                Entity(name="tom", initial_value=Decimal(15)),
                Entity(name="jane", initial_value=Decimal(5)),
            ],
            operations=[],
            query=Query(target="difference", question="compare", compare_a="tom", compare_b="jane"),
            raw_text="Tom has 15, Jane has 5. How many more does Tom have?",
        ),
        # Allocation
        ProblemSpec(
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
            raw_text="Split $100, Alice gets 2x Bob. How much does Alice get?",
        ),
    ]

    print("\nRunning tests with pre-built specs (no LLM):")
    print("-" * 60)

    for i, spec in enumerate(test_specs, 1):
        result = executor.execute_from_spec(spec)
        print(f"\nTest {i}: {spec.problem_type.value}")
        print(f"  Problem: {spec.raw_text[:50]}...")
        print(f"  {result.summary()}")
        if result.trace:
            print(f"  Trace steps: {len(result.trace.steps)}")

    # Batch execution
    print("\n" + "=" * 60)
    print("Batch Execution:")
    batch_result = executor.execute_batch(test_specs)
    print(f"  Total: {batch_result['total']}")
    print(f"  Success: {batch_result['success']} ({batch_result['success_rate']:.0%})")
    print(f"  Verified: {batch_result['verified']} ({batch_result['verified_rate']:.0%})")
