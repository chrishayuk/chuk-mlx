"""
CSP-CoT Evaluator.

Compares CSP-CoT against English CoT baseline on GSM-8K.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from .gsm8k_loader import GSM8KProblem
from ..schema.problem import ProblemSpec
from ..schema.verifier import VerificationStatus
from ..pipeline.executor import CSPCoTExecutor, ExecutionResult


@dataclass
class ProblemResult:
    """Result for a single problem."""

    problem: GSM8KProblem
    expected_answer: int

    # CSP-CoT results
    csp_cot_answer: Decimal | None = None
    csp_cot_correct: bool = False
    csp_cot_verified: bool = False
    csp_cot_time_ms: float = 0
    csp_cot_error: str | None = None

    # English CoT results (if run)
    english_cot_answer: int | None = None
    english_cot_correct: bool = False
    english_cot_time_ms: float = 0

    # Parsed spec (for debugging)
    spec: ProblemSpec | None = None


@dataclass
class EvaluationResult:
    """Aggregate evaluation results."""

    total: int = 0

    # CSP-CoT metrics
    csp_cot_correct: int = 0
    csp_cot_verified: int = 0
    csp_cot_parse_success: int = 0
    csp_cot_avg_time_ms: float = 0

    # English CoT metrics
    english_cot_correct: int = 0
    english_cot_avg_time_ms: float = 0

    # Per-problem results
    results: list[ProblemResult] = field(default_factory=list)

    @property
    def csp_cot_accuracy(self) -> float:
        return self.csp_cot_correct / self.total if self.total > 0 else 0

    @property
    def csp_cot_verified_rate(self) -> float:
        return self.csp_cot_verified / self.total if self.total > 0 else 0

    @property
    def english_cot_accuracy(self) -> float:
        return self.english_cot_correct / self.total if self.total > 0 else 0

    @property
    def improvement(self) -> float:
        """CSP-CoT improvement over English CoT (percentage points)."""
        return (self.csp_cot_accuracy - self.english_cot_accuracy) * 100

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"GSM-8K Evaluation Results (n={self.total})",
            "=" * 50,
            "",
            "CSP-CoT:",
            f"  Accuracy:    {self.csp_cot_correct}/{self.total} ({self.csp_cot_accuracy:.1%})",
            f"  Verified:    {self.csp_cot_verified}/{self.total} ({self.csp_cot_verified_rate:.1%})",
            f"  Parse rate:  {self.csp_cot_parse_success}/{self.total}",
            f"  Avg time:    {self.csp_cot_avg_time_ms:.1f}ms",
            "",
            "English CoT:",
            f"  Accuracy:    {self.english_cot_correct}/{self.total} ({self.english_cot_accuracy:.1%})",
            f"  Avg time:    {self.english_cot_avg_time_ms:.1f}ms",
            "",
            f"Improvement:   {self.improvement:+.1f} percentage points",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "total": self.total,
            "csp_cot": {
                "correct": self.csp_cot_correct,
                "accuracy": self.csp_cot_accuracy,
                "verified": self.csp_cot_verified,
                "verified_rate": self.csp_cot_verified_rate,
                "parse_success": self.csp_cot_parse_success,
                "avg_time_ms": self.csp_cot_avg_time_ms,
            },
            "english_cot": {
                "correct": self.english_cot_correct,
                "accuracy": self.english_cot_accuracy,
                "avg_time_ms": self.english_cot_avg_time_ms,
            },
            "improvement_pp": self.improvement,
        }


class CSPCoTEvaluator:
    """
    Evaluates CSP-CoT on GSM-8K benchmark.

    Compares against English CoT baseline and reports:
    - Accuracy (correct final answer)
    - Verifiability (valid trace that proves answer)
    - Error detection (invalid traces caught)
    """

    def __init__(
        self,
        model=None,
        tokenizer=None,
        english_cot_model=None,
        english_cot_tokenizer=None,
    ):
        """
        Initialize evaluator.

        Args:
            model: Model for CSP-CoT parsing (optional)
            tokenizer: Tokenizer for CSP-CoT parsing (optional)
            english_cot_model: Model for English CoT baseline (optional)
            english_cot_tokenizer: Tokenizer for English CoT baseline (optional)
        """
        self.executor = CSPCoTExecutor(model=model, tokenizer=tokenizer)
        self.english_cot_model = english_cot_model
        self.english_cot_tokenizer = english_cot_tokenizer

    def evaluate(
        self,
        problems: list[GSM8KProblem],
        specs: list[ProblemSpec] | None = None,
        run_english_cot: bool = False,
        verbose: bool = True,
    ) -> EvaluationResult:
        """
        Evaluate on a set of GSM-8K problems.

        Args:
            problems: List of GSM-8K problems
            specs: Optional pre-parsed specs (skips LLM parsing)
            run_english_cot: Whether to run English CoT baseline
            verbose: Print progress

        Returns:
            EvaluationResult with metrics and per-problem results
        """
        result = EvaluationResult(total=len(problems))
        total_csp_time = 0
        total_english_time = 0

        for i, problem in enumerate(problems):
            if verbose:
                print(f"\n[{i+1}/{len(problems)}] {problem.question[:50]}...")

            pr = ProblemResult(
                problem=problem,
                expected_answer=problem.answer,
            )

            # Run CSP-CoT
            start = time.perf_counter()

            if specs and i < len(specs):
                # Use pre-parsed spec
                exec_result = self.executor.execute_from_spec(specs[i])
                pr.spec = specs[i]
            else:
                # Parse with LLM
                exec_result = self.executor.execute(problem.question)
                pr.spec = exec_result.spec

            pr.csp_cot_time_ms = (time.perf_counter() - start) * 1000
            total_csp_time += pr.csp_cot_time_ms

            if exec_result.success:
                result.csp_cot_parse_success += 1
                pr.csp_cot_answer = exec_result.answer

                # Check correctness
                if exec_result.answer is not None:
                    # Compare with tolerance for floating point
                    expected = Decimal(problem.answer)
                    actual = exec_result.answer
                    pr.csp_cot_correct = abs(actual - expected) < Decimal("0.01")

                    if pr.csp_cot_correct:
                        result.csp_cot_correct += 1

                # Check verification
                if exec_result.verification:
                    pr.csp_cot_verified = exec_result.verification.status == VerificationStatus.VALID
                    if pr.csp_cot_verified:
                        result.csp_cot_verified += 1
            else:
                pr.csp_cot_error = exec_result.error

            # Run English CoT baseline
            if run_english_cot and self.english_cot_model:
                start = time.perf_counter()
                english_answer = self._run_english_cot(problem.question)
                pr.english_cot_time_ms = (time.perf_counter() - start) * 1000
                total_english_time += pr.english_cot_time_ms

                pr.english_cot_answer = english_answer
                if english_answer is not None:
                    pr.english_cot_correct = english_answer == problem.answer
                    if pr.english_cot_correct:
                        result.english_cot_correct += 1

            result.results.append(pr)

            if verbose:
                status = "CORRECT" if pr.csp_cot_correct else "WRONG"
                verified = "VERIFIED" if pr.csp_cot_verified else "unverified"
                print(f"  CSP-CoT: {pr.csp_cot_answer} ({status}, {verified})")
                print(f"  Expected: {problem.answer}")

        # Compute averages
        result.csp_cot_avg_time_ms = total_csp_time / len(problems) if problems else 0
        result.english_cot_avg_time_ms = total_english_time / len(problems) if problems else 0

        return result

    def _run_english_cot(self, question: str) -> int | None:
        """
        Run English Chain-of-Thought baseline.

        Uses standard "Let's think step by step" prompting.
        """
        if self.english_cot_model is None:
            return None

        prompt = f"""Solve this math problem step by step. End with "The answer is [number]."

Question: {question}

Let's think step by step:"""

        try:
            import mlx.core as mx

            tokens = self.english_cot_tokenizer.encode(prompt)
            input_ids = mx.array([tokens])

            generated = []
            for _ in range(500):
                output = self.english_cot_model(input_ids)
                logits = output.logits if hasattr(output, "logits") else output
                next_token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
                generated.append(next_token)
                input_ids = mx.concatenate(
                    [input_ids, mx.array([[next_token]])], axis=1
                )

                decoded = self.english_cot_tokenizer.decode(generated)
                if "The answer is" in decoded or len(generated) > 400:
                    break

            response = self.english_cot_tokenizer.decode(generated)

            # Extract answer
            import re
            match = re.search(r"(?:answer is|=)\s*\$?(-?\d+)", response, re.IGNORECASE)
            if match:
                return int(match.group(1))

        except Exception as e:
            print(f"English CoT error: {e}")

        return None


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    from .gsm8k_loader import get_sample_problems
    from ..schema.problem import Entity, Operation, Query, OperationType, ProblemType
    from decimal import Decimal

    print("CSP-CoT Evaluator Tests")
    print("=" * 60)

    # Create evaluator without model (use pre-parsed specs)
    evaluator = CSPCoTEvaluator()

    # Get sample problems
    problems = get_sample_problems(3)

    # Create corresponding specs manually
    specs = [
        # Janet's ducks: 16 - 3 - 4 = 9, then 9 * 2 = 18
        ProblemSpec(
            problem_type=ProblemType.ARITHMETIC_CHAIN,
            entities=[Entity(name="eggs", initial_value=Decimal(16))],
            operations=[
                Operation(type=OperationType.SUBTRACT, target="eggs", amount=Decimal(3)),
                Operation(type=OperationType.SUBTRACT, target="eggs", amount=Decimal(4)),
                Operation(type=OperationType.MULTIPLY, target="eggs", factor=Decimal(2)),
            ],
            query=Query(target="eggs"),
            raw_text=problems[0].question,
        ),
        # Robe: 2 + 1 = 3
        ProblemSpec(
            problem_type=ProblemType.ARITHMETIC_CHAIN,
            entities=[Entity(name="bolts", initial_value=Decimal(2))],
            operations=[
                Operation(type=OperationType.ADD, target="bolts", amount=Decimal(1)),
            ],
            query=Query(target="bolts"),
            raw_text=problems[1].question,
        ),
        # Josh profit: 80000*1.5 = 120000 increase, 80000+120000 = 200000 new value
        # 200000 - 130000 = 70000 profit
        ProblemSpec(
            problem_type=ProblemType.ARITHMETIC_CHAIN,
            entities=[Entity(name="profit", initial_value=Decimal(200000))],
            operations=[
                Operation(type=OperationType.SUBTRACT, target="profit", amount=Decimal(130000)),
            ],
            query=Query(target="profit"),
            raw_text=problems[2].question,
        ),
    ]

    # Run evaluation
    result = evaluator.evaluate(problems, specs=specs, verbose=True)

    print("\n" + result.summary())
