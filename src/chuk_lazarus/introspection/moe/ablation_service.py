"""Service layer for ablation benchmarking.

Provides high-level benchmarking logic that uses the low-level ablation functions.
This separates CLI concerns from business logic.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, computed_field


class BenchmarkProblemResult(BaseModel):
    """Result of running a single benchmark problem."""

    model_config = ConfigDict(frozen=True)

    prompt: str = Field(..., description="The benchmark prompt")
    expected_answer: int = Field(..., description="Expected integer answer")
    normal_output: str = Field(..., description="Model output without ablation")
    ablated_output: str = Field(..., description="Model output with ablation")
    normal_correct: bool = Field(..., description="Whether normal output is correct")
    ablated_correct: bool = Field(..., description="Whether ablated output is correct")

    @computed_field
    @property
    def status(self) -> str:
        """Get status description."""
        if self.normal_correct and not self.ablated_correct:
            return "BROKEN"
        elif not self.normal_correct and self.ablated_correct:
            return "FIXED"
        return ""


class AblationBenchmarkResult(BaseModel):
    """Result of running ablation benchmark across problems."""

    model_config = ConfigDict(frozen=True)

    expert_indices: list[int] = Field(..., description="Expert indices being ablated")
    problems: list[BenchmarkProblemResult] = Field(
        default_factory=list, description="Individual problem results"
    )

    @computed_field
    @property
    def normal_correct_count(self) -> int:
        """Count of problems correct without ablation."""
        return sum(1 for p in self.problems if p.normal_correct)

    @computed_field
    @property
    def ablated_correct_count(self) -> int:
        """Count of problems correct with ablation."""
        return sum(1 for p in self.problems if p.ablated_correct)

    @computed_field
    @property
    def normal_accuracy(self) -> float:
        """Accuracy without ablation."""
        if not self.problems:
            return 0.0
        return self.normal_correct_count / len(self.problems)

    @computed_field
    @property
    def ablated_accuracy(self) -> float:
        """Accuracy with ablation."""
        if not self.problems:
            return 0.0
        return self.ablated_correct_count / len(self.problems)

    @computed_field
    @property
    def accuracy_diff(self) -> int:
        """Difference in correct answers (negative means ablation hurt)."""
        return self.ablated_correct_count - self.normal_correct_count

    @computed_field
    @property
    def broken_count(self) -> int:
        """Count of problems broken by ablation."""
        return sum(1 for p in self.problems if p.status == "BROKEN")

    @computed_field
    @property
    def fixed_count(self) -> int:
        """Count of problems fixed by ablation."""
        return sum(1 for p in self.problems if p.status == "FIXED")


class AblationBenchmarkService:
    """Service for running ablation benchmarks.

    This service encapsulates the business logic for benchmarking
    expert ablation effects on model performance.
    """

    @staticmethod
    def check_answer(output: str, expected: int) -> bool:
        """Check if output contains the expected answer.

        Args:
            output: Model output text.
            expected: Expected integer answer.

        Returns:
            True if the first number in output matches expected.
        """
        match = re.search(r"-?\d+", output)
        if match:
            try:
                return int(match.group()) == expected
            except ValueError:
                pass
        return False

    @staticmethod
    def create_problem_result(
        prompt: str,
        expected_answer: int,
        normal_output: str,
        ablated_output: str,
    ) -> BenchmarkProblemResult:
        """Create a benchmark problem result.

        Args:
            prompt: The benchmark prompt.
            expected_answer: The expected integer answer.
            normal_output: Model output without ablation.
            ablated_output: Model output with ablation.

        Returns:
            BenchmarkProblemResult with correctness computed.
        """
        return BenchmarkProblemResult(
            prompt=prompt,
            expected_answer=expected_answer,
            normal_output=normal_output,
            ablated_output=ablated_output,
            normal_correct=AblationBenchmarkService.check_answer(normal_output, expected_answer),
            ablated_correct=AblationBenchmarkService.check_answer(ablated_output, expected_answer),
        )

    @staticmethod
    def format_summary(result: AblationBenchmarkResult) -> str:
        """Format a summary of benchmark results.

        Args:
            result: The benchmark result to summarize.

        Returns:
            Multi-line summary string.
        """
        lines = []
        total = len(result.problems)

        lines.append(
            f"Normal accuracy:  {result.normal_correct_count}/{total} "
            f"({100 * result.normal_accuracy:.0f}%)"
        )
        lines.append(
            f"Ablated accuracy: {result.ablated_correct_count}/{total} "
            f"({100 * result.ablated_accuracy:.0f}%)"
        )

        num_experts = len(result.expert_indices)
        if result.accuracy_diff < 0:
            lines.append(
                f"\nRemoving {num_experts} expert(s) caused "
                f"{-result.accuracy_diff} additional failures"
            )
        elif result.accuracy_diff > 0:
            lines.append(
                f"\nRemoving {num_experts} expert(s) improved {result.accuracy_diff} cases!"
            )
        else:
            lines.append("\nNo change in accuracy!")

        return "\n".join(lines)
