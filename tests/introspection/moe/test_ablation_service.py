"""Tests for ablation_service.py to improve coverage."""

from chuk_lazarus.introspection.moe.ablation_service import (
    AblationBenchmarkResult,
    AblationBenchmarkService,
    BenchmarkProblemResult,
)


class TestBenchmarkProblemResult:
    """Tests for BenchmarkProblemResult class."""

    def test_status_broken(self):
        """Test status returns BROKEN when normal correct but ablated wrong."""
        result = BenchmarkProblemResult(
            prompt="2 + 2 = ?",
            expected_answer=4,
            normal_output="4",
            ablated_output="5",
            normal_correct=True,
            ablated_correct=False,
        )
        assert result.status == "BROKEN"

    def test_status_fixed(self):
        """Test status returns FIXED when normal wrong but ablated correct (lines 32-33)."""
        result = BenchmarkProblemResult(
            prompt="2 + 2 = ?",
            expected_answer=4,
            normal_output="5",
            ablated_output="4",
            normal_correct=False,
            ablated_correct=True,
        )
        assert result.status == "FIXED"

    def test_status_empty_both_correct(self):
        """Test status returns empty when both are correct."""
        result = BenchmarkProblemResult(
            prompt="2 + 2 = ?",
            expected_answer=4,
            normal_output="4",
            ablated_output="4",
            normal_correct=True,
            ablated_correct=True,
        )
        assert result.status == ""

    def test_status_empty_both_wrong(self):
        """Test status returns empty when both are wrong."""
        result = BenchmarkProblemResult(
            prompt="2 + 2 = ?",
            expected_answer=4,
            normal_output="5",
            ablated_output="6",
            normal_correct=False,
            ablated_correct=False,
        )
        assert result.status == ""


class TestAblationBenchmarkResult:
    """Tests for AblationBenchmarkResult class."""

    def test_empty_problems_list(self):
        """Test computed fields with empty problems list (lines 63-65, 71-73)."""
        result = AblationBenchmarkResult(
            expert_indices=[0, 1],
            problems=[],
        )
        assert result.normal_correct_count == 0
        assert result.ablated_correct_count == 0
        assert result.normal_accuracy == 0.0
        assert result.ablated_accuracy == 0.0
        assert result.accuracy_diff == 0
        assert result.broken_count == 0
        assert result.fixed_count == 0

    def test_normal_correct_count(self):
        """Test normal_correct_count computed field."""
        problems = [
            BenchmarkProblemResult(
                prompt="1",
                expected_answer=1,
                normal_output="1",
                ablated_output="2",
                normal_correct=True,
                ablated_correct=False,
            ),
            BenchmarkProblemResult(
                prompt="2",
                expected_answer=2,
                normal_output="3",
                ablated_output="2",
                normal_correct=False,
                ablated_correct=True,
            ),
            BenchmarkProblemResult(
                prompt="3",
                expected_answer=3,
                normal_output="3",
                ablated_output="3",
                normal_correct=True,
                ablated_correct=True,
            ),
        ]
        result = AblationBenchmarkResult(expert_indices=[0], problems=problems)
        assert result.normal_correct_count == 2  # First and third

    def test_ablated_correct_count(self):
        """Test ablated_correct_count computed field."""
        problems = [
            BenchmarkProblemResult(
                prompt="1",
                expected_answer=1,
                normal_output="1",
                ablated_output="2",
                normal_correct=True,
                ablated_correct=False,
            ),
            BenchmarkProblemResult(
                prompt="2",
                expected_answer=2,
                normal_output="3",
                ablated_output="2",
                normal_correct=False,
                ablated_correct=True,
            ),
            BenchmarkProblemResult(
                prompt="3",
                expected_answer=3,
                normal_output="3",
                ablated_output="3",
                normal_correct=True,
                ablated_correct=True,
            ),
        ]
        result = AblationBenchmarkResult(expert_indices=[0], problems=problems)
        assert result.ablated_correct_count == 2  # Second and third

    def test_normal_accuracy(self):
        """Test normal_accuracy computed field."""
        problems = [
            BenchmarkProblemResult(
                prompt="1",
                expected_answer=1,
                normal_output="1",
                ablated_output="2",
                normal_correct=True,
                ablated_correct=False,
            ),
            BenchmarkProblemResult(
                prompt="2",
                expected_answer=2,
                normal_output="3",
                ablated_output="2",
                normal_correct=False,
                ablated_correct=True,
            ),
        ]
        result = AblationBenchmarkResult(expert_indices=[0], problems=problems)
        assert result.normal_accuracy == 0.5  # 1/2

    def test_ablated_accuracy(self):
        """Test ablated_accuracy computed field."""
        problems = [
            BenchmarkProblemResult(
                prompt="1",
                expected_answer=1,
                normal_output="1",
                ablated_output="1",
                normal_correct=True,
                ablated_correct=True,
            ),
            BenchmarkProblemResult(
                prompt="2",
                expected_answer=2,
                normal_output="3",
                ablated_output="2",
                normal_correct=False,
                ablated_correct=True,
            ),
            BenchmarkProblemResult(
                prompt="3",
                expected_answer=3,
                normal_output="4",
                ablated_output="2",
                normal_correct=False,
                ablated_correct=False,
            ),
        ]
        result = AblationBenchmarkResult(expert_indices=[0], problems=problems)
        assert abs(result.ablated_accuracy - 2 / 3) < 0.001  # 2/3

    def test_accuracy_diff_negative(self):
        """Test accuracy_diff when ablation hurts performance (line 79)."""
        problems = [
            BenchmarkProblemResult(
                prompt="1",
                expected_answer=1,
                normal_output="1",
                ablated_output="2",
                normal_correct=True,
                ablated_correct=False,
            ),
            BenchmarkProblemResult(
                prompt="2",
                expected_answer=2,
                normal_output="2",
                ablated_output="3",
                normal_correct=True,
                ablated_correct=False,
            ),
        ]
        result = AblationBenchmarkResult(expert_indices=[0], problems=problems)
        assert result.accuracy_diff == -2  # 0 ablated - 2 normal = -2

    def test_accuracy_diff_positive(self):
        """Test accuracy_diff when ablation helps performance."""
        problems = [
            BenchmarkProblemResult(
                prompt="1",
                expected_answer=1,
                normal_output="2",
                ablated_output="1",
                normal_correct=False,
                ablated_correct=True,
            ),
            BenchmarkProblemResult(
                prompt="2",
                expected_answer=2,
                normal_output="3",
                ablated_output="2",
                normal_correct=False,
                ablated_correct=True,
            ),
        ]
        result = AblationBenchmarkResult(expert_indices=[0], problems=problems)
        assert result.accuracy_diff == 2  # 2 ablated - 0 normal = 2

    def test_broken_count(self):
        """Test broken_count computed field (line 85)."""
        problems = [
            BenchmarkProblemResult(
                prompt="1",
                expected_answer=1,
                normal_output="1",
                ablated_output="2",
                normal_correct=True,
                ablated_correct=False,
            ),
            BenchmarkProblemResult(
                prompt="2",
                expected_answer=2,
                normal_output="2",
                ablated_output="3",
                normal_correct=True,
                ablated_correct=False,
            ),
            BenchmarkProblemResult(
                prompt="3",
                expected_answer=3,
                normal_output="3",
                ablated_output="3",
                normal_correct=True,
                ablated_correct=True,
            ),
        ]
        result = AblationBenchmarkResult(expert_indices=[0], problems=problems)
        assert result.broken_count == 2  # First and second are BROKEN

    def test_fixed_count(self):
        """Test fixed_count computed field (line 91)."""
        problems = [
            BenchmarkProblemResult(
                prompt="1",
                expected_answer=1,
                normal_output="2",
                ablated_output="1",
                normal_correct=False,
                ablated_correct=True,
            ),
            BenchmarkProblemResult(
                prompt="2",
                expected_answer=2,
                normal_output="3",
                ablated_output="2",
                normal_correct=False,
                ablated_correct=True,
            ),
            BenchmarkProblemResult(
                prompt="3",
                expected_answer=3,
                normal_output="4",
                ablated_output="5",
                normal_correct=False,
                ablated_correct=False,
            ),
        ]
        result = AblationBenchmarkResult(expert_indices=[0], problems=problems)
        assert result.fixed_count == 2  # First and second are FIXED

    def test_broken_count_zero(self):
        """Test broken_count when count is 0 (line 85)."""
        problems = [
            BenchmarkProblemResult(
                prompt="1",
                expected_answer=1,
                normal_output="2",
                ablated_output="1",
                normal_correct=False,
                ablated_correct=True,
            ),
        ]
        result = AblationBenchmarkResult(expert_indices=[0], problems=problems)
        assert result.broken_count == 0

    def test_fixed_count_zero(self):
        """Test fixed_count when count is 0 (line 91)."""
        problems = [
            BenchmarkProblemResult(
                prompt="1",
                expected_answer=1,
                normal_output="1",
                ablated_output="2",
                normal_correct=True,
                ablated_correct=False,
            ),
        ]
        result = AblationBenchmarkResult(expert_indices=[0], problems=problems)
        assert result.fixed_count == 0


class TestAblationBenchmarkService:
    """Tests for AblationBenchmarkService class."""

    def test_check_answer_correct(self):
        """Test check_answer with correct answer."""
        assert AblationBenchmarkService.check_answer("The answer is 42", 42) is True

    def test_check_answer_incorrect(self):
        """Test check_answer with incorrect answer."""
        assert AblationBenchmarkService.check_answer("The answer is 42", 100) is False

    def test_check_answer_no_number(self):
        """Test check_answer when output has no number."""
        assert AblationBenchmarkService.check_answer("No number here", 42) is False

    def test_check_answer_negative_number(self):
        """Test check_answer with negative number."""
        assert AblationBenchmarkService.check_answer("-5 is the result", -5) is True

    def test_check_answer_first_number_wins(self):
        """Test check_answer uses first number found."""
        assert AblationBenchmarkService.check_answer("10 then 42", 10) is True
        assert AblationBenchmarkService.check_answer("10 then 42", 42) is False

    def test_check_answer_empty_string(self):
        """Test check_answer with empty string (lines 112-118)."""
        assert AblationBenchmarkService.check_answer("", 42) is False

    def test_check_answer_value_error_handling(self):
        """Test check_answer handles regex match edge cases (lines 114-117)."""
        # This tests when regex finds something but int() might fail
        # In practice, -?\d+ should always produce valid int, but we test the path
        assert AblationBenchmarkService.check_answer("abc", 0) is False

    def test_create_problem_result(self):
        """Test create_problem_result factory method."""
        result = AblationBenchmarkService.create_problem_result(
            prompt="2 + 2 = ?",
            expected_answer=4,
            normal_output="4",
            ablated_output="5",
        )
        assert isinstance(result, BenchmarkProblemResult)
        assert result.prompt == "2 + 2 = ?"
        assert result.expected_answer == 4
        assert result.normal_output == "4"
        assert result.ablated_output == "5"
        assert result.normal_correct is True
        assert result.ablated_correct is False

    def test_create_problem_result_fixed_status(self):
        """Test create_problem_result with FIXED status scenario (line 138)."""
        result = AblationBenchmarkService.create_problem_result(
            prompt="2 + 2 = ?",
            expected_answer=4,
            normal_output="5",  # Wrong
            ablated_output="4",  # Correct
        )
        assert result.normal_correct is False
        assert result.ablated_correct is True
        assert result.status == "FIXED"


class TestFormatSummary:
    """Tests for format_summary method."""

    def test_format_summary_negative_diff(self):
        """Test format_summary when ablation hurts (lines 170-174)."""
        problems = [
            BenchmarkProblemResult(
                prompt="1",
                expected_answer=1,
                normal_output="1",
                ablated_output="2",
                normal_correct=True,
                ablated_correct=False,
            ),
            BenchmarkProblemResult(
                prompt="2",
                expected_answer=2,
                normal_output="2",
                ablated_output="3",
                normal_correct=True,
                ablated_correct=False,
            ),
        ]
        result = AblationBenchmarkResult(expert_indices=[0], problems=problems)
        summary = AblationBenchmarkService.format_summary(result)

        assert "Normal accuracy:" in summary
        assert "2/2" in summary  # Both correct normally
        assert "Ablated accuracy:" in summary
        assert "0/2" in summary  # Neither correct with ablation
        assert "caused" in summary
        assert "additional failures" in summary

    def test_format_summary_positive_diff(self):
        """Test format_summary when ablation helps (lines 175-178)."""
        problems = [
            BenchmarkProblemResult(
                prompt="1",
                expected_answer=1,
                normal_output="2",
                ablated_output="1",
                normal_correct=False,
                ablated_correct=True,
            ),
            BenchmarkProblemResult(
                prompt="2",
                expected_answer=2,
                normal_output="3",
                ablated_output="2",
                normal_correct=False,
                ablated_correct=True,
            ),
        ]
        result = AblationBenchmarkResult(expert_indices=[0], problems=problems)
        summary = AblationBenchmarkService.format_summary(result)

        assert "Normal accuracy:" in summary
        assert "0/2" in summary  # Neither correct normally
        assert "Ablated accuracy:" in summary
        assert "2/2" in summary  # Both correct with ablation
        assert "improved" in summary

    def test_format_summary_zero_diff(self):
        """Test format_summary when no change (lines 179-180)."""
        problems = [
            BenchmarkProblemResult(
                prompt="1",
                expected_answer=1,
                normal_output="1",
                ablated_output="1",
                normal_correct=True,
                ablated_correct=True,
            ),
            BenchmarkProblemResult(
                prompt="2",
                expected_answer=2,
                normal_output="3",
                ablated_output="4",
                normal_correct=False,
                ablated_correct=False,
            ),
        ]
        result = AblationBenchmarkResult(expert_indices=[0], problems=problems)
        summary = AblationBenchmarkService.format_summary(result)

        assert "Normal accuracy:" in summary
        assert "1/2" in summary
        assert "Ablated accuracy:" in summary
        assert "1/2" in summary
        assert "No change in accuracy!" in summary

    def test_format_summary_multiple_experts(self):
        """Test format_summary with multiple experts."""
        problems = [
            BenchmarkProblemResult(
                prompt="1",
                expected_answer=1,
                normal_output="1",
                ablated_output="2",
                normal_correct=True,
                ablated_correct=False,
            ),
        ]
        result = AblationBenchmarkResult(expert_indices=[0, 1, 2], problems=problems)
        summary = AblationBenchmarkService.format_summary(result)

        assert "Removing 3 expert(s)" in summary

    def test_format_summary_single_expert(self):
        """Test format_summary with single expert."""
        problems = [
            BenchmarkProblemResult(
                prompt="1",
                expected_answer=1,
                normal_output="1",
                ablated_output="2",
                normal_correct=True,
                ablated_correct=False,
            ),
        ]
        result = AblationBenchmarkResult(expert_indices=[5], problems=problems)
        summary = AblationBenchmarkService.format_summary(result)

        assert "Removing 1 expert(s)" in summary

    def test_format_summary_accuracy_percentages(self):
        """Test format_summary shows correct percentages."""
        problems = [
            BenchmarkProblemResult(
                prompt="1",
                expected_answer=1,
                normal_output="1",
                ablated_output="1",
                normal_correct=True,
                ablated_correct=True,
            ),
            BenchmarkProblemResult(
                prompt="2",
                expected_answer=2,
                normal_output="2",
                ablated_output="3",
                normal_correct=True,
                ablated_correct=False,
            ),
            BenchmarkProblemResult(
                prompt="3",
                expected_answer=3,
                normal_output="4",
                ablated_output="5",
                normal_correct=False,
                ablated_correct=False,
            ),
            BenchmarkProblemResult(
                prompt="4",
                expected_answer=4,
                normal_output="5",
                ablated_output="4",
                normal_correct=False,
                ablated_correct=True,
            ),
        ]
        result = AblationBenchmarkResult(expert_indices=[0], problems=problems)
        summary = AblationBenchmarkService.format_summary(result)

        # 2/4 normal correct = 50%
        assert "2/4" in summary
        assert "50%" in summary

    def test_format_summary_empty_problems(self):
        """Test format_summary with empty problems."""
        result = AblationBenchmarkResult(expert_indices=[0], problems=[])
        summary = AblationBenchmarkService.format_summary(result)

        assert "0/0" in summary
        assert "0%" in summary
        assert "No change in accuracy!" in summary
