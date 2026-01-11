"""Tests for dataset Pydantic models."""

import pytest
from pydantic import ValidationError

from chuk_lazarus.introspection.datasets.models import (
    ArithmeticBenchmark,
    ArithmeticProblem,
    ContextTest,
    ContextTestDataset,
    PatternCategory,
    PatternDiscoveryDataset,
    UncertaintyDataset,
    UncertaintyPromptsSection,
)


class TestArithmeticProblem:
    """Tests for ArithmeticProblem model."""

    def test_create_valid_problem(self):
        """Test creating a valid problem."""
        problem = ArithmeticProblem(
            prompt="2 + 2 = ",
            answer=4,
            operation="addition",
        )
        assert problem.prompt == "2 + 2 = "
        assert problem.answer == 4
        assert problem.operation == "addition"

    def test_problem_is_frozen(self):
        """Test that problem is immutable."""
        problem = ArithmeticProblem(
            prompt="2 + 2 = ",
            answer=4,
            operation="addition",
        )
        with pytest.raises(ValidationError):
            problem.answer = 5


class TestArithmeticBenchmark:
    """Tests for ArithmeticBenchmark model."""

    @pytest.fixture
    def sample_benchmark(self):
        """Create a sample benchmark."""
        return ArithmeticBenchmark(
            version="1.0.0",
            description="Test benchmark",
            problems={
                "simple": [
                    ArithmeticProblem(prompt="2 + 2 = ", answer=4, operation="addition"),
                    ArithmeticProblem(prompt="3 * 3 = ", answer=9, operation="multiplication"),
                ],
                "hard": [
                    ArithmeticProblem(
                        prompt="127 * 89 = ", answer=11303, operation="multiplication"
                    ),
                ],
            },
        )

    def test_get_all_problems(self, sample_benchmark):
        """Test getting all problems flattened."""
        all_problems = sample_benchmark.get_all_problems()
        assert len(all_problems) == 3

    def test_get_by_difficulty(self, sample_benchmark):
        """Test getting problems by difficulty."""
        simple = sample_benchmark.get_by_difficulty("simple")
        assert len(simple) == 2

        hard = sample_benchmark.get_by_difficulty("hard")
        assert len(hard) == 1

        nonexistent = sample_benchmark.get_by_difficulty("extreme")
        assert len(nonexistent) == 0

    def test_get_prompts(self, sample_benchmark):
        """Test getting just prompt strings."""
        all_prompts = sample_benchmark.get_prompts()
        assert len(all_prompts) == 3
        assert "2 + 2 = " in all_prompts

        simple_prompts = sample_benchmark.get_prompts("simple")
        assert len(simple_prompts) == 2


class TestUncertaintyDataset:
    """Tests for UncertaintyDataset model."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample uncertainty dataset."""
        return UncertaintyDataset(
            version="1.0.0",
            description="Test dataset",
            working_prompts=UncertaintyPromptsSection(
                description="Working",
                prompts=["100 - 37 = ", "50 + 25 = "],
            ),
            broken_prompts=UncertaintyPromptsSection(
                description="Broken",
                prompts=["100 - 37 =", "50 + 25 ="],
            ),
        )

    def test_working_property(self, sample_dataset):
        """Test working prompts property."""
        assert len(sample_dataset.working) == 2
        assert "100 - 37 = " in sample_dataset.working

    def test_broken_property(self, sample_dataset):
        """Test broken prompts property."""
        assert len(sample_dataset.broken) == 2
        assert "100 - 37 =" in sample_dataset.broken


class TestContextTestDataset:
    """Tests for ContextTestDataset model."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample context test dataset."""
        return ContextTestDataset(
            version="1.0.0",
            description="Test dataset",
            target_token="127",
            tests=[
                ContextTest(prompt="111 127", context_type="number"),
                ContextTest(prompt="abc 127", context_type="word"),
            ],
        )

    def test_get_by_context_type(self, sample_dataset):
        """Test filtering by context type."""
        numbers = sample_dataset.get_by_context_type("number")
        assert len(numbers) == 1
        assert numbers[0].prompt == "111 127"

    def test_get_prompts(self, sample_dataset):
        """Test getting just prompts."""
        prompts = sample_dataset.get_prompts()
        assert len(prompts) == 2
        assert "111 127" in prompts


class TestPatternDiscoveryDataset:
    """Tests for PatternDiscoveryDataset model."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample pattern discovery dataset."""
        return PatternDiscoveryDataset(
            version="1.0.0",
            description="Test dataset",
            categories={
                "numbers": PatternCategory(
                    description="Number patterns",
                    prompts=["1", "42", "127"],
                ),
                "words": PatternCategory(
                    description="Word patterns",
                    prompts=["hello", "world"],
                ),
            },
        )

    def test_get_category(self, sample_dataset):
        """Test getting a specific category."""
        numbers = sample_dataset.get_category("numbers")
        assert numbers is not None
        assert len(numbers.prompts) == 3

        nonexistent = sample_dataset.get_category("other")
        assert nonexistent is None

    def test_get_category_names(self, sample_dataset):
        """Test getting category names."""
        names = sample_dataset.get_category_names()
        assert "numbers" in names
        assert "words" in names

    def test_get_all_prompts(self, sample_dataset):
        """Test getting all prompts with categories."""
        all_prompts = sample_dataset.get_all_prompts()
        assert len(all_prompts) == 5
        assert ("numbers", "42") in all_prompts

    def test_get_prompts_for_category(self, sample_dataset):
        """Test getting prompts for a category."""
        prompts = sample_dataset.get_prompts_for_category("numbers")
        assert len(prompts) == 3
        assert "127" in prompts
