"""Tests for arithmetic Pydantic models."""

import pytest
from pydantic import ValidationError

from chuk_lazarus.introspection.enums import ArithmeticOperator, Difficulty
from chuk_lazarus.introspection.models.arithmetic import (
    ArithmeticStats,
    ArithmeticTestCase,
    ArithmeticTestResult,
    ArithmeticTestSuite,
    ParsedArithmeticPrompt,
)


class TestParsedArithmeticPrompt:
    """Tests for ParsedArithmeticPrompt model."""

    def test_instantiation_with_required_fields(self):
        """Test creating model with only required field."""
        prompt = ParsedArithmeticPrompt(prompt="2 + 3 = ")
        assert prompt.prompt == "2 + 3 = "
        assert prompt.operand_a is None
        assert prompt.operand_b is None
        assert prompt.operator is None
        assert prompt.result is None

    def test_instantiation_with_all_fields(self):
        """Test creating model with all fields."""
        prompt = ParsedArithmeticPrompt(
            prompt="2 + 3 = 5",
            operand_a=2,
            operand_b=3,
            operator=ArithmeticOperator.ADD,
            result=5,
        )
        assert prompt.prompt == "2 + 3 = 5"
        assert prompt.operand_a == 2
        assert prompt.operand_b == 3
        assert prompt.operator == ArithmeticOperator.ADD
        assert prompt.result == 5

    def test_is_arithmetic_property_true(self):
        """Test is_arithmetic property returns True for valid arithmetic."""
        prompt = ParsedArithmeticPrompt(
            prompt="2 + 3",
            operand_a=2,
            operand_b=3,
            operator=ArithmeticOperator.ADD,
        )
        assert prompt.is_arithmetic is True

    def test_is_arithmetic_property_false(self):
        """Test is_arithmetic property returns False for incomplete data."""
        prompt = ParsedArithmeticPrompt(prompt="Hello world")
        assert prompt.is_arithmetic is False

    def test_is_arithmetic_property_false_missing_operator(self):
        """Test is_arithmetic returns False when operator is missing."""
        prompt = ParsedArithmeticPrompt(
            prompt="2 3",
            operand_a=2,
            operand_b=3,
        )
        assert prompt.is_arithmetic is False

    def test_expected_result_for_addition(self):
        """Test expected_result computes addition correctly."""
        prompt = ParsedArithmeticPrompt(
            prompt="2 + 3",
            operand_a=2,
            operand_b=3,
            operator=ArithmeticOperator.ADD,
        )
        assert prompt.expected_result == 5

    def test_expected_result_for_multiplication(self):
        """Test expected_result computes multiplication correctly."""
        prompt = ParsedArithmeticPrompt(
            prompt="4 * 5",
            operand_a=4,
            operand_b=5,
            operator=ArithmeticOperator.MULTIPLY,
        )
        assert prompt.expected_result == 20

    def test_expected_result_for_subtraction(self):
        """Test expected_result computes subtraction correctly."""
        prompt = ParsedArithmeticPrompt(
            prompt="10 - 3",
            operand_a=10,
            operand_b=3,
            operator=ArithmeticOperator.SUBTRACT,
        )
        assert prompt.expected_result == 7

    def test_expected_result_for_division(self):
        """Test expected_result computes division correctly."""
        prompt = ParsedArithmeticPrompt(
            prompt="10 / 2",
            operand_a=10,
            operand_b=2,
            operator=ArithmeticOperator.DIVIDE,
        )
        assert prompt.expected_result == 5

    def test_expected_result_none_for_non_arithmetic(self):
        """Test expected_result returns None for non-arithmetic prompt."""
        prompt = ParsedArithmeticPrompt(prompt="What is the meaning of life?")
        assert prompt.expected_result is None

    def test_expected_result_handles_division_by_zero(self):
        """Test expected_result handles division by zero gracefully."""
        prompt = ParsedArithmeticPrompt(
            prompt="10 / 0",
            operand_a=10,
            operand_b=0,
            operator=ArithmeticOperator.DIVIDE,
        )
        # Should return None due to exception handling
        assert prompt.expected_result is None

    def test_parse_with_result(self):
        """Test parsing prompt with result included."""
        parsed = ParsedArithmeticPrompt.parse("2 + 3 = 5")
        assert parsed.prompt == "2 + 3 = 5"
        assert parsed.operand_a == 2
        assert parsed.operand_b == 3
        assert parsed.operator == ArithmeticOperator.ADD
        assert parsed.result == 5

    def test_parse_without_result(self):
        """Test parsing prompt without result."""
        parsed = ParsedArithmeticPrompt.parse("4 * 5 = ")
        assert parsed.prompt == "4 * 5 = "
        assert parsed.operand_a == 4
        assert parsed.operand_b == 5
        assert parsed.operator == ArithmeticOperator.MULTIPLY
        assert parsed.result is None

    def test_parse_with_explicit_result(self):
        """Test parsing prompt with explicit result parameter."""
        parsed = ParsedArithmeticPrompt.parse("4 * 5 = ", explicit_result=20)
        assert parsed.operand_a == 4
        assert parsed.operand_b == 5
        assert parsed.operator == ArithmeticOperator.MULTIPLY
        assert parsed.result == 20

    def test_parse_multiplication_with_x(self):
        """Test parsing multiplication with 'x' operator."""
        parsed = ParsedArithmeticPrompt.parse("3 x 4 = 12")
        assert parsed.operand_a == 3
        assert parsed.operand_b == 4
        assert parsed.operator == ArithmeticOperator.MULTIPLY
        assert parsed.result == 12

    def test_parse_multiplication_with_unicode(self):
        """Test parsing multiplication with unicode '×' operator."""
        parsed = ParsedArithmeticPrompt.parse("3 × 4 = 12")
        assert parsed.operand_a == 3
        assert parsed.operand_b == 4
        assert parsed.operator == ArithmeticOperator.MULTIPLY
        assert parsed.result == 12

    def test_parse_division_with_unicode(self):
        """Test parsing division with unicode '÷' operator."""
        parsed = ParsedArithmeticPrompt.parse("12 ÷ 3 = 4")
        assert parsed.operand_a == 12
        assert parsed.operand_b == 3
        assert parsed.operator == ArithmeticOperator.DIVIDE
        assert parsed.result == 4

    def test_parse_non_arithmetic(self):
        """Test parsing non-arithmetic prompt."""
        parsed = ParsedArithmeticPrompt.parse("Hello world")
        assert parsed.prompt == "Hello world"
        assert parsed.is_arithmetic is False
        assert parsed.expected_result is None

    def test_parse_with_whitespace_variations(self):
        """Test parsing handles various whitespace patterns."""
        parsed = ParsedArithmeticPrompt.parse("2+3=5")
        assert parsed.operand_a == 2
        assert parsed.operand_b == 3
        assert parsed.result == 5


class TestArithmeticTestCase:
    """Tests for ArithmeticTestCase model."""

    def test_instantiation_with_all_fields(self):
        """Test creating test case with all required fields."""
        test_case = ArithmeticTestCase(
            prompt="2 + 3 = ",
            expected="5",
            operator=ArithmeticOperator.ADD,
            difficulty=Difficulty.EASY,
            magnitude=1,
        )
        assert test_case.prompt == "2 + 3 = "
        assert test_case.expected == "5"
        assert test_case.operator == ArithmeticOperator.ADD
        assert test_case.difficulty == Difficulty.EASY
        assert test_case.magnitude == 1

    def test_missing_required_field_raises_error(self):
        """Test that missing required field raises ValidationError."""
        with pytest.raises(ValidationError):
            ArithmeticTestCase(
                prompt="2 + 3 = ",
                expected="5",
                # Missing operator, difficulty, magnitude
            )


class TestArithmeticTestResult:
    """Tests for ArithmeticTestResult model."""

    def test_instantiation_with_required_fields(self):
        """Test creating result with required fields."""
        result = ArithmeticTestResult(
            prompt="2 + 3 = ",
            expected="5",
            operator=ArithmeticOperator.ADD,
            difficulty=Difficulty.EASY,
            magnitude=1,
            final_prediction="5",
            correct=True,
        )
        assert result.prompt == "2 + 3 = "
        assert result.expected == "5"
        assert result.final_prediction == "5"
        assert result.correct is True
        assert result.emergence_layer is None
        assert result.peak_layer is None
        assert result.peak_probability == 0.0

    def test_instantiation_with_all_fields(self):
        """Test creating result with all fields."""
        result = ArithmeticTestResult(
            prompt="2 + 3 = ",
            expected="5",
            operator=ArithmeticOperator.ADD,
            difficulty=Difficulty.EASY,
            magnitude=1,
            final_prediction="5",
            correct=True,
            emergence_layer=3,
            peak_layer=5,
            peak_probability=0.95,
        )
        assert result.emergence_layer == 3
        assert result.peak_layer == 5
        assert result.peak_probability == 0.95

    def test_default_values(self):
        """Test default values for optional fields."""
        result = ArithmeticTestResult(
            prompt="2 + 3 = ",
            expected="5",
            operator=ArithmeticOperator.ADD,
            difficulty=Difficulty.EASY,
            magnitude=1,
            final_prediction="4",
            correct=False,
        )
        assert result.emergence_layer is None
        assert result.peak_layer is None
        assert result.peak_probability == 0.0


class TestArithmeticStats:
    """Tests for ArithmeticStats model."""

    def test_instantiation_with_defaults(self):
        """Test creating stats with default values."""
        stats = ArithmeticStats()
        assert stats.correct == 0
        assert stats.total == 0
        assert stats.emergence_layers == []

    def test_instantiation_with_values(self):
        """Test creating stats with specific values."""
        stats = ArithmeticStats(
            correct=8,
            total=10,
            emergence_layers=[2, 3, 3, 4, 2, 5, 3, 4],
        )
        assert stats.correct == 8
        assert stats.total == 10
        assert len(stats.emergence_layers) == 8

    def test_accuracy_property(self):
        """Test accuracy computation."""
        stats = ArithmeticStats(correct=8, total=10)
        assert stats.accuracy == 0.8

    def test_accuracy_property_zero_total(self):
        """Test accuracy returns 0 when total is 0."""
        stats = ArithmeticStats(correct=0, total=0)
        assert stats.accuracy == 0.0

    def test_avg_emergence_layer_property(self):
        """Test average emergence layer computation."""
        stats = ArithmeticStats(
            correct=4,
            total=4,
            emergence_layers=[2, 3, 4, 5],
        )
        assert stats.avg_emergence_layer == 3.5

    def test_avg_emergence_layer_none_when_empty(self):
        """Test avg_emergence_layer returns None when no data."""
        stats = ArithmeticStats(correct=0, total=0)
        assert stats.avg_emergence_layer is None


class TestArithmeticTestSuite:
    """Tests for ArithmeticTestSuite model."""

    def test_instantiation_with_defaults(self):
        """Test creating suite with default values."""
        suite = ArithmeticTestSuite()
        assert suite.model_id == ""
        assert suite.num_layers == 0
        assert suite.total_tests == 0
        assert suite.test_cases == []
        assert suite.results == []
        assert suite.stats_by_operation == {}
        assert suite.stats_by_difficulty == {}
        assert suite.stats_by_magnitude == {}

    def test_instantiation_with_values(self):
        """Test creating suite with specific values."""
        test_case = ArithmeticTestCase(
            prompt="2 + 3 = ",
            expected="5",
            operator=ArithmeticOperator.ADD,
            difficulty=Difficulty.EASY,
            magnitude=1,
        )
        suite = ArithmeticTestSuite(
            model_id="test-model",
            num_layers=12,
            total_tests=1,
            test_cases=[test_case],
        )
        assert suite.model_id == "test-model"
        assert suite.num_layers == 12
        assert suite.total_tests == 1
        assert len(suite.test_cases) == 1

    def test_generate_test_cases_all_operations(self):
        """Test generating test cases for all operations."""
        suite = ArithmeticTestSuite.generate_test_cases()
        assert len(suite.test_cases) > 0
        assert suite.total_tests == len(suite.test_cases)

        # Verify we have different operations
        operators = {tc.operator for tc in suite.test_cases}
        assert ArithmeticOperator.ADD in operators
        assert ArithmeticOperator.MULTIPLY in operators

    def test_generate_test_cases_single_operation(self):
        """Test generating test cases for single operation."""
        suite = ArithmeticTestSuite.generate_test_cases(operations=["add"])
        operators = {tc.operator for tc in suite.test_cases}
        assert operators == {ArithmeticOperator.ADD}

    def test_generate_test_cases_easy_only(self):
        """Test generating only easy difficulty test cases."""
        suite = ArithmeticTestSuite.generate_test_cases(difficulty=Difficulty.EASY)
        difficulties = {tc.difficulty for tc in suite.test_cases}
        assert difficulties == {Difficulty.EASY}

    def test_generate_test_cases_medium_only(self):
        """Test generating only medium difficulty test cases."""
        suite = ArithmeticTestSuite.generate_test_cases(difficulty=Difficulty.MEDIUM)
        difficulties = {tc.difficulty for tc in suite.test_cases}
        assert difficulties == {Difficulty.MEDIUM}

    def test_generate_test_cases_hard_only(self):
        """Test generating only hard difficulty test cases."""
        suite = ArithmeticTestSuite.generate_test_cases(difficulty=Difficulty.HARD)
        difficulties = {tc.difficulty for tc in suite.test_cases}
        assert difficulties == {Difficulty.HARD}

    def test_generate_test_cases_quick_mode(self):
        """Test quick mode reduces number of tests."""
        suite_full = ArithmeticTestSuite.generate_test_cases()
        suite_quick = ArithmeticTestSuite.generate_test_cases(quick=True)
        assert len(suite_quick.test_cases) < len(suite_full.test_cases)
        # Quick mode takes every 3rd test (using slicing [::3])
        # For n items, [::3] returns (n + 2) // 3 items
        expected_quick = (len(suite_full.test_cases) + 2) // 3
        assert len(suite_quick.test_cases) == expected_quick

    def test_generate_test_cases_multiple_operations(self):
        """Test generating test cases for multiple specific operations."""
        suite = ArithmeticTestSuite.generate_test_cases(operations=["add", "mul"])
        operators = {tc.operator for tc in suite.test_cases}
        assert ArithmeticOperator.ADD in operators
        assert ArithmeticOperator.MULTIPLY in operators
        assert ArithmeticOperator.SUBTRACT not in operators
        assert ArithmeticOperator.DIVIDE not in operators

    def test_generate_test_cases_all_difficulties(self):
        """Test generating test cases includes all difficulties when None."""
        suite = ArithmeticTestSuite.generate_test_cases(difficulty=None)
        difficulties = {tc.difficulty for tc in suite.test_cases}
        assert Difficulty.EASY in difficulties
        assert Difficulty.MEDIUM in difficulties
        assert Difficulty.HARD in difficulties

    def test_generate_test_cases_magnitude_values(self):
        """Test generated test cases have correct magnitude values."""
        suite = ArithmeticTestSuite.generate_test_cases()
        easy_cases = [tc for tc in suite.test_cases if tc.difficulty == Difficulty.EASY]
        medium_cases = [tc for tc in suite.test_cases if tc.difficulty == Difficulty.MEDIUM]
        hard_cases = [tc for tc in suite.test_cases if tc.difficulty == Difficulty.HARD]

        # Easy cases should be 1-digit
        for tc in easy_cases:
            assert tc.magnitude == 1

        # Medium cases should be 2-digit
        for tc in medium_cases:
            assert tc.magnitude == 2

        # Hard cases should be 3-digit
        for tc in hard_cases:
            assert tc.magnitude == 3

    def test_generate_test_cases_expected_answers(self):
        """Test generated test cases have correct expected answers."""
        suite = ArithmeticTestSuite.generate_test_cases(
            operations=["add"], difficulty=Difficulty.EASY
        )
        # Check a few specific cases
        case_1_plus_1 = next((tc for tc in suite.test_cases if tc.prompt == "1 + 1 = "), None)
        if case_1_plus_1:
            assert case_1_plus_1.expected == "2"

        case_2_plus_3 = next((tc for tc in suite.test_cases if tc.prompt == "2 + 3 = "), None)
        if case_2_plus_3:
            assert case_2_plus_3.expected == "5"
