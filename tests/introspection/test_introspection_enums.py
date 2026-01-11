"""Tests for introspection enums."""

import pytest

from chuk_lazarus.introspection.enums import (
    ArithmeticOperator,
    ComputeStrategy,
    ConfidenceLevel,
    Difficulty,
    FactType,
    FormatDiagnosis,
    Region,
)


class TestFactType:
    """Tests for FactType enum."""

    def test_multiplication(self):
        """Test MULTIPLICATION type."""
        assert FactType.MULTIPLICATION.value == "multiplication"

    def test_addition(self):
        """Test ADDITION type."""
        assert FactType.ADDITION.value == "addition"

    def test_capitals(self):
        """Test CAPITALS type."""
        assert FactType.CAPITALS.value == "capitals"

    def test_elements(self):
        """Test ELEMENTS type."""
        assert FactType.ELEMENTS.value == "elements"

    def test_custom(self):
        """Test CUSTOM type."""
        assert FactType.CUSTOM.value == "custom"


class TestRegion:
    """Tests for Region enum."""

    def test_all_regions(self):
        """Test all regions."""
        assert Region.EUROPE.value == "europe"
        assert Region.ASIA.value == "asia"
        assert Region.AMERICAS.value == "americas"
        assert Region.AFRICA.value == "africa"
        assert Region.OCEANIA.value == "oceania"
        assert Region.OTHER.value == "other"


class TestArithmeticOperator:
    """Tests for ArithmeticOperator enum."""

    def test_operators(self):
        """Test operator values."""
        assert ArithmeticOperator.ADD.value == "+"
        assert ArithmeticOperator.SUBTRACT.value == "-"
        assert ArithmeticOperator.MULTIPLY.value == "*"
        assert ArithmeticOperator.DIVIDE.value == "/"

    def test_from_string_basic(self):
        """Test parsing operators from strings."""
        assert ArithmeticOperator.from_string("+") == ArithmeticOperator.ADD
        assert ArithmeticOperator.from_string("-") == ArithmeticOperator.SUBTRACT
        assert ArithmeticOperator.from_string("*") == ArithmeticOperator.MULTIPLY
        assert ArithmeticOperator.from_string("/") == ArithmeticOperator.DIVIDE

    def test_from_string_aliases(self):
        """Test parsing operator aliases."""
        assert ArithmeticOperator.from_string("x") == ArithmeticOperator.MULTIPLY
        assert ArithmeticOperator.from_string("×") == ArithmeticOperator.MULTIPLY
        assert ArithmeticOperator.from_string("÷") == ArithmeticOperator.DIVIDE

    def test_from_string_unknown(self):
        """Test parsing unknown operator raises error."""
        with pytest.raises(ValueError, match="Unknown operator"):
            ArithmeticOperator.from_string("^")

    def test_compute_add(self):
        """Test addition computation."""
        assert ArithmeticOperator.ADD.compute(2, 3) == 5
        assert ArithmeticOperator.ADD.compute(2.5, 3.5) == 6.0

    def test_compute_subtract(self):
        """Test subtraction computation."""
        assert ArithmeticOperator.SUBTRACT.compute(5, 3) == 2
        assert ArithmeticOperator.SUBTRACT.compute(5.5, 2.5) == 3.0

    def test_compute_multiply(self):
        """Test multiplication computation."""
        assert ArithmeticOperator.MULTIPLY.compute(4, 3) == 12
        assert ArithmeticOperator.MULTIPLY.compute(2.5, 4) == 10.0

    def test_compute_divide(self):
        """Test division computation."""
        assert ArithmeticOperator.DIVIDE.compute(10, 2) == 5
        assert ArithmeticOperator.DIVIDE.compute(10.0, 4.0) == 2.5

    def test_compute_divide_by_zero(self):
        """Test division by zero raises error."""
        with pytest.raises(ValueError, match="Division by zero"):
            ArithmeticOperator.DIVIDE.compute(10, 0)


class TestDifficulty:
    """Tests for Difficulty enum."""

    def test_all_levels(self):
        """Test all difficulty levels."""
        assert Difficulty.EASY.value == "easy"
        assert Difficulty.MEDIUM.value == "medium"
        assert Difficulty.HARD.value == "hard"


class TestComputeStrategy:
    """Tests for ComputeStrategy enum."""

    def test_strategies(self):
        """Test all strategies."""
        assert ComputeStrategy.DIRECT.value == "direct"
        assert ComputeStrategy.CHAIN_OF_THOUGHT.value == "cot"
        assert ComputeStrategy.UNKNOWN.value == "unknown"


class TestConfidenceLevel:
    """Tests for ConfidenceLevel enum."""

    def test_levels(self):
        """Test all confidence levels."""
        assert ConfidenceLevel.CONFIDENT.value == "confident"
        assert ConfidenceLevel.UNCERTAIN.value == "uncertain"
        assert ConfidenceLevel.UNKNOWN.value == "unknown"


class TestFormatDiagnosis:
    """Tests for FormatDiagnosis enum."""

    def test_diagnoses(self):
        """Test all format diagnoses."""
        assert FormatDiagnosis.SPACE_LOCK_ONLY.value == "space_lock_only"
        assert FormatDiagnosis.ONSET_ROUTING.value == "onset_routing"
        assert FormatDiagnosis.COMPUTE_BLOCKED.value == "compute_blocked"
        assert FormatDiagnosis.BOTH_FAIL.value == "both_fail"
        assert FormatDiagnosis.WEIRD.value == "weird"
