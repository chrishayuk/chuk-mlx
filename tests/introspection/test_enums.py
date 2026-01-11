"""Tests for introspection enums module."""

import pytest

from chuk_lazarus.introspection.enums import (
    ArithmeticOperator,
    CommutativityLevel,
    ComputeStrategy,
    ConfidenceLevel,
    CriterionType,
    Difficulty,
    DirectionMethod,
    FactType,
    FormatDiagnosis,
    InvocationMethod,
    MemorizationLevel,
    NeuronRole,
    OverrideMode,
    PatchEffect,
    Region,
    TestStatus,
)


class TestFactType:
    """Tests for FactType enum."""

    def test_values(self):
        assert FactType.MULTIPLICATION.value == "multiplication"
        assert FactType.ADDITION.value == "addition"
        assert FactType.CAPITALS.value == "capitals"
        assert FactType.ELEMENTS.value == "elements"
        assert FactType.CUSTOM.value == "custom"

    def test_string_enum(self):
        # Should be string-based
        assert isinstance(FactType.MULTIPLICATION, str)


class TestRegion:
    """Tests for Region enum."""

    def test_values(self):
        assert Region.EUROPE.value == "europe"
        assert Region.ASIA.value == "asia"
        assert Region.AMERICAS.value == "americas"
        assert Region.AFRICA.value == "africa"
        assert Region.OCEANIA.value == "oceania"
        assert Region.OTHER.value == "other"


class TestArithmeticOperator:
    """Tests for ArithmeticOperator enum."""

    def test_values(self):
        assert ArithmeticOperator.ADD.value == "+"
        assert ArithmeticOperator.SUBTRACT.value == "-"
        assert ArithmeticOperator.MULTIPLY.value == "*"
        assert ArithmeticOperator.DIVIDE.value == "/"

    def test_from_string_basic(self):
        assert ArithmeticOperator.from_string("+") == ArithmeticOperator.ADD
        assert ArithmeticOperator.from_string("-") == ArithmeticOperator.SUBTRACT
        assert ArithmeticOperator.from_string("*") == ArithmeticOperator.MULTIPLY
        assert ArithmeticOperator.from_string("/") == ArithmeticOperator.DIVIDE

    def test_from_string_aliases(self):
        # Multiplication aliases
        assert ArithmeticOperator.from_string("x") == ArithmeticOperator.MULTIPLY
        assert ArithmeticOperator.from_string("×") == ArithmeticOperator.MULTIPLY

        # Division alias
        assert ArithmeticOperator.from_string("÷") == ArithmeticOperator.DIVIDE

    def test_from_string_invalid(self):
        with pytest.raises(ValueError, match="Unknown operator"):
            ArithmeticOperator.from_string("%")

    def test_compute_add(self):
        result = ArithmeticOperator.ADD.compute(5, 3)
        assert result == 8

    def test_compute_add_float(self):
        result = ArithmeticOperator.ADD.compute(5.5, 3.2)
        assert result == 8.7

    def test_compute_subtract(self):
        result = ArithmeticOperator.SUBTRACT.compute(10, 3)
        assert result == 7

    def test_compute_multiply(self):
        result = ArithmeticOperator.MULTIPLY.compute(7, 8)
        assert result == 56

    def test_compute_divide_int(self):
        # Integer division
        result = ArithmeticOperator.DIVIDE.compute(15, 3)
        assert result == 5
        assert isinstance(result, int)

    def test_compute_divide_float(self):
        # Float division
        result = ArithmeticOperator.DIVIDE.compute(7.0, 2.0)
        assert result == 3.5

    def test_compute_divide_by_zero(self):
        with pytest.raises(ValueError, match="Division by zero"):
            ArithmeticOperator.DIVIDE.compute(10, 0)

    def test_compute_mixed_types(self):
        # Int / float = float division
        result = ArithmeticOperator.DIVIDE.compute(7, 2.0)
        assert result == 3.5


class TestDifficulty:
    """Tests for Difficulty enum."""

    def test_values(self):
        assert Difficulty.EASY.value == "easy"
        assert Difficulty.MEDIUM.value == "medium"
        assert Difficulty.HARD.value == "hard"


class TestComputeStrategy:
    """Tests for ComputeStrategy enum."""

    def test_values(self):
        assert ComputeStrategy.DIRECT.value == "direct"
        assert ComputeStrategy.CHAIN_OF_THOUGHT.value == "cot"
        assert ComputeStrategy.UNKNOWN.value == "unknown"


class TestConfidenceLevel:
    """Tests for ConfidenceLevel enum."""

    def test_values(self):
        assert ConfidenceLevel.CONFIDENT.value == "confident"
        assert ConfidenceLevel.UNCERTAIN.value == "uncertain"
        assert ConfidenceLevel.UNKNOWN.value == "unknown"


class TestFormatDiagnosis:
    """Tests for FormatDiagnosis enum."""

    def test_values(self):
        assert FormatDiagnosis.SPACE_LOCK_ONLY.value == "space_lock_only"
        assert FormatDiagnosis.ONSET_ROUTING.value == "onset_routing"
        assert FormatDiagnosis.COMPUTE_BLOCKED.value == "compute_blocked"
        assert FormatDiagnosis.BOTH_FAIL.value == "both_fail"
        assert FormatDiagnosis.WEIRD.value == "weird"
        assert FormatDiagnosis.MINOR_DIFFERENCE.value == "minor_difference"


class TestInvocationMethod:
    """Tests for InvocationMethod enum."""

    def test_values(self):
        assert InvocationMethod.STEER.value == "steer"
        assert InvocationMethod.LINEAR.value == "linear"
        assert InvocationMethod.INTERPOLATE.value == "interpolate"
        assert InvocationMethod.EXTRAPOLATE.value == "extrapolate"


class TestDirectionMethod:
    """Tests for DirectionMethod enum."""

    def test_values(self):
        assert DirectionMethod.MEAN_DIFFERENCE.value == "difference"
        assert DirectionMethod.LOGISTIC.value == "logistic"
        assert DirectionMethod.PCA.value == "pca"
        assert DirectionMethod.RIDGE.value == "ridge"


class TestPatchEffect:
    """Tests for PatchEffect enum."""

    def test_values(self):
        assert PatchEffect.NO_CHANGE.value == "no_change"
        assert PatchEffect.TRANSFERRED.value == "transferred"
        assert PatchEffect.STILL_TARGET.value == "still_target"
        assert PatchEffect.CHANGED.value == "changed"


class TestCommutativityLevel:
    """Tests for CommutativityLevel enum."""

    def test_values(self):
        assert CommutativityLevel.PERFECT.value == "perfect"
        assert CommutativityLevel.HIGH.value == "high"
        assert CommutativityLevel.MODERATE.value == "moderate"
        assert CommutativityLevel.LOW.value == "low"


class TestTestStatus:
    """Tests for TestStatus enum."""

    def test_values(self):
        assert TestStatus.PASS.value == "pass"
        assert TestStatus.FAIL.value == "fail"
        assert TestStatus.IN_TRAINING.value == "in_training"
        assert TestStatus.NOVEL.value == "novel"


class TestMemorizationLevel:
    """Tests for MemorizationLevel enum."""

    def test_values(self):
        assert MemorizationLevel.MEMORIZED.value == "memorized"
        assert MemorizationLevel.PARTIAL.value == "partial"
        assert MemorizationLevel.WEAK.value == "weak"
        assert MemorizationLevel.NOT_MEMORIZED.value == "not_memorized"


class TestCriterionType:
    """Tests for CriterionType enum."""

    def test_values(self):
        assert CriterionType.FUNCTION_CALL.value == "function_call"
        assert CriterionType.SORRY.value == "sorry"
        assert CriterionType.POSITIVE.value == "positive"
        assert CriterionType.NEGATIVE.value == "negative"
        assert CriterionType.REFUSAL.value == "refusal"
        assert CriterionType.SUBSTRING.value == "substring"


class TestOverrideMode:
    """Tests for OverrideMode enum."""

    def test_values(self):
        assert OverrideMode.NONE.value == "none"
        assert OverrideMode.ARITHMETIC.value == "arithmetic"


class TestNeuronRole:
    """Tests for NeuronRole enum."""

    def test_values(self):
        assert NeuronRole.OPERAND_A.value == "operand_a"
        assert NeuronRole.OPERAND_B.value == "operand_b"
        assert NeuronRole.RESULT.value == "result"
        assert NeuronRole.OPERATOR.value == "operator"
        assert NeuronRole.POSITION.value == "position"
        assert NeuronRole.UNKNOWN.value == "unknown"


class TestEnumUsage:
    """Test practical enum usage patterns."""

    def test_string_comparison(self):
        # String enums should support string comparison
        assert FactType.MULTIPLICATION == "multiplication"
        assert Region.EUROPE == "europe"

    def test_enum_in_dict(self):
        # Enums should work as dict keys
        data = {
            FactType.MULTIPLICATION: "mult",
            FactType.ADDITION: "add",
        }
        assert data[FactType.MULTIPLICATION] == "mult"

    def test_enum_iteration(self):
        # Should be able to iterate over enum values
        operators = list(ArithmeticOperator)
        assert len(operators) == 4
        assert ArithmeticOperator.ADD in operators

    def test_compute_chain(self):
        # Test chaining operations
        result1 = ArithmeticOperator.ADD.compute(5, 3)  # 8
        result2 = ArithmeticOperator.MULTIPLY.compute(result1, 2)  # 16
        result3 = ArithmeticOperator.SUBTRACT.compute(result2, 6)  # 10
        assert result3 == 10
