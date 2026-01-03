"""Enums for introspection framework.

Centralizes all enum types to eliminate magic strings throughout the codebase.
"""

from enum import Enum, auto


class FactType(str, Enum):
    """Types of fact datasets for memory analysis."""

    MULTIPLICATION = "multiplication"
    ADDITION = "addition"
    CAPITALS = "capitals"
    ELEMENTS = "elements"
    CUSTOM = "custom"


class Region(str, Enum):
    """Geographic regions for categorization."""

    EUROPE = "europe"
    ASIA = "asia"
    AMERICAS = "americas"
    AFRICA = "africa"
    OCEANIA = "oceania"
    OTHER = "other"


class ArithmeticOperator(str, Enum):
    """Arithmetic operation types."""

    ADD = "+"
    SUBTRACT = "-"
    MULTIPLY = "*"
    DIVIDE = "/"

    @classmethod
    def from_string(cls, s: str) -> "ArithmeticOperator":
        """Parse operator from string, handling aliases."""
        mapping = {
            "+": cls.ADD,
            "-": cls.SUBTRACT,
            "*": cls.MULTIPLY,
            "x": cls.MULTIPLY,
            "×": cls.MULTIPLY,
            "/": cls.DIVIDE,
            "÷": cls.DIVIDE,
        }
        if s in mapping:
            return mapping[s]
        raise ValueError(f"Unknown operator: {s}")

    def compute(self, a: int | float, b: int | float) -> int | float:
        """Compute the result of the operation."""
        if self == ArithmeticOperator.ADD:
            return a + b
        elif self == ArithmeticOperator.SUBTRACT:
            return a - b
        elif self == ArithmeticOperator.MULTIPLY:
            return a * b
        elif self == ArithmeticOperator.DIVIDE:
            if b == 0:
                raise ValueError("Division by zero")
            return a // b if isinstance(a, int) and isinstance(b, int) else a / b
        raise ValueError(f"Unknown operator: {self}")


class Difficulty(str, Enum):
    """Difficulty levels for test cases."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ComputeStrategy(str, Enum):
    """Model computation strategy detection."""

    DIRECT = "direct"  # Model outputs answer directly
    CHAIN_OF_THOUGHT = "cot"  # Model uses reasoning steps
    UNKNOWN = "unknown"


class ConfidenceLevel(str, Enum):
    """Model confidence classification."""

    CONFIDENT = "confident"
    UNCERTAIN = "uncertain"
    UNKNOWN = "unknown"


class FormatDiagnosis(str, Enum):
    """Diagnosis of format sensitivity effects."""

    SPACE_LOCK_ONLY = "space_lock_only"  # Just adds space, same answer timing
    ONSET_ROUTING = "onset_routing"  # Answer delayed due to mode switch
    COMPUTE_BLOCKED = "compute_blocked"  # Answer not produced without space
    BOTH_FAIL = "both_fail"  # Neither format works
    WEIRD = "weird"  # Unexpected: no-space works but with-space fails
    MINOR_DIFFERENCE = "minor_difference"  # Small onset difference


class InvocationMethod(str, Enum):
    """Methods for circuit invocation."""

    STEER = "steer"  # Use direction to steer model
    LINEAR = "linear"  # Weighted average by distance
    INTERPOLATE = "interpolate"  # K-nearest neighbors interpolation
    EXTRAPOLATE = "extrapolate"  # Linear regression extrapolation


class DirectionMethod(str, Enum):
    """Methods for extracting steering directions."""

    MEAN_DIFFERENCE = "difference"  # Difference of class means
    LOGISTIC = "logistic"  # Logistic regression weights
    PCA = "pca"  # Principal component analysis
    RIDGE = "ridge"  # Ridge regression (for continuous targets)


class PatchEffect(str, Enum):
    """Effect of activation patching."""

    NO_CHANGE = "no_change"
    TRANSFERRED = "transferred"  # Source answer produced
    STILL_TARGET = "still_target"  # Target answer still produced
    CHANGED = "changed"  # Changed to something else


class CommutativityLevel(str, Enum):
    """Commutativity analysis interpretation."""

    PERFECT = "perfect"  # >0.999 similarity
    HIGH = "high"  # >0.99 similarity
    MODERATE = "moderate"  # >0.9 similarity
    LOW = "low"  # <0.9 similarity


class TestStatus(str, Enum):
    """Status of a test case."""

    PASS = "pass"
    FAIL = "fail"
    IN_TRAINING = "in_training"
    NOVEL = "novel"


class MemorizationLevel(str, Enum):
    """Classification of fact memorization."""

    MEMORIZED = "memorized"  # Rank 1, prob > 10%
    PARTIAL = "partial"  # Rank 2-5, prob > 1%
    WEAK = "weak"  # Rank 6-15, prob > 0.1%
    NOT_MEMORIZED = "not_memorized"  # Rank > 15 or prob < 0.1%


class CriterionType(str, Enum):
    """Built-in criterion types for ablation studies."""

    FUNCTION_CALL = "function_call"
    SORRY = "sorry"
    POSITIVE = "positive"
    NEGATIVE = "negative"
    REFUSAL = "refusal"
    SUBSTRING = "substring"


class OverrideMode(str, Enum):
    """Compute override modes."""

    NONE = "none"
    ARITHMETIC = "arithmetic"


class NeuronRole(str, Enum):
    """Roles that neurons can play in computations."""

    OPERAND_A = "operand_a"
    OPERAND_B = "operand_b"
    RESULT = "result"
    OPERATOR = "operator"
    POSITION = "position"
    UNKNOWN = "unknown"
