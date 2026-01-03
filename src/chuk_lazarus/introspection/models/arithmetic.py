"""Pydantic models for arithmetic analysis."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..enums import ArithmeticOperator, Difficulty


class ParsedArithmeticPrompt(BaseModel):
    """A parsed arithmetic prompt with extracted components."""

    prompt: str = Field(description="Original prompt string")
    operand_a: int | None = Field(default=None, description="First operand")
    operand_b: int | None = Field(default=None, description="Second operand")
    operator: ArithmeticOperator | None = Field(default=None, description="Operator")
    result: int | None = Field(default=None, description="Expected result if provided")

    @property
    def is_arithmetic(self) -> bool:
        """Check if this is a valid arithmetic prompt."""
        return self.operand_a is not None and self.operand_b is not None and self.operator is not None

    @property
    def expected_result(self) -> int | None:
        """Compute expected result from operands and operator."""
        if not self.is_arithmetic:
            return None
        try:
            return int(self.operator.compute(self.operand_a, self.operand_b))
        except (ValueError, TypeError):
            return None

    @classmethod
    def parse(cls, prompt: str, explicit_result: int | None = None) -> "ParsedArithmeticPrompt":
        """Parse a prompt string into structured components."""
        import re

        # Try pattern with result: "A op B = C"
        pattern_with_result = re.compile(r"(\d+)\s*([+\-*/x×÷])\s*(\d+)\s*=\s*(\d+)")
        match = pattern_with_result.search(prompt)
        if match:
            a, op, b, result = match.groups()
            return cls(
                prompt=prompt,
                operand_a=int(a),
                operand_b=int(b),
                operator=ArithmeticOperator.from_string(op),
                result=int(result),
            )

        # Try pattern without result: "A op B ="
        pattern_no_result = re.compile(r"(\d+)\s*([+\-*/x×÷])\s*(\d+)\s*=")
        match = pattern_no_result.search(prompt)
        if match:
            a, op, b = match.groups()
            return cls(
                prompt=prompt,
                operand_a=int(a),
                operand_b=int(b),
                operator=ArithmeticOperator.from_string(op),
                result=explicit_result,
            )

        # Non-arithmetic prompt
        return cls(prompt=prompt, result=explicit_result)


class ArithmeticTestCase(BaseModel):
    """A single arithmetic test case."""

    prompt: str = Field(description="The prompt to test")
    expected: str = Field(description="Expected answer string")
    operator: ArithmeticOperator = Field(description="Operation type")
    difficulty: Difficulty = Field(description="Difficulty level")
    magnitude: int = Field(description="Number of digits in operands")


class ArithmeticTestResult(BaseModel):
    """Result of a single arithmetic test."""

    prompt: str = Field(description="The prompt tested")
    expected: str = Field(description="Expected answer")
    operator: ArithmeticOperator = Field(description="Operation type")
    difficulty: Difficulty = Field(description="Difficulty level")
    magnitude: int = Field(description="Operand digit count")
    final_prediction: str = Field(description="Model's final prediction")
    correct: bool = Field(description="Whether prediction was correct")
    emergence_layer: int | None = Field(default=None, description="Layer where answer first emerges")
    peak_layer: int | None = Field(default=None, description="Layer with highest answer probability")
    peak_probability: float = Field(default=0.0, description="Highest probability achieved")


class ArithmeticStats(BaseModel):
    """Aggregated statistics for arithmetic tests."""

    correct: int = Field(default=0, description="Number of correct answers")
    total: int = Field(default=0, description="Total number of tests")
    emergence_layers: list[int] = Field(default_factory=list, description="Emergence layers for correct answers")

    @property
    def accuracy(self) -> float:
        """Compute accuracy as a fraction."""
        return self.correct / self.total if self.total > 0 else 0.0

    @property
    def avg_emergence_layer(self) -> float | None:
        """Compute average emergence layer."""
        if not self.emergence_layers:
            return None
        return sum(self.emergence_layers) / len(self.emergence_layers)


class ArithmeticTestSuite(BaseModel):
    """A complete arithmetic test suite with results."""

    model_id: str = Field(default="", description="Model identifier")
    num_layers: int = Field(default=0, description="Number of model layers")
    total_tests: int = Field(default=0, description="Total number of tests run")
    test_cases: list[ArithmeticTestCase] = Field(default_factory=list)
    results: list[ArithmeticTestResult] = Field(default_factory=list)
    stats_by_operation: dict[str, ArithmeticStats] = Field(default_factory=dict)
    stats_by_difficulty: dict[str, ArithmeticStats] = Field(default_factory=dict)
    stats_by_magnitude: dict[int, ArithmeticStats] = Field(default_factory=dict)

    @classmethod
    def generate_test_cases(
        cls,
        operations: list[str] | None = None,
        difficulty: Difficulty | None = None,
        quick: bool = False,
    ) -> "ArithmeticTestSuite":
        """Generate standard arithmetic test cases.

        Args:
            operations: List of operations to include (add, mul, sub, div). None = all.
            difficulty: Filter to specific difficulty. None = all.
            quick: If True, take every 3rd test.

        Returns:
            ArithmeticTestSuite with test_cases populated.
        """
        if operations is None:
            operations = ["add", "mul", "sub", "div"]

        include_easy = difficulty is None or difficulty == Difficulty.EASY
        include_medium = difficulty is None or difficulty == Difficulty.MEDIUM
        include_hard = difficulty is None or difficulty == Difficulty.HARD

        tests: list[ArithmeticTestCase] = []

        if include_easy:
            # Easy addition (1-digit)
            if "add" in operations:
                tests.extend([
                    ArithmeticTestCase(prompt="1 + 1 = ", expected="2", operator=ArithmeticOperator.ADD, difficulty=Difficulty.EASY, magnitude=1),
                    ArithmeticTestCase(prompt="2 + 3 = ", expected="5", operator=ArithmeticOperator.ADD, difficulty=Difficulty.EASY, magnitude=1),
                    ArithmeticTestCase(prompt="4 + 5 = ", expected="9", operator=ArithmeticOperator.ADD, difficulty=Difficulty.EASY, magnitude=1),
                    ArithmeticTestCase(prompt="7 + 2 = ", expected="9", operator=ArithmeticOperator.ADD, difficulty=Difficulty.EASY, magnitude=1),
                ])
            # Easy multiplication
            if "mul" in operations:
                tests.extend([
                    ArithmeticTestCase(prompt="2 * 3 = ", expected="6", operator=ArithmeticOperator.MULTIPLY, difficulty=Difficulty.EASY, magnitude=1),
                    ArithmeticTestCase(prompt="4 * 5 = ", expected="20", operator=ArithmeticOperator.MULTIPLY, difficulty=Difficulty.EASY, magnitude=1),
                    ArithmeticTestCase(prompt="7 * 8 = ", expected="56", operator=ArithmeticOperator.MULTIPLY, difficulty=Difficulty.EASY, magnitude=1),
                ])
            # Easy subtraction and division
            if "sub" in operations:
                tests.append(
                    ArithmeticTestCase(prompt="10 - 3 = ", expected="7", operator=ArithmeticOperator.SUBTRACT, difficulty=Difficulty.EASY, magnitude=1),
                )
            if "div" in operations:
                tests.append(
                    ArithmeticTestCase(prompt="10 / 2 = ", expected="5", operator=ArithmeticOperator.DIVIDE, difficulty=Difficulty.EASY, magnitude=1),
                )

        if include_medium:
            # Medium addition (2-digit)
            if "add" in operations:
                tests.extend([
                    ArithmeticTestCase(prompt="12 + 34 = ", expected="46", operator=ArithmeticOperator.ADD, difficulty=Difficulty.MEDIUM, magnitude=2),
                    ArithmeticTestCase(prompt="25 + 17 = ", expected="42", operator=ArithmeticOperator.ADD, difficulty=Difficulty.MEDIUM, magnitude=2),
                    ArithmeticTestCase(prompt="99 + 11 = ", expected="110", operator=ArithmeticOperator.ADD, difficulty=Difficulty.MEDIUM, magnitude=2),
                ])
            # Medium multiplication
            if "mul" in operations:
                tests.extend([
                    ArithmeticTestCase(prompt="12 * 12 = ", expected="144", operator=ArithmeticOperator.MULTIPLY, difficulty=Difficulty.MEDIUM, magnitude=2),
                    ArithmeticTestCase(prompt="25 * 4 = ", expected="100", operator=ArithmeticOperator.MULTIPLY, difficulty=Difficulty.MEDIUM, magnitude=2),
                ])
            # Medium subtraction and division
            if "sub" in operations:
                tests.append(
                    ArithmeticTestCase(prompt="100 - 37 = ", expected="63", operator=ArithmeticOperator.SUBTRACT, difficulty=Difficulty.MEDIUM, magnitude=2),
                )
            if "div" in operations:
                tests.append(
                    ArithmeticTestCase(prompt="100 / 4 = ", expected="25", operator=ArithmeticOperator.DIVIDE, difficulty=Difficulty.MEDIUM, magnitude=2),
                )

        if include_hard:
            # Hard addition (3-digit)
            if "add" in operations:
                tests.extend([
                    ArithmeticTestCase(prompt="156 + 287 = ", expected="443", operator=ArithmeticOperator.ADD, difficulty=Difficulty.HARD, magnitude=3),
                    ArithmeticTestCase(prompt="999 + 111 = ", expected="1110", operator=ArithmeticOperator.ADD, difficulty=Difficulty.HARD, magnitude=3),
                ])
            # Hard multiplication
            if "mul" in operations:
                tests.extend([
                    ArithmeticTestCase(prompt="123 * 456 = ", expected="56088", operator=ArithmeticOperator.MULTIPLY, difficulty=Difficulty.HARD, magnitude=3),
                    ArithmeticTestCase(prompt="347 * 892 = ", expected="309524", operator=ArithmeticOperator.MULTIPLY, difficulty=Difficulty.HARD, magnitude=3),
                ])

        if quick:
            tests = tests[::3]  # Take every 3rd test

        return cls(test_cases=tests, total_tests=len(tests))
