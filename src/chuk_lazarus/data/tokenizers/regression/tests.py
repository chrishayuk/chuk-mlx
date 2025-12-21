"""Token regression test framework."""

from enum import Enum
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field


class TokenizerProtocol(Protocol):
    """Protocol for tokenizer compatibility."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...


class TestAssertion(str, Enum):
    """Types of test assertions."""

    MAX_TOKENS = "max_tokens"
    MIN_TOKENS = "min_tokens"
    EXACT_TOKENS = "exact_tokens"
    CONTAINS_TOKEN = "contains_token"
    NOT_CONTAINS_TOKEN = "not_contains_token"
    ROUNDTRIP_LOSSLESS = "roundtrip_lossless"
    TOKEN_SEQUENCE = "token_sequence"


class TokenTest(BaseModel):
    """A single tokenization test case."""

    name: str = Field(description="Test name")
    text: str = Field(description="Text to tokenize")
    assertion: TestAssertion = Field(description="Type of assertion")
    expected: int | list[int] | str | None = Field(
        default=None, description="Expected value for assertion"
    )
    description: str = Field(default="", description="Test description")
    add_special_tokens: bool = Field(default=False, description="Add special tokens")


class TokenTestResult(BaseModel):
    """Result of a single test."""

    test_name: str = Field(description="Test name")
    passed: bool = Field(description="Whether test passed")
    assertion: TestAssertion = Field(description="Assertion type")
    expected: str = Field(description="Expected value")
    actual: str = Field(description="Actual value")
    message: str = Field(default="", description="Error message if failed")


class TokenTestSuite(BaseModel):
    """A suite of tokenization tests."""

    name: str = Field(description="Suite name")
    description: str = Field(default="", description="Suite description")
    tests: list[TokenTest] = Field(default_factory=list, description="Test cases")


class TestSuiteResult(BaseModel):
    """Result of running a test suite."""

    suite_name: str = Field(description="Suite name")
    total_tests: int = Field(ge=0, description="Total tests run")
    passed: int = Field(ge=0, description="Tests passed")
    failed: int = Field(ge=0, description="Tests failed")
    pass_rate: float = Field(ge=0.0, le=1.0, description="Pass rate")
    results: list[TokenTestResult] = Field(default_factory=list, description="Individual results")
    failures: list[TokenTestResult] = Field(default_factory=list, description="Failed tests")


def _run_single_test(
    test: TokenTest,
    tokenizer: TokenizerProtocol,
) -> TokenTestResult:
    """Run a single test case."""
    try:
        token_ids = tokenizer.encode(test.text, add_special_tokens=test.add_special_tokens)
        num_tokens = len(token_ids)

        if test.assertion == TestAssertion.MAX_TOKENS:
            max_expected = int(test.expected)
            passed = num_tokens <= max_expected
            return TokenTestResult(
                test_name=test.name,
                passed=passed,
                assertion=test.assertion,
                expected=f"<= {max_expected}",
                actual=str(num_tokens),
                message="" if passed else f"Got {num_tokens} tokens, expected <= {max_expected}",
            )

        elif test.assertion == TestAssertion.MIN_TOKENS:
            min_expected = int(test.expected)
            passed = num_tokens >= min_expected
            return TokenTestResult(
                test_name=test.name,
                passed=passed,
                assertion=test.assertion,
                expected=f">= {min_expected}",
                actual=str(num_tokens),
                message="" if passed else f"Got {num_tokens} tokens, expected >= {min_expected}",
            )

        elif test.assertion == TestAssertion.EXACT_TOKENS:
            expected = int(test.expected)
            passed = num_tokens == expected
            return TokenTestResult(
                test_name=test.name,
                passed=passed,
                assertion=test.assertion,
                expected=str(expected),
                actual=str(num_tokens),
                message="" if passed else f"Got {num_tokens} tokens, expected {expected}",
            )

        elif test.assertion == TestAssertion.CONTAINS_TOKEN:
            expected_id = int(test.expected)
            passed = expected_id in token_ids
            return TokenTestResult(
                test_name=test.name,
                passed=passed,
                assertion=test.assertion,
                expected=f"contains {expected_id}",
                actual=str(token_ids[:10]) + ("..." if len(token_ids) > 10 else ""),
                message="" if passed else f"Token {expected_id} not found in output",
            )

        elif test.assertion == TestAssertion.NOT_CONTAINS_TOKEN:
            excluded_id = int(test.expected)
            passed = excluded_id not in token_ids
            return TokenTestResult(
                test_name=test.name,
                passed=passed,
                assertion=test.assertion,
                expected=f"not contains {excluded_id}",
                actual=str(token_ids[:10]) + ("..." if len(token_ids) > 10 else ""),
                message="" if passed else f"Token {excluded_id} should not be present",
            )

        elif test.assertion == TestAssertion.ROUNDTRIP_LOSSLESS:
            decoded = tokenizer.decode(token_ids)
            passed = decoded == test.text
            return TokenTestResult(
                test_name=test.name,
                passed=passed,
                assertion=test.assertion,
                expected=test.text[:50],
                actual=decoded[:50],
                message="" if passed else "Roundtrip is not lossless",
            )

        elif test.assertion == TestAssertion.TOKEN_SEQUENCE:
            expected_seq = test.expected if isinstance(test.expected, list) else []
            passed = token_ids == expected_seq
            return TokenTestResult(
                test_name=test.name,
                passed=passed,
                assertion=test.assertion,
                expected=str(expected_seq),
                actual=str(token_ids),
                message="" if passed else "Token sequence mismatch",
            )

        else:
            return TokenTestResult(
                test_name=test.name,
                passed=False,
                assertion=test.assertion,
                expected="",
                actual="",
                message=f"Unknown assertion type: {test.assertion}",
            )

    except Exception as e:
        return TokenTestResult(
            test_name=test.name,
            passed=False,
            assertion=test.assertion,
            expected=str(test.expected),
            actual="ERROR",
            message=str(e),
        )


def run_token_tests(
    suite: TokenTestSuite,
    tokenizer: TokenizerProtocol,
) -> TestSuiteResult:
    """
    Run a test suite against a tokenizer.

    Args:
        suite: Test suite to run
        tokenizer: Tokenizer to test

    Returns:
        TestSuiteResult with all results
    """
    results = []
    failures = []

    for test in suite.tests:
        result = _run_single_test(test, tokenizer)
        results.append(result)
        if not result.passed:
            failures.append(result)

    passed = len(results) - len(failures)
    pass_rate = passed / len(results) if results else 1.0

    return TestSuiteResult(
        suite_name=suite.name,
        total_tests=len(results),
        passed=passed,
        failed=len(failures),
        pass_rate=pass_rate,
        results=results,
        failures=failures,
    )


def create_test_suite(
    name: str,
    tests: list[dict],
    description: str = "",
) -> TokenTestSuite:
    """
    Create a test suite from a list of test definitions.

    Args:
        name: Suite name
        tests: List of test dicts
        description: Suite description

    Returns:
        TokenTestSuite
    """
    test_cases = []
    for t in tests:
        test_cases.append(
            TokenTest(
                name=t.get("name", "unnamed"),
                text=t.get("text", ""),
                assertion=TestAssertion(t.get("assertion", "max_tokens")),
                expected=t.get("expected"),
                description=t.get("description", ""),
                add_special_tokens=t.get("add_special_tokens", False),
            )
        )

    return TokenTestSuite(name=name, description=description, tests=test_cases)


def load_tests_from_yaml(path: str | Path) -> TokenTestSuite:
    """
    Load test suite from YAML file.

    Expected format:
    ```yaml
    name: My Test Suite
    description: Tests for my tokenizer
    tests:
      - name: math_symbols
        text: "σ_LT = √(σ² × L)"
        assertion: max_tokens
        expected: 8
      - name: roundtrip
        text: "Hello world"
        assertion: roundtrip_lossless
    ```

    Args:
        path: Path to YAML file

    Returns:
        TokenTestSuite
    """
    import yaml

    path = Path(path)
    with path.open() as f:
        data = yaml.safe_load(f)

    return create_test_suite(
        name=data.get("name", path.stem),
        tests=data.get("tests", []),
        description=data.get("description", ""),
    )
