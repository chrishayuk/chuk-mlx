"""Tests for regression tests module."""

import tempfile
from pathlib import Path

from chuk_lazarus.data.tokenizers.regression.tests import (
    TestAssertion,
    TestSuiteResult,
    TokenTest,
    TokenTestResult,
    TokenTestSuite,
    create_test_suite,
    load_tests_from_yaml,
    run_token_tests,
)


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self, vocab: dict[str, int] | None = None):
        self._vocab = vocab or {
            "<pad>": 0,
            "<unk>": 1,
            "hello": 2,
            "world": 3,
            "the": 4,
            "test": 5,
        }
        self._id_to_token = {v: k for k, v in self._vocab.items()}

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        words = text.lower().split()
        return [self._vocab.get(w, 1) for w in words]

    def decode(self, ids: list[int]) -> str:
        return " ".join(self._id_to_token.get(i, "<unk>") for i in ids)


class TestTestAssertionEnum:
    """Tests for TestAssertion enum."""

    def test_all_assertions(self):
        assert TestAssertion.MAX_TOKENS == "max_tokens"
        assert TestAssertion.MIN_TOKENS == "min_tokens"
        assert TestAssertion.EXACT_TOKENS == "exact_tokens"
        assert TestAssertion.CONTAINS_TOKEN == "contains_token"
        assert TestAssertion.NOT_CONTAINS_TOKEN == "not_contains_token"
        assert TestAssertion.ROUNDTRIP_LOSSLESS == "roundtrip_lossless"
        assert TestAssertion.TOKEN_SEQUENCE == "token_sequence"


class TestTokenTestModel:
    """Tests for TokenTest model."""

    def test_valid_test(self):
        test = TokenTest(
            name="test_hello",
            text="hello world",
            assertion=TestAssertion.MAX_TOKENS,
            expected=5,
        )
        assert test.name == "test_hello"
        assert test.assertion == TestAssertion.MAX_TOKENS

    def test_test_with_description(self):
        test = TokenTest(
            name="test_desc",
            text="hello",
            assertion=TestAssertion.EXACT_TOKENS,
            expected=1,
            description="Test that hello is one token",
        )
        assert test.description == "Test that hello is one token"

    def test_test_defaults(self):
        test = TokenTest(
            name="test",
            text="hello",
            assertion=TestAssertion.ROUNDTRIP_LOSSLESS,
        )
        assert test.expected is None
        assert test.add_special_tokens is False


class TestTokenTestResultModel:
    """Tests for TokenTestResult model."""

    def test_passed_result(self):
        result = TokenTestResult(
            test_name="test_hello",
            passed=True,
            assertion=TestAssertion.MAX_TOKENS,
            expected="<= 5",
            actual="3",
        )
        assert result.passed is True

    def test_failed_result(self):
        result = TokenTestResult(
            test_name="test_hello",
            passed=False,
            assertion=TestAssertion.MAX_TOKENS,
            expected="<= 5",
            actual="10",
            message="Got 10 tokens, expected <= 5",
        )
        assert result.passed is False
        assert len(result.message) > 0


class TestTokenTestSuiteModel:
    """Tests for TokenTestSuite model."""

    def test_valid_suite(self):
        tests = [
            TokenTest(
                name="test1",
                text="hello",
                assertion=TestAssertion.EXACT_TOKENS,
                expected=1,
            )
        ]
        suite = TokenTestSuite(
            name="Test Suite",
            description="My tests",
            tests=tests,
        )
        assert suite.name == "Test Suite"
        assert len(suite.tests) == 1


class TestTestSuiteResultModel:
    """Tests for TestSuiteResult model."""

    def test_all_passed(self):
        results = [
            TokenTestResult(
                test_name="test1",
                passed=True,
                assertion=TestAssertion.MAX_TOKENS,
                expected="<= 5",
                actual="3",
            )
        ]
        suite_result = TestSuiteResult(
            suite_name="Suite",
            total_tests=1,
            passed=1,
            failed=0,
            pass_rate=1.0,
            results=results,
            failures=[],
        )
        assert suite_result.pass_rate == 1.0

    def test_some_failed(self):
        result1 = TokenTestResult(
            test_name="test1",
            passed=True,
            assertion=TestAssertion.MAX_TOKENS,
            expected="<= 5",
            actual="3",
        )
        result2 = TokenTestResult(
            test_name="test2",
            passed=False,
            assertion=TestAssertion.MAX_TOKENS,
            expected="<= 5",
            actual="10",
            message="Failed",
        )
        suite_result = TestSuiteResult(
            suite_name="Suite",
            total_tests=2,
            passed=1,
            failed=1,
            pass_rate=0.5,
            results=[result1, result2],
            failures=[result2],
        )
        assert suite_result.pass_rate == 0.5
        assert len(suite_result.failures) == 1


class TestRunTokenTests:
    """Tests for run_token_tests function."""

    def test_max_tokens_pass(self):
        tokenizer = MockTokenizer()
        suite = TokenTestSuite(
            name="Test",
            tests=[
                TokenTest(
                    name="max_test",
                    text="hello world",
                    assertion=TestAssertion.MAX_TOKENS,
                    expected=5,
                )
            ],
        )
        result = run_token_tests(suite, tokenizer)
        assert result.passed == 1

    def test_max_tokens_fail(self):
        tokenizer = MockTokenizer()
        suite = TokenTestSuite(
            name="Test",
            tests=[
                TokenTest(
                    name="max_test",
                    text="hello world the test",
                    assertion=TestAssertion.MAX_TOKENS,
                    expected=2,  # Only allow 2 tokens
                )
            ],
        )
        result = run_token_tests(suite, tokenizer)
        assert result.failed == 1

    def test_min_tokens_pass(self):
        tokenizer = MockTokenizer()
        suite = TokenTestSuite(
            name="Test",
            tests=[
                TokenTest(
                    name="min_test",
                    text="hello world the",
                    assertion=TestAssertion.MIN_TOKENS,
                    expected=2,
                )
            ],
        )
        result = run_token_tests(suite, tokenizer)
        assert result.passed == 1

    def test_min_tokens_fail(self):
        tokenizer = MockTokenizer()
        suite = TokenTestSuite(
            name="Test",
            tests=[
                TokenTest(
                    name="min_test",
                    text="hello",
                    assertion=TestAssertion.MIN_TOKENS,
                    expected=5,
                )
            ],
        )
        result = run_token_tests(suite, tokenizer)
        assert result.failed == 1

    def test_exact_tokens_pass(self):
        tokenizer = MockTokenizer()
        suite = TokenTestSuite(
            name="Test",
            tests=[
                TokenTest(
                    name="exact_test",
                    text="hello world",
                    assertion=TestAssertion.EXACT_TOKENS,
                    expected=2,
                )
            ],
        )
        result = run_token_tests(suite, tokenizer)
        assert result.passed == 1

    def test_exact_tokens_fail(self):
        tokenizer = MockTokenizer()
        suite = TokenTestSuite(
            name="Test",
            tests=[
                TokenTest(
                    name="exact_test",
                    text="hello world",
                    assertion=TestAssertion.EXACT_TOKENS,
                    expected=5,
                )
            ],
        )
        result = run_token_tests(suite, tokenizer)
        assert result.failed == 1

    def test_contains_token_pass(self):
        tokenizer = MockTokenizer()
        suite = TokenTestSuite(
            name="Test",
            tests=[
                TokenTest(
                    name="contains_test",
                    text="hello world",
                    assertion=TestAssertion.CONTAINS_TOKEN,
                    expected=2,  # Token ID for "hello"
                )
            ],
        )
        result = run_token_tests(suite, tokenizer)
        assert result.passed == 1

    def test_contains_token_fail(self):
        tokenizer = MockTokenizer()
        suite = TokenTestSuite(
            name="Test",
            tests=[
                TokenTest(
                    name="contains_test",
                    text="hello world",
                    assertion=TestAssertion.CONTAINS_TOKEN,
                    expected=999,  # Non-existent token
                )
            ],
        )
        result = run_token_tests(suite, tokenizer)
        assert result.failed == 1

    def test_not_contains_token_pass(self):
        tokenizer = MockTokenizer()
        suite = TokenTestSuite(
            name="Test",
            tests=[
                TokenTest(
                    name="not_contains_test",
                    text="hello world",
                    assertion=TestAssertion.NOT_CONTAINS_TOKEN,
                    expected=999,  # Token not present
                )
            ],
        )
        result = run_token_tests(suite, tokenizer)
        assert result.passed == 1

    def test_not_contains_token_fail(self):
        tokenizer = MockTokenizer()
        suite = TokenTestSuite(
            name="Test",
            tests=[
                TokenTest(
                    name="not_contains_test",
                    text="hello world",
                    assertion=TestAssertion.NOT_CONTAINS_TOKEN,
                    expected=2,  # Token IS present
                )
            ],
        )
        result = run_token_tests(suite, tokenizer)
        assert result.failed == 1

    def test_roundtrip_lossless_pass(self):
        tokenizer = MockTokenizer()
        suite = TokenTestSuite(
            name="Test",
            tests=[
                TokenTest(
                    name="roundtrip_test",
                    text="hello world",
                    assertion=TestAssertion.ROUNDTRIP_LOSSLESS,
                )
            ],
        )
        result = run_token_tests(suite, tokenizer)
        assert result.passed == 1

    def test_roundtrip_lossless_fail(self):
        """Test roundtrip failure when decode differs."""

        class LossyTokenizer:
            def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
                return [1, 2, 3]

            def decode(self, ids: list[int]) -> str:
                return "different text"

        tokenizer = LossyTokenizer()
        suite = TokenTestSuite(
            name="Test",
            tests=[
                TokenTest(
                    name="roundtrip_test",
                    text="hello world",
                    assertion=TestAssertion.ROUNDTRIP_LOSSLESS,
                )
            ],
        )
        result = run_token_tests(suite, tokenizer)
        assert result.failed == 1

    def test_token_sequence_pass(self):
        tokenizer = MockTokenizer()
        suite = TokenTestSuite(
            name="Test",
            tests=[
                TokenTest(
                    name="sequence_test",
                    text="hello world",
                    assertion=TestAssertion.TOKEN_SEQUENCE,
                    expected=[2, 3],  # hello=2, world=3
                )
            ],
        )
        result = run_token_tests(suite, tokenizer)
        assert result.passed == 1

    def test_token_sequence_fail(self):
        tokenizer = MockTokenizer()
        suite = TokenTestSuite(
            name="Test",
            tests=[
                TokenTest(
                    name="sequence_test",
                    text="hello world",
                    assertion=TestAssertion.TOKEN_SEQUENCE,
                    expected=[3, 2],  # Wrong order
                )
            ],
        )
        result = run_token_tests(suite, tokenizer)
        assert result.failed == 1

    def test_empty_suite(self):
        tokenizer = MockTokenizer()
        suite = TokenTestSuite(name="Empty", tests=[])
        result = run_token_tests(suite, tokenizer)
        assert result.total_tests == 0
        assert result.pass_rate == 1.0

    def test_multiple_tests(self):
        tokenizer = MockTokenizer()
        suite = TokenTestSuite(
            name="Multi",
            tests=[
                TokenTest(
                    name="test1",
                    text="hello",
                    assertion=TestAssertion.EXACT_TOKENS,
                    expected=1,
                ),
                TokenTest(
                    name="test2",
                    text="hello world",
                    assertion=TestAssertion.MAX_TOKENS,
                    expected=5,
                ),
            ],
        )
        result = run_token_tests(suite, tokenizer)
        assert result.total_tests == 2

    def test_exception_handling(self):
        """Test that exceptions are caught and reported as failures."""

        class ErrorTokenizer:
            def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
                raise ValueError("Encoding error")

            def decode(self, ids: list[int]) -> str:
                return ""

        tokenizer = ErrorTokenizer()
        suite = TokenTestSuite(
            name="Test",
            tests=[
                TokenTest(
                    name="error_test",
                    text="hello",
                    assertion=TestAssertion.MAX_TOKENS,
                    expected=5,
                )
            ],
        )
        result = run_token_tests(suite, tokenizer)
        assert result.failed == 1
        assert "error" in result.failures[0].message.lower()


class TestCreateTestSuite:
    """Tests for create_test_suite function."""

    def test_basic_creation(self):
        tests = [{"name": "test1", "text": "hello", "assertion": "max_tokens", "expected": 5}]
        suite = create_test_suite("MySuite", tests)
        assert suite.name == "MySuite"
        assert len(suite.tests) == 1

    def test_with_description(self):
        suite = create_test_suite("MySuite", [], description="My test suite")
        assert suite.description == "My test suite"

    def test_multiple_tests(self):
        tests = [
            {"name": "test1", "text": "hello", "assertion": "max_tokens", "expected": 5},
            {"name": "test2", "text": "world", "assertion": "exact_tokens", "expected": 1},
        ]
        suite = create_test_suite("MySuite", tests)
        assert len(suite.tests) == 2

    def test_default_values(self):
        tests = [{"text": "hello", "assertion": "max_tokens"}]
        suite = create_test_suite("MySuite", tests)
        assert suite.tests[0].name == "unnamed"


class TestLoadTestsFromYaml:
    """Tests for load_tests_from_yaml function."""

    def test_load_valid_yaml(self):
        yaml_content = """
name: Test Suite
description: My tests
tests:
  - name: max_test
    text: "hello world"
    assertion: max_tokens
    expected: 5
  - name: roundtrip
    text: "foo bar"
    assertion: roundtrip_lossless
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            suite = load_tests_from_yaml(f.name)

        assert suite.name == "Test Suite"
        assert len(suite.tests) == 2

    def test_load_minimal_yaml(self):
        yaml_content = """
tests:
  - name: test
    text: "hello"
    assertion: exact_tokens
    expected: 1
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            suite = load_tests_from_yaml(f.name)

        assert len(suite.tests) == 1

    def test_load_from_path(self):
        yaml_content = """
name: Path Test
tests:
  - name: test
    text: "hello"
    assertion: max_tokens
    expected: 5
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            suite = load_tests_from_yaml(Path(f.name))

        assert suite.name == "Path Test"
