"""
Token Regression Tests Example

Demonstrates the token regression testing framework:
- Define test cases with assertions
- Run tests against tokenizers
- Generate test suites from YAML

Uses Pydantic models for all data structures.
"""

import tempfile
from pathlib import Path

from chuk_lazarus.data.tokenizers.regression import (
    TestAssertion,
    TestSuiteResult,
    TokenTest,
    TokenTestSuite,
    create_test_suite,
    load_tests_from_yaml,
    run_token_tests,
)
from chuk_lazarus.utils.tokenizer_loader import load_tokenizer


def demo_basic_tests():
    """Demonstrate basic token tests."""
    print("=" * 60)
    print("Basic Token Tests")
    print("=" * 60)

    tokenizer = load_tokenizer("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # Create individual test cases
    tests = [
        TokenTest(
            name="hello_world_tokens",
            text="Hello, world!",
            assertion=TestAssertion.MAX_TOKENS,
            expected=10,
            description="Hello world should tokenize to <= 10 tokens",
        ),
        TokenTest(
            name="roundtrip_lossless",
            text="The quick brown fox",
            assertion=TestAssertion.ROUNDTRIP_LOSSLESS,
            description="Text should survive encode-decode roundtrip",
        ),
        TokenTest(
            name="min_tokens",
            text="A",
            assertion=TestAssertion.MIN_TOKENS,
            expected=1,
            description="Single character should be at least 1 token",
        ),
    ]

    print(f"\nCreated {len(tests)} test cases:")
    for test in tests:
        print(f"  - {test.name}: {test.assertion.value}")

    # Create test suite
    suite = TokenTestSuite(
        name="basic_tests",
        description="Basic tokenization tests",
        tests=tests,
    )

    # Run tests
    result: TestSuiteResult = run_token_tests(suite, tokenizer)

    print("\nTest results:")
    print(f"  Total:  {result.total_tests}")
    print(f"  Passed: {result.passed}")
    print(f"  Failed: {result.failed}")

    for test_result in result.results:
        status = "PASS" if test_result.passed else "FAIL"
        print(f"  [{status}] {test_result.test_name}: {test_result.message}")


def demo_assertion_types():
    """Demonstrate all assertion types."""
    print("\n" + "=" * 60)
    print("Assertion Types")
    print("=" * 60)

    tokenizer = load_tokenizer("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # Test each assertion type
    # Get token IDs for contains tests - encode without special tokens first
    hello_tokens = tokenizer.encode("Hello world")
    # Pick a token we know exists (use 2nd token to avoid BOS)
    known_token_id = hello_tokens[1] if len(hello_tokens) > 1 else hello_tokens[0]

    tests = [
        # Token count assertions
        TokenTest(
            name="max_tokens",
            text="Hello",
            assertion=TestAssertion.MAX_TOKENS,
            expected=5,
        ),
        TokenTest(
            name="min_tokens",
            text="Hello world, how are you?",
            assertion=TestAssertion.MIN_TOKENS,
            expected=3,
        ),
        TokenTest(
            name="exact_tokens",
            text="Hi",
            assertion=TestAssertion.EXACT_TOKENS,
            expected=1,  # May need adjustment based on tokenizer
        ),
        # Content assertions - use token IDs, not strings
        TokenTest(
            name="contains_token",
            text="Hello world",
            assertion=TestAssertion.CONTAINS_TOKEN,
            expected=known_token_id,  # Token ID we know is in the output
        ),
        TokenTest(
            name="not_contains",
            text="Hello world",
            assertion=TestAssertion.NOT_CONTAINS_TOKEN,
            expected=0,  # Token ID 0 (usually pad) should not be in "Hello world"
        ),
        # Roundtrip
        TokenTest(
            name="roundtrip",
            text="Machine learning is great!",
            assertion=TestAssertion.ROUNDTRIP_LOSSLESS,
        ),
    ]

    suite = TokenTestSuite(name="assertion_demo", tests=tests)
    result = run_token_tests(suite, tokenizer)

    print("\nAssertion results:")
    for r in result.results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.test_name}")
        print(f"      Assertion: {tests[result.results.index(r)].assertion.value}")
        print(f"      Message: {r.message}")


def demo_create_test_suite():
    """Demonstrate test suite creation helper."""
    print("\n" + "=" * 60)
    print("Test Suite Creation")
    print("=" * 60)

    tokenizer = load_tokenizer("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # Use helper function
    suite = create_test_suite(
        name="quick_suite",
        description="Quickly created test suite",
        tests=[
            {
                "name": "test1",
                "text": "Hello",
                "assertion": "max_tokens",
                "expected": 5,
            },
            {
                "name": "test2",
                "text": "World",
                "assertion": "roundtrip_lossless",
            },
        ],
    )

    print(f"\nCreated suite: {suite.name}")
    print(f"  Description: {suite.description}")
    print(f"  Tests: {len(suite.tests)}")

    result = run_token_tests(suite, tokenizer)
    print(f"\nResults: {result.passed}/{result.total_tests} passed")


def demo_yaml_tests():
    """Demonstrate loading tests from YAML."""
    print("\n" + "=" * 60)
    print("YAML Test Loading")
    print("=" * 60)

    # Create a temporary YAML file
    yaml_content = """
name: yaml_test_suite
description: Tests loaded from YAML file

tests:
  - name: greeting_tokens
    text: "Hello, how are you?"
    assertion: max_tokens
    expected: 10
    description: Greeting should be under 10 tokens

  - name: roundtrip_test
    text: "The quick brown fox jumps over the lazy dog."
    assertion: roundtrip_lossless
    description: Classic pangram should roundtrip perfectly

  - name: special_chars
    text: "Testing! @#$% symbols..."
    assertion: roundtrip_lossless
    description: Special characters should survive roundtrip

  - name: numbers
    text: "The year is 2024 and pi is 3.14159"
    assertion: min_tokens
    expected: 5
    description: Numbers should tokenize to multiple tokens
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        yaml_path = f.name

    try:
        tokenizer = load_tokenizer("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

        # Load from file path
        suite = load_tests_from_yaml(yaml_path)

        print(f"\nLoaded suite: {suite.name}")
        print(f"  Description: {suite.description}")
        print(f"  Tests: {len(suite.tests)}")

        for test in suite.tests:
            print(f"\n  Test: {test.name}")
            print(f"    Input: {test.text[:30]}...")
            print(f"    Assertion: {test.assertion.value}")

        # Run the tests
        result = run_token_tests(suite, tokenizer)

        print("\nResults:")
        for r in result.results:
            status = "PASS" if r.passed else "FAIL"
            print(f"  [{status}] {r.test_name}")

    finally:
        Path(yaml_path).unlink()


def demo_yaml_string():
    """Demonstrate loading tests from YAML via temp file."""
    print("\n" + "=" * 60)
    print("YAML String Loading")
    print("=" * 60)

    yaml_string = """
name: inline_suite
description: Suite from inline YAML

tests:
  - name: simple_test
    text: "Hello"
    assertion: max_tokens
    expected: 3
"""

    # Write to temp file and load
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_string)
        yaml_path = f.name

    try:
        tokenizer = load_tokenizer("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

        suite = load_tests_from_yaml(yaml_path)
        print(f"\nLoaded inline suite: {suite.name}")

        result = run_token_tests(suite, tokenizer)
        print(f"Results: {result.passed}/{result.total_tests} passed")
    finally:
        Path(yaml_path).unlink()


def demo_token_sequence():
    """Demonstrate token sequence assertion."""
    print("\n" + "=" * 60)
    print("Token Sequence Testing")
    print("=" * 60)

    tokenizer = load_tokenizer("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # First, see what tokens we get (without special tokens to match test runner)
    test_text = "Hello world"
    tokens = tokenizer.encode(test_text, add_special_tokens=False)
    print(f"\nText: '{test_text}'")
    print(f"Tokens (no special): {tokens}")

    # Create test with expected sequence
    # The test runner uses add_special_tokens=False by default
    test = TokenTest(
        name="sequence_test",
        text=test_text,
        assertion=TestAssertion.TOKEN_SEQUENCE,
        expected=tokens,  # Use actual tokens as expected
    )

    suite = TokenTestSuite(name="sequence_suite", tests=[test])
    result = run_token_tests(suite, tokenizer)

    print(f"\nSequence test: {'PASS' if result.passed == 1 else 'FAIL'}")


def demo_comprehensive_suite():
    """Demonstrate a comprehensive test suite."""
    print("\n" + "=" * 60)
    print("Comprehensive Test Suite")
    print("=" * 60)

    tokenizer = load_tokenizer("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # Create comprehensive test suite for tokenizer validation
    tests = [
        # Basic functionality
        TokenTest(
            name="empty_string",
            text="",
            assertion=TestAssertion.EXACT_TOKENS,
            expected=0,
            description="Empty string should produce 0 tokens",
        ),
        TokenTest(
            name="single_word",
            text="hello",
            assertion=TestAssertion.MIN_TOKENS,
            expected=1,
            description="Single word should be at least 1 token",
        ),
        # Roundtrip tests
        TokenTest(
            name="ascii_roundtrip",
            text="The quick brown fox jumps over the lazy dog.",
            assertion=TestAssertion.ROUNDTRIP_LOSSLESS,
        ),
        TokenTest(
            name="numbers_roundtrip",
            text="12345 67890",
            assertion=TestAssertion.ROUNDTRIP_LOSSLESS,
        ),
        TokenTest(
            name="punctuation_roundtrip",
            text="Hello! How are you? I'm fine, thanks.",
            assertion=TestAssertion.ROUNDTRIP_LOSSLESS,
        ),
        # Edge cases
        TokenTest(
            name="whitespace",
            text="   ",
            assertion=TestAssertion.MAX_TOKENS,
            expected=5,
        ),
        TokenTest(
            name="newlines",
            text="Line1\nLine2\nLine3",
            assertion=TestAssertion.ROUNDTRIP_LOSSLESS,
        ),
        # No UNK tokens for common text
        # UNK token ID is typically 0, 1, or a special ID - check tokenizer's unk_token_id
        TokenTest(
            name="no_unk_common",
            text="This is common English text.",
            assertion=TestAssertion.NOT_CONTAINS_TOKEN,
            expected=tokenizer.unk_token_id
            if hasattr(tokenizer, "unk_token_id") and tokenizer.unk_token_id is not None
            else 0,
        ),
    ]

    suite = TokenTestSuite(
        name="comprehensive_validation",
        description="Comprehensive tokenizer validation suite",
        tests=tests,
    )

    result = run_token_tests(suite, tokenizer)

    print("\nComprehensive test results:")
    print(f"  Total:  {result.total_tests}")
    print(f"  Passed: {result.passed}")
    print(f"  Failed: {result.failed}")
    print(f"  Pass rate: {result.passed / result.total_tests:.0%}")

    if result.failed > 0:
        print("\nFailed tests:")
        for r in result.results:
            if not r.passed:
                print(f"  - {r.test_name}: {r.message}")


def main():
    """Run all regression test demos."""
    print("Token Regression Tests Demo")
    print("=" * 60)

    demo_basic_tests()
    demo_assertion_types()
    demo_create_test_suite()
    demo_yaml_tests()
    demo_yaml_string()
    demo_token_sequence()
    demo_comprehensive_suite()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
