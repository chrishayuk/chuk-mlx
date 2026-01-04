"""Token regression tests command handler."""

import logging
import sys

from .._types import RegressionResult, RegressionRunConfig

logger = logging.getLogger(__name__)


def regression_run(config: RegressionRunConfig) -> RegressionResult:
    """Run token regression tests.

    Args:
        config: Regression test configuration.

    Returns:
        Regression result with test outcomes.
    """
    from .....data.tokenizers.regression import load_tests_from_yaml, run_token_tests
    from .....utils.tokenizer_loader import load_tokenizer

    logger.info(f"Loading tokenizer: {config.tokenizer}")
    tokenizer = load_tokenizer(config.tokenizer)

    logger.info(f"Loading tests from: {config.tests}")
    suite = load_tests_from_yaml(config.tests)

    logger.info(f"Running {len(suite.tests)} tests...")
    result = run_token_tests(suite, tokenizer)

    print("\n=== Regression Test Results ===")
    print(f"Suite: {suite.name}")
    print(f"Tests: {result.total_tests}")
    print(f"Passed: {result.passed}")
    print(f"Failed: {result.failed}")

    failures = []
    if result.failed > 0:
        print("\nFailed tests:")
        for test_result in result.results:
            if not test_result.passed:
                msg = f"{test_result.test_name}: {test_result.message}"
                failures.append(msg)
                print(f"  - {msg}")
        sys.exit(1)
    else:
        print("\nAll tests passed!")

    return RegressionResult(
        suite_name=suite.name,
        total_tests=result.total_tests,
        passed=result.passed,
        failed=result.failed,
        failures=failures,
    )
