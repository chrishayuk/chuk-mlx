"""Test fixtures for dataset tests."""

import pytest


@pytest.fixture
def sample_arithmetic_problem():
    """Sample arithmetic problem for testing."""
    from chuk_lazarus.introspection.datasets.models import ArithmeticProblem

    return ArithmeticProblem(
        prompt="127 * 89 = ",
        answer=11303,
        operation="multiplication",
    )


@pytest.fixture
def sample_context_test():
    """Sample context test for testing."""
    from chuk_lazarus.introspection.datasets.models import ContextTest

    return ContextTest(
        prompt="111 127",
        context_type="number",
        description="Number followed by target",
    )
