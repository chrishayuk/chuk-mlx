"""Fixtures for moe_expert handler tests."""

import pytest


# Override the autouse fixture from parent conftest to allow real moe imports
@pytest.fixture(autouse=True)
def setup_introspection_module():
    """Override parent fixture to allow real moe module imports."""
    yield None
