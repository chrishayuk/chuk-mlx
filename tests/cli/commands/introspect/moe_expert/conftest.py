"""Fixtures for moe_expert CLI tests.

This module overrides the autouse setup_introspection_module fixture
from the parent conftest to properly handle moe module imports.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest


# Override the autouse fixture from parent conftest to allow moe imports
@pytest.fixture(autouse=True)
def setup_introspection_module():
    """Override parent fixture to allow real moe module imports.

    The parent conftest.py has an autouse fixture that mocks
    chuk_lazarus.introspection, which breaks imports for the moe
    subpackage. This fixture takes precedence due to pytest's
    fixture resolution order (closer fixtures win).
    """
    # Don't do any sys.modules manipulation - let real imports work
    yield None


@pytest.fixture
def mock_expert_router():
    """Mock ExpertRouter for handler tests."""
    mock_router = AsyncMock()
    mock_router.__aenter__ = AsyncMock(return_value=mock_router)
    mock_router.__aexit__ = AsyncMock(return_value=None)
    mock_router._moe_type = "gpt_oss_batched"
    mock_router._moe_layers = (0, 1, 2, 3, 4, 5, 6, 7)
    mock_router._num_experts = 32
    mock_router._num_experts_per_tok = 4
    return mock_router


@pytest.fixture
def mock_model():
    """Create a mock MLX model for testing."""
    mock = MagicMock()
    layers = []
    for i in range(8):
        layer = MagicMock()
        layer.mlp = MagicMock()
        layer.mlp.router = MagicMock()
        layer.mlp.router.num_experts = 32
        layer.mlp.router.num_experts_per_tok = 4
        layer.mlp.experts = MagicMock()
        layer.mlp.experts.gate_up_proj = MagicMock()
        layer.mlp.experts.down_proj = MagicMock()
        layers.append(layer)
    mock.model.layers = layers
    return mock


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    tokenizer.decode.return_value = "test output"
    tokenizer.eos_token_id = 2
    tokenizer.pad_token_id = 0
    return tokenizer
