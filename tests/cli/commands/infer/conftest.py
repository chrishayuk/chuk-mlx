"""Shared fixtures for infer CLI tests."""

import sys
from argparse import Namespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def setup_models_module():
    """Set up mock models module."""
    if "chuk_lazarus.models" not in sys.modules:
        mock_models_module = MagicMock()
        sys.modules["chuk_lazarus.models"] = mock_models_module
        yield
        if "chuk_lazarus.models" in sys.modules:
            del sys.modules["chuk_lazarus.models"]
    else:
        yield


@pytest.fixture
def mock_model():
    """Create a mock model."""
    model = MagicMock()
    model.load_adapter = MagicMock()
    model.generate = MagicMock(return_value="Generated response")
    return model


@pytest.fixture
def basic_infer_args():
    """Create basic inference arguments."""
    return Namespace(
        model="test-model",
        adapter=None,
        prompt="What is 2+2?",
        prompt_file=None,
        max_tokens=256,
        temperature=0.7,
    )
