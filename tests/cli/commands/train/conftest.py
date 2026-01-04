"""Shared fixtures for train CLI tests."""

import sys
from argparse import Namespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True, scope="module")
def setup_mock_modules():
    """Set up mock modules for imports."""
    modules_to_mock = [
        "chuk_lazarus.models",
        "chuk_lazarus.data",
        "chuk_lazarus.training",
        "chuk_lazarus.training.losses",
        "chuk_lazarus.data.generators",
    ]

    original_modules = {}
    for module_name in modules_to_mock:
        if module_name not in sys.modules:
            original_modules[module_name] = None
            sys.modules[module_name] = MagicMock()

    yield

    for module_name, original in original_modules.items():
        if original is None and module_name in sys.modules:
            del sys.modules[module_name]


@pytest.fixture
def mock_model():
    """Create a mock model."""
    model = MagicMock()
    model.model = MagicMock()
    model.tokenizer = MagicMock()
    return model


@pytest.fixture
def mock_trainer():
    """Create a mock trainer."""
    trainer = MagicMock()
    trainer.train = MagicMock()
    return trainer


@pytest.fixture
def mock_dataset():
    """Create a mock dataset."""
    return MagicMock()


@pytest.fixture
def sft_args():
    """Create SFT training arguments."""
    return Namespace(
        model="test-model",
        data="/path/to/train.jsonl",
        eval_data=None,
        output="./checkpoints/sft",
        epochs=3,
        batch_size=4,
        learning_rate=1e-5,
        max_length=512,
        use_lora=False,
        lora_rank=8,
        mask_prompt=False,
        log_interval=10,
    )


@pytest.fixture
def dpo_args():
    """Create DPO training arguments."""
    return Namespace(
        model="test-model",
        ref_model=None,
        data="/path/to/preferences.jsonl",
        eval_data=None,
        output="./checkpoints/dpo",
        epochs=3,
        batch_size=4,
        learning_rate=1e-6,
        beta=0.1,
        max_length=512,
        use_lora=False,
        lora_rank=8,
    )


@pytest.fixture
def datagen_args():
    """Create data generation arguments."""
    return Namespace(
        type="math",
        output="./data/generated",
        sft_samples=10000,
        dpo_samples=5000,
        seed=42,
    )
