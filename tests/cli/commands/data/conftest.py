"""Shared fixtures for data CLI tests."""

from argparse import Namespace
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    tokenizer.decode.return_value = "test output"
    tokenizer.pad_token_id = 0
    return tokenizer


@pytest.fixture
def mock_length_cache():
    """Create a mock LengthCache."""
    cache = MagicMock()
    cache.__len__.return_value = 100
    cache.tokenizer_hash = "test_hash_123"
    cache.get_all.return_value = {
        "sample_0": 10,
        "sample_1": 20,
        "sample_2": 30,
        "sample_3": 15,
        "sample_4": 25,
    }

    # Make it work as async context manager
    async def mock_aenter(self):
        return self

    async def mock_aexit(self, exc_type, exc_val, exc_tb):
        return None

    cache.__aenter__ = mock_aenter
    cache.__aexit__ = mock_aexit
    cache.add = AsyncMock()

    return cache


@pytest.fixture
def mock_batch_plan():
    """Create a mock BatchPlan."""
    plan = MagicMock()
    plan.num_epochs = 2
    plan.total_microbatches = 50
    plan.fingerprint = "test_fingerprint_abc123"

    # Mock metadata
    plan.meta = MagicMock()
    plan.meta.created_at = "2026-01-03T12:00:00"
    plan.meta.dataset_hash = "dataset_hash_123"
    plan.meta.tokenizer_hash = "tokenizer_hash_456"
    plan.meta.token_budget = 2048
    plan.meta.bucket_edges = [128, 256, 512, 1024]
    plan.meta.mode = "predictable"
    plan.meta.pad_policy = "pad_to_bucket"
    plan.meta.overflow_max = 2048
    plan.meta.seed = 42

    # Mock epoch plan
    epoch_plan = MagicMock()
    epoch_plan.num_microbatches = 25
    epoch_plan.total_samples = 500
    epoch_plan.total_tokens = 10000

    # Mock microbatch
    microbatch = MagicMock()
    microbatch.batch_size = 8
    microbatch.bucket_id = 0
    microbatch.max_len = 128
    microbatch.samples = ["sample_0", "sample_1"]

    epoch_plan.microbatches = [microbatch] * 5

    plan.get_epoch.return_value = epoch_plan
    plan.iter_epoch.return_value = iter([microbatch] * 25)
    plan.shard.return_value = plan

    return plan


@pytest.fixture
def length_build_args(tmp_path):
    """Create arguments for length build command."""
    return Namespace(
        tokenizer="gpt2",
        dataset=str(tmp_path / "test.jsonl"),
        output=str(tmp_path / "cache.db"),
    )


@pytest.fixture
def length_stats_args(tmp_path):
    """Create arguments for length stats command."""
    return Namespace(cache=str(tmp_path / "cache.db"))


@pytest.fixture
def batchplan_build_args(tmp_path):
    """Create arguments for batchplan build command."""
    return Namespace(
        lengths=str(tmp_path / "cache.db"),
        bucket_edges="128,256,512,1024",
        token_budget=2048,
        overflow_max=2048,
        predictable=True,
        seed=42,
        epochs=2,
        output=str(tmp_path / "plan.msgpack"),
        dataset_hash="dataset_123",
    )


@pytest.fixture
def batchplan_info_args(tmp_path):
    """Create arguments for batchplan info command."""
    return Namespace(
        plan=str(tmp_path / "plan.msgpack"),
        rank=None,
        world_size=None,
        show_batches=None,
    )


@pytest.fixture
def batching_analyze_args(tmp_path):
    """Create arguments for batching analyze command."""
    return Namespace(
        cache=str(tmp_path / "cache.db"),
        bucket_edges="128,256,512",
        overflow_max=1024,
        output=None,
    )


@pytest.fixture
def batching_histogram_args(tmp_path):
    """Create arguments for batching histogram command."""
    return Namespace(
        cache=str(tmp_path / "cache.db"),
        bins=20,
        width=80,
    )


@pytest.fixture
def batching_suggest_args(tmp_path):
    """Create arguments for batching suggest command."""
    return Namespace(
        cache=str(tmp_path / "cache.db"),
        num_buckets=5,
        goal="waste",
        max_length=2048,
    )


@pytest.fixture
def batch_generate_args(tmp_path):
    """Create arguments for batch generate command."""
    return Namespace(
        plan=str(tmp_path / "plan.msgpack"),
        dataset=str(tmp_path / "dataset.jsonl"),
        tokenizer="gpt2",
        output=str(tmp_path / "batches"),
    )
