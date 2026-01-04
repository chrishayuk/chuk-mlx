"""Shared fixtures for gym CLI tests."""

from argparse import Namespace
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def gym_run_args():
    """Create basic arguments for gym run command."""
    return Namespace(
        tokenizer="gpt2",
        mock=True,
        num_episodes=5,
        steps_per_episode=10,
        difficulty_min=0.0,
        difficulty_max=1.0,
        success_rate=0.8,
        seed=42,
        buffer_size=1000,
        host="localhost",
        port=8023,
        transport="telnet",
        output_mode="json",
        timeout=10.0,
        retries=3,
        max_samples=None,
        output=None,
    )


@pytest.fixture
def bench_args():
    """Create basic arguments for bench_pipeline command."""
    return Namespace(
        dataset=None,
        tokenizer="gpt2",
        max_samples=100,
        num_samples=1000,
        seed=42,
        max_length=512,
        bucket_edges="64,128,256,512",
        token_budget=4096,
    )


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    tokenizer.decode.return_value = "test output"
    tokenizer.eos_token_id = 2
    tokenizer.pad_token_id = 0
    return tokenizer


@pytest.fixture
def mock_gym_sample():
    """Create a mock gym sample."""
    sample = MagicMock()
    sample.episode_id = "episode_1"
    sample.step = 1
    sample.reward = 1.0
    sample.done = False
    return sample


@pytest.fixture
def mock_replay_buffer():
    """Create a mock replay buffer."""
    buffer = MagicMock()
    buffer.size = 10
    buffer.success_rate = 0.75
    buffer.mean_difficulty = 0.5
    buffer.mean_reward = 0.8
    buffer.add = MagicMock()
    buffer.to_dict = MagicMock(
        return_value={
            "size": 10,
            "success_rate": 0.75,
            "samples": [],
        }
    )
    return buffer


@pytest.fixture
def mock_stream(mock_gym_sample):
    """Create a mock gym stream."""
    stream = MagicMock()
    stream.__aenter__ = AsyncMock(return_value=stream)
    stream.__aexit__ = AsyncMock(return_value=None)

    async def async_gen():
        for i in range(5):
            yield mock_gym_sample

    stream.__aiter__ = lambda self: async_gen()
    return stream
