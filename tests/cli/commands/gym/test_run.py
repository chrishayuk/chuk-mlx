"""Tests for gym run command."""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chuk_lazarus.cli.commands.gym._types import GymRunConfig
from chuk_lazarus.cli.commands.gym.run import gym_run, gym_run_cmd

LOAD_TOKENIZER_PATCH = "chuk_lazarus.utils.tokenizer_loader.load_tokenizer"
REPLAY_BUFFER_PATCH = "chuk_lazarus.data.batching.streaming.ReplayBuffer"
REPLAY_CONFIG_PATCH = "chuk_lazarus.data.batching.streaming.ReplayBufferConfig"
MOCK_STREAM_PATCH = "chuk_lazarus.data.batching.streaming.MockGymStream"


class TestGymRun:
    """Tests for gym_run async command."""

    @pytest.fixture
    def basic_config(self, gym_run_args):
        """Create basic gym config."""
        return GymRunConfig.from_args(gym_run_args)

    @pytest.mark.asyncio
    async def test_gym_run_mock_basic(
        self, basic_config, mock_tokenizer, mock_replay_buffer, mock_stream, capsys
    ):
        """Test basic gym run with mock stream."""
        with (
            patch(LOAD_TOKENIZER_PATCH, create=True, return_value=mock_tokenizer),
            patch(REPLAY_BUFFER_PATCH, create=True, return_value=mock_replay_buffer),
            patch(REPLAY_CONFIG_PATCH, create=True),
            patch(MOCK_STREAM_PATCH, create=True, return_value=mock_stream),
        ):
            result = await gym_run(basic_config)

            assert result.total_samples == 5
            assert result.total_episodes == 1
            assert result.buffer_size == 10

            captured = capsys.readouterr()
            assert "Gym Episode Streaming" in captured.out
            assert "Summary" in captured.out

    @pytest.mark.asyncio
    async def test_gym_run_with_max_samples(
        self, gym_run_args, mock_tokenizer, mock_replay_buffer, mock_gym_sample, capsys
    ):
        """Test gym run with max samples limit."""
        gym_run_args.max_samples = 3
        config = GymRunConfig.from_args(gym_run_args)

        # Create stream that yields more than max_samples
        stream = MagicMock()
        stream.__aenter__ = AsyncMock(return_value=stream)
        stream.__aexit__ = AsyncMock(return_value=None)

        async def async_gen():
            for i in range(10):
                yield mock_gym_sample

        stream.__aiter__ = lambda self: async_gen()

        with (
            patch(LOAD_TOKENIZER_PATCH, create=True, return_value=mock_tokenizer),
            patch(REPLAY_BUFFER_PATCH, create=True, return_value=mock_replay_buffer),
            patch(REPLAY_CONFIG_PATCH, create=True),
            patch(MOCK_STREAM_PATCH, create=True, return_value=stream),
        ):
            result = await gym_run(config)

            # Should stop at max_samples
            assert result.total_samples == 3

    @pytest.mark.asyncio
    async def test_gym_run_with_output(
        self, gym_run_args, mock_tokenizer, mock_replay_buffer, mock_stream, capsys
    ):
        """Test gym run saving output to file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            gym_run_args.output = output_path
            config = GymRunConfig.from_args(gym_run_args)

            with (
                patch(LOAD_TOKENIZER_PATCH, create=True, return_value=mock_tokenizer),
                patch(REPLAY_BUFFER_PATCH, create=True, return_value=mock_replay_buffer),
                patch(REPLAY_CONFIG_PATCH, create=True),
                patch(MOCK_STREAM_PATCH, create=True, return_value=mock_stream),
            ):
                result = await gym_run(config)

                assert result.output_path == Path(output_path)
                assert Path(output_path).exists()

                with open(output_path) as f:
                    data = json.load(f)
                    assert isinstance(data, dict)

                captured = capsys.readouterr()
                assert "Buffer saved to:" in captured.out
        finally:
            if Path(output_path).exists():
                Path(output_path).unlink()


class TestGymRunCmd:
    """Tests for gym_run_cmd CLI entry point."""

    @pytest.mark.asyncio
    async def test_gym_run_cmd(
        self, gym_run_args, mock_tokenizer, mock_replay_buffer, mock_stream, capsys
    ):
        """Test CLI entry point."""
        with (
            patch(LOAD_TOKENIZER_PATCH, create=True, return_value=mock_tokenizer),
            patch(REPLAY_BUFFER_PATCH, create=True, return_value=mock_replay_buffer),
            patch(REPLAY_CONFIG_PATCH, create=True),
            patch(MOCK_STREAM_PATCH, create=True, return_value=mock_stream),
        ):
            await gym_run_cmd(gym_run_args)

            captured = capsys.readouterr()
            assert "Gym Run Summary" in captured.out
