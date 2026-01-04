"""Tests for gym info command."""

from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest

from chuk_lazarus.cli.commands.gym.info import gym_info, gym_info_cmd

GYM_TRANSPORT_PATCH = "chuk_lazarus.data.batching.streaming.GymTransport"
GYM_OUTPUT_MODE_PATCH = "chuk_lazarus.data.batching.streaming.GymOutputMode"


class TestGymInfo:
    """Tests for gym_info async command."""

    @pytest.mark.asyncio
    async def test_gym_info_basic(self, capsys):
        """Test basic gym_info command."""
        with (
            patch(GYM_TRANSPORT_PATCH, create=True) as mock_transport,
            patch(GYM_OUTPUT_MODE_PATCH, create=True) as mock_mode,
        ):
            mock_transport.__iter__ = lambda self: iter(
                [
                    MagicMock(value="telnet"),
                    MagicMock(value="http"),
                ]
            )
            mock_mode.__iter__ = lambda self: iter(
                [
                    MagicMock(value="json"),
                    MagicMock(value="text"),
                ]
            )

            await gym_info()

            captured = capsys.readouterr()
            assert "Gym Stream Configuration" in captured.out
            assert "Supported Transports:" in captured.out
            assert "Supported Output Modes:" in captured.out

    @pytest.mark.asyncio
    async def test_gym_info_displays_defaults(self, capsys):
        """Test that gym_info displays default configuration values."""
        with (
            patch(GYM_TRANSPORT_PATCH, create=True) as mock_transport,
            patch(GYM_OUTPUT_MODE_PATCH, create=True) as mock_mode,
        ):
            mock_transport.__iter__ = lambda self: iter([])
            mock_mode.__iter__ = lambda self: iter([])

            await gym_info()

            captured = capsys.readouterr()
            assert "localhost" in captured.out
            assert "8023" in captured.out
            assert "telnet" in captured.out
            assert "json" in captured.out

    @pytest.mark.asyncio
    async def test_gym_info_displays_examples(self, capsys):
        """Test that gym_info displays usage examples."""
        with (
            patch(GYM_TRANSPORT_PATCH, create=True) as mock_transport,
            patch(GYM_OUTPUT_MODE_PATCH, create=True) as mock_mode,
        ):
            mock_transport.__iter__ = lambda self: iter([])
            mock_mode.__iter__ = lambda self: iter([])

            await gym_info()

            captured = capsys.readouterr()
            assert "lazarus gym run" in captured.out
            assert "--mock" in captured.out
            assert "--tokenizer gpt2" in captured.out


class TestGymInfoCmd:
    """Tests for gym_info_cmd CLI entry point."""

    @pytest.mark.asyncio
    async def test_gym_info_cmd(self, capsys):
        """Test CLI entry point."""
        args = Namespace()

        with (
            patch(GYM_TRANSPORT_PATCH, create=True) as mock_transport,
            patch(GYM_OUTPUT_MODE_PATCH, create=True) as mock_mode,
        ):
            mock_transport.__iter__ = lambda self: iter([])
            mock_mode.__iter__ = lambda self: iter([])

            await gym_info_cmd(args)

            captured = capsys.readouterr()
            assert "Gym Stream Configuration" in captured.out
