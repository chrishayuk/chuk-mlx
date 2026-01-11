"""Tests for chat handler."""

from argparse import Namespace
from unittest.mock import AsyncMock, patch

import pytest

from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.chat import (
    _async_chat,
    handle_chat,
)
from chuk_lazarus.introspection.moe.models import (
    ExpertChatResult,
    GenerationStats,
)


class TestHandleChat:
    """Tests for handle_chat function."""

    def test_handle_chat_calls_asyncio_run(self):
        """Test that handle_chat calls asyncio.run."""
        args = Namespace(
            model="test/model",
            expert=6,
            prompt="Test",
            max_tokens=100,
            temperature=0.0,
            raw=False,
            verbose=False,
        )

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.chat.asyncio"
        ) as mock_asyncio:
            handle_chat(args)
            mock_asyncio.run.assert_called_once()


class TestAsyncChat:
    """Tests for _async_chat function."""

    @pytest.mark.asyncio
    async def test_missing_expert_prints_error(self, capsys):
        """Test that missing expert prints error."""
        args = Namespace(model="test/model", prompt="Test")

        await _async_chat(args)

        captured = capsys.readouterr()
        assert "Error: --expert/-e is required" in captured.out

    @pytest.mark.asyncio
    async def test_missing_prompt_prints_error(self, capsys):
        """Test that missing prompt prints error."""
        args = Namespace(model="test/model", expert=6)

        await _async_chat(args)

        captured = capsys.readouterr()
        assert "Error: --prompt/-p is required" in captured.out

    @pytest.mark.asyncio
    async def test_successful_chat(self, capsys):
        """Test successful chat execution."""
        args = Namespace(
            model="test/model",
            expert=6,
            prompt="127 * 89 = ",
            max_tokens=100,
            temperature=0.0,
            raw=False,
            verbose=False,
        )

        mock_stats = GenerationStats(
            expert_idx=6,
            tokens_generated=10,
            layers_modified=8,
            moe_type="gpt_oss_batched",
        )
        mock_result = ExpertChatResult(
            prompt="127 * 89 = ",
            response="11303",
            expert_idx=6,
            stats=mock_stats,
        )

        mock_router = AsyncMock()
        mock_router.chat_with_expert = AsyncMock(return_value=mock_result)
        mock_router._moe_type = "gpt_oss_batched"
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.chat.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_chat(args)

            captured = capsys.readouterr()
            assert "CHAT WITH EXPERT 6" in captured.out
            assert "11303" in captured.out

    @pytest.mark.asyncio
    async def test_chat_with_verbose(self, capsys):
        """Test chat with verbose output."""
        args = Namespace(
            model="test/model",
            expert=6,
            prompt="Test",
            max_tokens=50,
            temperature=0.5,
            raw=True,
            verbose=True,
        )

        mock_stats = GenerationStats(
            expert_idx=6,
            tokens_generated=25,
            layers_modified=8,
            moe_type="gpt_oss_batched",
            prompt_tokens=5,
        )
        mock_result = ExpertChatResult(
            prompt="Test",
            response="Response",
            expert_idx=6,
            stats=mock_stats,
        )

        mock_router = AsyncMock()
        mock_router.chat_with_expert = AsyncMock(return_value=mock_result)
        mock_router._moe_type = "gpt_oss_batched"
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.chat.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_chat(args)

            captured = capsys.readouterr()
            assert "Statistics:" in captured.out
            assert "Tokens generated: 25" in captured.out
