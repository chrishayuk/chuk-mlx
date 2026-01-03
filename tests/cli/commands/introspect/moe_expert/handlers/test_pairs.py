"""Tests for pairs handler."""

from argparse import Namespace
from unittest.mock import AsyncMock, patch

import pytest

from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.pairs import (
    _async_pairs,
    handle_pairs,
)
from chuk_lazarus.introspection.moe.models import (
    ExpertChatResult,
    GenerationStats,
)


class TestHandlePairs:
    """Tests for handle_pairs function."""

    def test_handle_pairs_calls_asyncio_run(self):
        """Test that handle_pairs calls asyncio.run."""
        args = Namespace(
            model="test/model",
            experts="6,7",
            prompt="Test",
            max_tokens=100,
        )

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.pairs.asyncio"
        ) as mock_asyncio:
            handle_pairs(args)
            mock_asyncio.run.assert_called_once()


class TestAsyncPairs:
    """Tests for _async_pairs function."""

    @pytest.mark.asyncio
    async def test_missing_experts_prints_error(self, capsys):
        """Test that missing experts prints error."""
        args = Namespace(model="test/model", prompt="Test")

        await _async_pairs(args)

        captured = capsys.readouterr()
        assert "Error: --experts is required" in captured.out

    @pytest.mark.asyncio
    async def test_missing_prompt_prints_error(self, capsys):
        """Test that missing prompt prints error."""
        args = Namespace(model="test/model", experts="6,7")

        await _async_pairs(args)

        captured = capsys.readouterr()
        assert "Error: --prompt/-p is required" in captured.out

    @pytest.mark.asyncio
    async def test_invalid_experts_format_prints_error(self, capsys):
        """Test that invalid experts format prints error."""
        args = Namespace(
            model="test/model",
            experts="not,valid",
            prompt="Test",
        )

        await _async_pairs(args)

        captured = capsys.readouterr()
        assert "Error: Invalid experts format" in captured.out

    @pytest.mark.asyncio
    async def test_successful_pairs(self, capsys):
        """Test successful pairs execution."""
        args = Namespace(
            model="test/model",
            experts="6,7",
            prompt="Test prompt",
            max_tokens=100,
        )

        def make_result(expert_idx):
            stats = GenerationStats(
                expert_idx=expert_idx,
                tokens_generated=10,
                layers_modified=8,
                moe_type="gpt_oss_batched",
            )
            return ExpertChatResult(
                prompt="Test prompt",
                response=f"Response from {expert_idx}",
                expert_idx=expert_idx,
                stats=stats,
            )

        mock_router = AsyncMock()
        mock_router.chat_with_expert = AsyncMock(side_effect=[make_result(6), make_result(7)])
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.pairs.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_pairs(args)

            captured = capsys.readouterr()
            assert "EXPERT PAIRS TEST" in captured.out
            assert "Expert 6:" in captured.out
            assert "Expert 7:" in captured.out
