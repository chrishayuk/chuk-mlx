"""Tests for topk handler."""

from argparse import Namespace
from unittest.mock import AsyncMock, patch

import pytest

from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.topk import (
    _async_topk,
    handle_topk,
)
from chuk_lazarus.introspection.moe.models import TopKVariationResult


class TestHandleTopk:
    """Tests for handle_topk function."""

    def test_handle_topk_calls_asyncio_run(self):
        """Test that handle_topk calls asyncio.run."""
        args = Namespace(
            model="test/model",
            k=2,
            prompt="Test",
            max_tokens=100,
        )

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.topk.asyncio"
        ) as mock_asyncio:
            handle_topk(args)
            mock_asyncio.run.assert_called_once()


class TestAsyncTopk:
    """Tests for _async_topk function."""

    @pytest.mark.asyncio
    async def test_missing_k_prints_error(self, capsys):
        """Test that missing k prints error."""
        args = Namespace(model="test/model", prompt="Test")

        await _async_topk(args)

        captured = capsys.readouterr()
        assert "Error: --k is required" in captured.out

    @pytest.mark.asyncio
    async def test_missing_prompt_prints_error(self, capsys):
        """Test that missing prompt prints error."""
        args = Namespace(model="test/model", k=2)

        await _async_topk(args)

        captured = capsys.readouterr()
        assert "Error: --prompt/-p is required" in captured.out

    @pytest.mark.asyncio
    async def test_successful_topk(self, capsys):
        """Test successful topk execution."""
        args = Namespace(
            model="test/model",
            k=2,
            prompt="Test prompt",
            max_tokens=100,
        )

        mock_result = TopKVariationResult(
            prompt="Test prompt",
            k_value=2,
            default_k=4,
            response="Modified response",
            normal_response="Normal response",
        )

        mock_router = AsyncMock()
        mock_router.generate_with_topk = AsyncMock(return_value=mock_result)
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.topk.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_topk(args)

            captured = capsys.readouterr()
            assert "TOP-K EXPERIMENT" in captured.out
            assert "k=2" in captured.out
