"""Tests for compare handler."""

from argparse import Namespace
from unittest.mock import AsyncMock, patch

import pytest

from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.compare import (
    _async_compare,
    handle_compare,
)
from chuk_lazarus.introspection.moe.models import (
    ExpertChatResult,
    ExpertComparisonResult,
    GenerationStats,
)


class TestHandleCompare:
    """Tests for handle_compare function."""

    def test_handle_compare_calls_asyncio_run(self):
        """Test that handle_compare calls asyncio.run."""
        args = Namespace(
            model="test/model",
            experts="6,7,20",
            prompt="Test",
            max_tokens=100,
            verbose=False,
        )

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.compare.asyncio"
        ) as mock_asyncio:
            handle_compare(args)
            mock_asyncio.run.assert_called_once()


class TestAsyncCompare:
    """Tests for _async_compare function."""

    @pytest.mark.asyncio
    async def test_missing_experts_prints_error(self, capsys):
        """Test that missing experts prints error."""
        args = Namespace(model="test/model", prompt="Test")

        await _async_compare(args)

        captured = capsys.readouterr()
        assert "Error: --experts is required" in captured.out

    @pytest.mark.asyncio
    async def test_missing_prompt_prints_error(self, capsys):
        """Test that missing prompt prints error."""
        args = Namespace(model="test/model", experts="6,7,20")

        await _async_compare(args)

        captured = capsys.readouterr()
        assert "Error: --prompt/-p is required" in captured.out

    @pytest.mark.asyncio
    async def test_invalid_experts_format_prints_error(self, capsys):
        """Test that invalid experts format prints error."""
        args = Namespace(
            model="test/model",
            experts="not,valid,numbers",
            prompt="Test",
        )

        await _async_compare(args)

        captured = capsys.readouterr()
        assert "Error: Invalid experts format" in captured.out

    @pytest.mark.asyncio
    async def test_single_expert_prints_error(self, capsys):
        """Test that single expert prints error."""
        args = Namespace(
            model="test/model",
            experts="6",
            prompt="Test",
        )

        await _async_compare(args)

        captured = capsys.readouterr()
        assert "At least 2 experts required" in captured.out

    @pytest.mark.asyncio
    async def test_successful_compare(self, capsys):
        """Test successful compare execution."""
        args = Namespace(
            model="test/model",
            experts="6,7,20",
            prompt="Test prompt",
            max_tokens=100,
            verbose=False,
        )

        # Create mock results
        results = []
        for expert_idx in [6, 7, 20]:
            stats = GenerationStats(
                expert_idx=expert_idx,
                tokens_generated=15,
                layers_modified=8,
                moe_type="gpt_oss_batched",
            )
            results.append(
                ExpertChatResult(
                    prompt="Test prompt",
                    response=f"Response from {expert_idx}",
                    expert_idx=expert_idx,
                    stats=stats,
                )
            )

        mock_result = ExpertComparisonResult(
            prompt="Test prompt",
            expert_results=tuple(results),
        )

        mock_router = AsyncMock()
        mock_router.compare_experts = AsyncMock(return_value=mock_result)
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.compare.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_compare(args)

            captured = capsys.readouterr()
            assert "EXPERT COMPARISON" in captured.out
            assert "Expert 6" in captured.out
            assert "Expert 7" in captured.out
            assert "Expert 20" in captured.out

    @pytest.mark.asyncio
    async def test_compare_with_spaces_in_experts(self, capsys):
        """Test compare with spaces in experts string."""
        args = Namespace(
            model="test/model",
            experts="6, 7, 20",  # Spaces after commas
            prompt="Test prompt",
            max_tokens=100,
            verbose=False,
        )

        results = []
        for expert_idx in [6, 7, 20]:
            stats = GenerationStats(
                expert_idx=expert_idx,
                tokens_generated=15,
                layers_modified=8,
                moe_type="gpt_oss_batched",
            )
            results.append(
                ExpertChatResult(
                    prompt="Test prompt",
                    response=f"Response from {expert_idx}",
                    expert_idx=expert_idx,
                    stats=stats,
                )
            )

        mock_result = ExpertComparisonResult(
            prompt="Test prompt",
            expert_results=tuple(results),
        )

        mock_router = AsyncMock()
        mock_router.compare_experts = AsyncMock(return_value=mock_result)
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.compare.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_compare(args)

            # Should have called compare_experts with correct indices
            mock_router.compare_experts.assert_called_once()
            call_args = mock_router.compare_experts.call_args
            assert call_args[0][1] == [6, 7, 20]  # Trimmed spaces
