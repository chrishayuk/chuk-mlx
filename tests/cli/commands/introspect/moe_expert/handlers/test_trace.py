"""Tests for trace handler."""

from argparse import Namespace
from unittest.mock import AsyncMock, patch

import pytest

from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.trace import (
    _async_trace,
    handle_trace,
)
from chuk_lazarus.introspection.moe.enums import MoEArchitecture
from chuk_lazarus.introspection.moe.models import (
    LayerRouterWeights,
    MoEModelInfo,
    RouterWeightCapture,
)


class TestHandleTrace:
    """Tests for handle_trace function."""

    def test_handle_trace_calls_asyncio_run(self):
        """Test that handle_trace calls asyncio.run."""
        args = Namespace(
            model="test/model",
            prompt="Test",
        )

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.trace.asyncio"
        ) as mock_asyncio:
            handle_trace(args)
            mock_asyncio.run.assert_called_once()


class TestAsyncTrace:
    """Tests for _async_trace function."""

    @pytest.mark.asyncio
    async def test_missing_prompt_prints_error(self, capsys):
        """Test that missing prompt prints error."""
        args = Namespace(model="test/model")

        await _async_trace(args)

        captured = capsys.readouterr()
        assert "Error: --prompt/-p is required" in captured.out

    @pytest.mark.asyncio
    async def test_successful_trace(self, capsys):
        """Test successful trace execution."""
        args = Namespace(
            model="test/model",
            prompt="Hello world",
        )

        mock_info = MoEModelInfo(
            moe_layers=(0, 1, 2, 3, 4, 5, 6, 7),
            num_experts=32,
            num_experts_per_tok=4,
            total_layers=8,
            architecture=MoEArchitecture.GPT_OSS,
        )

        mock_weights = [
            LayerRouterWeights(
                layer_idx=0,
                positions=(
                    RouterWeightCapture(
                        layer_idx=0,
                        position_idx=0,
                        token="Hello",
                        expert_indices=(6, 7, 20, 1),
                        weights=(0.4, 0.3, 0.2, 0.1),
                    ),
                    RouterWeightCapture(
                        layer_idx=0,
                        position_idx=1,
                        token=" world",
                        expert_indices=(7, 6, 15, 3),
                        weights=(0.35, 0.3, 0.2, 0.15),
                    ),
                ),
            )
        ]

        mock_router = AsyncMock()
        mock_router.info = mock_info
        mock_router.capture_router_weights = AsyncMock(return_value=mock_weights)
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.trace.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_trace(args)

            captured = capsys.readouterr()
            assert "TOKEN-EXPERT TRACE" in captured.out
            assert "Layer 0:" in captured.out
            assert "Hello" in captured.out
            assert "E6" in captured.out

    @pytest.mark.asyncio
    async def test_trace_with_specific_layer(self, capsys):
        """Test trace with specific layer."""
        args = Namespace(
            model="test/model",
            prompt="Test",
            layer=3,
        )

        mock_info = MoEModelInfo(
            moe_layers=(0, 1, 2, 3, 4, 5, 6, 7),
            num_experts=32,
            num_experts_per_tok=4,
            total_layers=8,
            architecture=MoEArchitecture.GPT_OSS,
        )

        mock_weights = [
            LayerRouterWeights(
                layer_idx=3,
                positions=(
                    RouterWeightCapture(
                        layer_idx=3,
                        position_idx=0,
                        token="Test",
                        expert_indices=(10,),
                        weights=(1.0,),
                    ),
                ),
            )
        ]

        mock_router = AsyncMock()
        mock_router.info = mock_info
        mock_router.capture_router_weights = AsyncMock(return_value=mock_weights)
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.trace.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_trace(args)

            # Verify specific layer was requested
            mock_router.capture_router_weights.assert_called_once()
            call_args = mock_router.capture_router_weights.call_args
            assert call_args[1]["layers"] == [3]
