"""Tests for weights handler."""

from argparse import Namespace
from unittest.mock import AsyncMock, patch

import pytest

from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.weights import (
    _async_weights,
    handle_weights,
)
from chuk_lazarus.introspection.moe.models import (
    LayerRouterWeights,
    RouterWeightCapture,
)


class TestHandleWeights:
    """Tests for handle_weights function."""

    def test_handle_weights_calls_asyncio_run(self):
        """Test that handle_weights calls asyncio.run."""
        args = Namespace(
            model="test/model",
            prompt="Test",
        )

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.weights.asyncio"
        ) as mock_asyncio:
            handle_weights(args)
            mock_asyncio.run.assert_called_once()


class TestAsyncWeights:
    """Tests for _async_weights function."""

    @pytest.mark.asyncio
    async def test_missing_prompt_prints_error(self, capsys):
        """Test that missing prompt prints error."""
        args = Namespace(model="test/model")

        await _async_weights(args)

        captured = capsys.readouterr()
        assert "Error: --prompt/-p is required" in captured.out

    @pytest.mark.asyncio
    async def test_successful_weights(self, capsys):
        """Test successful weights capture."""
        args = Namespace(
            model="test/model",
            prompt="Hello world",
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
                ),
            )
        ]

        mock_router = AsyncMock()
        mock_router.capture_router_weights = AsyncMock(return_value=mock_weights)
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.weights.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_weights(args)

            captured = capsys.readouterr()
            assert "ROUTER WEIGHTS" in captured.out
            assert "Layer 0" in captured.out

    @pytest.mark.asyncio
    async def test_weights_with_specific_layer(self, capsys):
        """Test weights capture with specific layer."""
        args = Namespace(
            model="test/model",
            prompt="Test",
            layer=2,
        )

        mock_weights = [
            LayerRouterWeights(
                layer_idx=2,
                positions=(
                    RouterWeightCapture(
                        layer_idx=2,
                        position_idx=0,
                        token="Test",
                        expert_indices=(6,),
                        weights=(1.0,),
                    ),
                ),
            )
        ]

        mock_router = AsyncMock()
        mock_router.capture_router_weights = AsyncMock(return_value=mock_weights)
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.weights.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_weights(args)

            # Check that layer 2 was requested
            mock_router.capture_router_weights.assert_called_once()
            call_args = mock_router.capture_router_weights.call_args
            assert call_args[1]["layers"] == [2]
