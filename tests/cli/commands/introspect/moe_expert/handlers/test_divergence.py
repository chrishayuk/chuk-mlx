"""Tests for divergence handler."""

from argparse import Namespace
from unittest.mock import AsyncMock, patch

import pytest

from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.divergence import (
    _async_divergence,
    handle_divergence,
)
from chuk_lazarus.introspection.moe.enums import MoEArchitecture
from chuk_lazarus.introspection.moe.models import (
    LayerRouterWeights,
    MoEModelInfo,
    RouterWeightCapture,
)


class TestHandleDivergence:
    """Tests for handle_divergence function."""

    def test_handle_divergence_calls_asyncio_run(self):
        """Test that handle_divergence calls asyncio.run."""
        args = Namespace(
            model="test/model",
            prompt="Test",
        )

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.divergence.asyncio"
        ) as mock_asyncio:
            handle_divergence(args)
            mock_asyncio.run.assert_called_once()


class TestAsyncDivergence:
    """Tests for _async_divergence function."""

    @pytest.mark.asyncio
    async def test_missing_prompt_prints_error(self, capsys):
        """Test that missing prompt prints error."""
        args = Namespace(model="test/model")

        await _async_divergence(args)

        captured = capsys.readouterr()
        assert "Error: --prompt/-p is required" in captured.out

    @pytest.mark.asyncio
    async def test_successful_divergence(self, capsys):
        """Test successful divergence analysis."""
        args = Namespace(
            model="test/model",
            prompt="Hello world",
        )

        mock_info = MoEModelInfo(
            moe_layers=(0, 1, 2),
            num_experts=32,
            num_experts_per_tok=4,
            total_layers=4,
            architecture=MoEArchitecture.GPT_OSS,
        )

        # Create weights for multiple layers to test divergence
        mock_weights = [
            LayerRouterWeights(
                layer_idx=0,
                positions=(
                    RouterWeightCapture(
                        layer_idx=0,
                        position_idx=0,
                        token="Hello",
                        expert_indices=(6, 7),
                        weights=(0.6, 0.4),
                    ),
                ),
            ),
            LayerRouterWeights(
                layer_idx=1,
                positions=(
                    RouterWeightCapture(
                        layer_idx=1,
                        position_idx=0,
                        token="Hello",
                        expert_indices=(6, 8),  # Different from layer 0
                        weights=(0.5, 0.5),
                    ),
                ),
            ),
        ]

        mock_router = AsyncMock()
        mock_router.info = mock_info
        mock_router.capture_router_weights = AsyncMock(return_value=mock_weights)
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.divergence.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_divergence(args)

            captured = capsys.readouterr()
            assert "LAYER DIVERGENCE ANALYSIS" in captured.out
            assert "Adjacent layer agreement" in captured.out
            assert "L0 -> L1" in captured.out
