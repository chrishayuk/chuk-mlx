"""Tests for entropy handler."""

from argparse import Namespace
from unittest.mock import AsyncMock, patch

import pytest

from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.entropy import (
    _async_entropy,
    handle_entropy,
)
from chuk_lazarus.introspection.moe.enums import MoEArchitecture
from chuk_lazarus.introspection.moe.models import (
    LayerRouterWeights,
    MoEModelInfo,
    RouterWeightCapture,
)


class TestHandleEntropy:
    """Tests for handle_entropy function."""

    def test_handle_entropy_calls_asyncio_run(self):
        """Test that handle_entropy calls asyncio.run."""
        args = Namespace(
            model="test/model",
            prompt="Test",
        )

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.entropy.asyncio"
        ) as mock_asyncio:
            handle_entropy(args)
            mock_asyncio.run.assert_called_once()


class TestAsyncEntropy:
    """Tests for _async_entropy function."""

    @pytest.mark.asyncio
    async def test_missing_prompt_prints_error(self, capsys):
        """Test that missing prompt prints error."""
        args = Namespace(model="test/model")

        await _async_entropy(args)

        captured = capsys.readouterr()
        assert "Error: --prompt/-p is required" in captured.out

    @pytest.mark.asyncio
    async def test_successful_entropy_analysis(self, capsys):
        """Test successful entropy analysis."""
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
        mock_router.info = mock_info
        mock_router.capture_router_weights = AsyncMock(return_value=mock_weights)
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.entropy.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_entropy(args)

            captured = capsys.readouterr()
            assert "ROUTING ENTROPY ANALYSIS" in captured.out
            assert "Layer" in captured.out
            assert "Mean Entropy" in captured.out
