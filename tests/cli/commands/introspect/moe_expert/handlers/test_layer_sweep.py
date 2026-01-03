"""Tests for layer_sweep handler."""

from argparse import Namespace
from unittest.mock import AsyncMock, patch

import pytest

from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.layer_sweep import (
    _async_layer_sweep,
    handle_layer_sweep,
)
from chuk_lazarus.introspection.moe.enums import MoEArchitecture
from chuk_lazarus.introspection.moe.models import (
    CoactivationAnalysis,
    ExpertPair,
    MoEModelInfo,
)


class TestHandleLayerSweep:
    """Tests for handle_layer_sweep function."""

    def test_handle_layer_sweep_calls_asyncio_run(self):
        """Test that handle_layer_sweep calls asyncio.run."""
        args = Namespace(model="test/model")

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.layer_sweep.asyncio"
        ) as mock_asyncio:
            handle_layer_sweep(args)
            mock_asyncio.run.assert_called_once()


class TestAsyncLayerSweep:
    """Tests for _async_layer_sweep function."""

    @pytest.mark.asyncio
    async def test_successful_layer_sweep(self, capsys):
        """Test successful layer sweep."""
        args = Namespace(model="test/model")

        mock_info = MoEModelInfo(
            moe_layers=(0, 1, 2),
            num_experts=32,
            num_experts_per_tok=4,
            total_layers=4,
            architecture=MoEArchitecture.GPT_OSS,
        )

        mock_analysis = CoactivationAnalysis(
            layer_idx=0,
            total_activations=100,
            top_pairs=(
                ExpertPair(expert_a=6, expert_b=7, coactivation_count=25, coactivation_rate=0.25),
            ),
            generalist_experts=(6, 7),
        )

        mock_router = AsyncMock()
        mock_router.info = mock_info
        mock_router.analyze_coactivation = AsyncMock(return_value=mock_analysis)
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)

        with (
            patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.layer_sweep.ExpertRouter"
            ) as MockRouter,
            patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.layer_sweep.get_prompts_flat"
            ) as mock_prompts,
        ):
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)
            mock_prompts.return_value = [("cat", "prompt1"), ("cat", "prompt2")]

            await _async_layer_sweep(args)

            captured = capsys.readouterr()
            assert "LAYER SWEEP ANALYSIS" in captured.out
            assert "MoE layers: 3" in captured.out
            assert "L0" in captured.out

    @pytest.mark.asyncio
    async def test_layer_sweep_with_custom_prompts(self, capsys):
        """Test layer sweep with custom number of prompts."""
        args = Namespace(
            model="test/model",
            num_prompts=10,
        )

        mock_info = MoEModelInfo(
            moe_layers=(0,),
            num_experts=16,
            num_experts_per_tok=2,
            total_layers=1,
            architecture=MoEArchitecture.GPT_OSS,
        )

        mock_analysis = CoactivationAnalysis(
            layer_idx=0,
            total_activations=50,
            top_pairs=(),
            generalist_experts=(),
        )

        mock_router = AsyncMock()
        mock_router.info = mock_info
        mock_router.analyze_coactivation = AsyncMock(return_value=mock_analysis)
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)

        with (
            patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.layer_sweep.ExpertRouter"
            ) as MockRouter,
            patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.layer_sweep.get_prompts_flat"
            ) as mock_prompts,
        ):
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)
            mock_prompts.return_value = [("cat", f"prompt{i}") for i in range(100)]

            await _async_layer_sweep(args)

            captured = capsys.readouterr()
            assert "10 prompts" in captured.out
