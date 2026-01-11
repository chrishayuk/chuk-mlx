"""Tests for analyze handler."""

from argparse import Namespace
from unittest.mock import AsyncMock, patch

import pytest

from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.analyze import (
    _async_analyze,
    handle_analyze,
)
from chuk_lazarus.introspection.moe.enums import MoEArchitecture
from chuk_lazarus.introspection.moe.models import (
    CoactivationAnalysis,
    ExpertPair,
    MoEModelInfo,
)


class TestHandleAnalyze:
    """Tests for handle_analyze function."""

    def test_handle_analyze_calls_asyncio_run(self):
        """Test that handle_analyze calls asyncio.run."""
        args = Namespace(model="test/model")

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.analyze.asyncio"
        ) as mock_asyncio:
            handle_analyze(args)
            mock_asyncio.run.assert_called_once()


class TestAsyncAnalyze:
    """Tests for _async_analyze function."""

    @pytest.mark.asyncio
    async def test_successful_analyze(self, capsys):
        """Test successful analyze execution."""
        args = Namespace(
            model="test/model",
        )

        mock_info = MoEModelInfo(
            moe_layers=(0, 1, 2, 3, 4, 5, 6, 7),
            num_experts=32,
            num_experts_per_tok=4,
            total_layers=8,
            architecture=MoEArchitecture.GPT_OSS,
        )

        mock_analysis = CoactivationAnalysis(
            layer_idx=0,
            total_activations=100,
            top_pairs=(
                ExpertPair(
                    expert_a=6,
                    expert_b=7,
                    coactivation_count=25,
                    coactivation_rate=0.25,
                ),
                ExpertPair(
                    expert_a=6,
                    expert_b=20,
                    coactivation_count=15,
                    coactivation_rate=0.15,
                ),
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
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.analyze.ExpertRouter"
            ) as MockRouter,
            patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.analyze.get_prompts_flat"
            ) as mock_prompts,
        ):
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)
            mock_prompts.return_value = [("cat", "prompt1"), ("cat", "prompt2")]

            await _async_analyze(args)

            captured = capsys.readouterr()
            assert "EXPERT ROUTING ANALYSIS" in captured.out
            assert "gpt_oss" in captured.out
            assert "32 per layer" in captured.out

    @pytest.mark.asyncio
    async def test_analyze_with_specific_layer(self, capsys):
        """Test analyze with specific layer."""
        args = Namespace(
            model="test/model",
            layer=3,
            num_prompts=10,
        )

        mock_info = MoEModelInfo(
            moe_layers=(0, 1, 2, 3, 4, 5, 6, 7),
            num_experts=32,
            num_experts_per_tok=4,
            total_layers=8,
            architecture=MoEArchitecture.GPT_OSS,
        )

        mock_analysis = CoactivationAnalysis(
            layer_idx=3,
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
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.analyze.ExpertRouter"
            ) as MockRouter,
            patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.analyze.get_prompts_flat"
            ) as mock_prompts,
        ):
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)
            mock_prompts.return_value = [("cat", f"prompt{i}") for i in range(10)]

            await _async_analyze(args)

            # Verify layer 3 was used
            mock_router.analyze_coactivation.assert_called_once()
            call_args = mock_router.analyze_coactivation.call_args
            assert call_args[1]["layer_idx"] == 3
