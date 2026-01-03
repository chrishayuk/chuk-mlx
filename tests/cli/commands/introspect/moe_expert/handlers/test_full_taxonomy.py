"""Tests for full_taxonomy handler."""

from argparse import Namespace
from unittest.mock import AsyncMock, patch

import pytest

from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.full_taxonomy import (
    _async_full_taxonomy,
    handle_full_taxonomy,
)
from chuk_lazarus.introspection.moe.datasets.prompts import PromptCategory
from chuk_lazarus.introspection.moe.enums import MoEArchitecture
from chuk_lazarus.introspection.moe.models import (
    LayerRouterWeights,
    MoEModelInfo,
    RouterWeightCapture,
)


class TestHandleFullTaxonomy:
    """Tests for handle_full_taxonomy function."""

    def test_handle_full_taxonomy_calls_asyncio_run(self):
        """Test that handle_full_taxonomy calls asyncio.run."""
        args = Namespace(model="test/model")

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.full_taxonomy.asyncio"
        ) as mock_asyncio:
            handle_full_taxonomy(args)
            mock_asyncio.run.assert_called_once()


class TestAsyncFullTaxonomy:
    """Tests for _async_full_taxonomy function."""

    @pytest.mark.asyncio
    async def test_successful_full_taxonomy(self, capsys):
        """Test successful full taxonomy generation."""
        args = Namespace(model="test/model")

        mock_info = MoEModelInfo(
            moe_layers=(0, 1),
            num_experts=8,
            num_experts_per_tok=2,
            total_layers=2,
            architecture=MoEArchitecture.GPT_OSS,
        )

        mock_weights = [
            LayerRouterWeights(
                layer_idx=0,
                positions=(
                    RouterWeightCapture(
                        layer_idx=0,
                        position_idx=0,
                        token="test",
                        expert_indices=(0, 1),
                        weights=(0.6, 0.4),
                    ),
                ),
            )
        ]

        mock_router = AsyncMock()
        mock_router.info = mock_info
        mock_router.capture_router_weights = AsyncMock(return_value=mock_weights)
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)

        # Mock prompts data - use ARITHMETIC which exists in PromptCategory
        mock_prompts = [(PromptCategory.ARITHMETIC, "1+1=")]

        with (
            patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.full_taxonomy.ExpertRouter"
            ) as MockRouter,
            patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.full_taxonomy.get_prompts_flat"
            ) as mock_get_prompts,
        ):
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)
            mock_get_prompts.return_value = mock_prompts

            await _async_full_taxonomy(args)

            captured = capsys.readouterr()
            assert "EXPERT TAXONOMY" in captured.out
            assert "test/model" in captured.out

    @pytest.mark.asyncio
    async def test_full_taxonomy_with_verbose(self, capsys):
        """Test full taxonomy with verbose output."""
        args = Namespace(
            model="test/model",
            verbose=True,
            num_prompts=5,
        )

        mock_info = MoEModelInfo(
            moe_layers=(0,),
            num_experts=4,
            num_experts_per_tok=2,
            total_layers=1,
            architecture=MoEArchitecture.GPT_OSS,
        )

        mock_weights = [
            LayerRouterWeights(
                layer_idx=0,
                positions=(
                    RouterWeightCapture(
                        layer_idx=0,
                        position_idx=0,
                        token="math",
                        expert_indices=(0,),
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

        mock_prompts = [(PromptCategory.ARITHMETIC, f"prompt{i}") for i in range(5)]

        with (
            patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.full_taxonomy.ExpertRouter"
            ) as MockRouter,
            patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.full_taxonomy.get_prompts_flat"
            ) as mock_get_prompts,
        ):
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)
            mock_get_prompts.return_value = mock_prompts

            await _async_full_taxonomy(args)

            captured = capsys.readouterr()
            assert "EXPERT TAXONOMY" in captured.out
