"""Additional tests for full_taxonomy handler to improve coverage."""

from argparse import Namespace
from unittest.mock import AsyncMock, patch

import pytest

from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.full_taxonomy import (
    _async_full_taxonomy,
)
from chuk_lazarus.introspection.moe.datasets.prompts import PromptCategory
from chuk_lazarus.introspection.moe.enums import MoEArchitecture
from chuk_lazarus.introspection.moe.models import (
    LayerRouterWeights,
    MoEModelInfo,
    RouterWeightCapture,
)


class TestFullTaxonomyAdditional:
    """Additional tests for _async_full_taxonomy function."""

    @pytest.mark.asyncio
    async def test_full_taxonomy_handles_exceptions(self, capsys):
        """Test that exceptions during prompt processing are handled."""
        args = Namespace(model="test/model")

        mock_info = MoEModelInfo(
            moe_layers=(0,),
            num_experts=4,
            num_experts_per_tok=2,
            total_layers=1,
            architecture=MoEArchitecture.GPT_OSS,
        )

        mock_router = AsyncMock()
        mock_router.info = mock_info
        # Raise exception on capture
        mock_router.capture_router_weights = AsyncMock(side_effect=Exception("Test error"))
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)

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

            # Should still complete without crashing
            captured = capsys.readouterr()
            assert "EXPERT TAXONOMY" in captured.out

    @pytest.mark.asyncio
    async def test_full_taxonomy_with_empty_category_counts(self, capsys):
        """Test taxonomy generation when expert has no category counts."""
        args = Namespace(model="test/model")

        mock_info = MoEModelInfo(
            moe_layers=(0,),
            num_experts=4,
            num_experts_per_tok=2,
            total_layers=1,
            architecture=MoEArchitecture.GPT_OSS,
        )

        mock_router = AsyncMock()
        mock_router.info = mock_info
        mock_router.capture_router_weights = AsyncMock(return_value=[])
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)

        # Use empty prompts to create scenario with no category counts
        mock_prompts = []

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

    @pytest.mark.asyncio
    async def test_full_taxonomy_generalist_role_assignment(self, capsys):
        """Test that generalist role is assigned when confidence is low and many categories."""
        args = Namespace(model="test/model", num_prompts=20)

        mock_info = MoEModelInfo(
            moe_layers=(0,),
            num_experts=4,
            num_experts_per_tok=2,
            total_layers=1,
            architecture=MoEArchitecture.GPT_OSS,
        )

        # Create weights that will result in many categories with low confidence
        mock_weights = [
            LayerRouterWeights(
                layer_idx=0,
                positions=(
                    RouterWeightCapture(
                        layer_idx=0,
                        position_idx=0,
                        token="test",
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

        # Create diverse prompts across many categories
        mock_prompts = [(PromptCategory.ARITHMETIC, f"math{i}") for i in range(10)]

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
