"""Tests for role handler."""

from argparse import Namespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.role import (
    _async_role,
    handle_role,
)
from chuk_lazarus.introspection.moe.enums import MoEArchitecture
from chuk_lazarus.introspection.moe.models import (
    LayerRouterWeights,
    MoEModelInfo,
    RouterWeightCapture,
)


class TestHandleRole:
    """Tests for handle_role function."""

    def test_handle_role_calls_asyncio_run(self):
        """Test that handle_role calls asyncio.run."""
        args = Namespace(model="test/model")

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.role.asyncio"
        ) as mock_asyncio:
            handle_role(args)
            mock_asyncio.run.assert_called_once()


class TestAsyncRole:
    """Tests for _async_role function."""

    @pytest.mark.asyncio
    async def test_successful_role_analysis(self, capsys):
        """Test successful role analysis."""
        args = Namespace(model="test/model")

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
                        token="test",
                        expert_indices=(6, 7),
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

        # Mock prompts by group
        mock_cat_prompts = MagicMock()
        mock_cat_prompts.prompts = ["prompt1", "prompt2", "prompt3", "prompt4", "prompt5"]

        with (
            patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.role.ExpertRouter"
            ) as MockRouter,
            patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.role.get_prompts_by_group"
            ) as mock_get_prompts,
        ):
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)
            mock_get_prompts.return_value = [mock_cat_prompts]

            await _async_role(args)

            captured = capsys.readouterr()
            assert "LAYER ROLE ANALYSIS" in captured.out
            assert "Expert activation by category" in captured.out

    @pytest.mark.asyncio
    async def test_role_analysis_handles_exceptions(self, capsys):
        """Test role analysis handles exceptions during capture."""
        args = Namespace(model="test/model")

        mock_info = MoEModelInfo(
            moe_layers=(0,),
            num_experts=32,
            num_experts_per_tok=4,
            total_layers=1,
            architecture=MoEArchitecture.GPT_OSS,
        )

        mock_router = AsyncMock()
        mock_router.info = mock_info
        # Raise exception on capture
        mock_router.capture_router_weights = AsyncMock(side_effect=Exception("Test error"))
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)

        # Mock prompts by group
        mock_cat_prompts = MagicMock()
        mock_cat_prompts.prompts = ["prompt1", "prompt2", "prompt3", "prompt4", "prompt5"]

        with (
            patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.role.ExpertRouter"
            ) as MockRouter,
            patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.role.get_prompts_by_group"
            ) as mock_get_prompts,
        ):
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)
            mock_get_prompts.return_value = [mock_cat_prompts]

            await _async_role(args)

            # Should complete without crashing
            captured = capsys.readouterr()
            assert "LAYER ROLE ANALYSIS" in captured.out
