"""Tests for pattern_discovery handler."""

from argparse import Namespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.pattern_discovery import (
    _async_pattern_discovery,
    handle_pattern_discovery,
)
from chuk_lazarus.introspection.moe.enums import MoEArchitecture
from chuk_lazarus.introspection.moe.models import (
    LayerRouterWeights,
    MoEModelInfo,
    RouterWeightCapture,
)


class TestHandlePatternDiscovery:
    """Tests for handle_pattern_discovery function."""

    def test_handle_pattern_discovery_calls_asyncio_run(self):
        """Test that handle_pattern_discovery calls asyncio.run."""
        args = Namespace(model="test/model")

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.pattern_discovery.asyncio"
        ) as mock_asyncio:
            handle_pattern_discovery(args)
            mock_asyncio.run.assert_called_once()


class TestAsyncPatternDiscovery:
    """Tests for _async_pattern_discovery function."""

    @pytest.mark.asyncio
    async def test_successful_pattern_discovery(self, capsys):
        """Test successful pattern discovery."""
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
                        token="123",
                        expert_indices=(6, 7),
                        weights=(0.6, 0.4),
                    ),
                ),
            )
        ]

        # Mock pattern data
        mock_patterns_data = MagicMock()
        mock_patterns_data.get_category_names.return_value = ["numeric", "alpha"]

        mock_category = MagicMock()
        mock_category.prompts = ["123", "456"]
        mock_patterns_data.get_category.return_value = mock_category

        mock_router = AsyncMock()
        mock_router.info = mock_info
        mock_router.capture_router_weights = AsyncMock(return_value=mock_weights)
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)

        with (
            patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.pattern_discovery.ExpertRouter"
            ) as MockRouter,
            patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.pattern_discovery.get_pattern_discovery_prompts"
            ) as mock_get_patterns,
        ):
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)
            mock_get_patterns.return_value = mock_patterns_data

            await _async_pattern_discovery(args)

            captured = capsys.readouterr()
            assert "EXPERT PATTERN DISCOVERY" in captured.out
            assert "Pattern-Expert associations" in captured.out
