"""Tests for router_probe handler."""

from argparse import Namespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.router_probe import (
    _async_router_probe,
    handle_router_probe,
)
from chuk_lazarus.introspection.moe.enums import MoEArchitecture
from chuk_lazarus.introspection.moe.models import (
    LayerRouterWeights,
    MoEModelInfo,
    RouterWeightCapture,
)


class TestHandleRouterProbe:
    """Tests for handle_router_probe function."""

    def test_handle_router_probe_calls_asyncio_run(self):
        """Test that handle_router_probe calls asyncio.run."""
        args = Namespace(model="test/model")

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.router_probe.asyncio"
        ) as mock_asyncio:
            handle_router_probe(args)
            mock_asyncio.run.assert_called_once()


class TestAsyncRouterProbe:
    """Tests for _async_router_probe function."""

    @pytest.mark.asyncio
    async def test_successful_router_probe(self, capsys):
        """Test successful router probe."""
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
                        expert_indices=(6, 7, 20),
                        weights=(0.5, 0.3, 0.2),
                    ),
                ),
            )
        ]

        # Mock test data
        mock_test_data = MagicMock()
        mock_test = MagicMock()
        mock_test.prompt = "test prompt"
        mock_test.context_type = "neutral"
        mock_test_data.tests = [mock_test]

        mock_router = AsyncMock()
        mock_router.info = mock_info
        mock_router.capture_router_weights = AsyncMock(return_value=mock_weights)
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)

        with (
            patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.router_probe.ExpertRouter"
            ) as MockRouter,
            patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.router_probe.get_context_tests"
            ) as mock_get_tests,
        ):
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)
            mock_get_tests.return_value = mock_test_data

            await _async_router_probe(args)

            captured = capsys.readouterr()
            assert "ROUTER INPUT DECOMPOSITION" in captured.out
            assert "test/model" in captured.out
