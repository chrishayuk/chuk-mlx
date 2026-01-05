"""Tests for domain_test handler."""

from argparse import Namespace
from unittest.mock import AsyncMock, patch

import pytest

from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.domain_test import (
    _async_domain_test,
    handle_domain_test,
)
from chuk_lazarus.introspection.moe.enums import MoEArchitecture
from chuk_lazarus.introspection.moe.models import (
    LayerRouterWeights,
    MoEModelInfo,
    RouterWeightCapture,
)


class TestHandleDomainTest:
    """Tests for handle_domain_test function."""

    def test_handle_domain_test_calls_asyncio_run(self):
        """Test that handle_domain_test calls asyncio.run."""
        args = Namespace(model="test/model")

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.domain_test.asyncio"
        ) as mock_asyncio:
            handle_domain_test(args)
            mock_asyncio.run.assert_called_once()


class TestAsyncDomainTest:
    """Tests for _async_domain_test function."""

    @pytest.mark.asyncio
    async def test_successful_domain_test(self, capsys):
        """Test successful domain test execution."""
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

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.domain_test.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_domain_test(args)

            captured = capsys.readouterr()
            # Should output domain test results
            assert "DOMAIN" in captured.out or "test/model" in captured.out

    @pytest.mark.asyncio
    async def test_domain_test_with_layer(self, capsys):
        """Test domain test with specific layer."""
        args = Namespace(model="test/model", layer=5)

        mock_info = MoEModelInfo(
            moe_layers=(0, 5),
            num_experts=32,
            num_experts_per_tok=4,
            total_layers=10,
            architecture=MoEArchitecture.GPT_OSS,
        )

        mock_weights = [
            LayerRouterWeights(
                layer_idx=5,
                positions=(
                    RouterWeightCapture(
                        layer_idx=5,
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

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.domain_test.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_domain_test(args)

            captured = capsys.readouterr()
            assert "test/model" in captured.out or "Loading" in captured.out
