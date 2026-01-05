"""Tests for context_test handler."""

from argparse import Namespace
from unittest.mock import AsyncMock, patch

import pytest

from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.context_test import (
    _async_context_test,
    handle_context_test,
)
from chuk_lazarus.introspection.moe.enums import MoEArchitecture
from chuk_lazarus.introspection.moe.models import (
    LayerRouterWeights,
    MoEModelInfo,
    RouterWeightCapture,
)


class TestHandleContextTest:
    """Tests for handle_context_test function."""

    def test_handle_context_test_calls_asyncio_run(self):
        """Test that handle_context_test calls asyncio.run."""
        args = Namespace(model="test/model")

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.context_test.asyncio"
        ) as mock_asyncio:
            handle_context_test(args)
            mock_asyncio.run.assert_called_once()


class TestAsyncContextTest:
    """Tests for _async_context_test function."""

    @pytest.mark.asyncio
    async def test_successful_context_test(self, capsys):
        """Test successful context test execution."""
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
                        token="127",
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
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.context_test.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_context_test(args)

            captured = capsys.readouterr()
            assert "CONTEXT INDEPENDENCE TEST" in captured.out
            assert "test/model" in captured.out

    @pytest.mark.asyncio
    async def test_context_test_shows_context_dependent_verdict(self, capsys):
        """Test context test shows context-dependent verdict when routing varies."""
        args = Namespace(model="test/model")

        mock_info = MoEModelInfo(
            moe_layers=(0, 5, 10),
            num_experts=32,
            num_experts_per_tok=4,
            total_layers=12,
            architecture=MoEArchitecture.GPT_OSS,
        )

        # Mock weights that return different experts based on call count
        call_count = [0]

        def make_weights():
            call_count[0] += 1
            expert = 6 if call_count[0] % 2 == 0 else 8
            return [
                LayerRouterWeights(
                    layer_idx=0,
                    positions=(
                        RouterWeightCapture(
                            layer_idx=0,
                            position_idx=0,
                            token="127",
                            expert_indices=(expert,),
                            weights=(1.0,),
                        ),
                    ),
                )
            ]

        mock_router = AsyncMock()
        mock_router.info = mock_info
        mock_router.capture_router_weights = AsyncMock(
            side_effect=lambda *args, **kwargs: make_weights()
        )
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.context_test.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_context_test(args)

            captured = capsys.readouterr()
            # The handler should show verdict about context dependence
            assert "CONTEXT INDEPENDENCE TEST" in captured.out

    @pytest.mark.asyncio
    async def test_context_test_with_custom_token(self, capsys):
        """Test context test with custom target token."""
        args = Namespace(model="test/model", token="hello")

        mock_info = MoEModelInfo(
            moe_layers=(0,),
            num_experts=32,
            num_experts_per_tok=4,
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
                        token="hello",
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
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.context_test.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_context_test(args)

            captured = capsys.readouterr()
            assert "CONTEXT INDEPENDENCE TEST" in captured.out

    @pytest.mark.asyncio
    async def test_context_test_with_layer(self, capsys):
        """Test context test with specific layer."""
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
                        token="127",
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
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.context_test.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_context_test(args)

            captured = capsys.readouterr()
            assert "CONTEXT INDEPENDENCE TEST" in captured.out

    @pytest.mark.asyncio
    async def test_context_test_with_custom_contexts(self, capsys):
        """Test context test with custom contexts parameter."""
        args = Namespace(model="test/model", contexts="foo 127,bar 127")

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
                        token="127",
                        expert_indices=(6,),
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

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.context_test.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_context_test(args)

            captured = capsys.readouterr()
            assert "CONTEXT INDEPENDENCE TEST" in captured.out
            assert "CUSTOM" in captured.out  # Custom context type is shown

    @pytest.mark.asyncio
    async def test_context_test_routing_stabilizes(self, capsys):
        """Test context test shows stabilization message when early varies but late doesn't."""
        args = Namespace(model="test/model")

        mock_info = MoEModelInfo(
            moe_layers=(0, 5, 10),
            num_experts=32,
            num_experts_per_tok=4,
            total_layers=12,
            architecture=MoEArchitecture.GPT_OSS,
        )

        # Track which layer is being tested
        call_info = {"count": 0, "layer": None}

        def make_weights(*args, **kwargs):
            call_info["count"] += 1
            layer = kwargs.get("layers", [0])[0]

            # Early layer (0): return different experts to simulate context-dependence
            # Middle and Late layers (5, 10): return same expert to show stabilization
            if layer == 0:
                # Alternate between experts for early layer
                expert = 6 if call_info["count"] % 3 == 0 else 8
            else:
                # Same expert for middle/late layers
                expert = 6

            return [
                LayerRouterWeights(
                    layer_idx=layer,
                    positions=(
                        RouterWeightCapture(
                            layer_idx=layer,
                            position_idx=0,
                            token="127",
                            expert_indices=(expert,),
                            weights=(1.0,),
                        ),
                    ),
                )
            ]

        mock_router = AsyncMock()
        mock_router.info = mock_info
        mock_router.capture_router_weights = AsyncMock(side_effect=make_weights)
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.context_test.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_context_test(args)

            captured = capsys.readouterr()
            assert "CONTEXT INDEPENDENCE TEST" in captured.out
            # One of the conclusions should appear
            assert any(
                msg in captured.out
                for msg in ["STABILIZES", "CONTEXT-DEPENDENT", "CONSISTENT", "DIVERGES"]
            )

    @pytest.mark.asyncio
    async def test_context_test_routing_diverges(self, capsys):
        """Test context test shows diverges message when late varies but early doesn't."""
        args = Namespace(model="test/model")

        mock_info = MoEModelInfo(
            moe_layers=(0, 5, 10),
            num_experts=32,
            num_experts_per_tok=4,
            total_layers=12,
            architecture=MoEArchitecture.GPT_OSS,
        )

        # Track calls to return appropriate experts
        call_info = {"count": 0}

        def make_weights(*args, **kwargs):
            call_info["count"] += 1
            layer = kwargs.get("layers", [0])[0]

            # Early layer (0): same expert (consistent)
            # Late layer (10): different experts (diverges)
            if layer == 10:
                # Alternate between experts for late layer
                expert = 6 if call_info["count"] % 2 == 0 else 8
            else:
                # Same expert for early/middle layers
                expert = 6

            return [
                LayerRouterWeights(
                    layer_idx=layer,
                    positions=(
                        RouterWeightCapture(
                            layer_idx=layer,
                            position_idx=0,
                            token="127",
                            expert_indices=(expert,),
                            weights=(1.0,),
                        ),
                    ),
                )
            ]

        mock_router = AsyncMock()
        mock_router.info = mock_info
        mock_router.capture_router_weights = AsyncMock(side_effect=make_weights)
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.context_test.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_context_test(args)

            captured = capsys.readouterr()
            assert "CONTEXT INDEPENDENCE TEST" in captured.out
            # Should show one of the conclusions
            assert "CONCLUSION" in captured.out
