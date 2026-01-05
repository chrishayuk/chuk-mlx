"""Tests for context_window handler."""

from argparse import Namespace
from unittest.mock import AsyncMock, patch

import pytest

from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.context_window import (
    _async_context_window,
    handle_context_window,
)
from chuk_lazarus.introspection.moe.enums import MoEArchitecture
from chuk_lazarus.introspection.moe.models import (
    LayerRouterWeights,
    MoEModelInfo,
    RouterWeightCapture,
)


class TestHandleContextWindow:
    """Tests for handle_context_window function."""

    def test_handle_context_window_calls_asyncio_run(self):
        """Test that handle_context_window calls asyncio.run."""
        args = Namespace(model="test/model")

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.context_window.asyncio"
        ) as mock_asyncio:
            handle_context_window(args)
            mock_asyncio.run.assert_called_once()


class TestAsyncContextWindow:
    """Tests for _async_context_window function."""

    @pytest.mark.asyncio
    async def test_successful_context_window(self, capsys):
        """Test successful context window test execution."""
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
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.context_window.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_context_window(args)

            captured = capsys.readouterr()
            # Should output context window test results
            assert "CONTEXT" in captured.out or "test/model" in captured.out

    @pytest.mark.asyncio
    async def test_context_window_with_layer(self, capsys):
        """Test context window with specific layer."""
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
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.context_window.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_context_window(args)

            captured = capsys.readouterr()
            assert "test/model" in captured.out or "Loading" in captured.out

    @pytest.mark.asyncio
    async def test_context_window_with_test_name(self, capsys):
        """Test context window with specific test name."""
        args = Namespace(model="test/model", test="arithmetic_plus")

        mock_info = MoEModelInfo(
            moe_layers=(0, 5, 10),
            num_experts=32,
            num_experts_per_tok=4,
            total_layers=12,
            architecture=MoEArchitecture.GPT_OSS,
        )

        mock_weights = [
            LayerRouterWeights(
                layer_idx=0,
                positions=(
                    RouterWeightCapture(
                        layer_idx=0,
                        position_idx=0,
                        token="+",
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
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.context_window.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_context_window(args)

            captured = capsys.readouterr()
            assert "arithmetic_plus" in captured.out.lower() or "ARITHMETIC" in captured.out

    @pytest.mark.asyncio
    async def test_context_window_stable_routing_all_layers(self, capsys):
        """Test context window when routing is stable (trigram sufficient)."""
        args = Namespace(model="test/model")

        mock_info = MoEModelInfo(
            moe_layers=(0, 5, 10),
            num_experts=32,
            num_experts_per_tok=4,
            total_layers=12,
            architecture=MoEArchitecture.GPT_OSS,
        )

        # All weights return same expert (stable routing)
        mock_weights = [
            LayerRouterWeights(
                layer_idx=0,
                positions=(
                    RouterWeightCapture(
                        layer_idx=0,
                        position_idx=0,
                        token="+",
                        expert_indices=(5,),
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
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.context_window.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_context_window(args)

            captured = capsys.readouterr()
            assert (
                "TRIGRAM" in captured.out
                or "STABLE" in captured.out
                or "sufficient" in captured.out.lower()
            )

    @pytest.mark.asyncio
    async def test_context_window_extended_context_matters(self, capsys):
        """Test context window when extended context matters (different experts)."""
        args = Namespace(model="test/model", test="arithmetic_plus")

        mock_info = MoEModelInfo(
            moe_layers=(0, 5, 10),
            num_experts=32,
            num_experts_per_tok=4,
            total_layers=12,
            architecture=MoEArchitecture.GPT_OSS,
        )

        call_count = [0]

        def varying_weights(*args, **kwargs):
            """Return different experts for different calls."""
            call_count[0] += 1
            expert = call_count[0] % 8  # Different expert each time
            return [
                LayerRouterWeights(
                    layer_idx=0,
                    positions=(
                        RouterWeightCapture(
                            layer_idx=0,
                            position_idx=0,
                            token="+",
                            expert_indices=(expert, expert + 1),
                            weights=(0.6, 0.4),
                        ),
                    ),
                )
            ]

        mock_router = AsyncMock()
        mock_router.info = mock_info
        mock_router.capture_router_weights = AsyncMock(side_effect=varying_weights)
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.context_window.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_context_window(args)

            captured = capsys.readouterr()
            # Should see variation in routing
            assert "VARIES" in captured.out or "EXTENDED" in captured.out or "MIXED" in captured.out

    @pytest.mark.asyncio
    async def test_context_window_empty_weights(self, capsys):
        """Test context window when no weights returned."""
        args = Namespace(model="test/model", test="arithmetic_plus")

        mock_info = MoEModelInfo(
            moe_layers=(0,),
            num_experts=32,
            num_experts_per_tok=4,
            total_layers=4,
            architecture=MoEArchitecture.GPT_OSS,
        )

        mock_router = AsyncMock()
        mock_router.info = mock_info
        mock_router.capture_router_weights = AsyncMock(return_value=[])
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.context_window.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_context_window(args)

            captured = capsys.readouterr()
            # Should still produce output without crashing
            assert "test/model" in captured.out or "CONTEXT" in captured.out

    @pytest.mark.asyncio
    async def test_context_window_context_increases_with_depth(self, capsys):
        """Test finding: context sensitivity increases with depth."""
        args = Namespace(model="test/model", test="arithmetic_plus")

        mock_info = MoEModelInfo(
            moe_layers=(0, 5, 10),
            num_experts=32,
            num_experts_per_tok=4,
            total_layers=12,
            architecture=MoEArchitecture.GPT_OSS,
        )

        call_info = {"count": 0}

        def layer_dependent_weights(*args, **kwargs):
            """Early layers: same expert. Late layers: different experts."""
            call_info["count"] += 1
            layer = kwargs.get("layers", [0])[0]

            # Early layer (0): all same expert (trigram sufficient)
            # Late layer (10): varying experts (extended context matters)
            if layer == 10:
                expert = call_info["count"] % 5  # Vary
            else:
                expert = 5  # Stable

            return [
                LayerRouterWeights(
                    layer_idx=layer,
                    positions=(
                        RouterWeightCapture(
                            layer_idx=layer,
                            position_idx=0,
                            token="+",
                            expert_indices=(expert,),
                            weights=(1.0,),
                        ),
                    ),
                )
            ]

        mock_router = AsyncMock()
        mock_router.info = mock_info
        mock_router.capture_router_weights = AsyncMock(side_effect=layer_dependent_weights)
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.context_window.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_context_window(args)

            captured = capsys.readouterr()
            # Should see conclusions about layer phases
            assert "CONCLUSION" in captured.out

    @pytest.mark.asyncio
    async def test_context_window_routing_stabilizes_with_depth(self, capsys):
        """Test finding: routing stabilizes with depth (early varies, late stable)."""
        args = Namespace(model="test/model", test="arithmetic_plus")

        mock_info = MoEModelInfo(
            moe_layers=(0, 5, 10),
            num_experts=32,
            num_experts_per_tok=4,
            total_layers=12,
            architecture=MoEArchitecture.GPT_OSS,
        )

        call_info = {"count": 0}

        def stabilizing_weights(*args, **kwargs):
            """Early layers: varying. Late layers: stable."""
            call_info["count"] += 1
            layer = kwargs.get("layers", [0])[0]

            # Early layer (0): varying experts (extended context matters)
            # Late layer (10): same expert (trigram sufficient)
            if layer == 0:
                expert = call_info["count"] % 5  # Vary
            else:
                expert = 5  # Stable

            return [
                LayerRouterWeights(
                    layer_idx=layer,
                    positions=(
                        RouterWeightCapture(
                            layer_idx=layer,
                            position_idx=0,
                            token="+",
                            expert_indices=(expert,),
                            weights=(1.0,),
                        ),
                    ),
                )
            ]

        mock_router = AsyncMock()
        mock_router.info = mock_info
        mock_router.capture_router_weights = AsyncMock(side_effect=stabilizing_weights)
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.context_window.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_context_window(args)

            captured = capsys.readouterr()
            assert "CONCLUSION" in captured.out

    @pytest.mark.asyncio
    async def test_context_window_mixed_verdict(self, capsys):
        """Test mixed verdict when some tests stable, some not."""
        args = Namespace(model="test/model")

        mock_info = MoEModelInfo(
            moe_layers=(0, 5, 10),
            num_experts=32,
            num_experts_per_tok=4,
            total_layers=12,
            architecture=MoEArchitecture.GPT_OSS,
        )

        call_info = {"count": 0}

        def mixed_weights(*args, **kwargs):
            """Some tests stable, some not - triggers MIXED verdict."""
            call_info["count"] += 1
            prompt = args[0] if args else ""

            # Arithmetic tests: vary
            # Other tests: stable
            if "+" in prompt or "2" in prompt:
                expert = call_info["count"] % 4  # Vary
            else:
                expert = 5  # Stable

            return [
                LayerRouterWeights(
                    layer_idx=0,
                    positions=(
                        RouterWeightCapture(
                            layer_idx=0,
                            position_idx=0,
                            token="test",
                            expert_indices=(expert,),
                            weights=(1.0,),
                        ),
                    ),
                )
            ]

        mock_router = AsyncMock()
        mock_router.info = mock_info
        mock_router.capture_router_weights = AsyncMock(side_effect=mixed_weights)
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.context_window.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_context_window(args)

            captured = capsys.readouterr()
            # Should still produce output
            assert "CONCLUSION" in captured.out or "Layer" in captured.out
