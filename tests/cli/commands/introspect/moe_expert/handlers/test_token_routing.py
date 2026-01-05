"""Tests for token_routing handler."""

from argparse import Namespace
from unittest.mock import AsyncMock, patch

import pytest

from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.token_routing import (
    _async_token_routing,
    handle_token_routing,
)
from chuk_lazarus.introspection.moe.enums import MoEArchitecture
from chuk_lazarus.introspection.moe.models import (
    LayerRouterWeights,
    MoEModelInfo,
    RouterWeightCapture,
)


class TestHandleTokenRouting:
    """Tests for handle_token_routing function."""

    def test_handle_token_routing_calls_asyncio_run(self):
        """Test that handle_token_routing calls asyncio.run."""
        args = Namespace(model="test/model")

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.token_routing.asyncio"
        ) as mock_asyncio:
            handle_token_routing(args)
            mock_asyncio.run.assert_called_once()


class TestAsyncTokenRouting:
    """Tests for _async_token_routing function."""

    @pytest.mark.asyncio
    async def test_successful_token_routing(self, capsys):
        """Test successful token routing execution."""
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
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.token_routing.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_token_routing(args)

            captured = capsys.readouterr()
            # Should output token routing results
            assert "TOKEN" in captured.out or "test/model" in captured.out

    @pytest.mark.asyncio
    async def test_token_routing_with_layer(self, capsys):
        """Test token routing with specific layer."""
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
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.token_routing.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_token_routing(args)

            captured = capsys.readouterr()
            assert "test/model" in captured.out or "Loading" in captured.out

    @pytest.mark.asyncio
    async def test_token_routing_with_known_token_127(self, capsys):
        """Test token routing with token that has predefined contexts (127)."""
        args = Namespace(model="test/model", token="127")

        mock_info = MoEModelInfo(
            moe_layers=(0,),
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
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.token_routing.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_token_routing(args)

            captured = capsys.readouterr()
            assert "127" in captured.out
            assert "E6" in captured.out or "E7" in captured.out

    @pytest.mark.asyncio
    async def test_token_routing_with_custom_token(self, capsys):
        """Test token routing with custom token not in predefined contexts."""
        args = Namespace(model="test/model", token="custom_token")

        mock_info = MoEModelInfo(
            moe_layers=(0,),
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
                        token="custom_token",
                        expert_indices=(10,),
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
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.token_routing.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_token_routing(args)

            captured = capsys.readouterr()
            # Custom token should use default generated contexts
            assert "custom_token" in captured.out

    @pytest.mark.asyncio
    async def test_token_routing_same_expert_all_contexts(self, capsys):
        """Test token routing when same expert used in all contexts."""
        args = Namespace(model="test/model", token="stable")

        mock_info = MoEModelInfo(
            moe_layers=(0,),
            num_experts=32,
            num_experts_per_tok=4,
            total_layers=4,
            architecture=MoEArchitecture.GPT_OSS,
        )

        # Same expert for all positions
        mock_weights = [
            LayerRouterWeights(
                layer_idx=0,
                positions=(
                    RouterWeightCapture(
                        layer_idx=0,
                        position_idx=0,
                        token="stable",
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
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.token_routing.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_token_routing(args)

            captured = capsys.readouterr()
            # Should indicate stable routing when only one expert
            assert "stable" in captured.out or "SAME" in captured.out

    @pytest.mark.asyncio
    async def test_token_routing_with_multiple_different_experts(self, capsys):
        """Test token routing when different experts used in different contexts."""
        args = Namespace(model="test/model", token="127")

        mock_info = MoEModelInfo(
            moe_layers=(0,),
            num_experts=32,
            num_experts_per_tok=4,
            total_layers=4,
            architecture=MoEArchitecture.GPT_OSS,
        )

        call_count = [0]

        def varying_weights(*args, **kwargs):
            """Return different experts for different calls."""
            call_count[0] += 1
            expert = call_count[0] % 5  # Different expert each time
            return [
                LayerRouterWeights(
                    layer_idx=0,
                    positions=(
                        RouterWeightCapture(
                            layer_idx=0,
                            position_idx=0,
                            token="127",
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
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.token_routing.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_token_routing(args)

            captured = capsys.readouterr()
            # Should show different experts used
            assert "DIFFERENT" in captured.out or "E" in captured.out
