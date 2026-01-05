"""Tests for heatmap handler."""

from argparse import Namespace
from unittest.mock import AsyncMock, patch

import pytest

from chuk_lazarus.cli.commands.introspect.moe_expert.handlers.heatmap import (
    _async_heatmap,
    handle_heatmap,
)
from chuk_lazarus.introspection.moe.enums import MoEArchitecture
from chuk_lazarus.introspection.moe.models import (
    LayerRouterWeights,
    MoEModelInfo,
    RouterWeightCapture,
)


class TestHandleHeatmap:
    """Tests for handle_heatmap function."""

    def test_handle_heatmap_calls_asyncio_run(self):
        """Test that handle_heatmap calls asyncio.run."""
        args = Namespace(model="test/model", prompt="test")

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.heatmap.asyncio"
        ) as mock_asyncio:
            handle_heatmap(args)
            mock_asyncio.run.assert_called_once()


class TestAsyncHeatmap:
    """Tests for _async_heatmap function."""

    @pytest.mark.asyncio
    async def test_successful_heatmap(self, capsys):
        """Test successful heatmap generation."""
        args = Namespace(model="test/model", prompt="test prompt")

        mock_info = MoEModelInfo(
            moe_layers=(0, 1, 2),
            num_experts=8,
            num_experts_per_tok=2,
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
                        expert_indices=(0, 1),
                        weights=(0.6, 0.4),
                    ),
                    RouterWeightCapture(
                        layer_idx=0,
                        position_idx=1,
                        token="prompt",
                        expert_indices=(2, 3),
                        weights=(0.7, 0.3),
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
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.heatmap.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_heatmap(args)

            captured = capsys.readouterr()
            # Should output heatmap results
            assert "HEATMAP" in captured.out or "test/model" in captured.out

    @pytest.mark.asyncio
    async def test_heatmap_with_layer(self, capsys):
        """Test heatmap with specific layer."""
        args = Namespace(model="test/model", prompt="test", layer=5)

        mock_info = MoEModelInfo(
            moe_layers=(0, 5),
            num_experts=8,
            num_experts_per_tok=2,
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

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.heatmap.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_heatmap(args)

            captured = capsys.readouterr()
            assert "test/model" in captured.out or "Loading" in captured.out

    @pytest.mark.asyncio
    async def test_heatmap_with_empty_weights(self, capsys):
        """Test heatmap when no routing data captured."""
        args = Namespace(model="test/model", prompt="test")

        mock_info = MoEModelInfo(
            moe_layers=(0,),
            num_experts=8,
            num_experts_per_tok=2,
            total_layers=4,
            architecture=MoEArchitecture.GPT_OSS,
        )

        mock_router = AsyncMock()
        mock_router.info = mock_info
        mock_router.capture_router_weights = AsyncMock(return_value=[])
        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.heatmap.ExpertRouter"
        ) as MockRouter:
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_heatmap(args)

            captured = capsys.readouterr()
            assert "No routing data captured" in captured.out

    @pytest.mark.asyncio
    async def test_heatmap_save_to_file(self, capsys):
        """Test heatmap saves to file when output path specified."""
        args = Namespace(model="test/model", prompt="test", output="test.png")

        mock_info = MoEModelInfo(
            moe_layers=(0,),
            num_experts=8,
            num_experts_per_tok=2,
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

        with (
            patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.heatmap.ExpertRouter"
            ) as MockRouter,
            patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.heatmap.save_routing_heatmap"
            ) as mock_save,
        ):
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_heatmap(args)

            mock_save.assert_called_once()
            captured = capsys.readouterr()
            assert "Heatmap saved to: test.png" in captured.out

    @pytest.mark.asyncio
    async def test_heatmap_matplotlib_import_error(self, capsys):
        """Test heatmap falls back to ASCII on matplotlib ImportError."""
        args = Namespace(model="test/model", prompt="test", output="test.png")

        mock_info = MoEModelInfo(
            moe_layers=(0,),
            num_experts=8,
            num_experts_per_tok=2,
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

        with (
            patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.heatmap.ExpertRouter"
            ) as MockRouter,
            patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.heatmap.save_routing_heatmap",
                side_effect=ImportError("matplotlib not installed"),
            ),
            patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.heatmap.routing_heatmap_ascii",
                return_value="ASCII heatmap",
            ),
        ):
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_heatmap(args)

            captured = capsys.readouterr()
            assert "matplotlib not installed" in captured.out
            assert "ASCII heatmap" in captured.out

    @pytest.mark.asyncio
    async def test_heatmap_ascii_mode(self, capsys):
        """Test heatmap in explicit ASCII mode."""
        args = Namespace(model="test/model", prompt="test", ascii=True)

        mock_info = MoEModelInfo(
            moe_layers=(0,),
            num_experts=8,
            num_experts_per_tok=2,
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

        with (
            patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.heatmap.ExpertRouter"
            ) as MockRouter,
            patch(
                "chuk_lazarus.cli.commands.introspect.moe_expert.handlers.heatmap.routing_heatmap_ascii",
                return_value="ASCII output",
            ),
        ):
            MockRouter.from_pretrained = AsyncMock(return_value=mock_router)

            await _async_heatmap(args)

            captured = capsys.readouterr()
            assert "ASCII output" in captured.out
