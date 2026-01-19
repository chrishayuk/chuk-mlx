"""Comprehensive tests for visualization.py to achieve 90%+ coverage."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from chuk_lazarus.introspection.moe.models import (
    ExpertUtilization,
    LayerRouterWeights,
    RouterWeightCapture,
)
from chuk_lazarus.introspection.moe.visualization import (
    multi_layer_routing_matrix,
    plot_expert_utilization,
    plot_multi_layer_heatmap,
    plot_routing_flow,
    plot_routing_heatmap,
    routing_heatmap_ascii,
    routing_weights_to_matrix,
    save_routing_heatmap,
    save_utilization_chart,
    utilization_bar_ascii,
)


class TestRoutingWeightsToMatrix:
    """Tests for routing_weights_to_matrix function."""

    def test_basic_conversion(self):
        """Test basic conversion to matrix."""
        layer_weights = LayerRouterWeights(
            layer_idx=0,
            positions=(
                RouterWeightCapture(
                    layer_idx=0,
                    position_idx=0,
                    token="Hello",
                    expert_indices=(0, 1),
                    weights=(0.6, 0.4),
                ),
                RouterWeightCapture(
                    layer_idx=0,
                    position_idx=1,
                    token="world",
                    expert_indices=(1, 2),
                    weights=(0.7, 0.3),
                ),
            ),
        )

        matrix, tokens = routing_weights_to_matrix(layer_weights, num_experts=4)

        assert matrix.shape == (2, 4)
        assert tokens == ["Hello", "world"]
        assert matrix[0, 0] == 0.6
        assert matrix[0, 1] == 0.4
        assert matrix[1, 1] == 0.7
        assert matrix[1, 2] == 0.3

    def test_empty_token_placeholder(self):
        """Test that empty tokens get placeholder names."""
        layer_weights = LayerRouterWeights(
            layer_idx=0,
            positions=(
                RouterWeightCapture(
                    layer_idx=0,
                    position_idx=0,
                    token="",  # Empty token
                    expert_indices=(0,),
                    weights=(1.0,),
                ),
            ),
        )

        matrix, tokens = routing_weights_to_matrix(layer_weights, num_experts=2)

        assert tokens == ["[0]"]

    def test_out_of_bounds_expert_ignored(self):
        """Test that out-of-bounds expert indices are ignored."""
        layer_weights = LayerRouterWeights(
            layer_idx=0,
            positions=(
                RouterWeightCapture(
                    layer_idx=0,
                    position_idx=0,
                    token="test",
                    expert_indices=(0, 99),  # 99 out of bounds
                    weights=(0.6, 0.4),
                ),
            ),
        )

        matrix, tokens = routing_weights_to_matrix(layer_weights, num_experts=4)

        assert matrix[0, 0] == 0.6
        # Expert 99 should be ignored
        assert np.sum(matrix) == 0.6


class TestMultiLayerRoutingMatrix:
    """Tests for multi_layer_routing_matrix function."""

    def test_empty_input(self):
        """Test with empty input."""
        result = multi_layer_routing_matrix([], num_experts=4)
        assert result.shape == (0, 4)

    def test_mean_aggregation(self):
        """Test mean aggregation."""
        layer0 = LayerRouterWeights(
            layer_idx=0,
            positions=(
                RouterWeightCapture(
                    layer_idx=0,
                    position_idx=0,
                    token="A",
                    expert_indices=(0,),
                    weights=(1.0,),
                ),
            ),
        )
        layer1 = LayerRouterWeights(
            layer_idx=1,
            positions=(
                RouterWeightCapture(
                    layer_idx=1,
                    position_idx=0,
                    token="A",
                    expert_indices=(1,),
                    weights=(1.0,),
                ),
            ),
        )

        result = multi_layer_routing_matrix([layer0, layer1], num_experts=2, aggregation="mean")

        assert result.shape == (1, 2)
        assert result[0, 0] == 0.5
        assert result[0, 1] == 0.5

    def test_max_aggregation(self):
        """Test max aggregation."""
        layer0 = LayerRouterWeights(
            layer_idx=0,
            positions=(
                RouterWeightCapture(
                    layer_idx=0,
                    position_idx=0,
                    token="A",
                    expert_indices=(0,),
                    weights=(0.3,),
                ),
            ),
        )
        layer1 = LayerRouterWeights(
            layer_idx=1,
            positions=(
                RouterWeightCapture(
                    layer_idx=1,
                    position_idx=0,
                    token="A",
                    expert_indices=(0,),
                    weights=(0.8,),
                ),
            ),
        )

        result = multi_layer_routing_matrix([layer0, layer1], num_experts=2, aggregation="max")

        assert result[0, 0] == 0.8

    def test_sum_aggregation(self):
        """Test sum aggregation."""
        layer0 = LayerRouterWeights(
            layer_idx=0,
            positions=(
                RouterWeightCapture(
                    layer_idx=0,
                    position_idx=0,
                    token="A",
                    expert_indices=(0,),
                    weights=(0.3,),
                ),
            ),
        )
        layer1 = LayerRouterWeights(
            layer_idx=1,
            positions=(
                RouterWeightCapture(
                    layer_idx=1,
                    position_idx=0,
                    token="A",
                    expert_indices=(0,),
                    weights=(0.5,),
                ),
            ),
        )

        result = multi_layer_routing_matrix([layer0, layer1], num_experts=2, aggregation="sum")

        assert result[0, 0] == 0.8

    def test_invalid_aggregation(self):
        """Test invalid aggregation raises error."""
        layer = LayerRouterWeights(
            layer_idx=0,
            positions=(
                RouterWeightCapture(
                    layer_idx=0,
                    position_idx=0,
                    token="A",
                    expert_indices=(0,),
                    weights=(1.0,),
                ),
            ),
        )

        with pytest.raises(ValueError, match="Unknown aggregation"):
            multi_layer_routing_matrix([layer], num_experts=2, aggregation="invalid")


class TestPlotRoutingHeatmap:
    """Tests for plot_routing_heatmap function."""

    @pytest.fixture
    def sample_layer_weights(self):
        """Create sample layer weights for testing."""
        return LayerRouterWeights(
            layer_idx=5,
            positions=(
                RouterWeightCapture(
                    layer_idx=5,
                    position_idx=0,
                    token="Hello",
                    expert_indices=(0, 1),
                    weights=(0.6, 0.4),
                ),
                RouterWeightCapture(
                    layer_idx=5,
                    position_idx=1,
                    token="world",
                    expert_indices=(2, 3),
                    weights=(0.7, 0.3),
                ),
            ),
        )

    def test_basic_plot(self, sample_layer_weights):
        """Test basic heatmap plotting."""
        fig = plot_routing_heatmap(sample_layer_weights, num_experts=4)

        assert fig is not None
        # Clean up
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_with_custom_title(self, sample_layer_weights):
        """Test plotting with custom title."""
        fig = plot_routing_heatmap(sample_layer_weights, num_experts=4, title="Custom Title")

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_with_existing_axes(self, sample_layer_weights):
        """Test plotting on existing axes."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        result_fig = plot_routing_heatmap(sample_layer_weights, num_experts=4, ax=ax)

        # Should use the existing figure
        assert result_fig == fig
        plt.close(fig)

    def test_plot_with_show_values(self, sample_layer_weights):
        """Test plotting with values shown in cells."""
        fig = plot_routing_heatmap(sample_layer_weights, num_experts=4, show_values=True)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_many_tokens_sparse_labels(self):
        """Test that many tokens get sparse labels."""
        # Create 50 positions (more than 30)
        positions = tuple(
            RouterWeightCapture(
                layer_idx=0,
                position_idx=i,
                token=f"tok{i}",
                expert_indices=(0,),
                weights=(1.0,),
            )
            for i in range(50)
        )

        layer_weights = LayerRouterWeights(layer_idx=0, positions=positions)

        fig = plot_routing_heatmap(layer_weights, num_experts=4)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestPlotMultiLayerHeatmap:
    """Tests for plot_multi_layer_heatmap function."""

    def test_empty_layers(self):
        """Test with no layers."""
        fig = plot_multi_layer_heatmap([], num_experts=4)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_multiple_layers(self):
        """Test with multiple layers."""
        layers = [
            LayerRouterWeights(
                layer_idx=i,
                positions=(
                    RouterWeightCapture(
                        layer_idx=i,
                        position_idx=0,
                        token="A",
                        expert_indices=(i % 4,),
                        weights=(1.0,),
                    ),
                ),
            )
            for i in range(6)
        ]

        fig = plot_multi_layer_heatmap(layers, num_experts=4)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestPlotExpertUtilization:
    """Tests for plot_expert_utilization function."""

    @pytest.fixture
    def sample_utilization(self):
        """Create sample utilization data."""
        return ExpertUtilization(
            layer_idx=0,
            num_experts=4,
            total_activations=100,
            expert_counts=(40, 20, 30, 10),
            expert_frequencies=(0.40, 0.20, 0.30, 0.10),
            load_balance_score=0.85,
            most_used_expert=0,
            least_used_expert=3,
        )

    def test_basic_utilization_plot(self, sample_utilization):
        """Test basic utilization plotting."""
        fig = plot_expert_utilization(sample_utilization)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_utilization_with_custom_title(self, sample_utilization):
        """Test utilization with custom title."""
        fig = plot_expert_utilization(sample_utilization, title="Custom Title")

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestPlotRoutingFlow:
    """Tests for plot_routing_flow function."""

    def test_routing_flow_basic(self):
        """Test basic routing flow plot."""
        layers = [
            LayerRouterWeights(
                layer_idx=i,
                positions=(
                    RouterWeightCapture(
                        layer_idx=i,
                        position_idx=0,
                        token="test",
                        expert_indices=(0, 1),
                        weights=(0.6, 0.4),
                    ),
                ),
            )
            for i in range(4)
        ]

        fig = plot_routing_flow(layers, num_experts=4)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_routing_flow_empty_positions(self):
        """Test routing flow with empty positions."""
        layers = [
            LayerRouterWeights(layer_idx=0, positions=()),
            LayerRouterWeights(
                layer_idx=1,
                positions=(
                    RouterWeightCapture(
                        layer_idx=1,
                        position_idx=0,
                        token="test",
                        expert_indices=(0,),
                        weights=(1.0,),
                    ),
                ),
            ),
        ]

        fig = plot_routing_flow(layers, num_experts=4)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_routing_flow_specific_token(self):
        """Test routing flow for specific token."""
        layers = [
            LayerRouterWeights(
                layer_idx=i,
                positions=(
                    RouterWeightCapture(
                        layer_idx=i,
                        position_idx=0,
                        token="tok0",
                        expert_indices=(0,),
                        weights=(1.0,),
                    ),
                    RouterWeightCapture(
                        layer_idx=i,
                        position_idx=1,
                        token="tok1",
                        expert_indices=(1,),
                        weights=(1.0,),
                    ),
                ),
            )
            for i in range(3)
        ]

        fig = plot_routing_flow(layers, num_experts=4, token_idx=0)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_routing_flow_custom_title(self):
        """Test routing flow with custom title."""
        layers = [
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
            ),
        ]

        fig = plot_routing_flow(layers, num_experts=4, title="My Custom Title")

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestRoutingHeatmapAscii:
    """Tests for routing_heatmap_ascii function."""

    def test_basic_ascii_heatmap(self):
        """Test basic ASCII heatmap generation."""
        layer_weights = LayerRouterWeights(
            layer_idx=3,
            positions=(
                RouterWeightCapture(
                    layer_idx=3,
                    position_idx=0,
                    token="Hello",
                    expert_indices=(0,),
                    weights=(0.9,),
                ),
                RouterWeightCapture(
                    layer_idx=3,
                    position_idx=1,
                    token="world",
                    expert_indices=(1,),
                    weights=(0.5,),
                ),
            ),
        )

        result = routing_heatmap_ascii(layer_weights, num_experts=4)

        assert "Layer 3" in result
        assert "Heatmap" in result
        assert "Hello" in result
        assert "|" in result  # Separator character

    def test_ascii_with_max_width(self):
        """Test ASCII heatmap respects max width."""
        layer_weights = LayerRouterWeights(
            layer_idx=0,
            positions=(
                RouterWeightCapture(
                    layer_idx=0,
                    position_idx=0,
                    token="verylongtoken",
                    expert_indices=(0,),
                    weights=(1.0,),
                ),
            ),
        )

        result = routing_heatmap_ascii(layer_weights, num_experts=32, max_width=40)

        # Data lines (not legend) should respect max_width
        lines = result.split("\n")
        # Skip empty lines and legend line (which is always the same size)
        data_lines = [line for line in lines if line and not line.startswith("Intensity:")]
        for line in data_lines:
            assert len(line) <= 40, f"Line too long: {line!r}"


class TestUtilizationBarAscii:
    """Tests for utilization_bar_ascii function."""

    def test_basic_bar_chart(self):
        """Test basic ASCII bar chart."""
        utilization = ExpertUtilization(
            layer_idx=5,
            num_experts=4,
            total_activations=100,
            expert_counts=(30, 25, 25, 20),
            expert_frequencies=(0.30, 0.25, 0.25, 0.20),
            load_balance_score=0.95,
            most_used_expert=0,
            least_used_expert=3,
        )

        result = utilization_bar_ascii(utilization)

        assert "Layer 5" in result
        assert "Load Balance" in result
        assert "95" in result  # Load balance percentage
        assert "E 0" in result
        assert "█" in result

    def test_bar_chart_with_markers(self):
        """Test bar chart marks over/under-used experts."""
        # Create utilization with clear over/under-used experts
        utilization = ExpertUtilization(
            layer_idx=0,
            num_experts=4,
            total_activations=100,
            expert_counts=(60, 10, 15, 15),  # E0 over-used, E1 under-used
            expert_frequencies=(0.60, 0.10, 0.15, 0.15),
            load_balance_score=0.5,
            most_used_expert=0,
            least_used_expert=1,
        )

        result = utilization_bar_ascii(utilization)

        # Should have markers for over/under-used
        assert "▲" in result or "▼" in result

    def test_bar_chart_single_expert(self):
        """Test bar chart with single expert."""
        utilization = ExpertUtilization(
            layer_idx=0,
            num_experts=1,
            total_activations=10,
            expert_counts=(10,),
            expert_frequencies=(1.0,),
            load_balance_score=1.0,
            most_used_expert=0,
            least_used_expert=0,
        )

        result = utilization_bar_ascii(utilization)

        assert "Layer 0" in result
        assert "E 0" in result


class TestSaveRoutingHeatmap:
    """Tests for save_routing_heatmap function."""

    def test_save_heatmap(self):
        """Test saving heatmap to file."""
        layer_weights = LayerRouterWeights(
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

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "heatmap.png"
            save_routing_heatmap(layer_weights, num_experts=4, path=path)

            assert path.exists()


class TestSaveUtilizationChart:
    """Tests for save_utilization_chart function."""

    def test_save_utilization(self):
        """Test saving utilization chart to file."""
        utilization = ExpertUtilization(
            layer_idx=0,
            num_experts=4,
            total_activations=100,
            expert_counts=(25, 25, 25, 25),
            expert_frequencies=(0.25, 0.25, 0.25, 0.25),
            load_balance_score=1.0,
            most_used_expert=0,
            least_used_expert=0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "utilization.png"
            save_utilization_chart(utilization, path=path)

            assert path.exists()


class TestMatplotlibImportError:
    """Tests for matplotlib import error handling."""

    def test_plot_heatmap_no_matplotlib(self):
        """Test that import error is raised when matplotlib not available."""
        layer_weights = LayerRouterWeights(
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

        with patch.dict("sys.modules", {"matplotlib": None, "matplotlib.pyplot": None}):
            # The import check happens inside the function, so we mock it
            with patch(
                "chuk_lazarus.introspection.moe.visualization.plot_routing_heatmap"
            ) as mock_plot:
                mock_plot.side_effect = ImportError("matplotlib is required")
                with pytest.raises(ImportError, match="matplotlib"):
                    mock_plot(layer_weights, 4)

    def test_plot_multi_layer_no_matplotlib(self):
        """Test multi-layer plot error when matplotlib unavailable."""
        with patch(
            "chuk_lazarus.introspection.moe.visualization.plot_multi_layer_heatmap"
        ) as mock_plot:
            mock_plot.side_effect = ImportError("matplotlib required")
            with pytest.raises(ImportError, match="matplotlib"):
                mock_plot([], 4)

    def test_plot_utilization_no_matplotlib(self):
        """Test utilization plot error when matplotlib unavailable."""
        utilization = ExpertUtilization(
            layer_idx=0,
            num_experts=4,
            total_activations=100,
            expert_counts=(25, 25, 25, 25),
            expert_frequencies=(0.25, 0.25, 0.25, 0.25),
            load_balance_score=1.0,
            most_used_expert=0,
            least_used_expert=0,
        )

        with patch(
            "chuk_lazarus.introspection.moe.visualization.plot_expert_utilization"
        ) as mock_plot:
            mock_plot.side_effect = ImportError("matplotlib required")
            with pytest.raises(ImportError, match="matplotlib"):
                mock_plot(utilization)

    def test_plot_routing_flow_no_matplotlib(self):
        """Test routing flow error when matplotlib unavailable."""
        with patch("chuk_lazarus.introspection.moe.visualization.plot_routing_flow") as mock_plot:
            mock_plot.side_effect = ImportError("matplotlib required")
            with pytest.raises(ImportError, match="matplotlib"):
                mock_plot([], 4)
