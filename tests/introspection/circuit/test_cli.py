"""Tests for circuit CLI module."""

import argparse
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from chuk_lazarus.introspection.circuit.cli import (
    cmd_analyze,
    cmd_collect,
    cmd_dataset_create,
    cmd_dataset_show,
    cmd_directions,
    cmd_probes,
    cmd_probes_init,
    cmd_steer,
    cmd_visualize,
    main,
)


class TestCmdDatasetCreate:
    """Tests for cmd_dataset_create."""

    @patch("chuk_lazarus.introspection.circuit.dataset.create_tool_calling_dataset")
    def test_creates_dataset(self, mock_create):
        """Test dataset creation with default parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "dataset.json"

            mock_dataset = Mock()
            mock_dataset.summary.return_value = {
                "total": 100,
                "tool_calling": 50,
                "no_tool": 50,
                "by_category": {"search": 25, "calculate": 25},
                "by_tool": {"web_search": 25, "calculator": 25},
            }
            mock_create.return_value = mock_dataset

            args = argparse.Namespace(
                output=str(output_path),
                per_tool=25,
                no_tool=100,
                no_edge_cases=False,
                seed=42,
            )

            cmd_dataset_create(args)

            mock_create.assert_called_once_with(
                prompts_per_tool=25,
                no_tool_prompts=100,
                include_edge_cases=True,
                seed=42,
            )
            mock_dataset.save.assert_called_once()

    @patch("chuk_lazarus.introspection.circuit.dataset.create_tool_calling_dataset")
    def test_creates_dataset_no_edge_cases(self, mock_create):
        """Test dataset creation without edge cases."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "dataset.json"

            mock_dataset = Mock()
            mock_dataset.summary.return_value = {
                "total": 50,
                "tool_calling": 25,
                "no_tool": 25,
                "by_category": {},
                "by_tool": {},
            }
            mock_create.return_value = mock_dataset

            args = argparse.Namespace(
                output=str(output_path),
                per_tool=10,
                no_tool=25,
                no_edge_cases=True,
                seed=123,
            )

            cmd_dataset_create(args)

            mock_create.assert_called_once_with(
                prompts_per_tool=10,
                no_tool_prompts=25,
                include_edge_cases=False,
                seed=123,
            )


class TestCmdDatasetShow:
    """Tests for cmd_dataset_show."""

    @patch("chuk_lazarus.introspection.circuit.dataset.ToolPromptDataset")
    def test_shows_dataset_info(self, mock_dataset_class):
        """Test showing dataset information."""
        mock_dataset = Mock()
        mock_dataset.name = "test_dataset"
        mock_dataset.version = "1.0"
        mock_dataset.summary.return_value = {
            "total": 100,
            "tool_calling": 50,
            "no_tool": 50,
        }
        mock_dataset_class.load.return_value = mock_dataset

        args = argparse.Namespace(dataset="test.json", samples=0)

        cmd_dataset_show(args)

        mock_dataset_class.load.assert_called_once_with("test.json")
        mock_dataset.summary.assert_called_once()

    @patch("chuk_lazarus.introspection.circuit.dataset.ToolPromptDataset")
    def test_shows_samples(self, mock_dataset_class):
        """Test showing dataset samples."""
        mock_prompt = Mock()
        mock_prompt.expected_tool = "calculator"
        mock_prompt.category.value = "math"
        mock_prompt.text = "Calculate 2+2" + "x" * 100

        mock_dataset = Mock()
        mock_dataset.name = "test"
        mock_dataset.version = "1.0"
        mock_dataset.__len__ = Mock(return_value=10)
        mock_dataset.summary.return_value = {
            "total": 10,
            "tool_calling": 5,
            "no_tool": 5,
        }
        mock_dataset.sample.return_value = [mock_prompt]
        mock_dataset_class.load.return_value = mock_dataset

        args = argparse.Namespace(dataset="test.json", samples=5)

        cmd_dataset_show(args)

        mock_dataset.sample.assert_called_once_with(5, seed=42)

    @patch("chuk_lazarus.introspection.circuit.dataset.ToolPromptDataset")
    def test_shows_samples_no_tool(self, mock_dataset_class):
        """Test showing samples without expected tool."""
        mock_prompt = Mock()
        mock_prompt.expected_tool = None
        mock_prompt.category.value = "general"
        mock_prompt.text = "Hello world"

        mock_dataset = Mock()
        mock_dataset.name = "test"
        mock_dataset.version = "1.0"
        mock_dataset.__len__ = Mock(return_value=5)
        mock_dataset.summary.return_value = {
            "total": 5,
            "tool_calling": 0,
            "no_tool": 5,
        }
        mock_dataset.sample.return_value = [mock_prompt]
        mock_dataset_class.load.return_value = mock_dataset

        args = argparse.Namespace(dataset="test.json", samples=3)

        cmd_dataset_show(args)


class TestCmdCollect:
    """Tests for cmd_collect."""

    @patch("chuk_lazarus.introspection.circuit.collector.ActivationCollector")
    @patch("chuk_lazarus.introspection.circuit.dataset.ToolPromptDataset")
    @patch("chuk_lazarus.introspection.circuit.collector.CollectorConfig")
    def test_collect_with_all_layers(self, mock_config, mock_dataset_class, mock_collector_class):
        """Test collecting with all layers."""
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=10)
        mock_dataset_class.load.return_value = mock_dataset

        mock_collector = Mock()
        mock_collector.num_layers = 12
        mock_collector.hidden_size = 768
        mock_activations = Mock()
        mock_activations.captured_layers = [0, 1, 2]
        mock_activations.__len__ = Mock(return_value=10)
        mock_collector.collect.return_value = mock_activations
        mock_collector_class.from_pretrained.return_value = mock_collector

        args = argparse.Namespace(
            dataset="data.json",
            model="test-model",
            output="output",
            layers="all",
            attention=False,
            generate=0,
        )

        cmd_collect(args)

        mock_config.assert_called_once()
        call_kwargs = mock_config.call_args[1]
        assert call_kwargs["layers"] == "all"

    @patch("chuk_lazarus.introspection.circuit.collector.ActivationCollector")
    @patch("chuk_lazarus.introspection.circuit.dataset.ToolPromptDataset")
    @patch("chuk_lazarus.introspection.circuit.collector.CollectorConfig")
    def test_collect_with_decision_layers(
        self, mock_config, mock_dataset_class, mock_collector_class
    ):
        """Test collecting with decision layers."""
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=5)
        mock_dataset_class.load.return_value = mock_dataset

        mock_collector = Mock()
        mock_collector.num_layers = 12
        mock_collector.hidden_size = 512
        mock_activations = Mock()
        mock_activations.captured_layers = [8, 9, 10, 11]
        mock_activations.__len__ = Mock(return_value=5)
        mock_collector.collect.return_value = mock_activations
        mock_collector_class.from_pretrained.return_value = mock_collector

        args = argparse.Namespace(
            dataset="data.json",
            model="model",
            output="out",
            layers="decision",
            attention=True,
            generate=5,
        )

        cmd_collect(args)

        call_kwargs = mock_config.call_args[1]
        assert call_kwargs["layers"] == "decision"
        assert call_kwargs["capture_attention_weights"] is True

    @patch("chuk_lazarus.introspection.circuit.collector.ActivationCollector")
    @patch("chuk_lazarus.introspection.circuit.dataset.ToolPromptDataset")
    @patch("chuk_lazarus.introspection.circuit.collector.CollectorConfig")
    def test_collect_with_specific_layers(
        self, mock_config, mock_dataset_class, mock_collector_class
    ):
        """Test collecting with specific layer numbers."""
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=3)
        mock_dataset_class.load.return_value = mock_dataset

        mock_collector = Mock()
        mock_collector.num_layers = 12
        mock_collector.hidden_size = 256
        mock_activations = Mock()
        mock_activations.captured_layers = [5, 10]
        mock_activations.__len__ = Mock(return_value=3)
        mock_collector.collect.return_value = mock_activations
        mock_collector_class.from_pretrained.return_value = mock_collector

        args = argparse.Namespace(
            dataset="data.json",
            model="model",
            output="out",
            layers="5, 10",
            attention=False,
            generate=0,
        )

        cmd_collect(args)

        call_kwargs = mock_config.call_args[1]
        assert call_kwargs["layers"] == [5, 10]


class TestCmdAnalyze:
    """Tests for cmd_analyze."""

    @patch("chuk_lazarus.introspection.circuit.geometry.GeometryAnalyzer")
    @patch("chuk_lazarus.introspection.circuit.collector.CollectedActivations")
    def test_analyze_single_layer(self, mock_activations_class, mock_analyzer_class):
        """Test analyzing a single layer."""
        mock_activations = Mock()
        mock_activations.captured_layers = [8, 9, 10, 11]
        mock_activations.__len__ = Mock(return_value=100)
        mock_activations_class.load.return_value = mock_activations

        mock_result = Mock()
        mock_result.pca = Mock()
        mock_result.pca.intrinsic_dimensionality_90 = 10
        mock_result.pca.intrinsic_dimensionality_95 = 15
        mock_result.pca.explained_variance_ratio = [0.3, 0.2, 0.1]
        mock_result.binary_probe = Mock()
        mock_result.binary_probe.accuracy = 0.95
        mock_result.binary_probe.cv_mean = 0.93
        mock_result.binary_probe.cv_std = 0.02
        mock_result.category_probe = Mock()
        mock_result.category_probe.accuracy = 0.85

        mock_analyzer = Mock()
        mock_analyzer.analyze_layer.return_value = mock_result
        mock_analyzer_class.return_value = mock_analyzer

        args = argparse.Namespace(
            activations="act.safetensors",
            layer=10,
            umap=False,
            output=None,
        )

        cmd_analyze(args)

        mock_analyzer.analyze_layer.assert_called_once_with(10, include_umap=False)

    @patch("chuk_lazarus.introspection.circuit.geometry.GeometryAnalyzer")
    @patch("chuk_lazarus.introspection.circuit.collector.CollectedActivations")
    def test_analyze_all_layers(self, mock_activations_class, mock_analyzer_class):
        """Test analyzing all captured layers."""
        mock_activations = Mock()
        mock_activations.captured_layers = [0, 1, 2]
        mock_activations.__len__ = Mock(return_value=50)
        mock_activations_class.load.return_value = mock_activations

        mock_result = Mock()
        mock_result.pca = None
        mock_result.binary_probe = None
        mock_result.category_probe = None
        mock_result.summary.return_value = {"test": "data"}

        mock_analyzer = Mock()
        mock_analyzer.analyze_layer.return_value = mock_result
        mock_analyzer_class.return_value = mock_analyzer

        args = argparse.Namespace(
            activations="act.safetensors",
            layer=None,
            umap=True,
            output=None,
        )

        cmd_analyze(args)

        assert mock_analyzer.analyze_layer.call_count == 3
        mock_analyzer.print_layer_comparison.assert_called_once()

    @patch("chuk_lazarus.introspection.circuit.geometry.GeometryAnalyzer")
    @patch("chuk_lazarus.introspection.circuit.collector.CollectedActivations")
    def test_analyze_saves_output(self, mock_activations_class, mock_analyzer_class):
        """Test analyzing with output file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.json"

            mock_activations = Mock()
            mock_activations.captured_layers = [5]
            mock_activations.model_id = "test-model"
            mock_activations.__len__ = Mock(return_value=10)
            mock_activations_class.load.return_value = mock_activations

            mock_result = Mock()
            mock_result.pca = None
            mock_result.binary_probe = None
            mock_result.category_probe = None
            mock_result.summary.return_value = {"accuracy": 0.9}

            mock_analyzer = Mock()
            mock_analyzer.analyze_layer.return_value = mock_result
            mock_analyzer_class.return_value = mock_analyzer

            args = argparse.Namespace(
                activations="act.safetensors",
                layer=5,
                umap=False,
                output=str(output_path),
            )

            cmd_analyze(args)

            assert output_path.exists()
            with open(output_path) as f:
                data = json.load(f)
            assert data["model_id"] == "test-model"
            assert 5 in data["layers"] or "5" in data["layers"]


class TestCmdDirections:
    """Tests for cmd_directions."""

    @patch("chuk_lazarus.introspection.circuit.directions.DirectionExtractor")
    @patch("chuk_lazarus.introspection.circuit.collector.CollectedActivations")
    def test_extract_directions(self, mock_activations_class, mock_extractor_class):
        """Test extracting directions."""
        mock_activations = Mock()
        mock_activations.captured_layers = [8, 9, 10, 11]
        mock_activations.__len__ = Mock(return_value=100)
        mock_activations_class.load.return_value = mock_activations

        mock_direction = Mock()
        mock_direction.separation_score = 2.5
        mock_direction.accuracy = 0.95
        mock_direction.mean_projection_positive = 1.2
        mock_direction.mean_projection_negative = -1.3

        mock_extractor = Mock()
        mock_extractor.extract_tool_mode_direction.return_value = mock_direction
        mock_extractor_class.return_value = mock_extractor

        args = argparse.Namespace(
            activations="act.safetensors",
            layer=10,
            method="diff_means",
            per_tool=False,
            output=None,
        )

        cmd_directions(args)

        mock_extractor.extract_tool_mode_direction.assert_called_once()

    @patch("chuk_lazarus.introspection.circuit.directions.DirectionExtractor")
    @patch("chuk_lazarus.introspection.circuit.collector.CollectedActivations")
    def test_extract_directions_auto_layer(self, mock_activations_class, mock_extractor_class):
        """Test extracting directions with auto layer selection."""
        mock_activations = Mock()
        mock_activations.captured_layers = [0, 1, 2, 3, 4, 5]
        mock_activations.__len__ = Mock(return_value=50)
        mock_activations_class.load.return_value = mock_activations

        mock_direction = Mock()
        mock_direction.separation_score = 2.0
        mock_direction.accuracy = 0.9
        mock_direction.mean_projection_positive = 1.0
        mock_direction.mean_projection_negative = -1.0

        mock_extractor = Mock()
        mock_extractor.extract_tool_mode_direction.return_value = mock_direction
        mock_extractor_class.return_value = mock_extractor

        args = argparse.Namespace(
            activations="act.safetensors",
            layer=None,  # Auto-select middle
            method="lda",
            per_tool=False,
            output=None,
        )

        cmd_directions(args)

        # Should select middle layer (2 or 3 for 6 layers)
        call_args = mock_extractor.extract_tool_mode_direction.call_args
        layer_used = call_args[0][0]
        assert layer_used in [2, 3]

    @patch("chuk_lazarus.introspection.circuit.directions.DirectionExtractor")
    @patch("chuk_lazarus.introspection.circuit.collector.CollectedActivations")
    def test_extract_per_tool_directions(self, mock_activations_class, mock_extractor_class):
        """Test extracting per-tool directions."""
        import numpy as np

        mock_activations = Mock()
        mock_activations.captured_layers = [10]
        mock_activations.__len__ = Mock(return_value=100)
        mock_activations_class.load.return_value = mock_activations

        mock_direction = Mock()
        mock_direction.separation_score = 2.0
        mock_direction.accuracy = 0.9
        mock_direction.mean_projection_positive = 1.0
        mock_direction.mean_projection_negative = -1.0

        mock_tool_dir = Mock()
        mock_tool_dir.separation_score = 1.5

        mock_extractor = Mock()
        mock_extractor.extract_tool_mode_direction.return_value = mock_direction
        mock_extractor.extract_per_tool_directions.return_value = {
            "calculator": mock_tool_dir,
            "web_search": mock_tool_dir,
        }
        mock_extractor.check_orthogonality.return_value = np.array([[1.0, 0.1], [0.1, 1.0]])
        mock_extractor_class.return_value = mock_extractor

        args = argparse.Namespace(
            activations="act.safetensors",
            layer=10,
            method="diff_means",
            per_tool=True,
            output=None,
        )

        cmd_directions(args)

        mock_extractor.extract_per_tool_directions.assert_called_once()
        mock_extractor.check_orthogonality.assert_called_once()

    @patch("chuk_lazarus.introspection.circuit.directions.DirectionExtractor")
    @patch("chuk_lazarus.introspection.circuit.collector.CollectedActivations")
    def test_extract_directions_with_output(self, mock_activations_class, mock_extractor_class):
        """Test extracting directions and saving bundle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "directions.safetensors"

            mock_activations = Mock()
            mock_activations.captured_layers = [10]
            mock_activations.__len__ = Mock(return_value=100)
            mock_activations_class.load.return_value = mock_activations

            mock_direction = Mock()
            mock_direction.separation_score = 2.0
            mock_direction.accuracy = 0.9
            mock_direction.mean_projection_positive = 1.0
            mock_direction.mean_projection_negative = -1.0

            mock_bundle = Mock()
            mock_extractor = Mock()
            mock_extractor.extract_tool_mode_direction.return_value = mock_direction
            mock_extractor.create_bundle.return_value = mock_bundle
            mock_extractor_class.return_value = mock_extractor

            args = argparse.Namespace(
                activations="act.safetensors",
                layer=10,
                method="diff_means",
                per_tool=False,
                output=str(output_path),
            )

            cmd_directions(args)

            # Verify bundle creation and save
            mock_extractor.create_bundle.assert_called_once_with(10, include_per_tool=False)
            mock_bundle.save.assert_called_once_with(str(output_path))

    @patch("chuk_lazarus.introspection.circuit.directions.DirectionExtractor")
    @patch("chuk_lazarus.introspection.circuit.collector.CollectedActivations")
    def test_extract_directions_with_output_per_tool(
        self, mock_activations_class, mock_extractor_class
    ):
        """Test extracting directions with per-tool flag and saving bundle."""
        import numpy as np

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "directions.safetensors"

            mock_activations = Mock()
            mock_activations.captured_layers = [10]
            mock_activations.__len__ = Mock(return_value=100)
            mock_activations_class.load.return_value = mock_activations

            mock_direction = Mock()
            mock_direction.separation_score = 2.0
            mock_direction.accuracy = 0.9
            mock_direction.mean_projection_positive = 1.0
            mock_direction.mean_projection_negative = -1.0

            mock_tool_dir = Mock()
            mock_tool_dir.separation_score = 1.5

            mock_bundle = Mock()
            mock_extractor = Mock()
            mock_extractor.extract_tool_mode_direction.return_value = mock_direction
            mock_extractor.extract_per_tool_directions.return_value = {
                "calculator": mock_tool_dir,
            }
            mock_extractor.check_orthogonality.return_value = np.array([[1.0]])
            mock_extractor.create_bundle.return_value = mock_bundle
            mock_extractor_class.return_value = mock_extractor

            args = argparse.Namespace(
                activations="act.safetensors",
                layer=10,
                method="diff_means",
                per_tool=True,
                output=str(output_path),
            )

            cmd_directions(args)

            # Verify bundle creation with per_tool flag
            mock_extractor.create_bundle.assert_called_once_with(10, include_per_tool=True)
            mock_bundle.save.assert_called_once_with(str(output_path))


class TestCmdVisualize:
    """Tests for cmd_visualize.

    Note: These tests are skipped due to matplotlib/numpy incompatibility.
    The cmd_visualize function itself is tested in integration tests where the
    matplotlib backend is properly configured.
    """

    @pytest.mark.skip(reason="matplotlib/numpy incompatibility issue")
    @patch("matplotlib.pyplot")
    @patch("chuk_lazarus.introspection.circuit.geometry.GeometryAnalyzer")
    @patch("chuk_lazarus.introspection.circuit.collector.CollectedActivations")
    def test_visualize_pca(self, mock_activations_class, mock_analyzer_class, mock_plt):
        """Test PCA visualization."""
        import numpy as np

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            mock_activations = Mock()
            mock_activations.captured_layers = [10]
            mock_activations.hidden_size = 768
            mock_activations.__len__ = Mock(return_value=100)
            mock_activations_class.load.return_value = mock_activations

            # Mock PCA result
            mock_pca = Mock()
            mock_pca.explained_variance_ratio = np.random.rand(100)
            mock_pca.cumulative_variance = np.cumsum(mock_pca.explained_variance_ratio)
            mock_pca.intrinsic_dimensionality_90 = 50
            mock_pca.intrinsic_dimensionality_95 = 70

            mock_analyzer = Mock()
            mock_analyzer.compute_pca.return_value = mock_pca
            mock_analyzer_class.return_value = mock_analyzer

            # Mock matplotlib
            mock_fig = Mock()
            mock_ax1 = Mock()
            mock_ax2 = Mock()
            mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))

            args = argparse.Namespace(
                activations="act.safetensors",
                layer=10,
                output=str(output_dir),
                pca=True,
                umap=False,
                probes=False,
                all=False,
            )

            cmd_visualize(args)

            # Verify PCA computation
            mock_analyzer.compute_pca.assert_called_once_with(10, n_components=min(100, 768))

            # Verify plotting calls
            mock_ax1.plot.assert_called_once()
            mock_ax2.plot.assert_called_once()
            mock_plt.savefig.assert_called_once()

    @pytest.mark.skip(reason="matplotlib/numpy incompatibility issue")
    @patch("matplotlib.pyplot")
    @patch("chuk_lazarus.introspection.circuit.geometry.GeometryAnalyzer")
    @patch("chuk_lazarus.introspection.circuit.collector.CollectedActivations")
    def test_visualize_umap(self, mock_activations_class, mock_analyzer_class, mock_plt):
        """Test UMAP visualization."""
        import numpy as np

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            mock_activations = Mock()
            mock_activations.captured_layers = [10]
            mock_activations.hidden_size = 512
            mock_activations.__len__ = Mock(return_value=50)
            mock_activations_class.load.return_value = mock_activations

            # Mock UMAP result
            mock_umap = Mock()
            mock_umap.embedding = np.random.rand(50, 2)
            mock_umap.labels = np.random.randint(0, 2, 50)
            mock_umap.category_labels = ["cat1"] * 25 + ["cat2"] * 25

            mock_analyzer = Mock()
            mock_analyzer.compute_umap.return_value = mock_umap
            mock_analyzer_class.return_value = mock_analyzer

            # Mock matplotlib
            mock_fig = Mock()
            mock_ax1 = Mock()
            mock_ax2 = Mock()
            mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
            mock_cmap = Mock()
            mock_plt.cm.get_cmap.return_value = mock_cmap

            args = argparse.Namespace(
                activations="act.safetensors",
                layer=10,
                output=str(output_dir),
                pca=False,
                umap=True,
                probes=False,
                all=False,
            )

            cmd_visualize(args)

            # Verify UMAP computation
            mock_analyzer.compute_umap.assert_called_once_with(10)

            # Verify scatter plots were called
            assert mock_ax1.scatter.call_count >= 1
            assert mock_ax2.scatter.call_count >= 1

    @pytest.mark.skip(reason="matplotlib/numpy incompatibility issue")
    @patch("matplotlib.pyplot")
    @patch("chuk_lazarus.introspection.circuit.geometry.GeometryAnalyzer")
    @patch("chuk_lazarus.introspection.circuit.collector.CollectedActivations")
    def test_visualize_umap_import_error(
        self, mock_activations_class, mock_analyzer_class, mock_plt
    ):
        """Test UMAP visualization with ImportError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            mock_activations = Mock()
            mock_activations.captured_layers = [10]
            mock_activations.hidden_size = 512
            mock_activations.__len__ = Mock(return_value=50)
            mock_activations_class.load.return_value = mock_activations

            mock_analyzer = Mock()
            mock_analyzer.compute_umap.side_effect = ImportError("umap not installed")
            mock_analyzer_class.return_value = mock_analyzer

            args = argparse.Namespace(
                activations="act.safetensors",
                layer=10,
                output=str(output_dir),
                pca=False,
                umap=True,
                probes=False,
                all=False,
            )

            # Should not raise, just print message
            cmd_visualize(args)

    @pytest.mark.skip(reason="matplotlib/numpy incompatibility issue")
    @patch("matplotlib.pyplot")
    @patch("chuk_lazarus.introspection.circuit.geometry.GeometryAnalyzer")
    @patch("chuk_lazarus.introspection.circuit.collector.CollectedActivations")
    def test_visualize_probes(self, mock_activations_class, mock_analyzer_class, mock_plt):
        """Test probe accuracy visualization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            mock_activations = Mock()
            mock_activations.captured_layers = [8, 9, 10, 11]
            mock_activations.hidden_size = 768
            mock_activations.__len__ = Mock(return_value=100)
            mock_activations_class.load.return_value = mock_activations

            # Mock probe results
            mock_binary_probe = Mock()
            mock_binary_probe.accuracy = 0.95

            mock_cat_probe = Mock()
            mock_cat_probe.accuracy = 0.85

            mock_analyzer = Mock()
            mock_analyzer.train_probe.side_effect = [
                mock_binary_probe,
                mock_cat_probe,
                mock_binary_probe,
                mock_cat_probe,
                mock_binary_probe,
                mock_cat_probe,
                mock_binary_probe,
                mock_cat_probe,
            ]
            mock_analyzer_class.return_value = mock_analyzer

            # Mock matplotlib
            mock_fig = Mock()
            mock_ax = Mock()
            mock_plt.subplots.return_value = (mock_fig, mock_ax)

            args = argparse.Namespace(
                activations="act.safetensors",
                layer=None,
                output=str(output_dir),
                pca=False,
                umap=False,
                probes=True,
                all=False,
            )

            cmd_visualize(args)

            # Verify probes were trained for each layer
            assert mock_analyzer.train_probe.call_count == 8  # 4 layers * 2 probes each

            # Verify plotting
            assert mock_ax.plot.call_count == 2  # binary and category lines

    @pytest.mark.skip(reason="matplotlib/numpy incompatibility issue")
    @patch("matplotlib.pyplot")
    @patch("chuk_lazarus.introspection.circuit.geometry.GeometryAnalyzer")
    @patch("chuk_lazarus.introspection.circuit.collector.CollectedActivations")
    def test_visualize_all(self, mock_activations_class, mock_analyzer_class, mock_plt):
        """Test all visualizations together."""
        import numpy as np

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            mock_activations = Mock()
            mock_activations.captured_layers = [10, 11]
            mock_activations.hidden_size = 512
            mock_activations.__len__ = Mock(return_value=50)
            mock_activations_class.load.return_value = mock_activations

            # Mock PCA result
            mock_pca = Mock()
            mock_pca.explained_variance_ratio = np.random.rand(100)
            mock_pca.cumulative_variance = np.cumsum(mock_pca.explained_variance_ratio)
            mock_pca.intrinsic_dimensionality_90 = 40
            mock_pca.intrinsic_dimensionality_95 = 60

            # Mock UMAP result
            mock_umap = Mock()
            mock_umap.embedding = np.random.rand(50, 2)
            mock_umap.labels = np.random.randint(0, 2, 50)
            mock_umap.category_labels = ["cat1"] * 50

            # Mock probes
            mock_binary_probe = Mock()
            mock_binary_probe.accuracy = 0.9
            mock_cat_probe = Mock()
            mock_cat_probe.accuracy = 0.8

            mock_analyzer = Mock()
            mock_analyzer.compute_pca.return_value = mock_pca
            mock_analyzer.compute_umap.return_value = mock_umap
            mock_analyzer.train_probe.side_effect = [
                mock_binary_probe,
                mock_cat_probe,
                mock_binary_probe,
                mock_cat_probe,
            ]
            mock_analyzer_class.return_value = mock_analyzer

            # Mock matplotlib
            mock_fig = Mock()
            mock_ax1 = Mock()
            mock_ax2 = Mock()
            mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
            mock_plt.cm.get_cmap.return_value = Mock()

            args = argparse.Namespace(
                activations="act.safetensors",
                layer=11,
                output=str(output_dir),
                pca=False,
                umap=False,
                probes=False,
                all=True,  # All visualizations
            )

            cmd_visualize(args)

            # All should be called
            mock_analyzer.compute_pca.assert_called_once()
            mock_analyzer.compute_umap.assert_called_once()
            assert mock_analyzer.train_probe.call_count == 4  # 2 layers * 2 probes

    @pytest.mark.skip(reason="matplotlib/numpy incompatibility issue")
    @patch("matplotlib.pyplot")
    @patch("chuk_lazarus.introspection.circuit.geometry.GeometryAnalyzer")
    @patch("chuk_lazarus.introspection.circuit.collector.CollectedActivations")
    def test_visualize_default_output_dir(
        self, mock_activations_class, mock_analyzer_class, mock_plt
    ):
        """Test visualization with default output directory."""
        mock_activations = Mock()
        mock_activations.captured_layers = [10]
        mock_activations.hidden_size = 512
        mock_activations.__len__ = Mock(return_value=50)
        mock_activations_class.load.return_value = mock_activations

        mock_analyzer = Mock()
        mock_analyzer_class.return_value = mock_analyzer

        args = argparse.Namespace(
            activations="act.safetensors",
            layer=10,
            output=None,  # No output specified
            pca=False,
            umap=False,
            probes=False,
            all=False,
        )

        # Should use current directory
        cmd_visualize(args)

    @pytest.mark.skip(reason="matplotlib/numpy incompatibility issue")
    @patch("matplotlib.pyplot")
    @patch("chuk_lazarus.introspection.circuit.geometry.GeometryAnalyzer")
    @patch("chuk_lazarus.introspection.circuit.collector.CollectedActivations")
    def test_visualize_default_layer(self, mock_activations_class, mock_analyzer_class, mock_plt):
        """Test visualization with default layer (last captured layer)."""
        import numpy as np

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            mock_activations = Mock()
            mock_activations.captured_layers = [8, 9, 10, 11]
            mock_activations.hidden_size = 512
            mock_activations.__len__ = Mock(return_value=50)
            mock_activations_class.load.return_value = mock_activations

            # Mock PCA result
            mock_pca = Mock()
            mock_pca.explained_variance_ratio = np.random.rand(100)
            mock_pca.cumulative_variance = np.cumsum(mock_pca.explained_variance_ratio)
            mock_pca.intrinsic_dimensionality_90 = 40
            mock_pca.intrinsic_dimensionality_95 = 60

            mock_analyzer = Mock()
            mock_analyzer.compute_pca.return_value = mock_pca
            mock_analyzer_class.return_value = mock_analyzer

            # Mock matplotlib
            mock_fig = Mock()
            mock_ax1 = Mock()
            mock_ax2 = Mock()
            mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))

            args = argparse.Namespace(
                activations="act.safetensors",
                layer=None,  # Use default
                output=str(output_dir),
                pca=True,
                umap=False,
                probes=False,
                all=False,
            )

            cmd_visualize(args)

            # Should use last layer (11)
            mock_analyzer.compute_pca.assert_called_once_with(11, n_components=min(100, 512))


class TestCmdSteer:
    """Tests for cmd_steer."""

    def test_steer_placeholder(self, capsys):
        """Test steer command placeholder message."""
        args = argparse.Namespace(
            model="test-model",
            direction="dir.safetensors",
            strength=1.0,
        )

        cmd_steer(args)

        captured = capsys.readouterr()
        assert "not yet implemented" in captured.out


class TestCmdProbes:
    """Tests for cmd_probes."""

    @patch("chuk_lazarus.introspection.circuit.probes.ProbeBattery")
    def test_probes_run(self, mock_battery_class):
        """Test running probe battery."""
        mock_battery = Mock()
        mock_battery.num_layers = 12
        mock_battery.datasets = [Mock(), Mock()]
        mock_results = Mock()
        mock_battery.run_all_probes.return_value = mock_results
        mock_battery_class.from_pretrained.return_value = mock_battery

        args = argparse.Namespace(
            model="test-model",
            layers="0,5,10",
            datasets=None,
            category=None,
            threshold=0.75,
            no_stratigraphy=False,
            output=None,
        )

        cmd_probes(args)

        mock_battery.run_all_probes.assert_called_once()
        mock_battery.print_results_table.assert_called_once()
        mock_battery.print_stratigraphy.assert_called_once()

    @patch("chuk_lazarus.introspection.circuit.probes.ProbeBattery")
    def test_probes_run_auto_layers(self, mock_battery_class):
        """Test running probe battery with auto layer selection."""
        mock_battery = Mock()
        mock_battery.num_layers = 12
        mock_battery.datasets = [Mock()]
        mock_results = Mock()
        mock_battery.run_all_probes.return_value = mock_results
        mock_battery_class.from_pretrained.return_value = mock_battery

        args = argparse.Namespace(
            model="test-model",
            layers=None,  # Auto-select
            datasets=None,
            category="syntactic",
            threshold=0.8,
            no_stratigraphy=True,
            output=None,
        )

        cmd_probes(args)

        mock_battery.run_all_probes.assert_called_once()
        # Check that categories filter was passed
        call_kwargs = mock_battery.run_all_probes.call_args[1]
        assert call_kwargs["categories"] == ["syntactic"]

    @patch("chuk_lazarus.introspection.circuit.probes.ProbeBattery")
    def test_probes_run_with_custom_datasets(self, mock_battery_class):
        """Test running probe battery with custom dataset directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "datasets"
            dataset_dir.mkdir()

            mock_battery = Mock()
            mock_battery.num_layers = 8
            mock_battery.datasets = [Mock()]
            mock_results = Mock()
            mock_battery.run_all_probes.return_value = mock_results
            mock_battery_class.from_pretrained.return_value = mock_battery

            args = argparse.Namespace(
                model="test-model",
                layers=None,
                datasets=str(dataset_dir),  # Custom datasets path
                category=None,
                threshold=0.75,
                no_stratigraphy=False,
                output=None,
            )

            cmd_probes(args)

            # Verify dataset_dir was used
            call_args = mock_battery_class.from_pretrained.call_args
            assert call_args[0][1] == dataset_dir

    @patch("chuk_lazarus.introspection.circuit.probes.ProbeBattery")
    def test_probes_run_with_output(self, mock_battery_class):
        """Test running probe battery with output file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.json"

            mock_battery = Mock()
            mock_battery.num_layers = 12
            mock_battery.datasets = [Mock()]
            mock_results = Mock()
            mock_battery.run_all_probes.return_value = mock_results
            mock_battery_class.from_pretrained.return_value = mock_battery

            args = argparse.Namespace(
                model="test-model",
                layers="5,10",
                datasets=None,
                category=None,
                threshold=0.75,
                no_stratigraphy=False,
                output=str(output_path),  # Output file specified
            )

            cmd_probes(args)

            # Verify results.save was called
            mock_results.save.assert_called_once_with(str(output_path))

    @patch("chuk_lazarus.introspection.circuit.probes.ProbeBattery")
    def test_probes_run_multiple_categories(self, mock_battery_class):
        """Test running probe battery with multiple categories."""
        mock_battery = Mock()
        mock_battery.num_layers = 12
        mock_battery.datasets = [Mock()]
        mock_results = Mock()
        mock_battery.run_all_probes.return_value = mock_results
        mock_battery_class.from_pretrained.return_value = mock_battery

        args = argparse.Namespace(
            model="test-model",
            layers=None,
            datasets=None,
            category="syntactic,semantic,decision",  # Multiple categories
            threshold=0.75,
            no_stratigraphy=False,
            output=None,
        )

        cmd_probes(args)

        # Check that categories were parsed correctly
        call_kwargs = mock_battery.run_all_probes.call_args[1]
        assert call_kwargs["categories"] == ["syntactic", "semantic", "decision"]


class TestCmdProbesInit:
    """Tests for cmd_probes_init."""

    @pytest.mark.skip(reason="save_default_datasets not yet implemented")
    @patch("chuk_lazarus.introspection.circuit.cli.save_default_datasets")
    def test_probes_init(self, mock_save_datasets):
        """Test initializing probe datasets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "probe_datasets"

            args = argparse.Namespace(
                output=str(output_dir),
            )

            cmd_probes_init(args)

            # Verify save_default_datasets was called with correct path
            mock_save_datasets.assert_called_once_with(output_dir)


class TestMain:
    """Tests for main entry point."""

    @patch("sys.argv", ["circuit"])
    def test_main_no_command(self, capsys):
        """Test main with no command shows help."""
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0

    @patch("sys.argv", ["circuit", "steer", "-m", "model", "-d", "dir"])
    def test_main_steer_command(self, capsys):
        """Test main routes to steer command."""
        main()
        captured = capsys.readouterr()
        assert "not yet implemented" in captured.out

    @patch("chuk_lazarus.introspection.circuit.dataset.create_tool_calling_dataset")
    @patch("sys.argv", ["circuit", "dataset", "create", "-o", "out.json"])
    def test_main_dataset_create(self, mock_create):
        """Test main routes to dataset create."""
        mock_dataset = Mock()
        mock_dataset.summary.return_value = {
            "total": 10,
            "tool_calling": 5,
            "no_tool": 5,
            "by_category": {},
            "by_tool": {},
        }
        mock_create.return_value = mock_dataset

        main()

        mock_create.assert_called_once()

    @patch("chuk_lazarus.introspection.circuit.dataset.ToolPromptDataset")
    @patch("sys.argv", ["circuit", "dataset", "show", "test.json"])
    def test_main_dataset_show(self, mock_dataset_class):
        """Test main routes to dataset show."""
        mock_dataset = Mock()
        mock_dataset.name = "test"
        mock_dataset.version = "1.0"
        mock_dataset.__len__ = Mock(return_value=10)
        mock_dataset.summary.return_value = {
            "total": 10,
            "tool_calling": 5,
            "no_tool": 5,
        }
        mock_dataset.sample.return_value = []
        mock_dataset_class.load.return_value = mock_dataset

        main()

        mock_dataset_class.load.assert_called_once_with("test.json")

    @patch("sys.argv", ["circuit", "dataset"])
    def test_main_dataset_no_subcommand(self, capsys):
        """Test dataset command without subcommand shows help."""
        main()
        # Should print help, not crash

    @patch("chuk_lazarus.introspection.circuit.collector.ActivationCollector")
    @patch("chuk_lazarus.introspection.circuit.dataset.ToolPromptDataset")
    @patch("chuk_lazarus.introspection.circuit.collector.CollectorConfig")
    @patch(
        "sys.argv",
        ["circuit", "collect", "-m", "model", "-d", "data.json", "-o", "out"],
    )
    def test_main_collect_command(self, mock_config, mock_dataset_class, mock_collector_class):
        """Test main routes to collect command."""
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=10)
        mock_dataset_class.load.return_value = mock_dataset

        mock_collector = Mock()
        mock_collector.num_layers = 12
        mock_collector.hidden_size = 768
        mock_activations = Mock()
        mock_activations.captured_layers = [10, 11]
        mock_activations.__len__ = Mock(return_value=10)
        mock_collector.collect.return_value = mock_activations
        mock_collector_class.from_pretrained.return_value = mock_collector

        main()

        mock_collector_class.from_pretrained.assert_called_once()

    @patch("chuk_lazarus.introspection.circuit.geometry.GeometryAnalyzer")
    @patch("chuk_lazarus.introspection.circuit.collector.CollectedActivations")
    @patch("sys.argv", ["circuit", "analyze", "-a", "act.safetensors"])
    def test_main_analyze_command(self, mock_activations_class, mock_analyzer_class):
        """Test main routes to analyze command."""
        mock_activations = Mock()
        mock_activations.captured_layers = [10]
        mock_activations.__len__ = Mock(return_value=100)
        mock_activations_class.load.return_value = mock_activations

        mock_result = Mock()
        mock_result.pca = None
        mock_result.binary_probe = None
        mock_result.category_probe = None
        mock_analyzer = Mock()
        mock_analyzer.analyze_layer.return_value = mock_result
        mock_analyzer_class.return_value = mock_analyzer

        main()

        mock_activations_class.load.assert_called_once_with("act.safetensors")

    @patch("chuk_lazarus.introspection.circuit.directions.DirectionExtractor")
    @patch("chuk_lazarus.introspection.circuit.collector.CollectedActivations")
    @patch("sys.argv", ["circuit", "directions", "-a", "act.safetensors"])
    def test_main_directions_command(self, mock_activations_class, mock_extractor_class):
        """Test main routes to directions command."""
        mock_activations = Mock()
        mock_activations.captured_layers = [10]
        mock_activations.__len__ = Mock(return_value=100)
        mock_activations_class.load.return_value = mock_activations

        mock_direction = Mock()
        mock_direction.separation_score = 2.0
        mock_direction.accuracy = 0.9
        mock_direction.mean_projection_positive = 1.0
        mock_direction.mean_projection_negative = -1.0
        mock_extractor = Mock()
        mock_extractor.extract_tool_mode_direction.return_value = mock_direction
        mock_extractor_class.return_value = mock_extractor

        main()

        mock_activations_class.load.assert_called_once()

    @pytest.mark.skip(reason="matplotlib/numpy incompatibility issue")
    @patch("matplotlib.pyplot")
    @patch("chuk_lazarus.introspection.circuit.geometry.GeometryAnalyzer")
    @patch("chuk_lazarus.introspection.circuit.collector.CollectedActivations")
    @patch("sys.argv", ["circuit", "visualize", "-a", "act.safetensors"])
    def test_main_visualize_command(self, mock_activations_class, mock_analyzer_class, mock_plt):
        """Test main routes to visualize command."""
        mock_activations = Mock()
        mock_activations.captured_layers = [10]
        mock_activations.hidden_size = 512
        mock_activations.__len__ = Mock(return_value=50)
        mock_activations_class.load.return_value = mock_activations

        mock_analyzer = Mock()
        mock_analyzer_class.return_value = mock_analyzer

        main()

        mock_activations_class.load.assert_called_once()

    @patch("chuk_lazarus.introspection.circuit.probes.ProbeBattery")
    @patch("sys.argv", ["circuit", "probes", "run", "-m", "model"])
    def test_main_probes_run_command(self, mock_battery_class):
        """Test main routes to probes run command."""
        mock_battery = Mock()
        mock_battery.num_layers = 12
        mock_battery.datasets = [Mock()]
        mock_results = Mock()
        mock_battery.run_all_probes.return_value = mock_results
        mock_battery_class.from_pretrained.return_value = mock_battery

        main()

        mock_battery_class.from_pretrained.assert_called_once()

    @pytest.mark.skip(reason="save_default_datasets not yet implemented")
    @patch("chuk_lazarus.introspection.circuit.cli.save_default_datasets")
    @patch("sys.argv", ["circuit", "probes", "init", "-o", "datasets"])
    def test_main_probes_init_command(self, mock_save_datasets):
        """Test main routes to probes init command."""
        main()

        mock_save_datasets.assert_called_once()

    @patch("sys.argv", ["circuit", "probes"])
    def test_main_probes_no_subcommand(self, capsys):
        """Test probes command without subcommand shows help."""
        main()
        # Should print help, not crash

    @patch("sys.argv", ["circuit", "unknown"])
    def test_main_unknown_command(self, capsys):
        """Test main with unknown command shows help."""
        # Unknown commands cause argparse to exit with status 2
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 2
