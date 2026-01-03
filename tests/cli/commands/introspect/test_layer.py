"""Tests for introspect layer CLI commands."""

import tempfile
from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest


class TestIntrospectLayer:
    """Tests for introspect_layer command."""

    @pytest.fixture
    def layer_args(self):
        """Create arguments for layer command."""
        return Namespace(
            model="test-model",
            prompts="test1|test2",
            labels=None,
            layers=None,
            attention=False,
            output=None,
        )

    def test_layer_basic(self, layer_args, capsys):
        """Test basic layer analysis."""
        from chuk_lazarus.cli.commands.introspect import introspect_layer

        with patch("chuk_lazarus.introspection.LayerAnalyzer") as mock_cls:
            mock_analyzer = MagicMock()
            mock_cls.from_pretrained.return_value = mock_analyzer

            # Mock result
            mock_result = MagicMock()
            mock_result.layers = [0, 4, 8]
            mock_result.representations = {}
            mock_result.clusters = None
            mock_analyzer.analyze_representations.return_value = mock_result

            introspect_layer(layer_args)

            captured = capsys.readouterr()
            assert "Loading model" in captured.out

    def test_layer_from_file(self, layer_args):
        """Test loading prompts from file."""
        from chuk_lazarus.cli.commands.introspect import introspect_layer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("prompt1\nprompt2\n")
            f.flush()

            layer_args.prompts = f"@{f.name}"

            with patch("chuk_lazarus.introspection.LayerAnalyzer") as mock_cls:
                mock_analyzer = MagicMock()
                mock_cls.from_pretrained.return_value = mock_analyzer

                mock_result = MagicMock()
                mock_result.layers = [0]
                mock_result.representations = {}
                mock_result.clusters = None
                mock_analyzer.analyze_representations.return_value = mock_result

                introspect_layer(layer_args)

    def test_layer_with_labels(self, layer_args, capsys):
        """Test layer analysis with labels."""
        from chuk_lazarus.cli.commands.introspect import introspect_layer

        layer_args.labels = "A,B"

        with patch("chuk_lazarus.introspection.LayerAnalyzer") as mock_cls:
            mock_analyzer = MagicMock()
            mock_cls.from_pretrained.return_value = mock_analyzer

            mock_result = MagicMock()
            mock_result.layers = [0]
            mock_result.representations = {}

            # Mock cluster results
            mock_cluster = MagicMock()
            mock_cluster.within_cluster_similarity = {"A": 0.9, "B": 0.85}
            mock_cluster.between_cluster_similarity = {("A", "B"): 0.5}
            mock_cluster.separation_score = 0.35
            mock_result.clusters = {0: mock_cluster}

            mock_analyzer.analyze_representations.return_value = mock_result

            introspect_layer(layer_args)

            captured = capsys.readouterr()
            assert "Loading model" in captured.out

    def test_layer_specific_layers(self, layer_args):
        """Test specifying layers to analyze."""
        from chuk_lazarus.cli.commands.introspect import introspect_layer

        layer_args.layers = "4,8,12"

        with patch("chuk_lazarus.introspection.LayerAnalyzer") as mock_cls:
            mock_analyzer = MagicMock()
            mock_cls.from_pretrained.return_value = mock_analyzer

            mock_result = MagicMock()
            mock_result.layers = [4, 8, 12]
            mock_result.representations = {}
            mock_result.clusters = None
            mock_analyzer.analyze_representations.return_value = mock_result

            introspect_layer(layer_args)

            # Check that analyze_representations was called with correct layers
            call_args = mock_analyzer.analyze_representations.call_args
            assert call_args[1]["layers"] == [4, 8, 12]

    def test_layer_with_attention(self, layer_args, capsys):
        """Test layer analysis with attention."""
        from chuk_lazarus.cli.commands.introspect import introspect_layer

        layer_args.attention = True

        with patch("chuk_lazarus.introspection.LayerAnalyzer") as mock_cls:
            mock_analyzer = MagicMock()
            mock_cls.from_pretrained.return_value = mock_analyzer

            mock_result = MagicMock()
            mock_result.layers = [0]
            mock_result.representations = {}
            mock_result.clusters = None
            mock_analyzer.analyze_representations.return_value = mock_result
            mock_analyzer.analyze_attention.return_value = {}

            introspect_layer(layer_args)

            captured = capsys.readouterr()
            assert "Attention Analysis" in captured.out or "Loading" in captured.out


class TestIntrospectFormatSensitivity:
    """Tests for introspect_format_sensitivity command."""

    @pytest.fixture
    def format_args(self):
        """Create arguments for format sensitivity command."""
        return Namespace(
            model="test-model",
            prompts="test1|test2",
            layers=None,
        )

    def test_format_sensitivity_basic(self, format_args, capsys):
        """Test basic format sensitivity analysis."""
        from chuk_lazarus.cli.commands.introspect import introspect_format_sensitivity

        with patch("chuk_lazarus.introspection.analyze_format_sensitivity") as mock_fn:
            mock_result = MagicMock()
            mock_result.layers = [0, 4]

            mock_cluster = MagicMock()
            mock_cluster.separation_score = 0.05
            mock_result.clusters = {0: mock_cluster, 4: mock_cluster}

            mock_fn.return_value = mock_result

            introspect_format_sensitivity(format_args)

            captured = capsys.readouterr()
            assert "Format sensitivity" in captured.out

    def test_format_sensitivity_with_layers(self, format_args):
        """Test format sensitivity with specific layers."""
        from chuk_lazarus.cli.commands.introspect import introspect_format_sensitivity

        format_args.layers = "4,8"

        with patch("chuk_lazarus.introspection.analyze_format_sensitivity") as mock_fn:
            mock_result = MagicMock()
            mock_result.layers = [4, 8]
            mock_result.clusters = {}
            mock_fn.return_value = mock_result

            introspect_format_sensitivity(format_args)

            # Check layers were passed correctly
            call_args = mock_fn.call_args
            assert call_args[1]["layers"] == [4, 8]
