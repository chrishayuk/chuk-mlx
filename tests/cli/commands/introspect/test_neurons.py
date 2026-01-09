"""Tests for introspect neurons CLI commands."""

import tempfile
from argparse import Namespace

import numpy as np
import pytest


class TestNeuronAnalysisConfig:
    """Tests for NeuronAnalysisConfig."""

    def test_from_args_basic(self):
        """Test creating config from args."""
        from chuk_lazarus.cli.commands.introspect._types import NeuronAnalysisConfig

        args = Namespace(
            model="test-model",
            layer=12,
            layers=None,
            prompts="2+2=|47*47=",
            neurons="100,200",
            from_direction=None,
            top_k=10,
            labels=None,
            output=None,
            steer=None,
            strength=None,
            auto_discover=False,
            neuron_names=None,
        )

        config = NeuronAnalysisConfig.from_args(args)

        assert config.model == "test-model"
        assert config.layer == 12
        assert config.prompts == "2+2=|47*47="
        assert config.neurons == "100,200"
        assert config.top_k == 10

    def test_from_args_with_layers(self):
        """Test creating config with layers string."""
        from chuk_lazarus.cli.commands.introspect._types import NeuronAnalysisConfig

        args = Namespace(
            model="test-model",
            layer=None,
            layers="4,8,12",
            prompts="test",
            neurons="100",
            from_direction=None,
            top_k=10,
            labels=None,
            output=None,
            steer=None,
            strength=None,
            auto_discover=False,
            neuron_names=None,
        )

        config = NeuronAnalysisConfig.from_args(args)

        assert config.layers == "4,8,12"

    def test_from_args_with_auto_discover(self):
        """Test creating config with auto-discover."""
        from chuk_lazarus.cli.commands.introspect._types import NeuronAnalysisConfig

        args = Namespace(
            model="test-model",
            layer=12,
            layers=None,
            prompts="easy|hard",
            neurons=None,
            from_direction=None,
            top_k=5,
            labels="easy|hard",
            output=None,
            steer=None,
            strength=None,
            auto_discover=True,
            neuron_names=None,
        )

        config = NeuronAnalysisConfig.from_args(args)

        assert config.auto_discover is True
        assert config.labels == "easy|hard"


class TestIntrospectNeurons:
    """Tests for introspect_neurons command."""

    @pytest.fixture
    def neurons_args(self):
        """Create arguments for neurons command."""
        return Namespace(
            model="test-model",
            layer=12,
            layers=None,
            prompts="2+2=|47*47=",
            neurons="100,200",
            from_direction=None,
            top_k=10,
            labels=None,
            output=None,
            steer=None,
            strength=None,
            auto_discover=False,
            neuron_names=None,
        )

    def test_neurons_basic(self, neurons_args, capsys):
        """Test basic neuron analysis."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_neurons

        introspect_neurons(neurons_args)

        captured = capsys.readouterr()
        assert "Loading model" in captured.out

    def test_neurons_no_layer_specified(self, neurons_args, capsys):
        """Test error when no layer specified."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_neurons

        neurons_args.layer = None
        neurons_args.layers = None

        introspect_neurons(neurons_args)

        captured = capsys.readouterr()
        assert "ERROR: Must specify --layer or --layers" in captured.out

    def test_neurons_with_layers_string(self, neurons_args, capsys):
        """Test neurons with multiple layers specified as comma-separated string."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_neurons

        neurons_args.layer = None
        neurons_args.layers = "4,8,12"

        introspect_neurons(neurons_args)

        captured = capsys.readouterr()
        assert "Analyzing layers: [4, 8, 12]" in captured.out

    def test_neurons_with_labels(self, neurons_args, capsys):
        """Test neuron analysis with labels."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_neurons

        neurons_args.labels = "easy|hard"

        introspect_neurons(neurons_args)

        captured = capsys.readouterr()
        assert "Loading" in captured.out

    def test_neurons_with_neuron_names(self, neurons_args, capsys):
        """Test neuron analysis with custom neuron names."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_neurons

        neurons_args.neuron_names = "carry_detector|result_encoder"

        introspect_neurons(neurons_args)

        captured = capsys.readouterr()
        assert "Neuron names:" in captured.out

    def test_neurons_neuron_names_mismatch(self, neurons_args, capsys):
        """Test warning when neuron name count doesn't match neuron count."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_neurons

        neurons_args.neuron_names = "only_one_name"  # 1 name for 2 neurons

        introspect_neurons(neurons_args)

        captured = capsys.readouterr()
        # Names won't be used if count doesn't match
        assert "Neuron names:" not in captured.out

    def test_neurons_no_source_error(self, neurons_args, capsys):
        """Test error when no neuron source specified."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_neurons

        neurons_args.neurons = None
        neurons_args.from_direction = None
        neurons_args.auto_discover = False
        neurons_args.labels = None

        introspect_neurons(neurons_args)

        captured = capsys.readouterr()
        assert "ERROR: Must specify --neurons, --from-direction, or --auto-discover" in captured.out

    def test_neurons_auto_discover_with_labels(self, neurons_args, capsys):
        """Test auto-discover mode with labels."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_neurons

        neurons_args.neurons = None
        neurons_args.labels = "easy|hard"
        neurons_args.auto_discover = True

        introspect_neurons(neurons_args)

        captured = capsys.readouterr()
        assert "Auto-discovering discriminative neurons" in captured.out

    def test_neurons_auto_discover_inferred(self, neurons_args, capsys):
        """Test that auto-discover is inferred when labels but no neurons."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_neurons

        neurons_args.neurons = None
        neurons_args.labels = "cat1|cat2"
        # Don't set auto_discover=True - it should be inferred

        introspect_neurons(neurons_args)

        captured = capsys.readouterr()
        assert "Auto-discovering discriminative neurons" in captured.out

    def test_neurons_label_count_mismatch(self, neurons_args, capsys):
        """Test warning when label count doesn't match prompt count."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_neurons

        neurons_args.labels = "easy"  # Only 1 label for 2 prompts

        introspect_neurons(neurons_args)

        captured = capsys.readouterr()
        assert "Warning: 1 labels for 2 prompts" in captured.out


class TestDirectionComparisonConfig:
    """Tests for DirectionComparisonConfig."""

    def test_from_args(self):
        """Test creating config from args."""
        from chuk_lazarus.cli.commands.introspect._types import DirectionComparisonConfig

        args = Namespace(
            files=["dir1.npz", "dir2.npz"],
            threshold=0.1,
            output=None,
        )

        config = DirectionComparisonConfig.from_args(args)

        assert len(config.files) == 2
        assert config.threshold == 0.1


class TestIntrospectDirections:
    """Tests for introspect_directions command."""

    @pytest.fixture
    def directions_args(self):
        """Create arguments for directions command."""
        return Namespace(
            files=["dir1.npz", "dir2.npz"],
            threshold=0.1,
            output=None,
        )

    def test_directions_single_file_error(self, directions_args, capsys):
        """Test error when only one file provided."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_directions

        directions_args.files = ["single.npz"]

        introspect_directions(directions_args)

        captured = capsys.readouterr()
        assert "ERROR: Need at least 2 direction files" in captured.out

    def test_directions_file_not_found(self, directions_args, capsys):
        """Test error when file doesn't exist."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_directions

        directions_args.files = ["/nonexistent/path.npz", "/another/fake.npz"]

        introspect_directions(directions_args)

        captured = capsys.readouterr()
        assert "ERROR: File not found" in captured.out

    def test_directions_basic(self, directions_args, capsys):
        """Test comparing direction vectors."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_directions

        # Create direction files
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f1:
            np.savez(f1.name, direction=np.random.randn(768).astype(np.float32), layer=12)
            directions_args.files[0] = f1.name

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f2:
            np.savez(f2.name, direction=np.random.randn(768).astype(np.float32), layer=12)
            directions_args.files[1] = f2.name

        introspect_directions(directions_args)

        captured = capsys.readouterr()
        assert "Loading" in captured.out or "COSINE SIMILARITY MATRIX" in captured.out

    def test_directions_with_labels(self, directions_args, capsys):
        """Test directions with label metadata."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_directions

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f1:
            np.savez(
                f1.name,
                direction=np.random.randn(768).astype(np.float32),
                layer=12,
                label_positive="positive",
                label_negative="negative",
            )
            directions_args.files[0] = f1.name

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f2:
            np.savez(
                f2.name,
                direction=np.random.randn(768).astype(np.float32),
                layer=12,
                label_positive="correct",
                label_negative="wrong",
            )
            directions_args.files[1] = f2.name

        introspect_directions(directions_args)

        captured = capsys.readouterr()
        assert "negative->positive" in captured.out
        assert "wrong->correct" in captured.out

    def test_directions_aligned_vectors(self, directions_args, capsys):
        """Test with highly aligned direction vectors."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_directions

        base_direction = np.random.randn(768).astype(np.float32)
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f1:
            np.savez(f1.name, direction=base_direction, layer=12)
            directions_args.files[0] = f1.name

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f2:
            # Same direction = highly aligned
            np.savez(f2.name, direction=base_direction * 0.9, layer=12)
            directions_args.files[1] = f2.name

        introspect_directions(directions_args)

        captured = capsys.readouterr()
        assert "Aligned" in captured.out or "HIGHLY correlated" in captured.out

    def test_directions_save_output(self, directions_args, capsys):
        """Test saving direction comparison results."""
        import json
        from pathlib import Path

        from chuk_lazarus.cli.commands.introspect.neurons import introspect_directions

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f1:
            np.savez(f1.name, direction=np.random.randn(768).astype(np.float32), layer=12)
            directions_args.files[0] = f1.name

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f2:
            np.savez(f2.name, direction=np.random.randn(768).astype(np.float32), layer=12)
            directions_args.files[1] = f2.name

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as out:
            directions_args.output = out.name

        introspect_directions(directions_args)

        captured = capsys.readouterr()
        assert "Results saved to" in captured.out

        if Path(directions_args.output).exists():
            with open(directions_args.output) as f:
                data = json.load(f)
                assert "pairs" in data


class TestDirectionPairSimilarity:
    """Tests for DirectionPairSimilarity result type."""

    def test_basic_creation(self):
        """Test creating pair similarity result."""
        from chuk_lazarus.cli.commands.introspect._types import DirectionPairSimilarity

        pair = DirectionPairSimilarity(
            name_a="positive->negative",
            name_b="correct->wrong",
            cosine_similarity=0.8,
            orthogonal=False,
        )

        assert pair.name_a == "positive->negative"
        assert pair.cosine_similarity == 0.8
        assert pair.orthogonal is False


class TestParseLayers:
    """Tests for layer parsing utility."""

    def test_parse_layers_string(self):
        """Test parsing layer string."""
        from chuk_lazarus.cli.commands.introspect._types import parse_layers_string

        layers = parse_layers_string("4,8,12")
        assert layers == [4, 8, 12]

    def test_parse_layers_single(self):
        """Test parsing single layer."""
        from chuk_lazarus.cli.commands.introspect._types import parse_layers_string

        layers = parse_layers_string("6")
        assert layers == [6]

    def test_parse_layers_with_spaces(self):
        """Test parsing layers with spaces."""
        from chuk_lazarus.cli.commands.introspect._types import parse_layers_string

        layers = parse_layers_string("4, 8, 12")
        assert layers == [4, 8, 12]
