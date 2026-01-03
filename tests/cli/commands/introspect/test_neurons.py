"""Tests for introspect neurons CLI commands."""

import tempfile
from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest


class TestIntrospectNeurons:
    """Tests for introspect_neurons command."""

    @pytest.fixture
    def neurons_args(self):
        """Create arguments for neurons command."""
        return Namespace(
            model="test-model",
            layer=12,
            layers=None,  # Both layer and layers are supported
            prompts="2+2=|47*47=",
            neurons="100,200",  # Use indices within 768-dim hidden state
            from_direction=None,
            top_k=10,
            labels=None,
            output=None,
        )

    def test_neurons_basic(self, neurons_args, mock_ablation_study, capsys):
        """Test basic neuron analysis."""
        from chuk_lazarus.cli.commands.introspect import introspect_neurons

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls:
            mock_hooks = MagicMock()

            import mlx.core as mx

            mock_hooks.state.hidden_states = {12: mx.zeros((1, 1, 768))}
            mock_hooks.forward.return_value = None
            mock_hooks_cls.return_value = mock_hooks

            introspect_neurons(neurons_args)

            captured = capsys.readouterr()
            assert "Loading model" in captured.out

    def test_neurons_from_direction(self, neurons_args, mock_ablation_study):
        """Test loading neurons from direction file."""
        from chuk_lazarus.cli.commands.introspect import introspect_neurons

        import numpy as np

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            direction = np.random.randn(768).astype(np.float32)
            np.savez(f.name, direction=direction, layer=12, top_neurons=[808, 1190])
            neurons_args.from_direction = f.name
            neurons_args.neurons = None

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls:
            mock_hooks = MagicMock()

            import mlx.core as mx

            mock_hooks.state.hidden_states = {12: mx.zeros((1, 1, 768))}
            mock_hooks_cls.return_value = mock_hooks

            introspect_neurons(neurons_args)

    def test_neurons_with_labels(self, neurons_args, mock_ablation_study, capsys):
        """Test neuron analysis with labels."""
        from chuk_lazarus.cli.commands.introspect import introspect_neurons

        neurons_args.labels = "easy|hard"

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls:
            mock_hooks = MagicMock()

            import mlx.core as mx

            mock_hooks.state.hidden_states = {12: mx.zeros((1, 1, 768))}
            mock_hooks_cls.return_value = mock_hooks

            introspect_neurons(neurons_args)

            captured = capsys.readouterr()
            assert "Loading" in captured.out

    def test_neurons_save_output(self, neurons_args, mock_ablation_study):
        """Test saving neuron analysis results."""
        from chuk_lazarus.cli.commands.introspect import introspect_neurons

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            neurons_args.output = f.name

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls:
            mock_hooks = MagicMock()

            import mlx.core as mx

            mock_hooks.state.hidden_states = {12: mx.zeros((1, 1, 768))}
            mock_hooks_cls.return_value = mock_hooks

            introspect_neurons(neurons_args)


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

    def test_directions_basic(self, directions_args, capsys):
        """Test comparing direction vectors."""
        from chuk_lazarus.cli.commands.introspect import introspect_directions

        import numpy as np

        # Create direction files
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f1:
            np.savez(f1.name, direction=np.random.randn(768).astype(np.float32), layer=12)
            directions_args.files[0] = f1.name

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f2:
            np.savez(f2.name, direction=np.random.randn(768).astype(np.float32), layer=12)
            directions_args.files[1] = f2.name

        introspect_directions(directions_args)

        captured = capsys.readouterr()
        # Should show similarity matrix
        assert "Loading" in captured.out or "Direction" in captured.out or "Cosine" in captured.out


class TestIntrospectOperandDirections:
    """Tests for introspect_operand_directions command."""

    @pytest.fixture
    def operand_args(self):
        """Create arguments for operand directions command."""
        return Namespace(
            model="test-model",
            layers=None,
            operation="*",
            digits=None,
            output=None,
        )

    def test_operand_directions_basic(self, operand_args, mock_ablation_study, capsys):
        """Test basic operand direction analysis."""
        from chuk_lazarus.cli.commands.introspect import introspect_operand_directions
        from collections import defaultdict

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls:
            mock_hooks = MagicMock()

            import mlx.core as mx

            # Use defaultdict to provide hidden states for any requested layer
            def make_hidden_states():
                return mx.zeros((1, 1, 768))

            mock_hooks.state.hidden_states = defaultdict(make_hidden_states)
            mock_hooks.forward.return_value = None
            mock_hooks_cls.return_value = mock_hooks

            introspect_operand_directions(operand_args)

            captured = capsys.readouterr()
            assert "Loading model" in captured.out

    def test_operand_directions_specific_layers(self, operand_args, mock_ablation_study):
        """Test with specific layers."""
        from chuk_lazarus.cli.commands.introspect import introspect_operand_directions
        from collections import defaultdict

        operand_args.layers = "4,8,12"

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls:
            mock_hooks = MagicMock()

            import mlx.core as mx

            def make_hidden_states():
                return mx.zeros((1, 1, 768))

            mock_hooks.state.hidden_states = defaultdict(make_hidden_states)
            mock_hooks_cls.return_value = mock_hooks

            introspect_operand_directions(operand_args)

    def test_operand_directions_addition(self, operand_args, mock_ablation_study):
        """Test with addition operation."""
        from chuk_lazarus.cli.commands.introspect import introspect_operand_directions
        from collections import defaultdict

        operand_args.operation = "+"

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls:
            mock_hooks = MagicMock()

            import mlx.core as mx

            def make_hidden_states():
                return mx.zeros((1, 1, 768))

            mock_hooks.state.hidden_states = defaultdict(make_hidden_states)
            mock_hooks_cls.return_value = mock_hooks

            introspect_operand_directions(operand_args)

    def test_operand_directions_custom_digits(self, operand_args, mock_ablation_study):
        """Test with custom digit range."""
        from chuk_lazarus.cli.commands.introspect import introspect_operand_directions
        from collections import defaultdict

        operand_args.digits = "2,3,5,7"

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls:
            mock_hooks = MagicMock()

            import mlx.core as mx

            def make_hidden_states():
                return mx.zeros((1, 1, 768))

            mock_hooks.state.hidden_states = defaultdict(make_hidden_states)
            mock_hooks_cls.return_value = mock_hooks

            introspect_operand_directions(operand_args)

    def test_operand_directions_save_output(self, operand_args, mock_ablation_study):
        """Test saving operand analysis results."""
        from chuk_lazarus.cli.commands.introspect import introspect_operand_directions
        from collections import defaultdict

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            operand_args.output = f.name

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls:
            mock_hooks = MagicMock()

            import mlx.core as mx

            def make_hidden_states():
                return mx.zeros((1, 1, 768))

            mock_hooks.state.hidden_states = defaultdict(make_hidden_states)
            mock_hooks_cls.return_value = mock_hooks

            introspect_operand_directions(operand_args)
