"""Tests for introspect steering CLI commands."""

import tempfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestIntrospectSteer:
    """Tests for introspect_steer command."""

    @pytest.fixture
    def steer_args(self):
        """Create arguments for steer command."""
        return Namespace(
            model="test-model",
            extract=False,
            direction=None,
            neuron=None,
            positive=None,
            negative=None,
            prompts="test prompt",
            layer=None,
            coefficient=1.0,
            compare=None,
            name=None,
            positive_label=None,
            negative_label=None,
            max_tokens=10,
            temperature=0.0,
            output=None,
        )

    def test_steer_extract_direction(self, steer_args, mock_activation_steering, capsys):
        """Test extracting direction from contrastive prompts."""
        from chuk_lazarus.cli.commands.introspect import introspect_steer

        steer_args.extract = True
        steer_args.positive = "good prompt"
        steer_args.negative = "bad prompt"

        with patch("chuk_lazarus.introspection.hooks.ModelHooks") as mock_hooks:
            mock_hook_instance = MagicMock()
            mock_hooks.return_value = mock_hook_instance

            # Mock hidden states
            import mlx.core as mx

            mock_hook_instance.state.hidden_states = {6: mx.zeros((1, 1, 768))}
            mock_hook_instance.forward.return_value = None

            introspect_steer(steer_args)

            captured = capsys.readouterr()
            assert "Loading model" in captured.out
            assert "Extracting direction" in captured.out

    def test_steer_extract_and_save(self, steer_args, mock_activation_steering):
        """Test extracting and saving direction."""
        from chuk_lazarus.cli.commands.introspect import introspect_steer

        steer_args.extract = True
        steer_args.positive = "good"
        steer_args.negative = "bad"

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            steer_args.output = f.name

        with patch("chuk_lazarus.introspection.hooks.ModelHooks") as mock_hooks:
            mock_hook_instance = MagicMock()
            mock_hooks.return_value = mock_hook_instance

            import mlx.core as mx

            mock_hook_instance.state.hidden_states = {6: mx.zeros((1, 1, 768))}

            introspect_steer(steer_args)

            # Check file was created
            assert Path(steer_args.output).exists()

    def test_steer_extract_missing_prompts(self, steer_args, mock_activation_steering, capsys):
        """Test that extract mode requires positive/negative prompts."""
        from chuk_lazarus.cli.commands.introspect import introspect_steer

        steer_args.extract = True
        steer_args.positive = None
        steer_args.negative = None

        with pytest.raises(SystemExit):
            introspect_steer(steer_args)

    def test_steer_apply_from_file(self, steer_args, mock_activation_steering, capsys):
        """Test applying steering from saved direction."""
        import numpy as np

        from chuk_lazarus.cli.commands.introspect import introspect_steer

        # Create a direction file
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez(
                f.name,
                direction=np.random.randn(768).astype(np.float32),
                layer=6,
            )
            steer_args.direction = f.name

        introspect_steer(steer_args)

        captured = capsys.readouterr()
        assert "Loading model" in captured.out
        assert "Loaded direction from" in captured.out

    def test_steer_apply_with_neuron(self, steer_args, mock_activation_steering, capsys):
        """Test steering single neuron."""
        from chuk_lazarus.cli.commands.introspect import introspect_steer

        steer_args.neuron = 42

        introspect_steer(steer_args)

        captured = capsys.readouterr()
        assert "Steering neuron 42" in captured.out

    def test_steer_compare_coefficients(self, steer_args, mock_activation_steering, capsys):
        """Test comparing multiple steering coefficients."""
        from chuk_lazarus.cli.commands.introspect import introspect_steer

        steer_args.compare = "-2,-1,0,1,2"
        steer_args.positive = "good"
        steer_args.negative = "bad"

        with patch("chuk_lazarus.introspection.hooks.ModelHooks") as mock_hooks:
            mock_hook_instance = MagicMock()
            mock_hooks.return_value = mock_hook_instance

            import mlx.core as mx

            mock_hook_instance.state.hidden_states = {6: mx.zeros((1, 1, 768))}

            introspect_steer(steer_args)

            captured = capsys.readouterr()
            assert "Comparing steering" in captured.out

    def test_steer_missing_direction_source(self, steer_args, mock_activation_steering, capsys):
        """Test error when no direction source provided."""
        from chuk_lazarus.cli.commands.introspect import introspect_steer

        # No direction, no neuron, no positive/negative
        with pytest.raises(SystemExit):
            introspect_steer(steer_args)

    def test_steer_from_json_direction(self, steer_args, mock_activation_steering, capsys):
        """Test loading direction from JSON file."""
        import json

        from chuk_lazarus.cli.commands.introspect import introspect_steer

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"direction": [0.1] * 768, "layer": 6}, f)
            steer_args.direction = f.name

        introspect_steer(steer_args)

        captured = capsys.readouterr()
        assert "Loading model" in captured.out

    def test_steer_save_results(self, steer_args, mock_activation_steering):
        """Test saving steering results."""
        from chuk_lazarus.cli.commands.introspect import introspect_steer

        steer_args.positive = "good"
        steer_args.negative = "bad"

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            steer_args.output = f.name

        with patch("chuk_lazarus.introspection.hooks.ModelHooks") as mock_hooks:
            mock_hook_instance = MagicMock()
            mock_hooks.return_value = mock_hook_instance

            import mlx.core as mx

            mock_hook_instance.state.hidden_states = {6: mx.zeros((1, 1, 768))}

            introspect_steer(steer_args)

            # Check results saved
            import json

            if Path(steer_args.output).exists():
                with open(steer_args.output) as f:
                    data = json.load(f)
                    assert isinstance(data, list)

    def test_steer_load_direction_with_metadata(self, steer_args, mock_activation_steering, capsys):
        """Test loading direction with all metadata fields (lines 131, 133, 136)."""
        import numpy as np

        from chuk_lazarus.cli.commands.introspect import introspect_steer

        # Create a direction file with full metadata
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez(
                f.name,
                direction=np.random.randn(768).astype(np.float32),
                layer=6,
                positive_prompt="good example",
                negative_prompt="bad example",
                norm=1.234,
                cosine_similarity=0.456,
            )
            steer_args.direction = f.name

        introspect_steer(steer_args)

        captured = capsys.readouterr()
        # Verify all metadata was printed
        assert "Loaded direction from" in captured.out
        assert "Positive: good example" in captured.out  # Line 131
        assert "Negative: bad example" in captured.out  # Line 133
        assert "Norm: 1.234" in captured.out  # Line 136

    def test_steer_unsupported_direction_format(self, steer_args, mock_activation_steering, capsys):
        """Test error handling for unsupported direction file format (lines 143-144)."""
        from chuk_lazarus.cli.commands.introspect import introspect_steer

        # Create a file with unsupported extension
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"test content")
            steer_args.direction = f.name

        with pytest.raises(SystemExit):
            introspect_steer(steer_args)

        captured = capsys.readouterr()
        assert "Unsupported direction format" in captured.out

    def test_steer_prompts_from_file(self, steer_args, mock_activation_steering, capsys):
        """Test loading prompts from file (lines 199-200)."""
        from chuk_lazarus.cli.commands.introspect import introspect_steer

        # Create a prompts file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("prompt 1\n")
            f.write("prompt 2\n")
            f.write("\n")  # Empty line to test stripping
            f.write("prompt 3\n")
            prompts_file = f.name

        steer_args.prompts = f"@{prompts_file}"
        steer_args.positive = "good"
        steer_args.negative = "bad"

        with patch("chuk_lazarus.introspection.hooks.ModelHooks") as mock_hooks:
            mock_hook_instance = MagicMock()
            mock_hooks.return_value = mock_hook_instance

            import mlx.core as mx

            mock_hook_instance.state.hidden_states = {6: mx.zeros((1, 1, 768))}

            introspect_steer(steer_args)

            captured = capsys.readouterr()
            # Verify prompts were loaded from file
            assert "prompt 1" in captured.out
            assert "prompt 2" in captured.out
            assert "prompt 3" in captured.out

    def test_steer_load_npz_without_layer(self, steer_args, mock_activation_steering):
        """Test loading npz direction without layer metadata."""
        import numpy as np

        from chuk_lazarus.cli.commands.introspect import introspect_steer

        # Create a direction file without layer
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez(
                f.name,
                direction=np.random.randn(768).astype(np.float32),
                # No layer metadata
            )
            steer_args.direction = f.name

        steer_args.layer = 10  # Should use this as fallback

        introspect_steer(steer_args)

        # Test passes if no exception is raised

    def test_steer_apply_on_the_fly_direction(self, steer_args, mock_activation_steering, capsys):
        """Test generating direction on-the-fly from positive/negative."""
        from chuk_lazarus.cli.commands.introspect import introspect_steer

        steer_args.positive = "happy"
        steer_args.negative = "sad"
        # No direction file or neuron

        with patch("chuk_lazarus.introspection.hooks.ModelHooks") as mock_hooks:
            mock_hook_instance = MagicMock()
            mock_hooks.return_value = mock_hook_instance

            import mlx.core as mx

            mock_hook_instance.state.hidden_states = {6: mx.zeros((1, 1, 768))}

            introspect_steer(steer_args)

            captured = capsys.readouterr()
            assert "Using on-the-fly direction" in captured.out

    def test_steer_compare_with_file_prompts(self, steer_args, mock_activation_steering, capsys):
        """Test compare mode with prompts from file."""
        from chuk_lazarus.cli.commands.introspect import introspect_steer

        # Create a prompts file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test prompt 1\n")
            f.write("test prompt 2\n")
            prompts_file = f.name

        steer_args.prompts = f"@{prompts_file}"
        steer_args.compare = "-1,0,1"
        steer_args.positive = "positive"
        steer_args.negative = "negative"

        with patch("chuk_lazarus.introspection.hooks.ModelHooks") as mock_hooks:
            mock_hook_instance = MagicMock()
            mock_hooks.return_value = mock_hook_instance

            import mlx.core as mx

            mock_hook_instance.state.hidden_states = {6: mx.zeros((1, 1, 768))}

            introspect_steer(steer_args)

            captured = capsys.readouterr()
            assert "Comparing steering" in captured.out
            assert "test prompt 1" in captured.out
            assert "test prompt 2" in captured.out
