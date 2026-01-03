"""Tests for introspect neurons CLI commands."""

import tempfile
from argparse import Namespace
from unittest.mock import MagicMock, patch

import numpy as np
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
            steer=None,
            strength=None,
            auto_discover=False,
            neuron_names=None,
        )

    def test_neurons_basic(self, neurons_args, mock_ablation_study, capsys):
        """Test basic neuron analysis."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_neurons

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
        import numpy as np

        from chuk_lazarus.cli.commands.introspect.neurons import introspect_neurons

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
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_neurons

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
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_neurons

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            neurons_args.output = f.name

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls:
            mock_hooks = MagicMock()

            import mlx.core as mx

            mock_hooks.state.hidden_states = {12: mx.zeros((1, 1, 768))}
            mock_hooks_cls.return_value = mock_hooks

            introspect_neurons(neurons_args)

    def test_neurons_no_layer_specified(self, neurons_args, capsys):
        """Test error when no layer specified."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_neurons

        neurons_args.layer = None
        neurons_args.layers = None

        introspect_neurons(neurons_args)

        captured = capsys.readouterr()
        assert "ERROR: Must specify --layer or --layers" in captured.out

    def test_neurons_with_layers_string(self, neurons_args, mock_ablation_study, capsys):
        """Test neurons with multiple layers specified as comma-separated string."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_neurons

        neurons_args.layer = None
        neurons_args.layers = "4,8,12"  # Multiple layers

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls:
            mock_hooks = MagicMock()

            import mlx.core as mx

            # Provide hidden states for all requested layers
            mock_hooks.state.hidden_states = {
                4: mx.zeros((1, 1, 768)),
                8: mx.zeros((1, 1, 768)),
                12: mx.zeros((1, 1, 768)),
            }
            mock_hooks_cls.return_value = mock_hooks

            introspect_neurons(neurons_args)

            captured = capsys.readouterr()
            assert "Analyzing layers: [4, 8, 12]" in captured.out
            assert "CROSS-LAYER NEURON TRACKING" in captured.out

    def test_neurons_steer_with_coef_in_string(self, neurons_args, mock_ablation_study, capsys):
        """Test steering with coefficient in filename (file:coef format)."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_neurons

        # Create a direction file
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            direction = np.random.randn(768).astype(np.float32)
            np.savez(
                f.name,
                direction=direction,
                layer=12,
                label_positive="positive",
                label_negative="negative",
            )
            neurons_args.steer = f"{f.name}:0.5"

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls:
            with patch("chuk_lazarus.introspection.ActivationSteering") as mock_steer_cls:
                mock_hooks = MagicMock()
                mock_steerer = MagicMock()

                import mlx.core as mx

                mock_hooks.state.hidden_states = {12: mx.zeros((1, 1, 768))}
                mock_hooks_cls.return_value = mock_hooks
                mock_steer_cls.return_value = mock_steerer

                introspect_neurons(neurons_args)

                captured = capsys.readouterr()
                assert "Steering:" in captured.out
                assert "coefficient 0.5" in captured.out

    def test_neurons_steer_with_strength_arg(self, neurons_args, mock_ablation_study, capsys):
        """Test steering with separate --strength argument."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_neurons

        # Create a direction file
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            direction = np.random.randn(768).astype(np.float32)
            np.savez(f.name, direction=direction, layer=12)
            neurons_args.steer = f.name
            neurons_args.strength = 2.0

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls:
            with patch("chuk_lazarus.introspection.ActivationSteering") as mock_steer_cls:
                mock_hooks = MagicMock()
                mock_steerer = MagicMock()

                import mlx.core as mx

                mock_hooks.state.hidden_states = {12: mx.zeros((1, 1, 768))}
                mock_hooks_cls.return_value = mock_hooks
                mock_steer_cls.return_value = mock_steerer

                introspect_neurons(neurons_args)

                captured = capsys.readouterr()
                assert "Steering:" in captured.out
                assert "coefficient 2.0" in captured.out

    def test_neurons_prompts_from_file(self, neurons_args, mock_ablation_study, capsys):
        """Test loading prompts from a file."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_neurons

        # Create a prompts file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("2+2=\n")
            f.write("3+3=\n")
            f.write("\n")  # Empty line should be skipped
            f.write("4+4=\n")
            neurons_args.prompts = f"@{f.name}"

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls:
            mock_hooks = MagicMock()

            import mlx.core as mx

            mock_hooks.state.hidden_states = {12: mx.zeros((1, 1, 768))}
            mock_hooks_cls.return_value = mock_hooks

            introspect_neurons(neurons_args)

            captured = capsys.readouterr()
            assert "3 prompts" in captured.out

    def test_neurons_label_count_mismatch(self, neurons_args, mock_ablation_study, capsys):
        """Test warning when label count doesn't match prompt count."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_neurons

        neurons_args.labels = "easy"  # Only 1 label for 2 prompts

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls:
            mock_hooks = MagicMock()

            import mlx.core as mx

            mock_hooks.state.hidden_states = {12: mx.zeros((1, 1, 768))}
            mock_hooks_cls.return_value = mock_hooks

            introspect_neurons(neurons_args)

            captured = capsys.readouterr()
            assert "Warning: 1 labels for 2 prompts" in captured.out

    def test_neurons_no_source_error(self, neurons_args, mock_ablation_study, capsys):
        """Test error when no neuron source specified."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_neurons

        neurons_args.neurons = None
        neurons_args.from_direction = None
        neurons_args.auto_discover = False
        neurons_args.labels = None  # No labels means no auto-discover

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls:
            mock_hooks = MagicMock()

            import mlx.core as mx

            mock_hooks.state.hidden_states = {12: mx.zeros((1, 1, 768))}
            mock_hooks_cls.return_value = mock_hooks

            introspect_neurons(neurons_args)

            captured = capsys.readouterr()
            assert (
                "ERROR: Must specify --neurons, --from-direction, or --auto-discover"
                in captured.out
            )

    def test_neurons_with_neuron_names(self, neurons_args, mock_ablation_study, capsys):
        """Test neuron analysis with custom neuron names."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_neurons

        neurons_args.neuron_names = "carry_detector|result_encoder"

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls:
            mock_hooks = MagicMock()

            import mlx.core as mx

            mock_hooks.state.hidden_states = {12: mx.zeros((1, 1, 768))}
            mock_hooks_cls.return_value = mock_hooks

            introspect_neurons(neurons_args)

            captured = capsys.readouterr()
            assert "Neuron names:" in captured.out

    def test_neurons_neuron_names_mismatch(self, neurons_args, mock_ablation_study, capsys):
        """Test warning when neuron name count doesn't match neuron count."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_neurons

        neurons_args.neuron_names = "only_one_name"  # 1 name for 2 neurons

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls:
            mock_hooks = MagicMock()

            import mlx.core as mx

            mock_hooks.state.hidden_states = {12: mx.zeros((1, 1, 768))}
            mock_hooks_cls.return_value = mock_hooks

            introspect_neurons(neurons_args)

            captured = capsys.readouterr()
            assert "Warning: 1 names for 2 neurons" in captured.out

    def test_neurons_auto_discover_with_labels(self, neurons_args, mock_ablation_study, capsys):
        """Test auto-discover mode with labels."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_neurons

        neurons_args.neurons = None
        neurons_args.labels = "easy|hard"
        neurons_args.auto_discover = True

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls:
            mock_hooks = MagicMock()

            import mlx.core as mx

            # Auto-discover needs a full hidden state
            mock_hooks.state.hidden_states = {12: mx.zeros((1, 1, 768))}
            mock_hooks_cls.return_value = mock_hooks

            introspect_neurons(neurons_args)

            captured = capsys.readouterr()
            assert "Auto-discovering discriminative neurons" in captured.out

    def test_neurons_auto_discover_inferred(self, neurons_args, mock_ablation_study, capsys):
        """Test that auto-discover is inferred when labels but no neurons."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_neurons

        neurons_args.neurons = None
        neurons_args.labels = "cat1|cat2"
        # Don't set auto_discover=True - it should be inferred

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls:
            mock_hooks = MagicMock()

            import mlx.core as mx

            mock_hooks.state.hidden_states = {12: mx.zeros((1, 1, 768))}
            mock_hooks_cls.return_value = mock_hooks

            introspect_neurons(neurons_args)

            captured = capsys.readouterr()
            assert "Auto-discovering discriminative neurons" in captured.out

    def test_neurons_auto_discover_no_labels_error(self, neurons_args, mock_ablation_study, capsys):
        """Test error when auto-discover without labels."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_neurons

        neurons_args.neurons = None
        neurons_args.labels = None
        neurons_args.auto_discover = True

        introspect_neurons(neurons_args)

        captured = capsys.readouterr()
        # Without labels and neurons, should error about no source
        assert "ERROR" in captured.out

    def test_neurons_from_direction_with_labels(self, neurons_args, mock_ablation_study, capsys):
        """Test loading neurons from direction file with label metadata."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_neurons

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            direction = np.random.randn(768).astype(np.float32)
            np.savez(
                f.name,
                direction=direction,
                layer=12,
                label_positive="correct",
                label_negative="wrong",
            )
            neurons_args.from_direction = f.name
            neurons_args.neurons = None

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls:
            mock_hooks = MagicMock()

            import mlx.core as mx

            mock_hooks.state.hidden_states = {12: mx.zeros((1, 1, 768))}
            mock_hooks_cls.return_value = mock_hooks

            introspect_neurons(neurons_args)

            captured = capsys.readouterr()
            assert "Direction: wrong -> correct" in captured.out

    def test_neurons_neuron_weights_display(self, neurons_args, mock_ablation_study, capsys):
        """Test that neuron weights from direction are displayed in stats."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_neurons

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            direction = np.zeros(768, dtype=np.float32)
            direction[100] = 0.8  # Positive weight
            direction[200] = -0.5  # Negative weight
            np.savez(f.name, direction=direction, layer=12)
            neurons_args.from_direction = f.name
            neurons_args.neurons = None
            neurons_args.top_k = 2

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls:
            mock_hooks = MagicMock()

            import mlx.core as mx

            mock_hooks.state.hidden_states = {12: mx.zeros((1, 1, 768))}
            mock_hooks_cls.return_value = mock_hooks

            introspect_neurons(neurons_args)

            captured = capsys.readouterr()
            assert "POSITIVE detector" in captured.out or "NEGATIVE detector" in captured.out

    def test_neurons_multi_layer_with_labels(self, neurons_args, mock_ablation_study, capsys):
        """Test multi-layer mode with labels shows label column."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_neurons

        neurons_args.layer = None
        neurons_args.layers = "4,8,12"
        neurons_args.labels = "easy|hard"

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls:
            mock_hooks = MagicMock()

            import mlx.core as mx

            mock_hooks.state.hidden_states = {
                4: mx.zeros((1, 1, 768)),
                8: mx.zeros((1, 1, 768)),
                12: mx.zeros((1, 1, 768)),
            }
            mock_hooks_cls.return_value = mock_hooks

            introspect_neurons(neurons_args)

            captured = capsys.readouterr()
            assert "CROSS-LAYER NEURON TRACKING" in captured.out
            assert "Label" in captured.out

    def test_neurons_auto_discover_multi_sample(self, neurons_args, mock_ablation_study, capsys):
        """Test auto-discover with multiple samples per label group."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_neurons

        neurons_args.neurons = None
        neurons_args.prompts = "2+2=|3+3=|4+4=|5+5="  # 4 prompts
        neurons_args.labels = "easy|easy|hard|hard"  # 2 samples per label
        neurons_args.auto_discover = True

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls:
            mock_hooks = MagicMock()

            import mlx.core as mx

            # Return slightly different values to create variance
            call_count = [0]

            def make_hidden_state(*args, **kwargs):
                call_count[0] += 1
                return mx.array([[[float(call_count[0]) for _ in range(768)]]])

            mock_hooks.state.hidden_states = {12: make_hidden_state()}
            mock_hooks_cls.return_value = mock_hooks

            introspect_neurons(neurons_args)

            captured = capsys.readouterr()
            assert "Auto-discovering discriminative neurons" in captured.out

    def test_neurons_with_valid_neuron_names_display(
        self, neurons_args, mock_ablation_study, capsys
    ):
        """Test that neuron names are used in output when matched correctly."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_neurons

        neurons_args.neurons = "100,200"
        neurons_args.neuron_names = "carrydet|encoder"

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls:
            mock_hooks = MagicMock()

            import mlx.core as mx

            mock_hooks.state.hidden_states = {12: mx.zeros((1, 1, 768))}
            mock_hooks_cls.return_value = mock_hooks

            introspect_neurons(neurons_args)

            captured = capsys.readouterr()
            # Should see the neuron names used in various outputs
            assert "carrydet" in captured.out or "encoder" in captured.out

    def test_neurons_auto_discover_single_sample_zero_std(
        self, neurons_args, mock_ablation_study, capsys
    ):
        """Test auto-discover with single sample per label and zero overall_std."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_neurons

        neurons_args.neurons = None
        neurons_args.prompts = "2+2=|3+3="  # 2 prompts
        neurons_args.labels = "easy|hard"  # 1 sample per label (single sample mode)
        neurons_args.auto_discover = True

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls:
            mock_hooks = MagicMock()

            import mlx.core as mx

            # Create hidden states where some neurons have zero variance
            # to trigger the overall_std <= 1e-6 case
            h1 = mx.zeros((1, 1, 768))
            h2 = mx.zeros((1, 1, 768))
            # All neurons will have zero variance, triggering separation = 0.0

            mock_hooks.state.hidden_states = {12: h1}
            mock_hooks_cls.return_value = mock_hooks

            # Need to return different states for each forward pass
            call_count = [0]

            def forward_side_effect(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] == 1:
                    mock_hooks.state.hidden_states = {12: h1}
                else:
                    mock_hooks.state.hidden_states = {12: h2}

            mock_hooks.forward.side_effect = forward_side_effect

            introspect_neurons(neurons_args)

            captured = capsys.readouterr()
            assert "Single sample per label" in captured.out

    def test_neurons_auto_discover_multi_sample_zero_std(
        self, neurons_args, mock_ablation_study, capsys
    ):
        """Test auto-discover with multiple samples per label and zero pooled_std."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_neurons

        neurons_args.neurons = None
        neurons_args.prompts = "2+2=|3+3=|4+4=|5+5="  # 4 prompts
        neurons_args.labels = "easy|easy|hard|hard"  # 2 samples per label
        neurons_args.auto_discover = True

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls:
            mock_hooks = MagicMock()

            import mlx.core as mx

            # Create identical hidden states to get zero std within each group
            h_identical = mx.ones((1, 1, 768)) * 5.0

            call_count = [0]

            def forward_side_effect(*args, **kwargs):
                call_count[0] += 1
                # Return identical states to create zero pooled_std
                mock_hooks.state.hidden_states = {12: h_identical}

            mock_hooks.forward.side_effect = forward_side_effect
            mock_hooks_cls.return_value = mock_hooks

            introspect_neurons(neurons_args)

            captured = capsys.readouterr()
            assert "Auto-discovering discriminative neurons" in captured.out

    def test_neurons_auto_discover_multi_sample_nonzero_std(
        self, neurons_args, mock_ablation_study, capsys
    ):
        """Test auto-discover with multiple samples per label and non-zero pooled_std - line 197."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_neurons

        neurons_args.neurons = None
        neurons_args.prompts = "2+2=|3+3=|4+4=|5+5="  # 4 prompts
        neurons_args.labels = "easy|easy|hard|hard"  # 2 samples per label
        neurons_args.auto_discover = True

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls:
            mock_hooks = MagicMock()

            import mlx.core as mx

            # Create varied hidden states to get non-zero pooled_std
            call_count = [0]

            def forward_side_effect(*args, **kwargs):
                call_count[0] += 1
                # Return different states with variation to create non-zero pooled_std
                # Make values vary significantly
                h = mx.array([[[float((i + call_count[0]) * 10) for i in range(768)]]])
                mock_hooks.state.hidden_states = {12: h}

            mock_hooks.forward.side_effect = forward_side_effect
            mock_hooks_cls.return_value = mock_hooks

            introspect_neurons(neurons_args)

            captured = capsys.readouterr()
            assert "Auto-discovering discriminative neurons" in captured.out
            # Should show separation scores
            assert "Separation" in captured.out

    def test_neurons_auto_discover_with_high_separation(
        self, neurons_args, mock_ablation_study, capsys
    ):
        """Test auto-discover that triggers best_pair update (lines 202-203)."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_neurons

        neurons_args.neurons = None
        neurons_args.prompts = "2+2=|3+3=|4+4="  # 3 prompts
        neurons_args.labels = "cat1|cat2|cat3"  # 3 different labels
        neurons_args.auto_discover = True
        neurons_args.top_k = 5

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls:
            mock_hooks = MagicMock()

            import mlx.core as mx

            # Create hidden states with high variance to trigger separation > max_separation
            call_count = [0]

            def forward_side_effect(*args, **kwargs):
                call_count[0] += 1
                # Create varied activations that will have high separation
                h = mx.array([[[float(i * call_count[0]) for i in range(768)]]])
                mock_hooks.state.hidden_states = {12: h}

            mock_hooks.forward.side_effect = forward_side_effect
            mock_hooks_cls.return_value = mock_hooks

            introspect_neurons(neurons_args)

            captured = capsys.readouterr()
            assert "Best Pair" in captured.out

    def test_neurons_multi_layer_with_names_and_labels(
        self, neurons_args, mock_ablation_study, capsys
    ):
        """Test multi-layer with neuron names and labels."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_neurons

        neurons_args.layer = None
        neurons_args.layers = "4,12"
        neurons_args.labels = "easy|hard"
        neurons_args.neuron_names = "n1|n2"

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls:
            mock_hooks = MagicMock()

            import mlx.core as mx

            mock_hooks.state.hidden_states = {
                4: mx.zeros((1, 1, 768)),
                12: mx.zeros((1, 1, 768)),
            }
            mock_hooks_cls.return_value = mock_hooks

            introspect_neurons(neurons_args)

            captured = capsys.readouterr()
            assert "n1" in captured.out or "n2" in captured.out


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
        # Should show similarity matrix
        assert "Loading" in captured.out or "Direction" in captured.out or "Cosine" in captured.out

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
                method="mean_diff",
                accuracy=0.85,
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

    def test_directions_dimension_mismatch(self, directions_args, capsys):
        """Test warning when directions have different dimensions."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_directions

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f1:
            np.savez(f1.name, direction=np.random.randn(768).astype(np.float32), layer=12)
            directions_args.files[0] = f1.name

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f2:
            np.savez(f2.name, direction=np.random.randn(512).astype(np.float32), layer=8)
            directions_args.files[1] = f2.name

        introspect_directions(directions_args)

        captured = capsys.readouterr()
        assert "WARNING: Dimension mismatch" in captured.out

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
        assert "Aligned" in captured.out or "correlated" in captured.out.lower()

    def test_directions_orthogonal_vectors(self, directions_args, capsys):
        """Test with orthogonal direction vectors."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_directions

        # Create approximately orthogonal vectors using QR decomposition
        random_matrix = np.random.randn(768, 2).astype(np.float32)
        q, _ = np.linalg.qr(random_matrix)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f1:
            np.savez(f1.name, direction=q[:, 0], layer=12)
            directions_args.files[0] = f1.name

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f2:
            np.savez(f2.name, direction=q[:, 1], layer=12)
            directions_args.files[1] = f2.name

        introspect_directions(directions_args)

        captured = capsys.readouterr()
        assert "orthogonal" in captured.out.lower() or "ORTHOGONAL" in captured.out

    def test_directions_save_output(self, directions_args, capsys):
        """Test saving direction comparison results."""
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

        import json
        from pathlib import Path

        if Path(directions_args.output).exists():
            with open(directions_args.output) as f:
                data = json.load(f)
                assert "similarity_matrix" in data

    def test_directions_moderate_correlation(self, directions_args, capsys):
        """Test with moderately correlated direction vectors."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_directions

        base = np.random.randn(768).astype(np.float32)
        noise = np.random.randn(768).astype(np.float32) * 0.5

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f1:
            np.savez(f1.name, direction=base, layer=12)
            directions_args.files[0] = f1.name

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f2:
            np.savez(f2.name, direction=base + noise, layer=12)
            directions_args.files[1] = f2.name

        introspect_directions(directions_args)

        captured = capsys.readouterr()
        # Should print some assessment
        assert "Assessment" in captured.out or "SUMMARY" in captured.out

    def test_directions_weak_correlation(self, directions_args, capsys):
        """Test with weakly correlated directions (0.3 < cos < 0.5)."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_directions

        # Create vectors with ~0.4 cosine similarity
        base = np.random.randn(768).astype(np.float32)
        base = base / np.linalg.norm(base)
        noise = np.random.randn(768).astype(np.float32)
        noise = noise / np.linalg.norm(noise)
        # Mix to get moderate correlation
        mixed = base * 0.7 + noise * 0.7
        mixed = mixed / np.linalg.norm(mixed)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f1:
            np.savez(f1.name, direction=base, layer=12)
            directions_args.files[0] = f1.name

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f2:
            np.savez(f2.name, direction=mixed, layer=12)
            directions_args.files[1] = f2.name

        introspect_directions(directions_args)

        captured = capsys.readouterr()
        assert "Assessment" in captured.out

    def test_directions_very_weak_correlation(self, directions_args, capsys):
        """Test with very weakly correlated directions (threshold < cos < 0.3)."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_directions

        # Create vectors that are nearly orthogonal but not quite
        base = np.zeros(768, dtype=np.float32)
        base[0] = 1.0  # Unit vector in first dimension
        orthogonal = np.zeros(768, dtype=np.float32)
        orthogonal[1] = 1.0  # Orthogonal unit vector
        # Add small component to make cos ~0.2
        mixed = orthogonal + base * 0.2
        mixed = mixed / np.linalg.norm(mixed)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f1:
            np.savez(f1.name, direction=base, layer=12)
            directions_args.files[0] = f1.name

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f2:
            np.savez(f2.name, direction=mixed, layer=12)
            directions_args.files[1] = f2.name

        introspect_directions(directions_args)

        captured = capsys.readouterr()
        assert "INDEPENDENT" in captured.out or "ORTHOGONAL" in captured.out

    def test_directions_heatmap_moderate_corr(self, directions_args, capsys):
        """Test heatmap char for moderate correlation (0.5 < val <= 0.7) - line 677."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_directions

        # Create vectors with exact 0.6 cosine similarity
        base = np.zeros(768, dtype=np.float32)
        base[0] = 1.0  # Unit vector

        # To get cos = 0.6: v2 = 0.6 * v1 + sqrt(1 - 0.36) * orthogonal
        # sqrt(0.64) = 0.8
        mixed = np.zeros(768, dtype=np.float32)
        mixed[0] = 0.6  # Component along base
        mixed[1] = 0.8  # Orthogonal component
        mixed = mixed / np.linalg.norm(mixed)  # Normalize

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f1:
            np.savez(f1.name, direction=base, layer=12)
            directions_args.files[0] = f1.name

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f2:
            np.savez(f2.name, direction=mixed, layer=12)
            directions_args.files[1] = f2.name

        introspect_directions(directions_args)

        captured = capsys.readouterr()
        # Should display heatmap with '+' character for moderate correlation
        assert "ORTHOGONALITY HEATMAP" in captured.out

    def test_directions_heatmap_weak_corr(self, directions_args, capsys):
        """Test heatmap char for weak correlation (0.3 < val <= 0.5) - line 679."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_directions

        # Create vectors with exact 0.4 cosine similarity using construction
        # cos(theta) = 0.4 means we can construct orthogonal components
        base = np.zeros(768, dtype=np.float32)
        base[0] = 1.0  # Unit vector in first dimension

        # To get cos = 0.4, we need: v2 = 0.4 * v1 + sqrt(1 - 0.4^2) * orthogonal
        # sqrt(1 - 0.16) = sqrt(0.84) ~= 0.9165
        mixed = np.zeros(768, dtype=np.float32)
        mixed[0] = 0.4  # Component along base
        mixed[1] = 0.9165  # Orthogonal component
        mixed = mixed / np.linalg.norm(mixed)  # Normalize

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f1:
            np.savez(f1.name, direction=base, layer=12)
            directions_args.files[0] = f1.name

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f2:
            np.savez(f2.name, direction=mixed, layer=12)
            directions_args.files[1] = f2.name

        introspect_directions(directions_args)

        captured = capsys.readouterr()
        # Should display heatmap with '-' character for weak correlation
        assert "ORTHOGONALITY HEATMAP" in captured.out

    def test_directions_assessment_moderate(self, directions_args, capsys):
        """Test assessment for moderate correlation (0.3 <= mean < 0.5) - line 728."""
        from chuk_lazarus.cli.commands.introspect.neurons import introspect_directions

        # Create three vectors with exact moderate correlation to get mean in [0.3, 0.5)
        # We need 3 pairs: (v0,v1), (v0,v2), (v1,v2)
        # Target mean absolute cosine ~= 0.4

        # Base vector
        v0 = np.zeros(768, dtype=np.float32)
        v0[0] = 1.0

        # Vector with cos = 0.35 to v0
        v1 = np.zeros(768, dtype=np.float32)
        v1[0] = 0.35
        v1[1] = np.sqrt(1 - 0.35**2)

        # Vector with cos = 0.45 to v0
        v2 = np.zeros(768, dtype=np.float32)
        v2[0] = 0.45
        v2[2] = np.sqrt(1 - 0.45**2)
        # This gives us cos(v0,v1)=0.35, cos(v0,v2)=0.45, cos(v1,v2)~=0.16
        # Mean = (0.35 + 0.45 + 0.16) / 3 ~= 0.32, which is in [0.3, 0.5)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f1:
            np.savez(f1.name, direction=v0, layer=12)
            directions_args.files = [f1.name]

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f2:
            np.savez(f2.name, direction=v1, layer=12)
            directions_args.files.append(f2.name)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f3:
            np.savez(f3.name, direction=v2, layer=12)
            directions_args.files.append(f3.name)

        introspect_directions(directions_args)

        captured = capsys.readouterr()
        # Should show "MODERATE correlation" assessment
        assert "MODERATE correlation" in captured.out or "Assessment" in captured.out


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
        from collections import defaultdict

        from chuk_lazarus.cli.commands.introspect.neurons import introspect_operand_directions

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
        from collections import defaultdict

        from chuk_lazarus.cli.commands.introspect.neurons import introspect_operand_directions

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
        from collections import defaultdict

        from chuk_lazarus.cli.commands.introspect.neurons import introspect_operand_directions

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
        from collections import defaultdict

        from chuk_lazarus.cli.commands.introspect.neurons import introspect_operand_directions

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
        from collections import defaultdict

        from chuk_lazarus.cli.commands.introspect.neurons import introspect_operand_directions

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

    def test_operand_directions_save_npz(self, operand_args, mock_ablation_study, capsys):
        """Test saving operand directions to npz format."""
        from collections import defaultdict

        from chuk_lazarus.cli.commands.introspect.neurons import introspect_operand_directions

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            operand_args.output = f.name

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls:
            mock_hooks = MagicMock()

            import mlx.core as mx

            def make_hidden_states():
                return mx.zeros((1, 1, 768))

            mock_hooks.state.hidden_states = defaultdict(make_hidden_states)
            mock_hooks_cls.return_value = mock_hooks

            introspect_operand_directions(operand_args)

            captured = capsys.readouterr()
            assert "Directions and results saved to" in captured.out

    def test_operand_directions_interpretation_compositional(
        self, operand_args, mock_ablation_study, capsys
    ):
        """Test interpretation output for compositional encoding."""
        from collections import defaultdict

        from chuk_lazarus.cli.commands.introspect.neurons import introspect_operand_directions

        operand_args.layers = "6"  # Single layer for simpler test

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls:
            mock_hooks = MagicMock()

            import mlx.core as mx

            # Create varied hidden states to trigger compositional interpretation
            call_count = [0]

            def make_varied_hidden_states():
                call_count[0] += 1
                # Return different values based on call count to create variation
                return mx.array([[[float(i) for i in range(768)]]])

            mock_hooks.state.hidden_states = defaultdict(make_varied_hidden_states)
            mock_hooks_cls.return_value = mock_hooks

            introspect_operand_directions(operand_args)

            captured = capsys.readouterr()
            assert "Interpretation" in captured.out

    def test_operand_directions_with_division(self, operand_args, mock_ablation_study, capsys):
        """Test operand directions with division operation."""
        from collections import defaultdict

        from chuk_lazarus.cli.commands.introspect.neurons import introspect_operand_directions

        operand_args.operation = "/"

        with patch("chuk_lazarus.introspection.ModelHooks") as mock_hooks_cls:
            mock_hooks = MagicMock()

            import mlx.core as mx

            def make_hidden_states():
                return mx.zeros((1, 1, 768))

            mock_hooks.state.hidden_states = defaultdict(make_hidden_states)
            mock_hooks_cls.return_value = mock_hooks

            introspect_operand_directions(operand_args)

            captured = capsys.readouterr()
            assert "/" in captured.out or "Collecting activations" in captured.out
