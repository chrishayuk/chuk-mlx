"""Tests for steering core module."""

from unittest.mock import Mock, patch

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from chuk_lazarus.introspection.steering.config import SteeringConfig
from chuk_lazarus.introspection.steering.core import ActivationSteering


# Mock classes
class MockAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.o_proj = nn.Linear(hidden_size, hidden_size)


class MockMLP(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.down_proj = nn.Linear(hidden_size, hidden_size)


class MockLayer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.self_attn = MockAttention(hidden_size)
        self.mlp = MockMLP(hidden_size)
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)

    def __call__(self, h, **kwargs):
        return h


class MockBackbone(nn.Module):
    def __init__(self, num_layers: int, hidden_size: int):
        super().__init__()
        self.layers = [MockLayer(hidden_size) for _ in range(num_layers)]


class MockModelOutput:
    def __init__(self, logits):
        self.logits = logits


class MockModel(nn.Module):
    def __init__(self, num_layers: int = 4, hidden_size: int = 64, vocab_size: int = 100):
        super().__init__()
        self.model = MockBackbone(num_layers, hidden_size)
        self._hidden_size = hidden_size
        self._vocab_size = vocab_size

    def __call__(self, input_ids):
        batch_size, seq_len = input_ids.shape
        logits = mx.random.normal((batch_size, seq_len, self._vocab_size))
        return MockModelOutput(logits)


class MockModelDirect(nn.Module):
    """Mock model with direct layers."""

    def __init__(self, num_layers: int = 4, hidden_size: int = 64, vocab_size: int = 100):
        super().__init__()
        self.layers = [MockLayer(hidden_size) for _ in range(num_layers)]
        self._hidden_size = hidden_size
        self._vocab_size = vocab_size

    def __call__(self, input_ids):
        batch_size, seq_len = input_ids.shape
        logits = mx.random.normal((batch_size, seq_len, self._vocab_size))
        return MockModelOutput(logits)


class MockTokenizer:
    def __init__(self, vocab_size: int = 100):
        self.vocab_size = vocab_size
        self.eos_token_id = 2

    def encode(self, text, **kwargs):
        return [[1, 2, 3]]

    def decode(self, ids, **kwargs):
        return "generated text"


class TestActivationSteering:
    """Tests for ActivationSteering class."""

    def test_init(self):
        """Test initialization."""
        model = MockModel()
        tokenizer = MockTokenizer()
        steerer = ActivationSteering(model, tokenizer)
        assert steerer.model is model
        assert steerer.tokenizer is tokenizer
        assert steerer.num_layers == 4

    def test_init_direct_layers(self):
        """Test initialization with direct layers."""
        model = MockModelDirect()
        tokenizer = MockTokenizer()
        steerer = ActivationSteering(model, tokenizer)
        assert steerer.num_layers == 4

    def test_init_with_model_id(self):
        """Test initialization with model_id."""
        model = MockModel()
        tokenizer = MockTokenizer()
        steerer = ActivationSteering(model, tokenizer, model_id="test-model")
        assert steerer.model_id == "test-model"

    def test_init_unsupported_structure(self):
        """Test initialization with unsupported structure."""

        class UnsupportedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.something = nn.Linear(10, 10)

        with pytest.raises(ValueError, match="Cannot detect model layer structure"):
            ActivationSteering(UnsupportedModel(), MockTokenizer())

    def test_add_direction(self):
        """Test adding a direction."""
        model = MockModel(hidden_size=64)
        steerer = ActivationSteering(model, MockTokenizer())

        direction = mx.random.normal((64,))
        steerer.add_direction(
            layer=0, direction=direction, name="test", positive_label="pos", negative_label="neg"
        )

        assert 0 in steerer.directions
        assert steerer.direction_info[0]["name"] == "test"
        assert steerer.direction_info[0]["positive_label"] == "pos"
        assert steerer.direction_info[0]["negative_label"] == "neg"

    def test_add_direction_numpy(self):
        """Test adding a direction as numpy array."""
        model = MockModel(hidden_size=64)
        steerer = ActivationSteering(model, MockTokenizer())

        direction = np.random.normal(size=(64,))
        steerer.add_direction(layer=0, direction=direction)

        assert 0 in steerer.directions
        assert isinstance(steerer.directions[0], mx.array)

    def test_clear_directions(self):
        """Test clearing directions."""
        model = MockModel(hidden_size=64)
        steerer = ActivationSteering(model, MockTokenizer())

        direction = mx.random.normal((64,))
        steerer.add_direction(layer=0, direction=direction)
        steerer.add_direction(layer=1, direction=direction)

        steerer.clear_directions()
        assert len(steerer.directions) == 0
        assert len(steerer.direction_info) == 0

    def test_generate_no_steering(self):
        """Test generation without directions."""
        model = MockModel()
        steerer = ActivationSteering(model, MockTokenizer())

        output = steerer.generate("test prompt")
        assert isinstance(output, str)

    def test_generate_with_steering(self):
        """Test generation with steering."""
        model = MockModel(hidden_size=64)
        steerer = ActivationSteering(model, MockTokenizer())

        direction = mx.random.normal((64,))
        steerer.add_direction(layer=0, direction=direction)

        config = SteeringConfig(layers=[0], coefficient=1.0, max_new_tokens=5)
        output = steerer.generate("test prompt", config=config)
        assert isinstance(output, str)

    def test_generate_with_coefficient_override(self):
        """Test generation with coefficient override."""
        model = MockModel(hidden_size=64)
        steerer = ActivationSteering(model, MockTokenizer())

        direction = mx.random.normal((64,))
        steerer.add_direction(layer=0, direction=direction)

        config = SteeringConfig(layers=[0], max_new_tokens=5)
        output = steerer.generate("test prompt", config=config, coefficient=2.0)
        assert isinstance(output, str)

    def test_generate_with_layer_override(self):
        """Test generation with steering layer override."""
        model = MockModel(hidden_size=64)
        steerer = ActivationSteering(model, MockTokenizer())

        direction = mx.random.normal((64,))
        steerer.add_direction(layer=0, direction=direction)
        steerer.add_direction(layer=1, direction=direction)

        config = SteeringConfig(max_new_tokens=5)
        output = steerer.generate("test prompt", config=config, steering_layers=[0])
        assert isinstance(output, str)

    def test_generate_with_temperature(self):
        """Test generation with non-zero temperature."""
        model = MockModel(hidden_size=64)
        steerer = ActivationSteering(model, MockTokenizer())

        direction = mx.random.normal((64,))
        steerer.add_direction(layer=0, direction=direction)

        config = SteeringConfig(layers=[0], temperature=0.5, max_new_tokens=5)
        output = steerer.generate("test prompt", config=config)
        assert isinstance(output, str)

    def test_compare_steering(self):
        """Test comparing steering effects."""
        model = MockModel(hidden_size=64)
        steerer = ActivationSteering(model, MockTokenizer())

        direction = mx.random.normal((64,))
        steerer.add_direction(layer=0, direction=direction)

        config = SteeringConfig(layers=[0], max_new_tokens=5)
        results = steerer.compare_steering("test prompt", config=config)

        assert isinstance(results, dict)
        assert -1.0 in results
        assert 0.0 in results
        assert 1.0 in results

    def test_compare_steering_custom_coefficients(self):
        """Test comparing with custom coefficients."""
        model = MockModel(hidden_size=64)
        steerer = ActivationSteering(model, MockTokenizer())

        direction = mx.random.normal((64,))
        steerer.add_direction(layer=0, direction=direction)

        config = SteeringConfig(layers=[0], max_new_tokens=5)
        results = steerer.compare_steering(
            "test prompt", coefficients=[-2.0, 0.0, 2.0], config=config
        )

        assert -2.0 in results
        assert 2.0 in results

    def test_sweep_layers(self):
        """Test sweeping across layers."""
        model = MockModel(hidden_size=64, num_layers=4)
        steerer = ActivationSteering(model, MockTokenizer())

        direction = mx.random.normal((64,))
        for i in range(4):
            steerer.add_direction(layer=i, direction=direction)

        config = SteeringConfig(max_new_tokens=5)
        results = steerer.sweep_layers("test prompt", config=config)

        assert isinstance(results, dict)
        assert len(results) == 4

    def test_sweep_layers_subset(self):
        """Test sweeping across a subset of layers."""
        model = MockModel(hidden_size=64, num_layers=4)
        steerer = ActivationSteering(model, MockTokenizer())

        direction = mx.random.normal((64,))
        for i in range(4):
            steerer.add_direction(layer=i, direction=direction)

        config = SteeringConfig(max_new_tokens=5)
        results = steerer.sweep_layers("test prompt", layers=[0, 2], config=config)

        assert len(results) == 2
        assert 0 in results
        assert 2 in results

    def test_print_comparison(self, capsys):
        """Test printing comparison."""
        model = MockModel(hidden_size=64)
        steerer = ActivationSteering(model, MockTokenizer())

        direction = mx.random.normal((64,))
        steerer.add_direction(
            layer=0, direction=direction, positive_label="positive", negative_label="negative"
        )

        config = SteeringConfig(layers=[0], max_new_tokens=5)
        steerer.print_comparison("test prompt", config=config)

        captured = capsys.readouterr()
        assert "STEERING COMPARISON" in captured.out
        assert "positive" in captured.out
        assert "negative" in captured.out

    def test_print_comparison_no_directions(self, capsys):
        """Test printing comparison with no direction info."""
        model = MockModel(hidden_size=64)
        steerer = ActivationSteering(model, MockTokenizer())

        steerer.print_comparison("test prompt")

        captured = capsys.readouterr()
        assert "STEERING COMPARISON" in captured.out

    def test_wrap_unwrap_layers(self):
        """Test that layers are properly wrapped and unwrapped."""
        model = MockModel(hidden_size=64)
        steerer = ActivationSteering(model, MockTokenizer())

        direction = mx.random.normal((64,))
        steerer.add_direction(layer=0, direction=direction)

        original_layer = steerer._layers[0]

        # Wrap
        steerer._wrap_layer(0, 1.0)
        assert steerer._layers[0] is not original_layer

        # Unwrap
        steerer._unwrap_layers()
        assert steerer._layers[0] is original_layer

    def test_wrap_layer_not_in_directions(self):
        """Test wrapping a layer that has no direction (should be no-op)."""
        model = MockModel(hidden_size=64)
        steerer = ActivationSteering(model, MockTokenizer())

        original_layer = steerer._layers[0]
        steerer._wrap_layer(0, 1.0)

        # Should not be wrapped since no direction for layer 0
        assert steerer._layers[0] is original_layer


class TestSteeredLayerWrapper:
    """Tests for the steered layer wrapper."""

    def test_wrapper_handles_hidden_states_output(self):
        """Test wrapper handles output with hidden_states attribute."""
        model = MockModel(hidden_size=64)
        steerer = ActivationSteering(model, MockTokenizer())

        direction = mx.random.normal((64,))
        steerer.add_direction(layer=0, direction=direction)

        # Generate will create wrappers
        config = SteeringConfig(layers=[0], max_new_tokens=3)
        output = steerer.generate("test", config=config)
        assert isinstance(output, str)

    def test_wrapper_handles_tuple_output(self):
        """Test wrapper handles tuple output."""

        class TupleOutputLayer(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.mlp = nn.Linear(hidden_size, hidden_size)
                self.self_attn = nn.Linear(hidden_size, hidden_size)

            def __call__(self, h, **kwargs):
                return (h, None)  # Tuple output

        class TupleOutputModel(nn.Module):
            def __init__(self, hidden_size=64, vocab_size=100):
                super().__init__()
                self.layers = [TupleOutputLayer(hidden_size) for _ in range(2)]
                self._hidden_size = hidden_size
                self._vocab_size = vocab_size

            def __call__(self, input_ids):
                batch_size, seq_len = input_ids.shape
                logits = mx.random.normal((batch_size, seq_len, self._vocab_size))
                return MockModelOutput(logits)

        model = TupleOutputModel()
        steerer = ActivationSteering(model, MockTokenizer())

        direction = mx.random.normal((64,))
        steerer.add_direction(layer=0, direction=direction)

        config = SteeringConfig(layers=[0], max_new_tokens=3)
        output = steerer.generate("test", config=config)
        assert isinstance(output, str)

    def test_wrapper_getattr(self):
        """Test wrapper forwards attribute access to wrapped layer."""
        model = MockModel(hidden_size=64)
        steerer = ActivationSteering(model, MockTokenizer())

        direction = mx.random.normal((64,))
        steerer.add_direction(layer=0, direction=direction)

        # Wrap the layer
        steerer._wrap_layer(0, 1.0)

        # Test that we can still access layer attributes through wrapper
        wrapped_layer = steerer._layers[0]
        assert hasattr(wrapped_layer, "mlp")
        assert hasattr(wrapped_layer, "self_attn")

        steerer._unwrap_layers()


class TestAddDirections:
    """Tests for add_directions method."""

    def test_add_directions_from_bundle(self):
        """Test adding multiple directions from a bundle."""
        model = MockModel(hidden_size=64)
        steerer = ActivationSteering(model, MockTokenizer())

        # Create mock bundle
        bundle = Mock()
        bundle.directions = {}

        # Add mock directions
        for layer in [0, 1, 2]:
            direction_obj = Mock()
            direction_obj.direction = mx.random.normal((64,))
            direction_obj.name = f"test_L{layer}"
            direction_obj.positive_label = f"pos_{layer}"
            direction_obj.negative_label = f"neg_{layer}"
            bundle.directions[layer] = direction_obj

        steerer.add_directions(bundle)

        # Verify all directions were added
        assert len(steerer.directions) == 3
        assert 0 in steerer.directions
        assert 1 in steerer.directions
        assert 2 in steerer.directions

        # Verify metadata
        assert steerer.direction_info[0]["name"] == "test_L0"
        assert steerer.direction_info[1]["positive_label"] == "pos_1"
        assert steerer.direction_info[2]["negative_label"] == "neg_2"

    def test_add_directions_empty_bundle(self):
        """Test adding directions from empty bundle."""
        model = MockModel(hidden_size=64)
        steerer = ActivationSteering(model, MockTokenizer())

        bundle = Mock()
        bundle.directions = {}

        steerer.add_directions(bundle)

        assert len(steerer.directions) == 0


class TestGenerateEdgeCases:
    """Tests for edge cases in generation."""

    def test_generate_stops_at_eos(self):
        """Test that generation stops when EOS token is generated."""

        class EOSTokenizer:
            def __init__(self):
                self.eos_token_id = 2

            def encode(self, text, **kwargs):
                return [[1]]

            def decode(self, ids, **kwargs):
                return "text"

        class EOSModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = MockBackbone(2, 64)
                self.call_count = 0

            def __call__(self, input_ids):
                # Return EOS token on second call
                self.call_count += 1
                batch_size, seq_len = input_ids.shape
                logits = mx.zeros((batch_size, seq_len, 100))
                if self.call_count >= 2:
                    # Make EOS token most likely - use array indexing
                    logits_array = mx.array(logits)
                    new_logits = []
                    for i in range(batch_size):
                        batch_logits = logits_array[i]
                        batch_logits = batch_logits.at[-1, 2].add(100.0)
                        new_logits.append(batch_logits)
                    logits = mx.stack(new_logits)
                return MockModelOutput(logits)

        model = EOSModel()
        steerer = ActivationSteering(model, EOSTokenizer())

        config = SteeringConfig(max_new_tokens=50, temperature=0)
        output = steerer.generate("test", config=config)

        # Should stop early due to EOS
        assert isinstance(output, str)
        # Model should have been called fewer than max_new_tokens times
        assert model.call_count < 50

    def test_generate_with_no_eos_token_id(self):
        """Test generation when tokenizer has no eos_token_id."""

        class NoEOSTokenizer:
            def encode(self, text, **kwargs):
                return [[1]]

            def decode(self, ids, **kwargs):
                return "text"

        model = MockModel()
        steerer = ActivationSteering(model, NoEOSTokenizer())

        config = SteeringConfig(max_new_tokens=3)
        output = steerer.generate("test", config=config)

        # Should complete without error
        assert isinstance(output, str)

    def test_generate_handles_model_without_logits_attribute(self):
        """Test generation with model that returns logits directly."""

        class DirectLogitsModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = MockBackbone(2, 64)

            def __call__(self, input_ids):
                batch_size, seq_len = input_ids.shape
                # Return logits directly, not wrapped in object
                return mx.random.normal((batch_size, seq_len, 100))

        model = DirectLogitsModel()
        steerer = ActivationSteering(model, MockTokenizer())

        config = SteeringConfig(max_new_tokens=3)
        output = steerer.generate("test", config=config)

        assert isinstance(output, str)

    def test_generate_with_position_steering(self):
        """Test generation with position-specific steering."""
        model = MockModel(hidden_size=64)
        steerer = ActivationSteering(model, MockTokenizer())

        direction = mx.random.normal((64,))
        steerer.add_direction(layer=0, direction=direction)

        # Steer only position 1
        config = SteeringConfig(layers=[0], position=1, max_new_tokens=5)
        output = steerer.generate("test", config=config)

        assert isinstance(output, str)

    def test_generate_default_config(self):
        """Test generation with default config (None)."""
        model = MockModel(hidden_size=64)
        steerer = ActivationSteering(model, MockTokenizer())

        direction = mx.random.normal((64,))
        steerer.add_direction(layer=0, direction=direction)

        # Pass None for config
        output = steerer.generate("test", config=None, steering_layers=[0])

        assert isinstance(output, str)


class TestFromPretrained:
    """Tests for from_pretrained class method."""

    @patch("chuk_lazarus.introspection.ablation.AblationStudy")
    def test_from_pretrained(self, mock_ablation_class):
        """Test loading model via from_pretrained."""
        # Setup mock
        mock_study = Mock()
        mock_study.adapter.model = MockModel()
        mock_study.adapter.tokenizer = MockTokenizer()
        mock_ablation_class.from_pretrained.return_value = mock_study

        # Load
        steerer = ActivationSteering.from_pretrained("test-model")

        # Verify
        mock_ablation_class.from_pretrained.assert_called_once_with("test-model")
        assert steerer.model is mock_study.adapter.model
        assert steerer.tokenizer is mock_study.adapter.tokenizer
        assert steerer.model_id == "test-model"

    @patch("chuk_lazarus.introspection.ablation.AblationStudy")
    def test_from_pretrained_initializes_correctly(self, mock_ablation_class):
        """Test that from_pretrained initializes all attributes."""
        mock_study = Mock()
        mock_study.adapter.model = MockModel(num_layers=6)
        mock_study.adapter.tokenizer = MockTokenizer()
        mock_ablation_class.from_pretrained.return_value = mock_study

        steerer = ActivationSteering.from_pretrained("test-model")

        assert steerer.num_layers == 6
        assert len(steerer.directions) == 0
        assert len(steerer.direction_info) == 0
        assert not steerer._is_steering


class TestLayerProbabilities:
    """Tests for get_layer_probabilities and related methods."""

    @patch("chuk_lazarus.introspection.hooks.ModelHooks")
    @patch("chuk_lazarus.introspection.logit_lens.LogitLens")
    def test_get_layer_probabilities(self, mock_lens_class, mock_hooks_class):
        """Test get_layer_probabilities method."""
        # Setup mocks
        mock_hooks = Mock()
        mock_hooks_class.return_value = mock_hooks

        mock_lens = Mock()
        mock_evolution = Mock()
        mock_evolution.layers = [0, 1, 2]
        mock_evolution.probabilities = [0.1, 0.5, 0.9]
        mock_lens.track_token.return_value = mock_evolution
        mock_lens_class.return_value = mock_lens

        # Setup steerer
        model = MockModel(hidden_size=64)
        steerer = ActivationSteering(model, MockTokenizer())
        direction = mx.random.normal((64,))
        steerer.add_direction(layer=0, direction=direction)

        # Call method
        config = SteeringConfig(layers=[0])
        result = steerer.get_layer_probabilities("test prompt", "token", config=config)

        # Verify
        assert isinstance(result, dict)
        assert result[0] == 0.1
        assert result[1] == 0.5
        assert result[2] == 0.9

    @patch("chuk_lazarus.introspection.hooks.ModelHooks")
    @patch("chuk_lazarus.introspection.logit_lens.LogitLens")
    def test_get_layer_probabilities_with_overrides(self, mock_lens_class, mock_hooks_class):
        """Test get_layer_probabilities with parameter overrides."""
        mock_hooks = Mock()
        mock_hooks_class.return_value = mock_hooks

        mock_lens = Mock()
        mock_evolution = Mock()
        mock_evolution.layers = [1]
        mock_evolution.probabilities = [0.7]
        mock_lens.track_token.return_value = mock_evolution
        mock_lens_class.return_value = mock_lens

        model = MockModel(hidden_size=64)
        steerer = ActivationSteering(model, MockTokenizer())
        direction = mx.random.normal((64,))
        steerer.add_direction(layer=1, direction=direction)

        result = steerer.get_layer_probabilities(
            "test prompt",
            "token",
            steering_layers=[1],
            coefficient=2.0,
        )

        assert 1 in result
        assert result[1] == 0.7

    @patch("chuk_lazarus.introspection.hooks.ModelHooks")
    @patch("chuk_lazarus.introspection.logit_lens.LogitLens")
    def test_compare_layer_dynamics(self, mock_lens_class, mock_hooks_class):
        """Test compare_layer_dynamics method."""
        mock_hooks = Mock()
        mock_hooks_class.return_value = mock_hooks

        mock_lens = Mock()
        mock_evolution = Mock()
        mock_evolution.layers = [0, 1]
        mock_evolution.probabilities = [0.3, 0.7]
        mock_lens.track_token.return_value = mock_evolution
        mock_lens_class.return_value = mock_lens

        model = MockModel(hidden_size=64)
        steerer = ActivationSteering(model, MockTokenizer())
        direction = mx.random.normal((64,))
        steerer.add_direction(layer=0, direction=direction)

        config = SteeringConfig(layers=[0])
        result = steerer.compare_layer_dynamics(
            "test prompt", "token", coefficients=[-1.0, 0.0, 1.0], config=config
        )

        # Should have results for each coefficient
        assert isinstance(result, dict)
        assert -1.0 in result
        assert 0.0 in result
        assert 1.0 in result

    @patch("chuk_lazarus.introspection.hooks.ModelHooks")
    @patch("chuk_lazarus.introspection.logit_lens.LogitLens")
    def test_compare_layer_dynamics_default_coefficients(self, mock_lens_class, mock_hooks_class):
        """Test compare_layer_dynamics with default coefficients."""
        mock_hooks = Mock()
        mock_hooks_class.return_value = mock_hooks

        mock_lens = Mock()
        mock_evolution = Mock()
        mock_evolution.layers = [0]
        mock_evolution.probabilities = [0.5]
        mock_lens.track_token.return_value = mock_evolution
        mock_lens_class.return_value = mock_lens

        model = MockModel(hidden_size=64)
        steerer = ActivationSteering(model, MockTokenizer())
        direction = mx.random.normal((64,))
        steerer.add_direction(layer=0, direction=direction)

        result = steerer.compare_layer_dynamics("test prompt", "token")

        # Default coefficients: [-2.0, 0.0, 2.0]
        assert -2.0 in result
        assert 0.0 in result
        assert 2.0 in result


class TestPrintMethods:
    """Tests for print methods."""

    @patch("chuk_lazarus.introspection.hooks.ModelHooks")
    @patch("chuk_lazarus.introspection.logit_lens.LogitLens")
    def test_print_layer_dynamics(self, mock_lens_class, mock_hooks_class, capsys):
        """Test print_layer_dynamics method."""
        mock_hooks = Mock()
        mock_hooks_class.return_value = mock_hooks

        mock_lens = Mock()
        mock_evolution = Mock()
        mock_evolution.layers = [0, 1, 2]
        mock_evolution.probabilities = [0.1, 0.5, 0.9]
        mock_lens.track_token.return_value = mock_evolution
        mock_lens_class.return_value = mock_lens

        model = MockModel(hidden_size=64)
        steerer = ActivationSteering(model, MockTokenizer())
        direction = mx.random.normal((64,))
        steerer.add_direction(layer=0, direction=direction)

        config = SteeringConfig(layers=[0])
        steerer.print_layer_dynamics("test prompt", "token", coefficients=[0.0], config=config)

        captured = capsys.readouterr()
        assert "LAYER DYNAMICS WITH STEERING" in captured.out
        assert "test prompt" in captured.out
        assert "token" in captured.out

    @patch("chuk_lazarus.introspection.hooks.ModelHooks")
    @patch("chuk_lazarus.introspection.logit_lens.LogitLens")
    def test_print_layer_dynamics_with_key_layers(self, mock_lens_class, mock_hooks_class, capsys):
        """Test print_layer_dynamics with custom key_layers."""
        mock_hooks = Mock()
        mock_hooks_class.return_value = mock_hooks

        mock_lens = Mock()
        mock_evolution = Mock()
        # Simulate many layers
        mock_evolution.layers = list(range(30))
        mock_evolution.probabilities = [0.5] * 30
        mock_lens.track_token.return_value = mock_evolution
        mock_lens_class.return_value = mock_lens

        model = MockModel(hidden_size=64)
        steerer = ActivationSteering(model, MockTokenizer())
        direction = mx.random.normal((64,))
        steerer.add_direction(layer=0, direction=direction)

        # Test with many layers (should sample)
        steerer.print_layer_dynamics("test prompt", "token", key_layers=[0, 5, 10])

        captured = capsys.readouterr()
        assert "LAYER DYNAMICS WITH STEERING" in captured.out

    @patch("chuk_lazarus.introspection.hooks.ModelHooks")
    @patch("chuk_lazarus.introspection.logit_lens.LogitLens")
    def test_print_layer_dynamics_auto_sampling(self, mock_lens_class, mock_hooks_class, capsys):
        """Test that print_layer_dynamics auto-samples when many layers."""
        mock_hooks = Mock()
        mock_hooks_class.return_value = mock_hooks

        mock_lens = Mock()
        mock_evolution = Mock()
        # Simulate 25 layers (should trigger sampling)
        mock_evolution.layers = list(range(25))
        mock_evolution.probabilities = [0.5] * 25
        mock_lens.track_token.return_value = mock_evolution
        mock_lens_class.return_value = mock_lens

        model = MockModel(hidden_size=64)
        steerer = ActivationSteering(model, MockTokenizer())
        direction = mx.random.normal((64,))
        steerer.add_direction(layer=0, direction=direction)

        steerer.print_layer_dynamics("test prompt", "token")

        captured = capsys.readouterr()
        # Should print successfully even with many layers
        assert "LAYER DYNAMICS WITH STEERING" in captured.out

    def test_print_comparison_with_labels(self, capsys):
        """Test print_comparison with custom labels."""
        model = MockModel(hidden_size=64)
        steerer = ActivationSteering(model, MockTokenizer())

        direction = mx.random.normal((64,))
        steerer.add_direction(
            layer=0, direction=direction, positive_label="arithmetic", negative_label="suppress"
        )

        config = SteeringConfig(layers=[0], max_new_tokens=3)
        steerer.print_comparison("test", coefficients=[0.0, 1.0], config=config)

        captured = capsys.readouterr()
        assert "arithmetic" in captured.out
        assert "suppress" in captured.out

    def test_print_comparison_multiple_coefficients(self, capsys):
        """Test print_comparison with multiple coefficients."""
        model = MockModel(hidden_size=64)
        steerer = ActivationSteering(model, MockTokenizer())

        direction = mx.random.normal((64,))
        steerer.add_direction(layer=0, direction=direction)

        config = SteeringConfig(layers=[0], max_new_tokens=3)
        steerer.print_comparison("test", coefficients=[-2.0, -1.0, 0.0, 1.0, 2.0], config=config)

        captured = capsys.readouterr()
        # Should print results for all coefficients
        assert "-2.0" in captured.out
        assert "-1.0" in captured.out
        assert "0.0" in captured.out
        assert "1.0" in captured.out
        assert "2.0" in captured.out


class TestSteeringState:
    """Tests for steering state management."""

    def test_is_steering_flag(self):
        """Test that _is_steering flag is managed correctly."""
        model = MockModel(hidden_size=64)
        steerer = ActivationSteering(model, MockTokenizer())

        direction = mx.random.normal((64,))
        steerer.add_direction(layer=0, direction=direction)

        assert not steerer._is_steering

        # During generation, flag should be set but then cleared
        config = SteeringConfig(layers=[0], max_new_tokens=2)
        steerer.generate("test", config=config)

        # After generation, should be back to False
        assert not steerer._is_steering

    def test_layers_restored_after_exception(self):
        """Test that layers are restored even if generation fails."""

        class FailingTokenizer:
            def encode(self, text, **kwargs):
                return [[1]]

            def decode(self, ids, **kwargs):
                raise RuntimeError("Decode failed")

        model = MockModel(hidden_size=64)
        steerer = ActivationSteering(model, FailingTokenizer())

        direction = mx.random.normal((64,))
        steerer.add_direction(layer=0, direction=direction)

        original_layer = steerer._layers[0]

        config = SteeringConfig(layers=[0], max_new_tokens=2)

        with pytest.raises(RuntimeError):
            steerer.generate("test", config=config)

        # Layers should still be restored
        assert steerer._layers[0] is original_layer
        assert not steerer._is_steering
        assert len(steerer._original_forwards) == 0

    def test_multiple_generations_reuse_directions(self):
        """Test that multiple generations can reuse the same directions."""
        model = MockModel(hidden_size=64)
        steerer = ActivationSteering(model, MockTokenizer())

        direction = mx.random.normal((64,))
        steerer.add_direction(layer=0, direction=direction)

        config = SteeringConfig(layers=[0], max_new_tokens=2)

        # Generate multiple times
        output1 = steerer.generate("test1", config=config)
        output2 = steerer.generate("test2", config=config)
        output3 = steerer.generate("test3", config=config)

        # All should succeed
        assert isinstance(output1, str)
        assert isinstance(output2, str)
        assert isinstance(output3, str)

        # Directions should still be there
        assert 0 in steerer.directions


class TestSweepLayers:
    """Tests for sweep_layers method."""

    def test_sweep_layers_all(self):
        """Test sweeping all layers with directions."""
        model = MockModel(hidden_size=64, num_layers=4)
        steerer = ActivationSteering(model, MockTokenizer())

        direction = mx.random.normal((64,))
        for i in range(4):
            steerer.add_direction(layer=i, direction=direction)

        config = SteeringConfig(max_new_tokens=3)
        results = steerer.sweep_layers("test prompt", config=config)

        assert len(results) == 4
        for i in range(4):
            assert i in results
            assert isinstance(results[i], str)

    def test_sweep_layers_with_coefficient(self):
        """Test sweeping layers with custom coefficient."""
        model = MockModel(hidden_size=64, num_layers=3)
        steerer = ActivationSteering(model, MockTokenizer())

        direction = mx.random.normal((64,))
        for i in range(3):
            steerer.add_direction(layer=i, direction=direction)

        config = SteeringConfig(max_new_tokens=3)
        results = steerer.sweep_layers("test prompt", coefficient=2.5, config=config)

        assert len(results) == 3
        assert all(isinstance(v, str) for v in results.values())
