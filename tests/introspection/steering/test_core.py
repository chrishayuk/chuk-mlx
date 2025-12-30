"""Tests for steering core module."""

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
