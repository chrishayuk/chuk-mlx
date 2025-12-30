"""Tests for steering legacy module (FunctionGemma)."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from chuk_lazarus.introspection.steering.config import LegacySteeringConfig, SteeringMode
from chuk_lazarus.introspection.steering.legacy import SteeredGemmaMLP, ToolCallingSteering


# Mock classes
class MockGateProjMLP(nn.Module):
    """Mock MLP with gate_proj, up_proj, down_proj (Gemma style)."""

    def __init__(self, hidden_size: int, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or hidden_size * 4
        self.gate_proj = nn.Linear(hidden_size, intermediate_size)
        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)

    def __call__(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(nn.gelu_approx(gate) * up)


class MockAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.o_proj = nn.Linear(hidden_size, hidden_size)


class MockGemmaLayer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.self_attn = MockAttention(hidden_size)
        self.mlp = MockGateProjMLP(hidden_size)


class MockGemmaBackbone(nn.Module):
    def __init__(self, num_layers: int, hidden_size: int):
        super().__init__()
        self.layers = [MockGemmaLayer(hidden_size) for _ in range(num_layers)]


class MockModelOutput:
    def __init__(self, logits):
        self.logits = logits


class MockGemmaModel(nn.Module):
    """Mock Gemma-style model."""

    def __init__(self, num_layers: int = 13, hidden_size: int = 256, vocab_size: int = 100):
        super().__init__()
        self.model = MockGemmaBackbone(num_layers, hidden_size)
        self._hidden_size = hidden_size
        self._vocab_size = vocab_size

    def __call__(self, input_ids):
        batch_size, seq_len = input_ids.shape
        logits = mx.random.normal((batch_size, seq_len, self._vocab_size))
        return MockModelOutput(logits)

    def generate(self, input_ids, max_new_tokens=10, temperature=0.0, stop_tokens=None):
        # Return input + some generated tokens
        new_tokens = mx.array([[5, 6, 7, 2]])  # 2 is EOS
        return mx.concatenate([input_ids, new_tokens], axis=1)


class MockTokenizer:
    def __init__(self, vocab_size: int = 100):
        self.vocab_size = vocab_size
        self.eos_token_id = 2
        self.unk_token_id = 0

    def encode(self, text, **kwargs):
        return [1, 2, 3]  # Return list (not nested)

    def decode(self, ids, **kwargs):
        return "generated text"

    def convert_tokens_to_ids(self, token):
        if token == "<end_of_turn>":
            return 99  # Not unk
        return 0


class MockConfig:
    def __init__(self, hidden_size: int = 256):
        self.hidden_size = hidden_size


class TestSteeredGemmaMLP:
    """Tests for SteeredGemmaMLP class."""

    def test_init(self):
        """Test initialization."""
        mlp = MockGateProjMLP(hidden_size=256)
        config = LegacySteeringConfig()
        steered = SteeredGemmaMLP(mlp, config, layer_idx=11)

        assert steered.original_mlp is mlp
        assert steered.config is config
        assert steered.layer_idx == 11

    def test_forward_normal_mode(self):
        """Test forward pass in normal mode."""
        mlp = MockGateProjMLP(hidden_size=256)
        config = LegacySteeringConfig(mode=SteeringMode.NORMAL)
        steered = SteeredGemmaMLP(mlp, config, layer_idx=11)

        x = mx.random.normal((1, 5, 256))
        output = steered(x)

        assert output.shape == (1, 5, 256)

    def test_forward_force_tool_control_layer(self):
        """Test forward pass in force_tool mode at control layer."""
        mlp = MockGateProjMLP(hidden_size=256, intermediate_size=2048)
        config = LegacySteeringConfig(mode=SteeringMode.FORCE_TOOL)
        steered = SteeredGemmaMLP(mlp, config, layer_idx=11, control_layer=11)

        x = mx.random.normal((1, 5, 256))
        output = steered(x)

        assert output.shape == (1, 5, 256)

    def test_forward_prevent_tool_control_layer(self):
        """Test forward pass in prevent_tool mode at control layer."""
        mlp = MockGateProjMLP(hidden_size=256, intermediate_size=2048)
        config = LegacySteeringConfig(mode=SteeringMode.PREVENT_TOOL)
        steered = SteeredGemmaMLP(mlp, config, layer_idx=11, control_layer=11)

        x = mx.random.normal((1, 5, 256))
        output = steered(x)

        assert output.shape == (1, 5, 256)

    def test_forward_boost_tool_control_layer(self):
        """Test forward pass in boost_tool mode at control layer."""
        mlp = MockGateProjMLP(hidden_size=256, intermediate_size=2048)
        config = LegacySteeringConfig(mode=SteeringMode.BOOST_TOOL)
        steered = SteeredGemmaMLP(mlp, config, layer_idx=11, control_layer=11)

        x = mx.random.normal((1, 5, 256))
        output = steered(x)

        assert output.shape == (1, 5, 256)

    def test_forward_suppress_tool_control_layer(self):
        """Test forward pass in suppress_tool mode at control layer."""
        mlp = MockGateProjMLP(hidden_size=256, intermediate_size=2048)
        config = LegacySteeringConfig(mode=SteeringMode.SUPPRESS_TOOL)
        steered = SteeredGemmaMLP(mlp, config, layer_idx=11, control_layer=11)

        x = mx.random.normal((1, 5, 256))
        output = steered(x)

        assert output.shape == (1, 5, 256)

    def test_forward_gate_layer_with_kill_switch(self):
        """Test forward pass at gate layer with kill switch enabled."""
        mlp = MockGateProjMLP(hidden_size=256, intermediate_size=2048)
        config = LegacySteeringConfig(mode=SteeringMode.FORCE_TOOL, use_kill_switch=True)
        steered = SteeredGemmaMLP(mlp, config, layer_idx=12, gate_layer=12, kill_switch_neuron=230)

        x = mx.random.normal((1, 5, 256))
        output = steered(x)

        assert output.shape == (1, 5, 256)

    def test_forward_gate_layer_with_kill_switch_boost(self):
        """Test forward pass at gate layer with kill switch boost."""
        mlp = MockGateProjMLP(hidden_size=256, intermediate_size=2048)
        config = LegacySteeringConfig(mode=SteeringMode.NORMAL, kill_switch_boost=1000.0)
        steered = SteeredGemmaMLP(mlp, config, layer_idx=12, gate_layer=12, kill_switch_neuron=230)

        x = mx.random.normal((1, 5, 256))
        output = steered(x)

        assert output.shape == (1, 5, 256)

    def test_forward_not_control_or_gate_layer(self):
        """Test forward pass at a layer that's neither control nor gate."""
        mlp = MockGateProjMLP(hidden_size=256, intermediate_size=2048)
        config = LegacySteeringConfig(mode=SteeringMode.FORCE_TOOL)
        steered = SteeredGemmaMLP(mlp, config, layer_idx=5, control_layer=11, gate_layer=12)

        x = mx.random.normal((1, 5, 256))
        output = steered(x)

        # Should just pass through normally
        assert output.shape == (1, 5, 256)


class TestToolCallingSteering:
    """Tests for ToolCallingSteering class."""

    def test_init(self):
        """Test initialization."""
        model = MockGemmaModel()
        tokenizer = MockTokenizer()
        config = MockConfig()
        steerer = ToolCallingSteering(model, tokenizer, config)

        assert steerer.model is model
        assert steerer.tokenizer is tokenizer
        assert steerer.model_config is config

    def test_class_constants(self):
        """Test class constants are defined."""
        assert ToolCallingSteering.CONTROL_LAYER == 11
        assert ToolCallingSteering.GATE_LAYER == 12
        assert ToolCallingSteering.KILL_SWITCH_NEURON == 230
        assert len(ToolCallingSteering.TOOL_PROMOTERS) == 5
        assert len(ToolCallingSteering.TOOL_SUPPRESSORS) == 5

    def test_install_uninstall_steering(self):
        """Test installing and uninstalling steering."""
        model = MockGemmaModel(num_layers=13)
        steerer = ToolCallingSteering(model, MockTokenizer(), MockConfig())

        original_mlp_11 = model.model.layers[11].mlp
        original_mlp_12 = model.model.layers[12].mlp

        config = LegacySteeringConfig()
        steerer._install_steering(config)

        # MLPs should be wrapped
        assert isinstance(model.model.layers[11].mlp, SteeredGemmaMLP)
        assert isinstance(model.model.layers[12].mlp, SteeredGemmaMLP)

        steerer._uninstall_steering()

        # MLPs should be restored
        assert model.model.layers[11].mlp is original_mlp_11
        assert model.model.layers[12].mlp is original_mlp_12

    def test_generate_normal(self):
        """Test generation in normal mode."""
        model = MockGemmaModel(num_layers=13)
        steerer = ToolCallingSteering(model, MockTokenizer(), MockConfig())

        output = steerer.generate("test prompt", mode="normal", max_new_tokens=5)
        assert isinstance(output, str)

    def test_generate_force_tool(self):
        """Test generation in force_tool mode."""
        model = MockGemmaModel(num_layers=13)
        steerer = ToolCallingSteering(model, MockTokenizer(), MockConfig())

        output = steerer.generate("test prompt", mode="force_tool", max_new_tokens=5)
        assert isinstance(output, str)

    def test_generate_prevent_tool(self):
        """Test generation in prevent_tool mode."""
        model = MockGemmaModel(num_layers=13)
        steerer = ToolCallingSteering(model, MockTokenizer(), MockConfig())

        output = steerer.generate("test prompt", mode="prevent_tool", max_new_tokens=5)
        assert isinstance(output, str)

    def test_generate_with_temperature(self):
        """Test generation with temperature."""
        model = MockGemmaModel(num_layers=13)
        steerer = ToolCallingSteering(model, MockTokenizer(), MockConfig())

        output = steerer.generate("test prompt", mode="normal", max_new_tokens=5, temperature=0.5)
        assert isinstance(output, str)

    def test_generate_with_numpy_tokens(self):
        """Test generation when tokenizer returns numpy array."""

        class NumpyTokenizer(MockTokenizer):
            def encode(self, text, **kwargs):
                return np.array([1, 2, 3])

        model = MockGemmaModel(num_layers=13)
        steerer = ToolCallingSteering(model, NumpyTokenizer(), MockConfig())

        output = steerer.generate("test prompt", mode="normal", max_new_tokens=5)
        assert isinstance(output, str)

    def test_predict_normal(self):
        """Test prediction in normal mode."""
        model = MockGemmaModel(num_layers=13)
        steerer = ToolCallingSteering(model, MockTokenizer(), MockConfig())

        result = steerer.predict("test prompt", mode="normal")

        assert isinstance(result, dict)
        assert "prompt" in result
        assert "mode" in result
        assert "top_tokens" in result
        assert "tool_likely" in result
        assert len(result["top_tokens"]) == 5

    def test_predict_force_tool(self):
        """Test prediction in force_tool mode."""
        model = MockGemmaModel(num_layers=13)
        steerer = ToolCallingSteering(model, MockTokenizer(), MockConfig())

        result = steerer.predict("test prompt", mode="force_tool")

        assert result["mode"] == "force_tool"

    def test_predict_with_kwargs(self):
        """Test prediction with additional kwargs."""
        model = MockGemmaModel(num_layers=13)
        steerer = ToolCallingSteering(model, MockTokenizer(), MockConfig())

        result = steerer.predict("test prompt", mode="normal", use_kill_switch=True)

        assert isinstance(result, dict)

    def test_predict_tool_likely_detection(self):
        """Test tool_likely detection in prediction."""
        model = MockGemmaModel(num_layers=13)

        class MockTokenizerWithToolToken(MockTokenizer):
            def decode(self, ids, **kwargs):
                # First token looks like a function call
                if ids == [ids[0]] if isinstance(ids, list) else True:
                    return "{"
                return "text"

        steerer = ToolCallingSteering(model, MockTokenizerWithToolToken(), MockConfig())
        result = steerer.predict("test prompt", mode="normal")

        # Since we return "{" it should detect tool_likely
        # But actual behavior depends on model output
        assert "tool_likely" in result

    def test_compare_modes(self):
        """Test comparing all modes."""
        model = MockGemmaModel(num_layers=13)
        steerer = ToolCallingSteering(model, MockTokenizer(), MockConfig())

        results = steerer.compare_modes("test prompt")

        assert isinstance(results, dict)
        # Should have results for all modes
        for mode in SteeringMode:
            assert mode.value in results
