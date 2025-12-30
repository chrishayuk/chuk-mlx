"""Tests for ablation adapter module."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from chuk_lazarus.introspection.ablation.adapter import ModelAdapter


class MockAttention(nn.Module):
    """Mock attention module."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.o_proj = nn.Linear(hidden_size, hidden_size)


class MockMLP(nn.Module):
    """Mock MLP module."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.down_proj = nn.Linear(hidden_size, hidden_size)


class MockMLPWithCProj(nn.Module):
    """Mock MLP with GPT-2 style c_proj."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.c_proj = nn.Linear(hidden_size, hidden_size)


class MockMLPWithW2(nn.Module):
    """Mock MLP with Llama style w2."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.w2 = nn.Linear(hidden_size, hidden_size)


class MockMoERouter(nn.Module):
    """Mock MoE router."""

    def __init__(self, hidden_size: int, num_experts: int = 4):
        super().__init__()
        self.weight = mx.random.normal((num_experts, hidden_size))


class MockMoEMLP(nn.Module):
    """Mock MoE MLP module."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.router = MockMoERouter(hidden_size)
        self.experts = [MockMLP(hidden_size) for _ in range(4)]


class MockFeedForward(nn.Module):
    """Mock feed forward module (alternative naming)."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.down_proj = nn.Linear(hidden_size, hidden_size)


class MockFeedForwardW2(nn.Module):
    """Mock feed forward with w2."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.w2 = nn.Linear(hidden_size, hidden_size)


class MockAttentionWithOutProj(nn.Module):
    """Mock attention with out_proj naming."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.out_proj = nn.Linear(hidden_size, hidden_size)


class MockAttentionWo(nn.Module):
    """Mock attention with wo (Llama style)."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.wo = nn.Linear(hidden_size, hidden_size)


class MockLayer(nn.Module):
    """Mock transformer layer."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.self_attn = MockAttention(hidden_size)
        self.mlp = MockMLP(hidden_size)


class MockLayerWithFeedForward(nn.Module):
    """Mock layer with feed_forward naming."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.self_attn = MockAttention(hidden_size)
        self.feed_forward = MockFeedForward(hidden_size)


class MockLayerWithFeedForwardW2(nn.Module):
    """Mock layer with feed_forward using w2."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.self_attn = MockAttention(hidden_size)
        self.feed_forward = MockFeedForwardW2(hidden_size)


class MockLayerCProj(nn.Module):
    """Mock layer with GPT-2 style naming."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.self_attn = MockAttention(hidden_size)
        self.mlp = MockMLPWithCProj(hidden_size)


class MockLayerW2(nn.Module):
    """Mock layer with Llama style w2."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.self_attn = MockAttention(hidden_size)
        self.mlp = MockMLPWithW2(hidden_size)


class MockMoELayer(nn.Module):
    """Mock MoE layer."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.self_attn = MockAttention(hidden_size)
        self.mlp = MockMoEMLP(hidden_size)


class MockLayerAttention(nn.Module):
    """Mock layer with 'attention' instead of 'self_attn'."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = MockAttention(hidden_size)
        self.mlp = MockMLP(hidden_size)


class MockLayerAttentionWo(nn.Module):
    """Mock layer with attention.wo."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = MockAttentionWo(hidden_size)
        self.mlp = MockMLP(hidden_size)


class MockLayerWithOutProj(nn.Module):
    """Mock layer with out_proj attention."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.self_attn = MockAttentionWithOutProj(hidden_size)
        self.mlp = MockMLP(hidden_size)


class MockBackbone(nn.Module):
    """Mock transformer backbone."""

    def __init__(self, num_layers: int, hidden_size: int, layer_class=None):
        super().__init__()
        layer_class = layer_class or MockLayer
        self.layers = [layer_class(hidden_size) for _ in range(num_layers)]


class MockModel(nn.Module):
    """Mock model with model.layers structure."""

    def __init__(self, num_layers: int = 4, hidden_size: int = 64, layer_class=None):
        super().__init__()
        self.model = MockBackbone(num_layers, hidden_size, layer_class)


class MockModelDirect(nn.Module):
    """Mock model with direct model.layers structure."""

    def __init__(self, num_layers: int = 4, hidden_size: int = 64, layer_class=None):
        super().__init__()
        layer_class = layer_class or MockLayer
        self.layers = [layer_class(hidden_size) for _ in range(num_layers)]


class MockTransformerH(nn.Module):
    """Mock transformer backbone with h attribute (GPT-2 style)."""

    def __init__(self, num_layers: int, hidden_size: int, layer_class=None):
        super().__init__()
        layer_class = layer_class or MockLayer
        self.h = [layer_class(hidden_size) for _ in range(num_layers)]


class MockModelGPT2(nn.Module):
    """Mock GPT-2 style model with transformer.h."""

    def __init__(self, num_layers: int = 4, hidden_size: int = 64):
        super().__init__()
        self.transformer = MockTransformerH(num_layers, hidden_size)


class MockTokenizer:
    """Mock tokenizer."""

    def __init__(self, vocab_size: int = 100):
        self.vocab_size = vocab_size
        self.eos_token_id = 2

    def encode(self, text, **kwargs):
        return [[1, 2, 3]]

    def decode(self, ids, **kwargs):
        return "generated text"


class MockConfig:
    """Mock model config."""

    def __init__(self, hidden_size: int = 64):
        self.hidden_size = hidden_size


class MockConfigDModel:
    """Mock model config with d_model."""

    def __init__(self, d_model: int = 64):
        self.d_model = d_model


class MockConfigNoSize:
    """Mock config without hidden_size."""

    pass


class MockModelOutput:
    """Mock model output with logits."""

    def __init__(self, logits):
        self.logits = logits


class MockModelWithGenerate(nn.Module):
    """Mock model with generate method."""

    def __init__(self, num_layers: int = 4, hidden_size: int = 64):
        super().__init__()
        self.model = MockBackbone(num_layers, hidden_size)
        self._hidden_size = hidden_size

    def generate(self, input_ids, max_new_tokens=10, temperature=0.0, stop_tokens=None):
        # Return input + some generated tokens
        new_tokens = mx.array([[5, 6, 7, 2]])  # 2 is EOS
        return mx.concatenate([input_ids, new_tokens], axis=1)


class MockModelCallable(nn.Module):
    """Mock model that's callable for manual generation."""

    def __init__(self, num_layers: int = 4, hidden_size: int = 64, vocab_size: int = 100):
        super().__init__()
        self.model = MockBackbone(num_layers, hidden_size)
        self._hidden_size = hidden_size
        self._vocab_size = vocab_size
        self._call_count = 0

    def __call__(self, input_ids):
        self._call_count += 1
        batch_size, seq_len = input_ids.shape
        # Return logits that favor token 5, then 6, then EOS (2)
        logits = mx.random.normal((batch_size, seq_len, self._vocab_size))
        # Make EOS very likely after a few tokens
        if self._call_count > 2:
            logits = logits.at[:, -1, 2].add(100.0)  # Boost EOS token
        return MockModelOutput(logits)


class TestModelAdapter:
    """Tests for ModelAdapter class."""

    def test_init_model_model_layers(self):
        """Test initialization with model.model.layers structure."""
        model = MockModel(num_layers=4, hidden_size=64)
        tokenizer = MockTokenizer()
        config = MockConfig()
        adapter = ModelAdapter(model, tokenizer, config)
        assert adapter.num_layers == 4

    def test_init_direct_layers(self):
        """Test initialization with direct model.layers structure."""
        model = MockModelDirect(num_layers=6, hidden_size=64)
        adapter = ModelAdapter(model, MockTokenizer(), MockConfig())
        assert adapter.num_layers == 6

    def test_init_gpt2_style(self):
        """Test initialization with GPT-2 style transformer.h structure."""
        model = MockModelGPT2(num_layers=5, hidden_size=64)
        adapter = ModelAdapter(model, MockTokenizer(), MockConfig())
        assert adapter.num_layers == 5

    def test_num_layers(self):
        """Test num_layers property."""
        model = MockModel(num_layers=8)
        adapter = ModelAdapter(model, MockTokenizer(), MockConfig())
        assert adapter.num_layers == 8

    def test_hidden_size(self):
        """Test hidden_size property."""
        model = MockModel(hidden_size=128)
        config = MockConfig(hidden_size=128)
        adapter = ModelAdapter(model, MockTokenizer(), config)
        assert adapter.hidden_size == 128

    def test_hidden_size_d_model(self):
        """Test hidden_size with d_model config."""
        model = MockModel(hidden_size=256)
        config = MockConfigDModel(d_model=256)
        adapter = ModelAdapter(model, MockTokenizer(), config)
        assert adapter.hidden_size == 256

    def test_hidden_size_no_config(self):
        """Test hidden_size raises when not in config."""
        model = MockModel(hidden_size=64)
        config = MockConfigNoSize()
        adapter = ModelAdapter(model, MockTokenizer(), config)
        with pytest.raises(ValueError, match="Cannot determine hidden size"):
            _ = adapter.hidden_size

    def test_get_layer(self):
        """Test get_layer method."""
        model = MockModel(num_layers=4)
        adapter = ModelAdapter(model, MockTokenizer(), MockConfig())
        layer = adapter.get_layer(2)
        assert isinstance(layer, MockLayer)

    def test_is_moe_layer_false(self):
        """Test is_moe_layer for non-MoE layer."""
        model = MockModel()
        adapter = ModelAdapter(model, MockTokenizer(), MockConfig())
        assert adapter.is_moe_layer(0) is False

    def test_is_moe_layer_true_with_router(self):
        """Test is_moe_layer for MoE layer with router."""
        model = MockModel(num_layers=4, hidden_size=64, layer_class=MockMoELayer)
        adapter = ModelAdapter(model, MockTokenizer(), MockConfig())
        assert adapter.is_moe_layer(0) is True

    def test_get_mlp_down_weight(self):
        """Test getting MLP down projection weight."""
        model = MockModel(hidden_size=64)
        adapter = ModelAdapter(model, MockTokenizer(), MockConfig(64))
        weight = adapter.get_mlp_down_weight(0)
        assert weight.shape == (64, 64)

    def test_get_mlp_down_weight_c_proj(self):
        """Test getting MLP down weight with c_proj naming."""
        model = MockModel(hidden_size=64, layer_class=MockLayerCProj)
        adapter = ModelAdapter(model, MockTokenizer(), MockConfig(64))
        weight = adapter.get_mlp_down_weight(0)
        assert weight.shape == (64, 64)

    def test_get_mlp_down_weight_w2(self):
        """Test getting MLP down weight with w2 naming."""
        model = MockModel(hidden_size=64, layer_class=MockLayerW2)
        adapter = ModelAdapter(model, MockTokenizer(), MockConfig(64))
        weight = adapter.get_mlp_down_weight(0)
        assert weight.shape == (64, 64)

    def test_get_mlp_down_weight_feed_forward(self):
        """Test getting MLP down weight with feed_forward naming."""
        model = MockModel(hidden_size=64, layer_class=MockLayerWithFeedForward)
        adapter = ModelAdapter(model, MockTokenizer(), MockConfig(64))
        weight = adapter.get_mlp_down_weight(0)
        assert weight.shape == (64, 64)

    def test_get_mlp_down_weight_feed_forward_w2(self):
        """Test getting MLP down weight with feed_forward.w2."""
        model = MockModel(hidden_size=64, layer_class=MockLayerWithFeedForwardW2)
        adapter = ModelAdapter(model, MockTokenizer(), MockConfig(64))
        weight = adapter.get_mlp_down_weight(0)
        assert weight.shape == (64, 64)

    def test_get_mlp_down_weight_moe_router(self):
        """Test getting MLP down weight from MoE router."""
        model = MockModel(hidden_size=64, layer_class=MockMoELayer)
        adapter = ModelAdapter(model, MockTokenizer(), MockConfig(64))
        weight = adapter.get_mlp_down_weight(0)
        # Router weight shape is (num_experts, hidden_size)
        assert weight.shape[1] == 64

    def test_set_mlp_down_weight(self):
        """Test setting MLP down projection weight."""
        model = MockModel(hidden_size=64)
        adapter = ModelAdapter(model, MockTokenizer(), MockConfig(64))
        new_weight = mx.zeros((64, 64))
        adapter.set_mlp_down_weight(0, new_weight)
        retrieved = adapter.get_mlp_down_weight(0)
        assert mx.allclose(retrieved, new_weight)

    def test_set_mlp_down_weight_c_proj(self):
        """Test setting MLP down weight with c_proj naming."""
        model = MockModel(hidden_size=64, layer_class=MockLayerCProj)
        adapter = ModelAdapter(model, MockTokenizer(), MockConfig(64))
        new_weight = mx.zeros((64, 64))
        adapter.set_mlp_down_weight(0, new_weight)
        retrieved = adapter.get_mlp_down_weight(0)
        assert mx.allclose(retrieved, new_weight)

    def test_set_mlp_down_weight_w2(self):
        """Test setting MLP down weight with w2 naming."""
        model = MockModel(hidden_size=64, layer_class=MockLayerW2)
        adapter = ModelAdapter(model, MockTokenizer(), MockConfig(64))
        new_weight = mx.zeros((64, 64))
        adapter.set_mlp_down_weight(0, new_weight)
        retrieved = adapter.get_mlp_down_weight(0)
        assert mx.allclose(retrieved, new_weight)

    def test_set_mlp_down_weight_feed_forward(self):
        """Test setting MLP down weight with feed_forward naming."""
        model = MockModel(hidden_size=64, layer_class=MockLayerWithFeedForward)
        adapter = ModelAdapter(model, MockTokenizer(), MockConfig(64))
        new_weight = mx.zeros((64, 64))
        adapter.set_mlp_down_weight(0, new_weight)
        retrieved = adapter.get_mlp_down_weight(0)
        assert mx.allclose(retrieved, new_weight)

    def test_set_mlp_down_weight_feed_forward_w2(self):
        """Test setting MLP down weight with feed_forward.w2."""
        model = MockModel(hidden_size=64, layer_class=MockLayerWithFeedForwardW2)
        adapter = ModelAdapter(model, MockTokenizer(), MockConfig(64))
        new_weight = mx.zeros((64, 64))
        adapter.set_mlp_down_weight(0, new_weight)
        retrieved = adapter.get_mlp_down_weight(0)
        assert mx.allclose(retrieved, new_weight)

    def test_get_attn_o_weight(self):
        """Test getting attention output projection weight."""
        model = MockModel(hidden_size=64)
        adapter = ModelAdapter(model, MockTokenizer(), MockConfig(64))
        weight = adapter.get_attn_o_weight(0)
        assert weight.shape == (64, 64)

    def test_get_attn_o_weight_out_proj(self):
        """Test getting attention output weight with out_proj naming."""
        model = MockModel(hidden_size=64, layer_class=MockLayerWithOutProj)
        adapter = ModelAdapter(model, MockTokenizer(), MockConfig(64))
        weight = adapter.get_attn_o_weight(0)
        assert weight.shape == (64, 64)

    def test_get_attn_o_weight_attention(self):
        """Test getting attention output weight with 'attention' naming."""
        model = MockModel(hidden_size=64, layer_class=MockLayerAttention)
        adapter = ModelAdapter(model, MockTokenizer(), MockConfig(64))
        weight = adapter.get_attn_o_weight(0)
        assert weight.shape == (64, 64)

    def test_get_attn_o_weight_wo(self):
        """Test getting attention output weight with wo naming."""
        model = MockModel(hidden_size=64, layer_class=MockLayerAttentionWo)
        adapter = ModelAdapter(model, MockTokenizer(), MockConfig(64))
        weight = adapter.get_attn_o_weight(0)
        assert weight.shape == (64, 64)

    def test_set_attn_o_weight(self):
        """Test setting attention output projection weight."""
        model = MockModel(hidden_size=64)
        adapter = ModelAdapter(model, MockTokenizer(), MockConfig(64))
        new_weight = mx.zeros((64, 64))
        adapter.set_attn_o_weight(0, new_weight)
        retrieved = adapter.get_attn_o_weight(0)
        assert mx.allclose(retrieved, new_weight)

    def test_set_attn_o_weight_out_proj(self):
        """Test setting attention output weight with out_proj."""
        model = MockModel(hidden_size=64, layer_class=MockLayerWithOutProj)
        adapter = ModelAdapter(model, MockTokenizer(), MockConfig(64))
        new_weight = mx.zeros((64, 64))
        adapter.set_attn_o_weight(0, new_weight)
        retrieved = adapter.get_attn_o_weight(0)
        assert mx.allclose(retrieved, new_weight)

    def test_set_attn_o_weight_attention(self):
        """Test setting attention output weight with 'attention' naming."""
        model = MockModel(hidden_size=64, layer_class=MockLayerAttention)
        adapter = ModelAdapter(model, MockTokenizer(), MockConfig(64))
        new_weight = mx.zeros((64, 64))
        adapter.set_attn_o_weight(0, new_weight)
        retrieved = adapter.get_attn_o_weight(0)
        assert mx.allclose(retrieved, new_weight)

    def test_set_attn_o_weight_wo(self):
        """Test setting attention output weight with wo naming."""
        model = MockModel(hidden_size=64, layer_class=MockLayerAttentionWo)
        adapter = ModelAdapter(model, MockTokenizer(), MockConfig(64))
        new_weight = mx.zeros((64, 64))
        adapter.set_attn_o_weight(0, new_weight)
        retrieved = adapter.get_attn_o_weight(0)
        assert mx.allclose(retrieved, new_weight)

    def test_generate_with_model_generate(self):
        """Test generate using model's built-in generate method."""
        model = MockModelWithGenerate(hidden_size=64)
        adapter = ModelAdapter(model, MockTokenizer(), MockConfig(64))
        input_ids = mx.array([[1, 2, 3]])
        output = adapter.generate(input_ids, max_new_tokens=10)
        assert isinstance(output, str)

    def test_manual_generate(self):
        """Test manual generation fallback."""
        model = MockModelCallable(hidden_size=64, vocab_size=100)
        adapter = ModelAdapter(model, MockTokenizer(), MockConfig(64))
        input_ids = mx.array([[1, 2, 3]])
        output = adapter._manual_generate(input_ids, max_new_tokens=5, temperature=0.0)
        assert isinstance(output, str)

    def test_manual_generate_with_temperature(self):
        """Test manual generation with non-zero temperature."""
        model = MockModelCallable(hidden_size=64, vocab_size=100)
        adapter = ModelAdapter(model, MockTokenizer(), MockConfig(64))
        input_ids = mx.array([[1, 2, 3]])
        output = adapter._manual_generate(input_ids, max_new_tokens=5, temperature=0.5)
        assert isinstance(output, str)


class TestModelAdapterUnsupportedArchitecture:
    """Tests for unsupported architectures."""

    class UnsupportedModel(nn.Module):
        """Model with unsupported structure."""

        def __init__(self):
            super().__init__()
            self.something = nn.Linear(10, 10)

    def test_raises_on_unsupported_architecture(self):
        """Test that unsupported architecture raises ValueError."""
        model = self.UnsupportedModel()
        with pytest.raises(ValueError, match="Cannot detect model architecture"):
            ModelAdapter(model, MockTokenizer(), MockConfig())


class TestModelAdapterEdgeCases:
    """Test edge cases and error handling."""

    class LayerNoMLP(nn.Module):
        """Layer without MLP."""

        def __init__(self, hidden_size: int):
            super().__init__()
            self.self_attn = MockAttention(hidden_size)

    class LayerNoAttn(nn.Module):
        """Layer without attention."""

        def __init__(self, hidden_size: int):
            super().__init__()
            self.mlp = MockMLP(hidden_size)

    class MLPNoWeight(nn.Module):
        """MLP without recognizable weight attributes."""

        def __init__(self, hidden_size: int):
            super().__init__()
            self.something = nn.Linear(hidden_size, hidden_size)

    class LayerMLPNoWeight(nn.Module):
        """Layer with MLP that has no recognizable weight."""

        def __init__(self, hidden_size: int):
            super().__init__()
            self.self_attn = MockAttention(hidden_size)
            self.mlp = TestModelAdapterEdgeCases.MLPNoWeight(hidden_size)

    class FeedForwardNoWeight(nn.Module):
        """Feed forward without recognizable weight."""

        def __init__(self, hidden_size: int):
            super().__init__()
            self.something = nn.Linear(hidden_size, hidden_size)

    class LayerFeedForwardNoWeight(nn.Module):
        """Layer with feed_forward that has no recognizable weight."""

        def __init__(self, hidden_size: int):
            super().__init__()
            self.self_attn = MockAttention(hidden_size)
            self.feed_forward = TestModelAdapterEdgeCases.FeedForwardNoWeight(hidden_size)

    class AttentionNoWeight(nn.Module):
        """Attention without recognizable weight."""

        def __init__(self, hidden_size: int):
            super().__init__()
            self.something = nn.Linear(hidden_size, hidden_size)

    class LayerAttentionNoWeight(nn.Module):
        """Layer with self_attn that has no recognizable weight."""

        def __init__(self, hidden_size: int):
            super().__init__()
            self.self_attn = TestModelAdapterEdgeCases.AttentionNoWeight(hidden_size)
            self.mlp = MockMLP(hidden_size)

    class LayerAttnNoWeight(nn.Module):
        """Layer with attention attr that has no recognizable weight."""

        def __init__(self, hidden_size: int):
            super().__init__()
            self.attention = TestModelAdapterEdgeCases.AttentionNoWeight(hidden_size)
            self.mlp = MockMLP(hidden_size)

    def test_get_mlp_down_weight_no_mlp(self):
        """Test error when layer has no MLP."""
        model = MockModel(hidden_size=64, layer_class=self.LayerNoMLP)
        adapter = ModelAdapter(model, MockTokenizer(), MockConfig(64))
        with pytest.raises(ValueError, match="Cannot find MLP"):
            adapter.get_mlp_down_weight(0)

    def test_get_mlp_down_weight_unrecognized(self):
        """Test error when MLP has unrecognized weight naming."""
        model = MockModel(hidden_size=64, layer_class=self.LayerMLPNoWeight)
        adapter = ModelAdapter(model, MockTokenizer(), MockConfig(64))
        with pytest.raises(ValueError, match="Cannot find MLP down projection"):
            adapter.get_mlp_down_weight(0)

    def test_set_mlp_down_weight_unrecognized(self):
        """Test error when setting MLP weight with unrecognized naming."""
        model = MockModel(hidden_size=64, layer_class=self.LayerMLPNoWeight)
        adapter = ModelAdapter(model, MockTokenizer(), MockConfig(64))
        with pytest.raises(ValueError, match="Cannot find MLP down projection"):
            adapter.set_mlp_down_weight(0, mx.zeros((64, 64)))

    def test_set_mlp_down_weight_feed_forward_unrecognized(self):
        """Test error when setting feed_forward weight with unrecognized naming."""
        model = MockModel(hidden_size=64, layer_class=self.LayerFeedForwardNoWeight)
        adapter = ModelAdapter(model, MockTokenizer(), MockConfig(64))
        with pytest.raises(ValueError, match="Cannot find MLP down projection"):
            adapter.set_mlp_down_weight(0, mx.zeros((64, 64)))

    def test_set_mlp_down_weight_no_mlp(self):
        """Test error when setting MLP weight on layer without MLP."""
        model = MockModel(hidden_size=64, layer_class=self.LayerNoMLP)
        adapter = ModelAdapter(model, MockTokenizer(), MockConfig(64))
        with pytest.raises(ValueError, match="Cannot find MLP"):
            adapter.set_mlp_down_weight(0, mx.zeros((64, 64)))

    def test_get_attn_o_weight_no_attn(self):
        """Test error when layer has no attention."""
        model = MockModel(hidden_size=64, layer_class=self.LayerNoAttn)
        adapter = ModelAdapter(model, MockTokenizer(), MockConfig(64))
        with pytest.raises(ValueError, match="Cannot find attention"):
            adapter.get_attn_o_weight(0)

    def test_get_attn_o_weight_unrecognized(self):
        """Test error when attention has unrecognized weight naming."""
        model = MockModel(hidden_size=64, layer_class=self.LayerAttentionNoWeight)
        adapter = ModelAdapter(model, MockTokenizer(), MockConfig(64))
        with pytest.raises(ValueError, match="Cannot find attention output projection"):
            adapter.get_attn_o_weight(0)

    def test_set_attn_o_weight_no_attn(self):
        """Test error when setting attention weight on layer without attention."""
        model = MockModel(hidden_size=64, layer_class=self.LayerNoAttn)
        adapter = ModelAdapter(model, MockTokenizer(), MockConfig(64))
        with pytest.raises(ValueError, match="Cannot find attention"):
            adapter.set_attn_o_weight(0, mx.zeros((64, 64)))

    def test_set_attn_o_weight_unrecognized(self):
        """Test error when setting attention weight with unrecognized naming."""
        model = MockModel(hidden_size=64, layer_class=self.LayerAttentionNoWeight)
        adapter = ModelAdapter(model, MockTokenizer(), MockConfig(64))
        with pytest.raises(ValueError, match="Cannot find attention output projection"):
            adapter.set_attn_o_weight(0, mx.zeros((64, 64)))

    def test_set_attn_o_weight_attention_unrecognized(self):
        """Test error when setting attention.x weight with unrecognized naming."""
        model = MockModel(hidden_size=64, layer_class=self.LayerAttnNoWeight)
        adapter = ModelAdapter(model, MockTokenizer(), MockConfig(64))
        with pytest.raises(ValueError, match="Cannot find attention output projection"):
            adapter.set_attn_o_weight(0, mx.zeros((64, 64)))
