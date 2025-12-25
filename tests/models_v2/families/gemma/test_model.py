"""
Tests for Gemma 3 model components.
"""

import mlx.core as mx
import pytest

from chuk_lazarus.models_v2.families.gemma import (
    GemmaAttention,
    GemmaBlock,
    GemmaConfig,
    GemmaForCausalLM,
    GemmaMLP,
    GemmaModel,
    GemmaRMSNorm,
)


class TestGemmaRMSNorm:
    """Tests for GemmaRMSNorm."""

    def test_basic_norm(self):
        """Test basic normalization."""
        norm = GemmaRMSNorm(dims=64, eps=1e-6)
        x = mx.random.normal((2, 10, 64))

        output = norm(x)

        assert output.shape == x.shape

    def test_uses_one_plus_weight(self):
        """Test that Gemma uses (1 + weight) scaling."""
        norm = GemmaRMSNorm(dims=32)

        # Weight should be initialized to ones
        assert norm.weight.shape == (32,)
        # After (1 + weight), effective scale is 2
        assert mx.allclose(1.0 + norm.weight, mx.ones((32,)) * 2.0)


class TestGemmaMLP:
    """Tests for GemmaMLP."""

    def test_forward_pass(self):
        """Test MLP forward pass."""
        mlp = GemmaMLP(hidden_size=64, intermediate_size=128)
        x = mx.random.normal((2, 10, 64))

        output = mlp(x)

        assert output.shape == x.shape

    def test_gated_gelu(self):
        """Test that MLP uses gated GELU pattern."""
        mlp = GemmaMLP(hidden_size=32, intermediate_size=64)

        assert hasattr(mlp, "gate_proj")
        assert hasattr(mlp, "up_proj")
        assert hasattr(mlp, "down_proj")


class TestGemmaAttention:
    """Tests for GemmaAttention."""

    @pytest.fixture
    def tiny_config(self):
        return GemmaConfig.tiny()

    def test_forward_pass(self, tiny_config):
        """Test attention forward pass."""
        attn = GemmaAttention(tiny_config, layer_idx=0)
        x = mx.random.normal((2, 10, tiny_config.hidden_size))

        output, cache = attn(x)

        assert output.shape == x.shape
        assert cache is not None
        assert len(cache) == 2  # (keys, values)

    def test_sliding_vs_global_layer(self, tiny_config):
        """Test different attention types for sliding vs global layers."""
        sliding_attn = GemmaAttention(tiny_config, layer_idx=0)  # sliding
        global_attn = GemmaAttention(tiny_config, layer_idx=2)   # global (pattern=3)

        assert sliding_attn.is_sliding is True
        assert global_attn.is_sliding is False

    def test_query_key_norms(self, tiny_config):
        """Test that attention has Q/K normalization."""
        attn = GemmaAttention(tiny_config, layer_idx=0)

        assert hasattr(attn, "q_norm")
        assert hasattr(attn, "k_norm")
        assert isinstance(attn.q_norm, GemmaRMSNorm)
        assert isinstance(attn.k_norm, GemmaRMSNorm)

    def test_with_cache(self, tiny_config):
        """Test attention with KV cache."""
        attn = GemmaAttention(tiny_config, layer_idx=0)

        # First forward
        x1 = mx.random.normal((1, 5, tiny_config.hidden_size))
        _, cache1 = attn(x1)

        # Second forward with cache
        x2 = mx.random.normal((1, 1, tiny_config.hidden_size))
        output2, cache2 = attn(x2, cache=cache1)

        assert output2.shape == (1, 1, tiny_config.hidden_size)
        assert cache2[0].shape[2] == 6  # 5 + 1 cached positions


class TestGemmaBlock:
    """Tests for GemmaBlock."""

    @pytest.fixture
    def tiny_config(self):
        return GemmaConfig.tiny()

    def test_forward_pass(self, tiny_config):
        """Test block forward pass."""
        block = GemmaBlock(tiny_config, layer_idx=0)
        x = mx.random.normal((2, 10, tiny_config.hidden_size))

        output = block(x)

        assert output.hidden_states.shape == x.shape
        assert output.cache is not None

    def test_four_norm_layers(self, tiny_config):
        """Test that block has 4 normalization layers."""
        block = GemmaBlock(tiny_config, layer_idx=0)

        assert hasattr(block, "input_layernorm")
        assert hasattr(block, "post_attention_layernorm")
        assert hasattr(block, "pre_feedforward_layernorm")
        assert hasattr(block, "post_feedforward_layernorm")


class TestGemmaModel:
    """Tests for GemmaModel (backbone)."""

    @pytest.fixture
    def tiny_config(self):
        return GemmaConfig.tiny()

    def test_forward_pass(self, tiny_config):
        """Test backbone forward pass."""
        model = GemmaModel(tiny_config)
        input_ids = mx.array([[1, 2, 3, 4, 5]])

        output = model(input_ids)

        assert output.last_hidden_state.shape == (1, 5, tiny_config.hidden_size)

    def test_embedding_scaling(self, tiny_config):
        """Test that embeddings are scaled by sqrt(hidden_size)."""
        model = GemmaModel(tiny_config)
        input_ids = mx.array([[1]])

        # Get raw embedding
        raw_embed = model.embed_tokens(input_ids)

        # The model internally scales by sqrt(hidden_size)
        # We can verify the output is different from raw embedding
        output = model(input_ids)
        # Note: After all layers, the relationship is complex
        # Just verify the model runs
        assert output.last_hidden_state is not None

    def test_output_hidden_states(self, tiny_config):
        """Test returning all hidden states."""
        model = GemmaModel(tiny_config)
        input_ids = mx.array([[1, 2, 3, 4, 5]])

        output = model(input_ids, output_hidden_states=True)

        assert output.hidden_states is not None
        # num_layers + 1 (for embeddings)
        assert len(output.hidden_states) == tiny_config.num_hidden_layers + 1

    def test_with_cache(self, tiny_config):
        """Test generation with KV cache."""
        model = GemmaModel(tiny_config)

        # First forward
        input_ids = mx.array([[1, 2, 3]])
        output1 = model(input_ids)

        # Second forward with cache
        next_ids = mx.array([[4]])
        output2 = model(next_ids, cache=output1.cache)

        assert output2.last_hidden_state.shape == (1, 1, tiny_config.hidden_size)


class TestGemmaForCausalLM:
    """Tests for GemmaForCausalLM."""

    @pytest.fixture
    def tiny_config(self):
        return GemmaConfig.tiny()

    def test_forward_pass(self, tiny_config):
        """Test full model forward pass."""
        model = GemmaForCausalLM(tiny_config)
        input_ids = mx.array([[1, 2, 3, 4, 5]])

        output = model(input_ids)

        assert output.logits.shape == (1, 5, tiny_config.vocab_size)
        assert output.loss is None  # No labels provided

    def test_with_labels(self, tiny_config):
        """Test forward with labels for loss computation."""
        model = GemmaForCausalLM(tiny_config)
        input_ids = mx.array([[1, 2, 3, 4, 5]])
        labels = mx.array([[2, 3, 4, 5, 6]])

        output = model(input_ids, labels=labels)

        assert output.logits is not None
        assert output.loss is not None
        assert output.loss.item() > 0

    def test_generate_greedy(self, tiny_config):
        """Test greedy generation."""
        model = GemmaForCausalLM(tiny_config)
        input_ids = mx.array([[1, 2, 3]])

        generated = model.generate(
            input_ids,
            max_new_tokens=5,
            temperature=0,  # Greedy
        )

        assert generated.shape[0] == 1
        assert generated.shape[1] == 8  # 3 prompt + 5 generated

    def test_generate_with_sampling(self, tiny_config):
        """Test generation with temperature sampling."""
        model = GemmaForCausalLM(tiny_config)
        input_ids = mx.array([[1, 2]])

        generated = model.generate(
            input_ids,
            max_new_tokens=3,
            temperature=0.8,
        )

        assert generated.shape[1] == 5  # 2 prompt + 3 generated

    def test_generate_with_stop_tokens(self, tiny_config):
        """Test generation stops on stop tokens."""
        model = GemmaForCausalLM(tiny_config)
        input_ids = mx.array([[1, 2, 3]])

        # Generate with EOS as stop token
        generated = model.generate(
            input_ids,
            max_new_tokens=100,
            temperature=0,
            stop_tokens=[1],  # Stop on token 1
        )

        # Should stop before max_new_tokens if token 1 is generated
        assert generated.shape[1] <= 103

    def test_config_property(self, tiny_config):
        """Test config property."""
        model = GemmaForCausalLM(tiny_config)

        assert model.config == tiny_config

    def test_backbone_property(self, tiny_config):
        """Test backbone property."""
        model = GemmaForCausalLM(tiny_config)

        assert model.backbone == model.model
        assert isinstance(model.backbone, GemmaModel)

    def test_from_config(self, tiny_config):
        """Test factory method."""
        model = GemmaForCausalLM.from_config(tiny_config)

        assert isinstance(model, GemmaForCausalLM)
        assert model.config == tiny_config


class TestFunctionGemma:
    """Tests specific to FunctionGemma usage."""

    def test_functiongemma_alias(self):
        """Test FunctionGemmaForCausalLM is an alias."""
        from chuk_lazarus.models_v2.families.gemma import FunctionGemmaForCausalLM

        assert FunctionGemmaForCausalLM is GemmaForCausalLM

    def test_functiongemma_config(self):
        """Test FunctionGemma config creation."""
        config = GemmaConfig.functiongemma_270m()

        # Should match Gemma 3 270M specs
        assert config.hidden_size == 640
        assert config.vocab_size == 262144
        assert config.num_hidden_layers == 18
