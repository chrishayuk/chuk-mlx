"""
Tests for Granite 4.x hybrid model.
"""

import mlx.core as mx
import mlx.nn as nn

from chuk_lazarus.models_v2.families.granite.config import GraniteHybridConfig
from chuk_lazarus.models_v2.families.granite.hybrid import (
    Granite4,
    GraniteHybrid,
    GraniteHybridAttention,
    GraniteHybridBlock,
    GraniteHybridForCausalLM,
    GraniteHybridModel,
    GraniteHybridMoE,
    GraniteMamba2Block,
)


class TestGraniteMamba2Block:
    """Tests for GraniteMamba2Block."""

    def test_creation(self):
        """Test block creation."""
        config = GraniteHybridConfig.tiny()
        block = GraniteMamba2Block(config, layer_idx=0)

        assert block.hidden_size == 64
        assert block.layer_idx == 0
        assert block.d_state == config.mamba_d_state
        assert block.d_conv == config.mamba_d_conv
        assert block.expand == config.mamba_expand
        assert block.n_heads == config.mamba_n_heads
        assert block.d_head == config.mamba_d_head

    def test_forward_pass(self):
        """Test forward pass."""
        config = GraniteHybridConfig.tiny()
        block = GraniteMamba2Block(config)

        x = mx.random.normal((2, 10, 64))
        output, cache = block(x)

        assert output.shape == (2, 10, 64)
        # Cache may be None for simplified implementation

    def test_forward_different_seq_lengths(self):
        """Test forward with different sequence lengths."""
        config = GraniteHybridConfig.tiny()
        block = GraniteMamba2Block(config)

        for seq_len in [1, 5, 10]:
            x = mx.random.normal((2, seq_len, 64))
            output, _ = block(x)
            assert output.shape == (2, seq_len, 64)

    def test_selective_scan(self):
        """Test selective scan method."""
        config = GraniteHybridConfig.tiny()
        block = GraniteMamba2Block(config)

        batch_size, seq_len = 2, 5
        n_heads = config.mamba_n_heads
        d_per_head = block.d_inner // n_heads

        x = mx.random.normal((batch_size, seq_len, n_heads, d_per_head))
        dt = mx.random.normal((batch_size, seq_len, n_heads))
        B = mx.random.normal((batch_size, seq_len, n_heads, config.mamba_d_state))
        C = mx.random.normal((batch_size, seq_len, n_heads, config.mamba_d_state))

        output = block._selective_scan(x, dt, B, C)
        assert output.shape == (batch_size, seq_len, n_heads, d_per_head)


class TestGraniteHybridAttention:
    """Tests for GraniteHybridAttention."""

    def test_creation(self):
        """Test attention creation."""
        config = GraniteHybridConfig.tiny()
        attn = GraniteHybridAttention(config, layer_idx=0)

        assert attn.hidden_size == 64
        assert attn.num_heads == 4
        assert attn.num_kv_heads == 2
        assert attn.head_dim == 16
        assert attn.attention_multiplier == 1.0

    def test_creation_with_rope(self):
        """Test attention creation with RoPE."""
        config = GraniteHybridConfig.tiny()
        config.position_embedding_type = "rope"
        attn = GraniteHybridAttention(config)

        assert attn.use_rope is True

    def test_creation_without_rope(self):
        """Test attention creation without RoPE."""
        config = GraniteHybridConfig.tiny()
        config.position_embedding_type = "nope"
        attn = GraniteHybridAttention(config)

        assert attn.use_rope is False

    def test_forward_pass(self):
        """Test forward pass."""
        config = GraniteHybridConfig.tiny()
        attn = GraniteHybridAttention(config)

        x = mx.random.normal((2, 10, 64))
        output, cache = attn(x)

        assert output.shape == (2, 10, 64)
        assert cache is not None

    def test_forward_with_mask(self):
        """Test forward with mask."""
        config = GraniteHybridConfig.tiny()
        attn = GraniteHybridAttention(config)

        x = mx.random.normal((2, 10, 64))
        mask = nn.MultiHeadAttention.create_additive_causal_mask(10)
        output, cache = attn(x, mask=mask)

        assert output.shape == (2, 10, 64)

    def test_forward_with_cache(self):
        """Test forward with cache."""
        config = GraniteHybridConfig.tiny()
        attn = GraniteHybridAttention(config)

        # First pass
        x1 = mx.random.normal((2, 10, 64))
        _, cache = attn(x1)

        # Second pass
        x2 = mx.random.normal((2, 1, 64))
        output, new_cache = attn(x2, cache=cache)

        assert output.shape == (2, 1, 64)

    def test_kv_repeat(self):
        """Test KV head repeat for GQA."""
        config = GraniteHybridConfig.tiny()
        config.num_attention_heads = 8
        config.num_key_value_heads = 2
        attn = GraniteHybridAttention(config)

        assert attn.n_rep == 4

        x = mx.random.normal((2, 5, 64))
        output, _ = attn(x)
        assert output.shape == (2, 5, 64)


class TestGraniteHybridMoE:
    """Tests for GraniteHybridMoE."""

    def test_creation(self):
        """Test MoE creation."""
        config = GraniteHybridConfig.tiny_moe()
        moe = GraniteHybridMoE(config)

        assert moe.hidden_size == 64
        assert moe.num_experts == 4
        assert moe.num_experts_per_tok == 2
        assert moe.has_shared_expert is True

    def test_creation_without_shared_expert(self):
        """Test MoE creation without shared expert."""
        config = GraniteHybridConfig.tiny_moe()
        config.shared_intermediate_size = 0
        moe = GraniteHybridMoE(config)

        assert moe.has_shared_expert is False

    def test_forward_pass(self):
        """Test forward pass."""
        config = GraniteHybridConfig.tiny_moe()
        moe = GraniteHybridMoE(config)

        x = mx.random.normal((2, 10, 64))
        output = moe(x)

        assert output.shape == (2, 10, 64)

    def test_forward_without_shared_expert(self):
        """Test forward without shared expert."""
        config = GraniteHybridConfig.tiny_moe()
        config.shared_intermediate_size = 0
        moe = GraniteHybridMoE(config)

        x = mx.random.normal((2, 10, 64))
        output = moe(x)

        assert output.shape == (2, 10, 64)


class TestGraniteHybridBlock:
    """Tests for GraniteHybridBlock."""

    def test_creation_attention(self):
        """Test block creation with attention."""
        config = GraniteHybridConfig.tiny()
        block = GraniteHybridBlock(config, layer_idx=0, layer_type="attention")

        assert block.layer_type == "attention"
        assert block.residual_multiplier == 1.0

    def test_creation_mamba(self):
        """Test block creation with mamba."""
        config = GraniteHybridConfig.tiny()
        block = GraniteHybridBlock(config, layer_idx=0, layer_type="mamba")

        assert block.layer_type == "mamba"

    def test_block_type_property_attention(self):
        """Test block_type property for attention."""
        from chuk_lazarus.models_v2.core.enums import BlockType

        config = GraniteHybridConfig.tiny()
        block = GraniteHybridBlock(config, layer_type="attention")

        assert block.block_type == BlockType.TRANSFORMER

    def test_block_type_property_mamba(self):
        """Test block_type property for mamba."""
        from chuk_lazarus.models_v2.core.enums import BlockType

        config = GraniteHybridConfig.tiny()
        block = GraniteHybridBlock(config, layer_type="mamba")

        assert block.block_type == BlockType.MAMBA

    def test_hidden_size_property(self):
        """Test hidden_size property."""
        config = GraniteHybridConfig.tiny()
        block = GraniteHybridBlock(config)

        assert block.hidden_size == 64

    def test_forward_attention(self):
        """Test forward pass for attention block."""
        config = GraniteHybridConfig.tiny()
        block = GraniteHybridBlock(config, layer_type="attention")

        x = mx.random.normal((2, 10, 64))
        mask = nn.MultiHeadAttention.create_additive_causal_mask(10)
        output = block(x, mask=mask)

        assert output.hidden_states.shape == (2, 10, 64)

    def test_forward_mamba(self):
        """Test forward pass for mamba block."""
        config = GraniteHybridConfig.tiny()
        block = GraniteHybridBlock(config, layer_type="mamba")

        x = mx.random.normal((2, 10, 64))
        output = block(x)

        assert output.hidden_states.shape == (2, 10, 64)

    def test_forward_with_moe(self):
        """Test forward pass with MoE."""
        config = GraniteHybridConfig.tiny_moe()
        block = GraniteHybridBlock(config, layer_type="attention")

        x = mx.random.normal((2, 10, 64))
        output = block(x)

        assert output.hidden_states.shape == (2, 10, 64)


class TestGraniteHybridModel:
    """Tests for GraniteHybridModel backbone."""

    def test_creation(self):
        """Test model creation."""
        config = GraniteHybridConfig.tiny()
        model = GraniteHybridModel(config)

        assert model.hidden_size == 64
        assert model.num_layers == 4
        assert model.vocab_size == 1000
        assert model.embedding_multiplier == 1.0

    def test_forward_pass(self):
        """Test forward pass."""
        config = GraniteHybridConfig.tiny()
        model = GraniteHybridModel(config)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        output = model(input_ids)

        assert output.last_hidden_state.shape == (1, 5, 64)
        assert output.cache is not None
        assert len(output.cache) == 4

    def test_forward_with_output_hidden_states(self):
        """Test forward with output_hidden_states=True."""
        config = GraniteHybridConfig.tiny()
        model = GraniteHybridModel(config)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        output = model(input_ids, output_hidden_states=True)

        assert output.hidden_states is not None
        assert len(output.hidden_states) == 5

    def test_forward_with_attention_mask(self):
        """Test forward with attention mask."""
        config = GraniteHybridConfig.tiny()
        model = GraniteHybridModel(config)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        mask = nn.MultiHeadAttention.create_additive_causal_mask(5)
        output = model(input_ids, attention_mask=mask)

        assert output.last_hidden_state.shape == (1, 5, 64)

    def test_forward_with_cache(self):
        """Test forward with cache."""
        config = GraniteHybridConfig.tiny()
        model = GraniteHybridModel(config)

        # First pass
        input_ids = mx.array([[1, 2, 3, 4, 5]])
        out1 = model(input_ids)

        # Second pass
        next_token = mx.array([[6]])
        out2 = model(next_token, cache=out1.cache)

        assert out2.last_hidden_state.shape == (1, 1, 64)

    def test_get_input_embeddings(self):
        """Test get_input_embeddings method."""
        config = GraniteHybridConfig.tiny()
        model = GraniteHybridModel(config)

        embeddings = model.get_input_embeddings()
        assert embeddings is model.embed_tokens

    def test_set_input_embeddings(self):
        """Test set_input_embeddings method."""
        config = GraniteHybridConfig.tiny()
        model = GraniteHybridModel(config)

        new_embed = nn.Embedding(500, 64)
        model.set_input_embeddings(new_embed)

        assert model.embed_tokens is new_embed


class TestGraniteHybridForCausalLM:
    """Tests for GraniteHybridForCausalLM."""

    def test_creation(self):
        """Test model creation."""
        config = GraniteHybridConfig.tiny()
        model = GraniteHybridForCausalLM(config)

        assert model.config is config
        assert model.logits_scaling == 1.0

    def test_creation_tied_embeddings(self):
        """Test model with tied embeddings."""
        config = GraniteHybridConfig.tiny()
        config.tie_word_embeddings = True
        model = GraniteHybridForCausalLM(config)

        assert model.lm_head is not None

    def test_creation_untied_embeddings(self):
        """Test model without tied embeddings."""
        config = GraniteHybridConfig.tiny()
        config.tie_word_embeddings = False
        model = GraniteHybridForCausalLM(config)

        assert model.lm_head is not None

    def test_backbone_property(self):
        """Test backbone property."""
        config = GraniteHybridConfig.tiny()
        model = GraniteHybridForCausalLM(config)

        assert model.backbone is model.model

    def test_forward_pass(self):
        """Test forward pass."""
        config = GraniteHybridConfig.tiny()
        model = GraniteHybridForCausalLM(config)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        output = model(input_ids)

        assert output.logits.shape == (1, 5, 1000)
        assert output.loss is None
        assert output.cache is not None

    def test_forward_with_labels(self):
        """Test forward with labels."""
        config = GraniteHybridConfig.tiny()
        model = GraniteHybridForCausalLM(config)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        labels = mx.array([[2, 3, 4, 5, 6]])
        output = model(input_ids, labels=labels)

        assert output.loss is not None

    def test_forward_with_logits_scaling(self):
        """Test logits scaling."""
        config = GraniteHybridConfig.tiny()
        config.logits_scaling = 2.0
        model = GraniteHybridForCausalLM(config)

        assert model.logits_scaling == 2.0

        input_ids = mx.array([[1, 2, 3]])
        output = model(input_ids)
        assert output.logits is not None

    def test_generate_basic(self):
        """Test basic generation."""
        config = GraniteHybridConfig.tiny()
        model = GraniteHybridForCausalLM(config)

        input_ids = mx.array([[1, 2, 3]])
        generated = model.generate(input_ids, max_new_tokens=5)

        assert generated.shape[0] == 1
        assert generated.shape[1] >= 3

    def test_generate_with_temperature(self):
        """Test generation with temperature."""
        config = GraniteHybridConfig.tiny()
        model = GraniteHybridForCausalLM(config)

        input_ids = mx.array([[1, 2, 3]])
        generated = model.generate(input_ids, max_new_tokens=3, temperature=0.5)

        assert generated.shape[1] >= 3

    def test_generate_with_top_k(self):
        """Test generation with top_k."""
        config = GraniteHybridConfig.tiny()
        model = GraniteHybridForCausalLM(config)

        input_ids = mx.array([[1, 2, 3]])
        generated = model.generate(input_ids, max_new_tokens=3, top_k=10)

        assert generated.shape[1] >= 3

    def test_generate_with_stop_tokens(self):
        """Test generation with stop tokens."""
        config = GraniteHybridConfig.tiny()
        model = GraniteHybridForCausalLM(config)

        input_ids = mx.array([[1, 2, 3]])
        generated = model.generate(input_ids, max_new_tokens=10, stop_tokens=[999])

        assert generated.shape[1] >= 3

    def test_from_config(self):
        """Test from_config class method."""
        config = GraniteHybridConfig.tiny()
        model = GraniteHybridForCausalLM.from_config(config)

        assert isinstance(model, GraniteHybridForCausalLM)

    def test_aliases(self):
        """Test model aliases."""
        config = GraniteHybridConfig.tiny()

        model1 = GraniteHybrid(config)
        assert isinstance(model1, GraniteHybridForCausalLM)

        model2 = Granite4(config)
        assert isinstance(model2, GraniteHybridForCausalLM)


class TestGraniteHybridGradients:
    """Tests for gradient flow."""

    def test_forward_backward(self):
        """Test forward-backward pass."""
        config = GraniteHybridConfig.tiny()
        model = GraniteHybridForCausalLM(config)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        labels = mx.array([[2, 3, 4, 5, 6]])

        def loss_fn(model, input_ids, labels):
            output = model(input_ids, labels=labels)
            return output.loss

        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad_fn(model, input_ids, labels)

        assert loss.item() > 0
        assert any(g is not None for g in grads.values())

    def test_moe_loss_computation(self):
        """Test that MoE model computes loss (gradient tests skipped due to MLX ArgsSort limitation)."""
        # Note: Full gradient tests are skipped because MLX's ArgSort operation
        # (used in MoE top-k routing) does not support VJP.
        # This is a known limitation: "Not implemented for ArgSort"
        config = GraniteHybridConfig.tiny_moe()
        model = GraniteHybridForCausalLM(config)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        labels = mx.array([[2, 3, 4, 5, 6]])

        output = model(input_ids, labels=labels)
        assert output.loss is not None
        assert output.loss.item() > 0
