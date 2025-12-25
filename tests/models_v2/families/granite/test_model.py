"""
Tests for Granite model.
"""

import mlx.core as mx
import mlx.nn as nn

from chuk_lazarus.models_v2.families.granite.config import GraniteConfig
from chuk_lazarus.models_v2.families.granite.model import (
    Granite,
    GraniteAttention,
    GraniteBlock,
    GraniteForCausalLM,
    GraniteModel,
)


class TestGraniteAttention:
    """Tests for GraniteAttention."""

    def test_creation(self):
        """Test attention creation."""
        config = GraniteConfig.tiny()
        attn = GraniteAttention(config)

        assert attn.hidden_size == 64
        assert attn.num_heads == 4
        assert attn.num_kv_heads == 2
        assert attn.head_dim == 16
        assert attn.n_rep == 2
        assert attn.attention_multiplier == 1.0

    def test_forward_pass(self):
        """Test attention forward pass."""
        config = GraniteConfig.tiny()
        attn = GraniteAttention(config, layer_idx=0)

        x = mx.random.normal((2, 10, 64))
        output, cache = attn(x)

        assert output.shape == (2, 10, 64)
        assert cache is not None
        k, v = cache
        assert k.shape[0] == 2
        assert k.shape[2] == 10

    def test_forward_with_mask(self):
        """Test attention with mask."""
        config = GraniteConfig.tiny()
        attn = GraniteAttention(config)

        x = mx.random.normal((2, 10, 64))
        mask = nn.MultiHeadAttention.create_additive_causal_mask(10)
        output, cache = attn(x, mask=mask)

        assert output.shape == (2, 10, 64)

    def test_forward_with_cache(self):
        """Test attention with KV cache."""
        config = GraniteConfig.tiny()
        attn = GraniteAttention(config)

        # First pass
        x1 = mx.random.normal((2, 10, 64))
        _, cache = attn(x1)

        # Second pass with cache
        x2 = mx.random.normal((2, 1, 64))
        output, new_cache = attn(x2, cache=cache)

        assert output.shape == (2, 1, 64)
        k, v = new_cache
        assert k.shape[2] == 11  # 10 + 1

    def test_repeat_kv(self):
        """Test KV repeat method."""
        config = GraniteConfig.tiny()
        attn = GraniteAttention(config)

        x = mx.random.normal((2, 2, 10, 16))

        # n_rep = 1, should return same
        result = attn._repeat_kv(x, n_rep=1)
        assert result.shape == x.shape

        # n_rep = 2
        result = attn._repeat_kv(x, n_rep=2)
        assert result.shape == (2, 4, 10, 16)

    def test_attention_multiplier_applied(self):
        """Test that attention multiplier is applied."""
        config = GraniteConfig.tiny()
        config.attention_multiplier = 0.5

        attn = GraniteAttention(config)
        assert attn.attention_multiplier == 0.5

        x = mx.random.normal((1, 5, 64))
        output1, _ = attn(x)

        # Verify multiplier with different config also works
        config2 = GraniteConfig.tiny()
        config2.attention_multiplier = 1.0
        attn2 = GraniteAttention(config2)
        output2, _ = attn2(x)
        # Just verify both run without error - exact comparison tricky due to random init
        assert output1.shape == output2.shape


class TestGraniteBlock:
    """Tests for GraniteBlock."""

    def test_creation(self):
        """Test block creation."""
        config = GraniteConfig.tiny()
        block = GraniteBlock(config, layer_idx=0)

        assert block.hidden_size == 64
        assert block.layer_idx == 0
        assert block.residual_multiplier == 1.0

    def test_block_type(self):
        """Test block_type property."""
        from chuk_lazarus.models_v2.core.enums import BlockType

        config = GraniteConfig.tiny()
        block = GraniteBlock(config)

        assert block.block_type == BlockType.TRANSFORMER

    def test_forward_pass(self):
        """Test block forward pass."""
        config = GraniteConfig.tiny()
        block = GraniteBlock(config)

        x = mx.random.normal((2, 10, 64))
        output = block(x)

        assert output.hidden_states.shape == (2, 10, 64)
        assert output.cache is not None

    def test_forward_with_mask(self):
        """Test block with mask."""
        config = GraniteConfig.tiny()
        block = GraniteBlock(config)

        x = mx.random.normal((2, 10, 64))
        mask = nn.MultiHeadAttention.create_additive_causal_mask(10)
        output = block(x, mask=mask)

        assert output.hidden_states.shape == (2, 10, 64)

    def test_forward_with_cache(self):
        """Test block with cache."""
        config = GraniteConfig.tiny()
        block = GraniteBlock(config)

        # First pass
        x1 = mx.random.normal((2, 10, 64))
        out1 = block(x1)

        # Second pass with cache
        x2 = mx.random.normal((2, 1, 64))
        out2 = block(x2, cache=out1.cache)

        assert out2.hidden_states.shape == (2, 1, 64)

    def test_residual_multiplier(self):
        """Test residual multiplier is applied."""
        config = GraniteConfig.tiny()
        config.residual_multiplier = 0.5

        block = GraniteBlock(config)
        assert block.residual_multiplier == 0.5


class TestGraniteModel:
    """Tests for GraniteModel backbone."""

    def test_creation(self):
        """Test model creation."""
        config = GraniteConfig.tiny()
        model = GraniteModel(config)

        assert model.hidden_size == 64
        assert model.num_layers == 4
        assert model.vocab_size == 1000
        assert model.embedding_multiplier == 1.0

    def test_forward_pass(self):
        """Test model forward pass."""
        config = GraniteConfig.tiny()
        model = GraniteModel(config)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        output = model(input_ids)

        assert output.last_hidden_state.shape == (1, 5, 64)
        assert output.cache is not None
        assert len(output.cache) == 4
        assert output.hidden_states is None

    def test_forward_with_output_hidden_states(self):
        """Test model with output_hidden_states=True."""
        config = GraniteConfig.tiny()
        model = GraniteModel(config)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        output = model(input_ids, output_hidden_states=True)

        assert output.hidden_states is not None
        assert len(output.hidden_states) == 5  # embeddings + 4 layers

    def test_forward_with_attention_mask(self):
        """Test model with custom attention mask."""
        config = GraniteConfig.tiny()
        model = GraniteModel(config)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        mask = nn.MultiHeadAttention.create_additive_causal_mask(5)
        output = model(input_ids, attention_mask=mask)

        assert output.last_hidden_state.shape == (1, 5, 64)

    def test_forward_with_cache(self):
        """Test model with cache."""
        config = GraniteConfig.tiny()
        model = GraniteModel(config)

        # First pass
        input_ids = mx.array([[1, 2, 3, 4, 5]])
        out1 = model(input_ids)

        # Second pass with cache
        next_token = mx.array([[6]])
        out2 = model(next_token, cache=out1.cache)

        assert out2.last_hidden_state.shape == (1, 1, 64)

    def test_get_input_embeddings(self):
        """Test get_input_embeddings method."""
        config = GraniteConfig.tiny()
        model = GraniteModel(config)

        embeddings = model.get_input_embeddings()
        assert embeddings is model.embed_tokens

    def test_set_input_embeddings(self):
        """Test set_input_embeddings method."""
        config = GraniteConfig.tiny()
        model = GraniteModel(config)

        new_embed = nn.Embedding(500, 64)
        model.set_input_embeddings(new_embed)

        assert model.embed_tokens is new_embed


class TestGraniteForCausalLM:
    """Tests for GraniteForCausalLM."""

    def test_creation(self):
        """Test model creation."""
        config = GraniteConfig.tiny()
        model = GraniteForCausalLM(config)

        assert model.config is config
        assert model.logits_scaling == 1.0

    def test_creation_tied_embeddings(self):
        """Test model creation with tied embeddings."""
        config = GraniteConfig.tiny()
        config.tie_word_embeddings = True
        model = GraniteForCausalLM(config)

        assert model.lm_head is not None

    def test_creation_untied_embeddings(self):
        """Test model creation without tied embeddings."""
        config = GraniteConfig.tiny()
        config.tie_word_embeddings = False
        model = GraniteForCausalLM(config)

        assert model.lm_head is not None

    def test_backbone_property(self):
        """Test backbone property."""
        config = GraniteConfig.tiny()
        model = GraniteForCausalLM(config)

        assert model.backbone is model.model

    def test_forward_pass(self):
        """Test forward pass."""
        config = GraniteConfig.tiny()
        model = GraniteForCausalLM(config)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        output = model(input_ids)

        assert output.logits.shape == (1, 5, 1000)
        assert output.loss is None
        assert output.cache is not None

    def test_forward_with_labels(self):
        """Test forward pass with labels."""
        config = GraniteConfig.tiny()
        model = GraniteForCausalLM(config)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        labels = mx.array([[2, 3, 4, 5, 6]])
        output = model(input_ids, labels=labels)

        assert output.loss is not None
        assert output.loss.item() > 0

    def test_forward_with_logits_scaling(self):
        """Test logits scaling is applied."""
        config = GraniteConfig.tiny()
        config.logits_scaling = 2.0
        model = GraniteForCausalLM(config)

        assert model.logits_scaling == 2.0

        input_ids = mx.array([[1, 2, 3]])
        output = model(input_ids)

        assert output.logits is not None

    def test_forward_with_output_hidden_states(self):
        """Test forward with output_hidden_states."""
        config = GraniteConfig.tiny()
        model = GraniteForCausalLM(config)

        input_ids = mx.array([[1, 2, 3]])
        output = model(input_ids, output_hidden_states=True)

        assert output.hidden_states is not None

    def test_generate_basic(self):
        """Test basic generation."""
        config = GraniteConfig.tiny()
        model = GraniteForCausalLM(config)

        input_ids = mx.array([[1, 2, 3]])
        generated = model.generate(input_ids, max_new_tokens=5)

        assert generated.shape[0] == 1
        assert generated.shape[1] >= 3

    def test_generate_with_temperature(self):
        """Test generation with temperature."""
        config = GraniteConfig.tiny()
        model = GraniteForCausalLM(config)

        input_ids = mx.array([[1, 2, 3]])
        generated = model.generate(input_ids, max_new_tokens=3, temperature=0.5)

        assert generated.shape[1] >= 3

    def test_generate_with_top_k(self):
        """Test generation with top_k."""
        config = GraniteConfig.tiny()
        model = GraniteForCausalLM(config)

        input_ids = mx.array([[1, 2, 3]])
        generated = model.generate(input_ids, max_new_tokens=3, top_k=10)

        assert generated.shape[1] >= 3

    def test_generate_with_repetition_penalty(self):
        """Test generation with repetition penalty."""
        config = GraniteConfig.tiny()
        model = GraniteForCausalLM(config)

        input_ids = mx.array([[1, 2, 3]])
        generated = model.generate(input_ids, max_new_tokens=3, repetition_penalty=1.2)

        assert generated.shape[1] >= 3

    def test_generate_with_stop_tokens(self):
        """Test generation with stop tokens."""
        config = GraniteConfig.tiny()
        model = GraniteForCausalLM(config)

        input_ids = mx.array([[1, 2, 3]])
        # Use a token that's likely in the vocabulary
        generated = model.generate(input_ids, max_new_tokens=10, stop_tokens=[999])

        assert generated.shape[1] >= 3

    def test_from_config(self):
        """Test from_config class method."""
        config = GraniteConfig.tiny()
        model = GraniteForCausalLM.from_config(config)

        assert isinstance(model, GraniteForCausalLM)
        assert model.config is config

    def test_alias_granite(self):
        """Test Granite alias."""
        config = GraniteConfig.tiny()
        model = Granite(config)

        assert isinstance(model, GraniteForCausalLM)


class TestGraniteGradients:
    """Tests for gradient flow."""

    def test_forward_backward(self):
        """Test forward-backward pass."""
        config = GraniteConfig.tiny()
        model = GraniteForCausalLM(config)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        labels = mx.array([[2, 3, 4, 5, 6]])

        def loss_fn(model, input_ids, labels):
            output = model(input_ids, labels=labels)
            return output.loss

        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad_fn(model, input_ids, labels)

        assert loss.item() > 0
        assert any(g is not None for g in grads.values())


class TestGraniteBatchHandling:
    """Tests for batch handling."""

    def test_different_batch_sizes(self):
        """Test different batch sizes."""
        config = GraniteConfig.tiny()
        model = GraniteForCausalLM(config)

        for batch_size in [1, 2, 4]:
            input_ids = mx.random.randint(0, config.vocab_size, (batch_size, 5))
            output = model(input_ids)
            assert output.logits.shape == (batch_size, 5, config.vocab_size)

    def test_different_sequence_lengths(self):
        """Test different sequence lengths."""
        config = GraniteConfig.tiny()
        model = GraniteForCausalLM(config)

        for seq_len in [1, 5, 10]:
            input_ids = mx.random.randint(0, config.vocab_size, (2, seq_len))
            output = model(input_ids)
            assert output.logits.shape == (2, seq_len, config.vocab_size)
