"""
Tests for Llama 4 model.
"""

import mlx.core as mx
import mlx.nn as nn

from chuk_lazarus.models_v2.families.llama4.config import Llama4TextConfig
from chuk_lazarus.models_v2.families.llama4.model import (
    Llama4,
    Llama4Block,
    Llama4ForCausalLM,
    Llama4Model,
)


class TestLlama4Block:
    """Tests for Llama4Block."""

    def test_creation(self):
        """Test block creation."""
        config = Llama4TextConfig.tiny()
        block = Llama4Block(config, layer_idx=0)

        assert block.hidden_size == 64
        assert block.layer_idx == 0

    def test_block_type(self):
        """Test block_type property."""
        from chuk_lazarus.models_v2.core.enums import BlockType

        config = Llama4TextConfig.tiny()
        block = Llama4Block(config)

        assert block.block_type == BlockType.TRANSFORMER

    def test_hidden_size_property(self):
        """Test hidden_size property."""
        config = Llama4TextConfig.tiny()
        block = Llama4Block(config)

        assert block.hidden_size == 64

    def test_forward_pass(self):
        """Test forward pass."""
        config = Llama4TextConfig.tiny()
        block = Llama4Block(config, layer_idx=1)

        x = mx.random.normal((2, 10, 64))
        output = block(x)

        assert output.hidden_states.shape == (2, 10, 64)
        assert output.cache is not None

    def test_forward_with_mask(self):
        """Test forward with mask."""
        config = Llama4TextConfig.tiny()
        block = Llama4Block(config, layer_idx=1)

        x = mx.random.normal((2, 10, 64))
        mask = nn.MultiHeadAttention.create_additive_causal_mask(10)
        output = block(x, mask=mask)

        assert output.hidden_states.shape == (2, 10, 64)

    def test_forward_with_cache(self):
        """Test forward with cache."""
        config = Llama4TextConfig.tiny()
        block = Llama4Block(config, layer_idx=1)

        # First pass
        x1 = mx.random.normal((2, 10, 64))
        out1 = block(x1)

        # Second pass
        x2 = mx.random.normal((2, 1, 64))
        out2 = block(x2, cache=out1.cache)

        assert out2.hidden_states.shape == (2, 1, 64)


class TestLlama4Model:
    """Tests for Llama4Model backbone."""

    def test_creation(self):
        """Test model creation."""
        config = Llama4TextConfig.tiny()
        model = Llama4Model(config)

        assert model.hidden_size == 64
        assert model.num_layers == 4
        assert model.vocab_size == 1000

    def test_forward_pass(self):
        """Test forward pass."""
        config = Llama4TextConfig.tiny()
        model = Llama4Model(config)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        output = model(input_ids)

        assert output.last_hidden_state.shape == (1, 5, 64)
        assert output.cache is not None
        assert len(output.cache) == 4
        assert output.hidden_states is None

    def test_forward_with_output_hidden_states(self):
        """Test forward with output_hidden_states=True."""
        config = Llama4TextConfig.tiny()
        model = Llama4Model(config)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        output = model(input_ids, output_hidden_states=True)

        assert output.hidden_states is not None
        assert len(output.hidden_states) == 5  # embeddings + 4 layers

    def test_forward_with_attention_mask(self):
        """Test forward with attention mask."""
        config = Llama4TextConfig.tiny()
        model = Llama4Model(config)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        mask = nn.MultiHeadAttention.create_additive_causal_mask(5)
        output = model(input_ids, attention_mask=mask)

        assert output.last_hidden_state.shape == (1, 5, 64)

    def test_forward_with_cache(self):
        """Test forward with cache."""
        config = Llama4TextConfig.tiny()
        model = Llama4Model(config)

        # First pass
        input_ids = mx.array([[1, 2, 3, 4, 5]])
        out1 = model(input_ids)

        # Second pass
        next_token = mx.array([[6]])
        out2 = model(next_token, cache=out1.cache)

        assert out2.last_hidden_state.shape == (1, 1, 64)

    def test_get_input_embeddings(self):
        """Test get_input_embeddings method."""
        config = Llama4TextConfig.tiny()
        model = Llama4Model(config)

        embeddings = model.get_input_embeddings()
        assert embeddings is model.embed_tokens

    def test_set_input_embeddings(self):
        """Test set_input_embeddings method."""
        config = Llama4TextConfig.tiny()
        model = Llama4Model(config)

        new_embed = nn.Embedding(500, 64)
        model.set_input_embeddings(new_embed)

        assert model.embed_tokens is new_embed


class TestLlama4ForCausalLM:
    """Tests for Llama4ForCausalLM."""

    def test_creation(self):
        """Test model creation."""
        config = Llama4TextConfig.tiny()
        model = Llama4ForCausalLM(config)

        assert model.config is config

    def test_creation_tied_embeddings(self):
        """Test model with tied embeddings."""
        config = Llama4TextConfig.tiny()
        config.tie_word_embeddings = True
        model = Llama4ForCausalLM(config)

        assert model.lm_head is not None

    def test_creation_untied_embeddings(self):
        """Test model without tied embeddings."""
        config = Llama4TextConfig.tiny()
        config.tie_word_embeddings = False
        model = Llama4ForCausalLM(config)

        assert model.lm_head is not None

    def test_backbone_property(self):
        """Test backbone property."""
        config = Llama4TextConfig.tiny()
        model = Llama4ForCausalLM(config)

        assert model.backbone is model.model

    def test_forward_pass(self):
        """Test forward pass."""
        config = Llama4TextConfig.tiny()
        model = Llama4ForCausalLM(config)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        output = model(input_ids)

        assert output.logits.shape == (1, 5, 1000)
        assert output.loss is None
        assert output.cache is not None

    def test_forward_with_labels(self):
        """Test forward with labels."""
        config = Llama4TextConfig.tiny()
        model = Llama4ForCausalLM(config)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        labels = mx.array([[2, 3, 4, 5, 6]])
        output = model(input_ids, labels=labels)

        assert output.loss is not None
        assert output.loss.item() > 0

    def test_forward_with_output_hidden_states(self):
        """Test forward with output_hidden_states."""
        config = Llama4TextConfig.tiny()
        model = Llama4ForCausalLM(config)

        input_ids = mx.array([[1, 2, 3]])
        output = model(input_ids, output_hidden_states=True)

        assert output.hidden_states is not None

    def test_generate_basic(self):
        """Test basic generation."""
        config = Llama4TextConfig.tiny()
        model = Llama4ForCausalLM(config)

        input_ids = mx.array([[1, 2, 3]])
        generated = model.generate(input_ids, max_new_tokens=5)

        assert generated.shape[0] == 1
        assert generated.shape[1] >= 3

    def test_generate_with_temperature(self):
        """Test generation with temperature."""
        config = Llama4TextConfig.tiny()
        model = Llama4ForCausalLM(config)

        input_ids = mx.array([[1, 2, 3]])
        generated = model.generate(input_ids, max_new_tokens=3, temperature=0.5)

        assert generated.shape[1] >= 3

    def test_generate_with_top_k(self):
        """Test generation with top_k."""
        config = Llama4TextConfig.tiny()
        model = Llama4ForCausalLM(config)

        input_ids = mx.array([[1, 2, 3]])
        generated = model.generate(input_ids, max_new_tokens=3, top_k=10)

        assert generated.shape[1] >= 3

    def test_generate_with_repetition_penalty(self):
        """Test generation with repetition penalty."""
        config = Llama4TextConfig.tiny()
        model = Llama4ForCausalLM(config)

        input_ids = mx.array([[1, 2, 3]])
        generated = model.generate(input_ids, max_new_tokens=3, repetition_penalty=1.2)

        assert generated.shape[1] >= 3

    def test_generate_with_stop_tokens(self):
        """Test generation with stop tokens."""
        config = Llama4TextConfig.tiny()
        model = Llama4ForCausalLM(config)

        input_ids = mx.array([[1, 2, 3]])
        generated = model.generate(input_ids, max_new_tokens=10, stop_tokens=[999])

        assert generated.shape[1] >= 3

    def test_from_config(self):
        """Test from_config class method."""
        config = Llama4TextConfig.tiny()
        model = Llama4ForCausalLM.from_config(config)

        assert isinstance(model, Llama4ForCausalLM)
        assert model.config is config

    def test_alias_llama4(self):
        """Test Llama4 alias."""
        config = Llama4TextConfig.tiny()
        model = Llama4(config)

        assert isinstance(model, Llama4ForCausalLM)


class TestLlama4Gradients:
    """Tests for gradient flow."""

    def test_loss_computation(self):
        """Test that loss can be computed (gradient flow tests skipped due to MoE gather_mm limitation)."""
        # Note: Full gradient tests are skipped because MLX's gather_mm operation
        # (used in the MoE layer) does not support VJP with respect to indices.
        # This is a known limitation: "Cannot calculate VJP with respect to indices"
        config = Llama4TextConfig.tiny()
        model = Llama4ForCausalLM(config)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        labels = mx.array([[2, 3, 4, 5, 6]])

        output = model(input_ids, labels=labels)
        assert output.loss is not None
        assert output.loss.item() > 0


class TestLlama4BatchHandling:
    """Tests for batch handling."""

    def test_different_batch_sizes(self):
        """Test different batch sizes."""
        config = Llama4TextConfig.tiny()
        model = Llama4ForCausalLM(config)

        for batch_size in [1, 2, 4]:
            input_ids = mx.random.randint(0, config.vocab_size, (batch_size, 5))
            output = model(input_ids)
            assert output.logits.shape == (batch_size, 5, config.vocab_size)

    def test_different_sequence_lengths(self):
        """Test different sequence lengths."""
        config = Llama4TextConfig.tiny()
        model = Llama4ForCausalLM(config)

        for seq_len in [1, 5, 10]:
            input_ids = mx.random.randint(0, config.vocab_size, (2, seq_len))
            output = model(input_ids)
            assert output.logits.shape == (2, seq_len, config.vocab_size)
