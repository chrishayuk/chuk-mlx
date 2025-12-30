"""
Tests for Qwen3 model family.

Tests Qwen3Config, Qwen3Model, and Qwen3ForCausalLM.
"""

import mlx.core as mx
import mlx.nn as nn

from chuk_lazarus.models_v2.families.qwen3 import (
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3Model,
)


class TestQwen3Config:
    """Tests for Qwen3Config."""

    def test_defaults(self):
        """Test default configuration values."""
        config = Qwen3Config()

        assert config.model_type == "qwen3"
        assert config.hidden_act == "silu"
        assert config.attention_bias is False
        assert config.mlp_bias is False

    def test_custom_config(self):
        """Test custom configuration."""
        config = Qwen3Config(
            vocab_size=50000,
            hidden_size=2048,
            num_hidden_layers=16,
        )

        assert config.vocab_size == 50000
        assert config.hidden_size == 2048
        assert config.num_hidden_layers == 16

    def test_gqa_config(self):
        """Test grouped-query attention configuration."""
        config = Qwen3Config(
            num_attention_heads=32,
            num_key_value_heads=8,
        )

        assert config.num_attention_heads == 32
        assert config.num_key_value_heads == 8

    def test_from_hf_config(self):
        """Test from_hf_config class method."""
        hf_config = {
            "model_type": "qwen3",
            "vocab_size": 151936,
            "hidden_size": 2048,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "intermediate_size": 5504,
            "attention_bias": False,
        }

        config = Qwen3Config.from_hf_config(hf_config)

        assert config.vocab_size == 151936
        assert config.hidden_size == 2048
        assert config.num_hidden_layers == 28
        assert config.attention_bias is False


class TestQwen3Model:
    """Tests for Qwen3Model."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        config = Qwen3Config(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
        )
        model = Qwen3Model(config)

        x = mx.random.randint(0, config.vocab_size, (1, 5))
        output = model(x)

        # Qwen3Model returns BackboneOutput with last_hidden_state
        hidden_states = output.last_hidden_state if hasattr(output, "last_hidden_state") else output
        assert hidden_states.shape == (1, 5, config.hidden_size)

    def test_batch_forward(self):
        """Test forward pass with batch."""
        config = Qwen3Config(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
        )
        model = Qwen3Model(config)

        x = mx.random.randint(0, config.vocab_size, (2, 10))
        output = model(x)

        hidden_states = output.last_hidden_state if hasattr(output, "last_hidden_state") else output
        assert hidden_states.shape == (2, 10, config.hidden_size)

    def test_different_sequence_lengths(self):
        """Test forward with different sequence lengths."""
        config = Qwen3Config(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
        )
        model = Qwen3Model(config)

        for seq_len in [1, 5, 10, 20]:
            x = mx.random.randint(0, config.vocab_size, (1, seq_len))
            output = model(x)
            hidden_states = (
                output.last_hidden_state if hasattr(output, "last_hidden_state") else output
            )
            assert hidden_states.shape == (1, seq_len, config.hidden_size)

    def test_layer_count(self):
        """Test model has correct number of layers."""
        config = Qwen3Config(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=6,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
        )
        model = Qwen3Model(config)

        assert len(model.layers) == config.num_hidden_layers


class TestQwen3ForCausalLM:
    """Tests for Qwen3ForCausalLM."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        config = Qwen3Config(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
        )
        model = Qwen3ForCausalLM(config)

        x = mx.random.randint(0, config.vocab_size, (1, 5))
        output = model(x)

        # Output should have logits
        if hasattr(output, "logits"):
            assert output.logits.shape == (1, 5, config.vocab_size)
        else:
            assert output.shape == (1, 5, config.vocab_size)

    def test_tied_embeddings(self):
        """Test model with tied embeddings."""
        config = Qwen3Config(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
            tie_word_embeddings=True,
        )
        model = Qwen3ForCausalLM(config)

        x = mx.random.randint(0, config.vocab_size, (1, 5))
        output = model(x)
        logits = output.logits if hasattr(output, "logits") else output

        assert logits.shape == (1, 5, config.vocab_size)

    def test_untied_embeddings(self):
        """Test model with untied embeddings."""
        config = Qwen3Config(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
            tie_word_embeddings=False,
        )
        model = Qwen3ForCausalLM(config)

        # Should have separate lm_head
        assert hasattr(model, "lm_head")

    def test_generate(self):
        """Test generation."""
        config = Qwen3Config(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
        )
        model = Qwen3ForCausalLM(config)

        input_ids = mx.array([[1, 2, 3]])
        output = model.generate(input_ids, max_new_tokens=5)

        # Should generate more tokens
        assert output.shape[1] >= input_ids.shape[1]

    def test_from_config(self):
        """Test from_config class method."""
        config = Qwen3Config(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
        )
        model = Qwen3ForCausalLM.from_config(config)

        assert isinstance(model, Qwen3ForCausalLM)


class TestQwen3Gradients:
    """Tests for gradient flow through Qwen3."""

    def test_forward_backward(self):
        """Test gradients flow correctly."""
        config = Qwen3Config(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
        )
        model = Qwen3ForCausalLM(config)

        x = mx.random.randint(0, config.vocab_size, (1, 5))

        def loss_fn(model, x):
            out = model(x)
            logits = out.logits if hasattr(out, "logits") else out
            return mx.mean(logits**2)

        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad_fn(model, x)

        assert loss.item() > 0


class TestQwen3BatchHandling:
    """Tests for batch dimension handling."""

    def test_different_batch_sizes(self):
        """Test model handles different batch sizes."""
        config = Qwen3Config(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
        )
        model = Qwen3ForCausalLM(config)

        for batch_size in [1, 2, 4]:
            x = mx.random.randint(0, config.vocab_size, (batch_size, 5))
            output = model(x)
            logits = output.logits if hasattr(output, "logits") else output
            assert logits.shape[0] == batch_size

    def test_different_sequence_lengths(self):
        """Test model handles different sequence lengths."""
        config = Qwen3Config(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
        )
        model = Qwen3ForCausalLM(config)

        for seq_len in [1, 5, 10, 20]:
            x = mx.random.randint(0, config.vocab_size, (2, seq_len))
            output = model(x)
            logits = output.logits if hasattr(output, "logits") else output
            assert logits.shape == (2, seq_len, config.vocab_size)
