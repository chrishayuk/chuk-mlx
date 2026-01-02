"""
Tests for Jamba model family.

Tests JambaConfig, JambaModel, and JambaForCausalLM.
"""

import mlx.core as mx
import mlx.nn as nn

from chuk_lazarus.models_v2.families.jamba import (
    JambaConfig,
    JambaForCausalLM,
    JambaModel,
)


class TestJambaConfig:
    """Tests for JambaConfig."""

    def test_defaults(self):
        """Test default configuration values."""
        config = JambaConfig()

        assert config.model_type == "jamba"
        assert config.attn_layer_period == 8
        assert config.expert_layer_period == 2
        assert config.num_experts == 16
        assert config.num_experts_per_tok == 2

    def test_custom_config(self):
        """Test custom configuration."""
        config = JambaConfig(
            vocab_size=50000,
            hidden_size=2048,
            num_hidden_layers=16,
            num_experts=8,
        )

        assert config.vocab_size == 50000
        assert config.hidden_size == 2048
        assert config.num_hidden_layers == 16
        assert config.num_experts == 8

    def test_tiny_preset(self):
        """Test tiny preset for testing."""
        config = JambaConfig.tiny()

        assert config.hidden_size == 64
        assert config.num_hidden_layers == 8
        assert config.num_experts == 4

    def test_jamba_v0_1_preset(self):
        """Test Jamba v0.1 preset."""
        config = JambaConfig.jamba_v0_1()

        assert config.hidden_size == 4096
        assert config.num_hidden_layers == 32
        assert config.num_experts == 16

    def test_jamba_1_5_mini_preset(self):
        """Test Jamba 1.5 Mini preset."""
        config = JambaConfig.jamba_1_5_mini()

        assert config.hidden_size == 4096
        assert config.num_hidden_layers == 32

    def test_jamba_1_5_large_preset(self):
        """Test Jamba 1.5 Large preset."""
        config = JambaConfig.jamba_1_5_large()

        assert config.hidden_size == 8192
        assert config.num_hidden_layers == 64

    def test_is_attention_layer(self):
        """Test is_attention_layer method."""
        config = JambaConfig(
            attn_layer_period=4,
            attn_layer_offset=2,
        )

        # Layer 2 should be attention (offset=2)
        assert config.is_attention_layer(2) is True
        # Layer 6 should be attention (2 + 4)
        assert config.is_attention_layer(6) is True
        # Layer 0, 1, 3, 4, 5 should be Mamba
        assert config.is_attention_layer(0) is False
        assert config.is_attention_layer(1) is False
        assert config.is_attention_layer(3) is False

    def test_is_moe_layer(self):
        """Test is_moe_layer method."""
        config = JambaConfig(
            expert_layer_period=2,
            expert_layer_offset=1,
            num_experts=4,
        )

        # Layer 1, 3, 5, 7 should be MoE
        assert config.is_moe_layer(1) is True
        assert config.is_moe_layer(3) is True
        # Layer 0, 2, 4, 6 should be dense
        assert config.is_moe_layer(0) is False
        assert config.is_moe_layer(2) is False

    def test_is_moe_layer_no_experts(self):
        """Test is_moe_layer returns False when num_experts <= 1."""
        config = JambaConfig(num_experts=1)

        # No MoE when only 1 expert
        assert config.is_moe_layer(1) is False
        assert config.is_moe_layer(3) is False

    def test_from_hf_config(self):
        """Test from_hf_config class method."""
        hf_config = {
            "model_type": "jamba",
            "vocab_size": 65536,
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 14336,
            "attn_layer_period": 8,
            "expert_layer_period": 2,
            "num_experts": 16,
            "num_experts_per_tok": 2,
        }

        config = JambaConfig.from_hf_config(hf_config)

        assert config.vocab_size == 65536
        assert config.hidden_size == 4096
        assert config.num_experts == 16


class TestJambaModel:
    """Tests for JambaModel."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        config = JambaConfig.tiny()
        model = JambaModel(config)

        x = mx.random.randint(0, config.vocab_size, (1, 5))
        output = model(x)

        # JambaModel returns BackboneOutput with last_hidden_state
        hidden_states = output.last_hidden_state if hasattr(output, "last_hidden_state") else output
        assert hidden_states.shape == (1, 5, config.hidden_size)

    def test_batch_forward(self):
        """Test forward pass with batch."""
        config = JambaConfig.tiny()
        model = JambaModel(config)

        x = mx.random.randint(0, config.vocab_size, (2, 10))
        output = model(x)

        hidden_states = output.last_hidden_state if hasattr(output, "last_hidden_state") else output
        assert hidden_states.shape == (2, 10, config.hidden_size)

    def test_different_sequence_lengths(self):
        """Test forward with different sequence lengths."""
        config = JambaConfig.tiny()
        model = JambaModel(config)

        for seq_len in [1, 5, 10, 20]:
            x = mx.random.randint(0, config.vocab_size, (1, seq_len))
            output = model(x)
            hidden_states = (
                output.last_hidden_state if hasattr(output, "last_hidden_state") else output
            )
            assert hidden_states.shape == (1, seq_len, config.hidden_size)

    def test_layer_count(self):
        """Test model has correct number of layers."""
        config = JambaConfig.tiny()
        model = JambaModel(config)

        assert len(model.layers) == config.num_hidden_layers


class TestJambaForCausalLM:
    """Tests for JambaForCausalLM."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        config = JambaConfig.tiny()
        model = JambaForCausalLM(config)

        x = mx.random.randint(0, config.vocab_size, (1, 5))
        output = model(x)

        # Output should have logits
        if hasattr(output, "logits"):
            assert output.logits.shape == (1, 5, config.vocab_size)
        else:
            assert output.shape == (1, 5, config.vocab_size)

    def test_untied_embeddings(self):
        """Test model with untied embeddings."""
        config = JambaConfig.tiny()
        config.tie_word_embeddings = False
        model = JambaForCausalLM(config)

        # Should have separate lm_head
        assert hasattr(model, "lm_head")

    def test_generate(self):
        """Test generation."""
        config = JambaConfig.tiny()
        model = JambaForCausalLM(config)

        input_ids = mx.array([[1, 2, 3]])
        output = model.generate(input_ids, max_new_tokens=5)

        # Should generate more tokens
        assert output.shape[1] >= input_ids.shape[1]

    def test_from_config(self):
        """Test from_config class method."""
        config = JambaConfig.tiny()
        model = JambaForCausalLM.from_config(config)

        assert isinstance(model, JambaForCausalLM)


class TestJambaGradients:
    """Tests for gradient flow through Jamba."""

    def test_forward_backward_no_moe(self):
        """Test gradients flow correctly with no MoE (simpler case)."""
        # Disable MoE to avoid routing issues in tiny model
        config = JambaConfig(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
            attn_layer_period=2,
            attn_layer_offset=1,
            num_experts=1,  # No MoE - just regular FFN
            num_experts_per_tok=1,
            mamba_d_state=8,
            mamba_d_conv=4,
            mamba_expand=2,
            mamba_dt_rank=16,
        )
        model = JambaForCausalLM(config)

        x = mx.random.randint(0, config.vocab_size, (1, 5))

        def loss_fn(model, x):
            out = model(x)
            logits = out.logits if hasattr(out, "logits") else out
            return mx.mean(logits**2)

        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad_fn(model, x)

        assert loss.item() > 0


class TestJambaBatchHandling:
    """Tests for batch dimension handling."""

    def test_different_batch_sizes(self):
        """Test model handles different batch sizes."""
        config = JambaConfig.tiny()
        model = JambaForCausalLM(config)

        for batch_size in [1, 2, 4]:
            x = mx.random.randint(0, config.vocab_size, (batch_size, 5))
            output = model(x)
            logits = output.logits if hasattr(output, "logits") else output
            assert logits.shape[0] == batch_size
