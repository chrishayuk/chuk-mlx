"""
Tests for Granite configuration.
"""

from chuk_lazarus.models_v2.families.granite.config import (
    GraniteConfig,
    GraniteHybridConfig,
)


class TestGraniteConfig:
    """Tests for GraniteConfig."""

    def test_defaults(self):
        """Test default configuration values."""
        config = GraniteConfig()

        assert config.model_type == "granite"
        assert config.embedding_multiplier == 12.0
        assert config.attention_multiplier == 1.0
        assert config.residual_multiplier == 1.0
        assert config.logits_scaling == 1.0
        assert config.hidden_act == "silu"
        assert config.rope_theta == 10000.0
        assert config.rms_norm_eps == 1e-5
        assert config.attention_dropout == 0.0
        assert config.attention_bias is False
        assert config.mlp_bias is False
        assert config.rope_scaling is None

    def test_granite_3_8b(self):
        """Test Granite 3.0 8B preset."""
        config = GraniteConfig.granite_3_8b()

        assert config.vocab_size == 49155
        assert config.hidden_size == 4096
        assert config.num_hidden_layers == 40
        assert config.num_attention_heads == 32
        assert config.num_key_value_heads == 8
        assert config.intermediate_size == 12800
        assert config.max_position_embeddings == 4096
        assert config.embedding_multiplier == 12.0
        assert config.attention_multiplier == 0.0078125
        assert config.residual_multiplier == 0.22
        assert config.logits_scaling == 16.0
        assert config.attention_dropout == 0.1
        assert config.tie_word_embeddings is True

    def test_granite_3_1_2b(self):
        """Test Granite 3.1 2B preset."""
        config = GraniteConfig.granite_3_1_2b()

        assert config.vocab_size == 49155
        assert config.hidden_size == 2048
        assert config.num_hidden_layers == 40
        assert config.num_attention_heads == 32
        assert config.num_key_value_heads == 8
        assert config.intermediate_size == 8192
        assert config.max_position_embeddings == 131072
        assert config.rope_theta == 5000000.0
        assert config.embedding_multiplier == 12.0
        assert config.attention_multiplier == 0.015625
        assert config.logits_scaling == 8.0

    def test_granite_3_1_8b(self):
        """Test Granite 3.1 8B preset."""
        config = GraniteConfig.granite_3_1_8b()

        assert config.vocab_size == 49155
        assert config.hidden_size == 4096
        assert config.num_hidden_layers == 40
        assert config.num_attention_heads == 32
        assert config.num_key_value_heads == 8
        assert config.intermediate_size == 12800
        assert config.max_position_embeddings == 131072
        assert config.rope_theta == 5000000.0

    def test_tiny(self):
        """Test tiny config for testing."""
        config = GraniteConfig.tiny()

        assert config.vocab_size == 1000
        assert config.hidden_size == 64
        assert config.num_hidden_layers == 4
        assert config.num_attention_heads == 4
        assert config.num_key_value_heads == 2
        assert config.intermediate_size == 128
        assert config.embedding_multiplier == 1.0
        assert config.attention_multiplier == 1.0
        assert config.residual_multiplier == 1.0
        assert config.logits_scaling == 1.0


class TestGraniteHybridConfig:
    """Tests for GraniteHybridConfig."""

    def test_defaults(self):
        """Test default configuration values."""
        config = GraniteHybridConfig()

        assert config.model_type == "granitemoehybrid"
        assert config.position_embedding_type == "rope"
        assert config.embedding_multiplier == 12.0
        assert config.attention_multiplier == 0.0078125
        assert config.residual_multiplier == 0.22
        assert config.logits_scaling == 6.0
        assert len(config.layer_types) == 40
        assert all(t == "attention" for t in config.layer_types)
        assert config.normalization_function == "rmsnorm"

    def test_mamba_defaults(self):
        """Test Mamba-2 default settings."""
        config = GraniteHybridConfig()

        assert config.mamba_d_state == 128
        assert config.mamba_d_conv == 4
        assert config.mamba_expand == 2
        assert config.mamba_n_heads == 48
        assert config.mamba_d_head == 64
        assert config.mamba_n_groups == 1
        assert config.mamba_chunk_size == 256
        assert config.mamba_conv_bias is True
        assert config.mamba_proj_bias is False

    def test_moe_defaults(self):
        """Test MoE default settings."""
        config = GraniteHybridConfig()

        assert config.num_local_experts == 0
        assert config.num_experts_per_tok == 0
        assert config.shared_intermediate_size == 0
        assert config.router_aux_loss_coef == 0.0
        assert config.output_router_logits is False

    def test_is_moe_property(self):
        """Test is_moe property."""
        # Dense model
        dense_config = GraniteHybridConfig(num_local_experts=0, num_experts_per_tok=0)
        assert dense_config.is_moe is False

        # MoE model
        moe_config = GraniteHybridConfig(num_local_experts=4, num_experts_per_tok=2)
        assert moe_config.is_moe is True

        # Only num_local_experts set
        partial_config = GraniteHybridConfig(num_local_experts=4, num_experts_per_tok=0)
        assert partial_config.is_moe is False

    def test_num_mamba_layers_property(self):
        """Test num_mamba_layers property."""
        # All attention
        attn_config = GraniteHybridConfig(layer_types=["attention"] * 10)
        assert attn_config.num_mamba_layers == 0
        assert attn_config.num_attention_layers == 10

        # Mixed
        mixed_config = GraniteHybridConfig(layer_types=["mamba", "mamba", "attention", "mamba"])
        assert mixed_config.num_mamba_layers == 3
        assert mixed_config.num_attention_layers == 1

    def test_granite_4_micro(self):
        """Test Granite 4.0 Micro preset."""
        config = GraniteHybridConfig.granite_4_micro()

        assert config.vocab_size == 100352
        assert config.hidden_size == 2560
        assert config.num_hidden_layers == 40
        assert config.num_attention_heads == 40
        assert config.num_key_value_heads == 8
        assert config.intermediate_size == 8192
        assert all(t == "attention" for t in config.layer_types)
        assert config.position_embedding_type == "rope"
        assert config.num_local_experts == 0
        assert config.is_moe is False
        assert config.tie_word_embeddings is True

    def test_granite_4_tiny(self):
        """Test Granite 4.0 Tiny preset."""
        config = GraniteHybridConfig.granite_4_tiny()

        assert config.vocab_size == 49160
        assert config.hidden_size == 1536
        assert config.num_hidden_layers == 40
        assert config.num_attention_heads == 12
        assert config.num_key_value_heads == 4
        assert config.position_embedding_type == "nope"
        assert config.num_local_experts == 62
        assert config.num_experts_per_tok == 6
        assert config.is_moe is True
        assert config.num_mamba_layers > 0
        assert config.num_attention_layers == 4

    def test_granite_4_small(self):
        """Test Granite 4.0 Small preset."""
        config = GraniteHybridConfig.granite_4_small()

        assert config.vocab_size == 49160
        assert config.hidden_size == 3072
        assert config.num_hidden_layers == 40
        assert config.num_local_experts == 62
        assert config.num_experts_per_tok == 6
        assert config.is_moe is True

    def test_tiny(self):
        """Test tiny config for testing."""
        config = GraniteHybridConfig.tiny()

        assert config.vocab_size == 1000
        assert config.hidden_size == 64
        assert config.num_hidden_layers == 4
        assert config.num_attention_heads == 4
        assert config.num_key_value_heads == 2
        assert config.layer_types == ["mamba", "mamba", "attention", "mamba"]
        assert config.num_mamba_layers == 3
        assert config.num_attention_layers == 1
        assert config.is_moe is False

    def test_tiny_moe(self):
        """Test tiny MoE config for testing."""
        config = GraniteHybridConfig.tiny_moe()

        assert config.vocab_size == 1000
        assert config.hidden_size == 64
        assert config.num_hidden_layers == 4
        assert config.num_local_experts == 4
        assert config.num_experts_per_tok == 2
        assert config.shared_intermediate_size == 64
        assert config.is_moe is True
