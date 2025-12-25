"""
Tests for Llama 4 configuration.
"""

from chuk_lazarus.models_v2.families.llama4.config import (
    Llama4Config,
    Llama4TextConfig,
    Llama4VisionConfig,
)


class TestLlama4TextConfig:
    """Tests for Llama4TextConfig."""

    def test_defaults(self):
        """Test default configuration values."""
        config = Llama4TextConfig()

        assert config.model_type == "llama4"
        assert config.hidden_act == "silu"
        assert config.rope_theta == 500000.0
        assert config.rms_norm_eps == 1e-5
        assert config.num_local_experts == 16
        assert config.num_experts_per_tok == 1
        assert config.intermediate_size_mlp == 16384
        assert config.moe_router_topk == 1
        assert config.no_rope_layers is None
        assert config.attention_chunk_size == 8192
        assert config.use_qk_norm is True
        assert config.attn_temperature_tuning is False
        assert config.rope_scaling is None

    def test_scout_17b(self):
        """Test Llama 4 Scout preset."""
        config = Llama4TextConfig.scout_17b()

        assert config.vocab_size == 202048
        assert config.hidden_size == 5120
        assert config.num_hidden_layers == 48
        assert config.num_attention_heads == 40
        assert config.num_key_value_heads == 8
        assert config.intermediate_size == 8192
        assert config.intermediate_size_mlp == 16384
        assert config.num_local_experts == 16
        assert config.num_experts_per_tok == 1
        assert config.max_position_embeddings == 131072
        assert config.rope_theta == 500000.0
        assert config.use_qk_norm is True
        assert config.tie_word_embeddings is False
        assert config.no_rope_layers == [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44]

    def test_maverick_17b(self):
        """Test Llama 4 Maverick preset."""
        config = Llama4TextConfig.maverick_17b()

        assert config.vocab_size == 202048
        assert config.hidden_size == 5120
        assert config.num_hidden_layers == 48
        assert config.num_attention_heads == 40
        assert config.num_key_value_heads == 8
        assert config.intermediate_size == 8192
        assert config.intermediate_size_mlp == 8192
        assert config.num_local_experts == 128
        assert config.num_experts_per_tok == 1
        assert config.use_qk_norm is True

    def test_tiny(self):
        """Test tiny config for testing."""
        config = Llama4TextConfig.tiny()

        assert config.vocab_size == 1000
        assert config.hidden_size == 64
        assert config.num_hidden_layers == 4
        assert config.num_attention_heads == 4
        assert config.num_key_value_heads == 2
        assert config.intermediate_size == 128
        assert config.intermediate_size_mlp == 256
        assert config.num_local_experts == 4
        assert config.num_experts_per_tok == 1
        assert config.max_position_embeddings == 256
        assert config.use_qk_norm is True
        assert config.no_rope_layers == [0]


class TestLlama4VisionConfig:
    """Tests for Llama4VisionConfig."""

    def test_defaults(self):
        """Test default configuration values."""
        config = Llama4VisionConfig()

        assert config.model_type == "llama4_vision"
        assert config.hidden_size == 1280
        assert config.num_hidden_layers == 32
        assert config.num_attention_heads == 16
        assert config.intermediate_size == 5120
        assert config.image_size == 560
        assert config.patch_size == 14
        assert config.num_channels == 3
        assert config.vision_output_dim == 5120
        assert config.pixel_shuffle_ratio == 0.5
        assert config.rms_norm_eps == 1e-5
        assert config.hidden_act == "gelu"

    def test_default_factory(self):
        """Test default factory method."""
        config = Llama4VisionConfig.default()

        assert config.model_type == "llama4_vision"
        assert config.hidden_size == 1280
        assert config.image_size == 560


class TestLlama4Config:
    """Tests for Llama4Config multimodal config."""

    def test_scout_multimodal(self):
        """Test Scout multimodal preset."""
        config = Llama4Config.scout_multimodal()

        assert config.model_type == "llama4"
        assert config.text_config is not None
        assert config.vision_config is not None
        assert config.text_config.hidden_size == 5120
        assert config.vision_config.hidden_size == 1280
        assert config.image_token_index == 128011
        assert config.image_token == "<|image|>"

    def test_scout_text_only(self):
        """Test Scout text-only preset."""
        config = Llama4Config.scout_text_only()

        assert config.text_config is not None
        assert config.vision_config is None

    def test_text_config_vocab_size(self):
        """Test accessing vocab_size via text_config."""
        config = Llama4Config.scout_text_only()
        assert config.text_config.vocab_size == 202048

    def test_text_config_hidden_size(self):
        """Test accessing hidden_size via text_config."""
        config = Llama4Config.scout_text_only()
        assert config.text_config.hidden_size == 5120

    def test_text_config_num_hidden_layers(self):
        """Test accessing num_hidden_layers via text_config."""
        config = Llama4Config.scout_text_only()
        assert config.text_config.num_hidden_layers == 48

    def test_text_config_num_attention_heads(self):
        """Test accessing num_attention_heads via text_config."""
        config = Llama4Config.scout_text_only()
        assert config.text_config.num_attention_heads == 40

    def test_text_config_num_key_value_heads(self):
        """Test accessing num_key_value_heads via text_config."""
        config = Llama4Config.scout_text_only()
        assert config.text_config.num_key_value_heads == 8

    def test_text_config_intermediate_size(self):
        """Test accessing intermediate_size via text_config."""
        config = Llama4Config.scout_text_only()
        assert config.text_config.intermediate_size == 8192

    def test_text_config_rms_norm_eps(self):
        """Test accessing rms_norm_eps via text_config."""
        config = Llama4Config.scout_text_only()
        assert config.text_config.rms_norm_eps == 1e-5

    def test_text_config_tie_word_embeddings(self):
        """Test accessing tie_word_embeddings via text_config."""
        config = Llama4Config.scout_text_only()
        assert config.text_config.tie_word_embeddings is False
