"""
Tests for StarCoder and StarCoder2 configuration.
"""

from chuk_lazarus.models_v2.families.starcoder2 import StarCoder2Config, StarCoderConfig


class TestStarCoder2Config:
    """Tests for StarCoder2Config."""

    def test_defaults(self):
        """Test default configuration values."""
        config = StarCoder2Config()

        # StarCoder2-specific defaults
        assert config.model_type == "starcoder2"
        assert config.hidden_act == "gelu_pytorch_tanh"
        assert config.rope_theta == 100000.0
        assert config.layer_norm_eps == 1e-5
        assert config.attention_bias is True
        assert config.mlp_bias is True
        assert config.sliding_window == 4096

    def test_custom_config(self):
        """Test custom configuration."""
        config = StarCoder2Config(
            vocab_size=50000,
            hidden_size=1024,
            num_hidden_layers=12,
            num_attention_heads=8,
            num_key_value_heads=4,
            intermediate_size=4096,
            rope_theta=50000.0,
            sliding_window=2048,
        )

        assert config.vocab_size == 50000
        assert config.hidden_size == 1024
        assert config.num_hidden_layers == 12
        assert config.num_attention_heads == 8
        assert config.num_key_value_heads == 4
        assert config.intermediate_size == 4096
        assert config.rope_theta == 50000.0
        assert config.sliding_window == 2048

    def test_tiny_preset(self):
        """Test tiny preset for testing."""
        config = StarCoder2Config.tiny()

        assert config.vocab_size == 1000
        assert config.hidden_size == 64
        assert config.num_hidden_layers == 2
        assert config.num_attention_heads == 4
        assert config.num_key_value_heads == 2
        assert config.intermediate_size == 128
        assert config.max_position_embeddings == 256
        assert config.sliding_window == 128

    def test_starcoder2_3b_preset(self):
        """Test StarCoder2 3B preset."""
        config = StarCoder2Config.starcoder2_3b()

        assert config.vocab_size == 49152
        assert config.hidden_size == 3072
        assert config.num_hidden_layers == 30
        assert config.num_attention_heads == 24
        assert config.num_key_value_heads == 2
        assert config.intermediate_size == 12288
        assert config.max_position_embeddings == 16384
        assert config.sliding_window == 4096

    def test_starcoder2_7b_preset(self):
        """Test StarCoder2 7B preset."""
        config = StarCoder2Config.starcoder2_7b()

        assert config.vocab_size == 49152
        assert config.hidden_size == 4608
        assert config.num_hidden_layers == 32
        assert config.num_attention_heads == 36
        assert config.num_key_value_heads == 4
        assert config.intermediate_size == 18432
        assert config.max_position_embeddings == 16384
        assert config.sliding_window == 4096

    def test_starcoder2_15b_preset(self):
        """Test StarCoder2 15B preset."""
        config = StarCoder2Config.starcoder2_15b()

        assert config.vocab_size == 49152
        assert config.hidden_size == 6144
        assert config.num_hidden_layers == 40
        assert config.num_attention_heads == 48
        assert config.num_key_value_heads == 4
        assert config.intermediate_size == 24576
        assert config.max_position_embeddings == 16384
        assert config.sliding_window == 4096

    def test_rope_scaling(self):
        """Test rope_scaling configuration."""
        config = StarCoder2Config(rope_scaling={"type": "linear", "factor": 2.0})
        assert config.rope_scaling == {"type": "linear", "factor": 2.0}

    def test_model_type(self):
        """Test model_type is set correctly."""
        config = StarCoder2Config()
        assert config.model_type == "starcoder2"

    def test_no_sliding_window(self):
        """Test disabling sliding window."""
        config = StarCoder2Config(sliding_window=None)
        assert config.sliding_window is None

    def test_bias_settings(self):
        """Test attention and MLP bias settings."""
        # Default with bias
        config_with_bias = StarCoder2Config()
        assert config_with_bias.attention_bias is True
        assert config_with_bias.mlp_bias is True

        # Without bias
        config_no_bias = StarCoder2Config(
            attention_bias=False,
            mlp_bias=False,
        )
        assert config_no_bias.attention_bias is False
        assert config_no_bias.mlp_bias is False


class TestStarCoderConfig:
    """Tests for StarCoder (original) configuration."""

    def test_defaults(self):
        """Test default configuration values."""
        config = StarCoderConfig()

        # StarCoder-specific defaults
        assert config.model_type == "gpt_bigcode"
        assert config.hidden_act == "gelu_pytorch_tanh"
        assert config.layer_norm_eps == 1e-5
        assert config.attention_bias is True
        assert config.mlp_bias is True
        assert config.sliding_window is None  # No sliding window
        assert config.use_learned_position_embeddings is True
        assert config.multi_query is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = StarCoderConfig(
            vocab_size=50000,
            hidden_size=1024,
            num_hidden_layers=12,
            num_attention_heads=8,
            num_key_value_heads=1,  # MQA
            intermediate_size=4096,
        )

        assert config.vocab_size == 50000
        assert config.hidden_size == 1024
        assert config.num_hidden_layers == 12
        assert config.num_attention_heads == 8
        assert config.num_key_value_heads == 1
        assert config.intermediate_size == 4096

    def test_tiny_preset(self):
        """Test tiny preset for testing."""
        config = StarCoderConfig.tiny()

        assert config.vocab_size == 1000
        assert config.hidden_size == 64
        assert config.num_hidden_layers == 2
        assert config.num_attention_heads == 4
        assert config.num_key_value_heads == 1  # MQA
        assert config.intermediate_size == 128
        assert config.max_position_embeddings == 256
        assert config.multi_query is True

    def test_starcoder_preset(self):
        """Test StarCoder 15.5B preset."""
        config = StarCoderConfig.starcoder()

        assert config.vocab_size == 49152
        assert config.hidden_size == 6144
        assert config.num_hidden_layers == 40
        assert config.num_attention_heads == 48
        assert config.num_key_value_heads == 1  # MQA
        assert config.intermediate_size == 24576
        assert config.max_position_embeddings == 8192

    def test_starcoderbase_preset(self):
        """Test StarCoderBase preset (same as StarCoder)."""
        config = StarCoderConfig.starcoderbase()

        # Same as starcoder
        assert config.vocab_size == 49152
        assert config.hidden_size == 6144
        assert config.num_hidden_layers == 40
        assert config.num_attention_heads == 48
        assert config.num_key_value_heads == 1

    def test_santacoder_preset(self):
        """Test SantaCoder 1.1B preset."""
        config = StarCoderConfig.santacoder()

        assert config.vocab_size == 49280  # Different vocab
        assert config.hidden_size == 2048
        assert config.num_hidden_layers == 24
        assert config.num_attention_heads == 16
        assert config.num_key_value_heads == 1  # MQA
        assert config.intermediate_size == 8192
        assert config.max_position_embeddings == 2048

    def test_model_type(self):
        """Test model_type is set correctly."""
        config = StarCoderConfig()
        assert config.model_type == "gpt_bigcode"

    def test_no_sliding_window(self):
        """Test StarCoder has no sliding window."""
        config = StarCoderConfig()
        assert config.sliding_window is None

    def test_learned_position_embeddings(self):
        """Test StarCoder uses learned position embeddings."""
        config = StarCoderConfig()
        assert config.use_learned_position_embeddings is True

    def test_multi_query_attention(self):
        """Test StarCoder uses MQA (num_key_value_heads=1)."""
        config = StarCoderConfig.starcoder()
        assert config.num_key_value_heads == 1
        assert config.multi_query is True

    def test_bias_settings(self):
        """Test attention and MLP bias settings."""
        # Default with bias
        config_with_bias = StarCoderConfig()
        assert config_with_bias.attention_bias is True
        assert config_with_bias.mlp_bias is True

        # Without bias
        config_no_bias = StarCoderConfig(
            attention_bias=False,
            mlp_bias=False,
        )
        assert config_no_bias.attention_bias is False
        assert config_no_bias.mlp_bias is False
