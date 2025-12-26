"""
Tests for GemmaConfig.
"""

from chuk_lazarus.models_v2.families.gemma import GemmaConfig


class TestGemmaConfig:
    """Tests for GemmaConfig."""

    def test_tiny_config(self):
        """Test tiny config for testing."""
        config = GemmaConfig.tiny()

        assert config.hidden_size == 64
        assert config.num_hidden_layers == 6
        assert config.num_attention_heads == 2
        assert config.num_key_value_heads == 1
        assert config.intermediate_size == 128
        assert config.head_dim == 32
        assert config.sliding_window_pattern == 3

    def test_gemma3_270m_config(self):
        """Test Gemma 3 270M config."""
        config = GemmaConfig.gemma3_270m()

        assert config.hidden_size == 640
        assert config.num_hidden_layers == 18
        assert config.num_attention_heads == 4
        assert config.num_key_value_heads == 1
        assert config.intermediate_size == 2048
        assert config.head_dim == 256
        assert config.vocab_size == 262144
        assert config.sliding_window == 512
        assert config.sliding_window_pattern == 6

    def test_functiongemma_config(self):
        """Test FunctionGemma config (same as 270M)."""
        config = GemmaConfig.functiongemma_270m()
        base_config = GemmaConfig.gemma3_270m()

        assert config.hidden_size == base_config.hidden_size
        assert config.num_hidden_layers == base_config.num_hidden_layers
        assert config.vocab_size == base_config.vocab_size

    def test_gemma3_1b_config(self):
        """Test Gemma 3 1B config."""
        config = GemmaConfig.gemma3_1b()

        assert config.hidden_size == 1152
        assert config.num_hidden_layers == 26
        assert config.num_attention_heads == 4
        assert config.intermediate_size == 6912

    def test_gemma3_4b_config(self):
        """Test Gemma 3 4B config."""
        config = GemmaConfig.gemma3_4b()

        assert config.hidden_size == 2560
        assert config.num_hidden_layers == 34
        assert config.num_attention_heads == 8

    def test_gemma3_12b_config(self):
        """Test Gemma 3 12B config."""
        config = GemmaConfig.gemma3_12b()

        assert config.hidden_size == 3840
        assert config.num_hidden_layers == 48
        assert config.num_attention_heads == 16

    def test_gemma3_27b_config(self):
        """Test Gemma 3 27B config."""
        config = GemmaConfig.gemma3_27b()

        assert config.hidden_size == 5120
        assert config.num_hidden_layers == 62
        assert config.num_attention_heads == 24

    def test_is_sliding_layer(self):
        """Test sliding layer detection."""
        config = GemmaConfig.tiny()  # pattern = 3

        # Pattern 3 means every 3rd layer (indices 2, 5, 8...) is global
        assert config.is_sliding_layer(0) is True  # sliding
        assert config.is_sliding_layer(1) is True  # sliding
        assert config.is_sliding_layer(2) is False  # global (3rd layer)
        assert config.is_sliding_layer(3) is True  # sliding
        assert config.is_sliding_layer(4) is True  # sliding
        assert config.is_sliding_layer(5) is False  # global (6th layer)

    def test_is_global_layer(self):
        """Test global layer detection."""
        config = GemmaConfig.gemma3_270m()  # pattern = 6

        # Pattern 6 means every 6th layer is global
        assert config.is_global_layer(4) is False  # layer 5 is sliding
        assert config.is_global_layer(5) is True  # layer 6 is global
        assert config.is_global_layer(6) is False  # layer 7 is sliding
        assert config.is_global_layer(11) is True  # layer 12 is global

    def test_gemma_specific_defaults(self):
        """Test Gemma-specific default values."""
        config = GemmaConfig.gemma3_270m()

        assert config.model_type == "gemma3_text"
        assert config.hidden_act == "gelu_pytorch_tanh"
        assert config.rope_theta == 1000000.0
        assert config.rope_local_base_freq == 10000.0
        assert config.rms_norm_eps == 1e-6
        assert config.query_pre_attn_scalar == 256.0

    def test_max_position_embeddings(self):
        """Test max position embeddings for different sizes."""
        config_270m = GemmaConfig.gemma3_270m()
        config_4b = GemmaConfig.gemma3_4b()

        assert config_270m.max_position_embeddings == 32768
        assert config_4b.max_position_embeddings == 131072  # Larger models have more context
