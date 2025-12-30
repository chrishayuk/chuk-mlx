"""Tests for GPT2Config."""

from chuk_lazarus.models_v2.families.constants import (
    DefaultNormEps,
    DefaultPositionEmbeddings,
    DefaultVocabSize,
    HFModelType,
)
from chuk_lazarus.models_v2.families.gpt2 import GPT2Config


class TestGPT2Config:
    """Tests for GPT2Config."""

    def test_tiny_config(self):
        """Test tiny config for testing."""
        config = GPT2Config.tiny()

        assert config.vocab_size == 1000
        assert config.hidden_size == 64
        assert config.num_hidden_layers == 2
        assert config.num_attention_heads == 4
        assert config.intermediate_size == 256
        assert config.max_position_embeddings == 256

    def test_gpt2_small_config(self):
        """Test GPT-2 Small (117M) config."""
        config = GPT2Config.gpt2_small()

        assert config.vocab_size == DefaultVocabSize.GPT2
        assert config.hidden_size == 768
        assert config.num_hidden_layers == 12
        assert config.num_attention_heads == 12
        assert config.intermediate_size == 3072  # 4 * 768
        assert config.max_position_embeddings == DefaultPositionEmbeddings.GPT2

    def test_gpt2_medium_config(self):
        """Test GPT-2 Medium (345M) config."""
        config = GPT2Config.gpt2_medium()

        assert config.vocab_size == DefaultVocabSize.GPT2
        assert config.hidden_size == 1024
        assert config.num_hidden_layers == 24
        assert config.num_attention_heads == 16
        assert config.intermediate_size == 4096

    def test_gpt2_large_config(self):
        """Test GPT-2 Large (762M) config."""
        config = GPT2Config.gpt2_large()

        assert config.vocab_size == DefaultVocabSize.GPT2
        assert config.hidden_size == 1280
        assert config.num_hidden_layers == 36
        assert config.num_attention_heads == 20
        assert config.intermediate_size == 5120

    def test_gpt2_xl_config(self):
        """Test GPT-2 XL (1.5B) config."""
        config = GPT2Config.gpt2_xl()

        assert config.vocab_size == DefaultVocabSize.GPT2
        assert config.hidden_size == 1600
        assert config.num_hidden_layers == 48
        assert config.num_attention_heads == 25
        assert config.intermediate_size == 6400

    def test_distilgpt2_config(self):
        """Test DistilGPT-2 config."""
        config = GPT2Config.distilgpt2()

        assert config.vocab_size == DefaultVocabSize.GPT2
        assert config.hidden_size == 768
        assert config.num_hidden_layers == 6  # Half of GPT-2 Small
        assert config.num_attention_heads == 12

    def test_gpt2_specific_defaults(self):
        """Test GPT-2-specific default values."""
        config = GPT2Config.tiny()

        assert config.model_type == HFModelType.GPT2.value
        assert config.hidden_act == "gelu_new"
        assert config.layer_norm_eps == DefaultNormEps.GPT2.value
        assert config.use_learned_position_embeddings is True
        assert config.attention_bias is True
        assert config.mlp_bias is True

    def test_from_hf_config(self):
        """Test creating config from HuggingFace config dict."""
        hf_config = {
            "model_type": "gpt2",
            "vocab_size": 50257,
            "n_embd": 768,
            "n_layer": 12,
            "n_head": 12,
            "n_inner": 3072,
            "n_positions": 1024,
        }

        config = GPT2Config.from_hf_config(hf_config)

        assert config.vocab_size == 50257
        assert config.hidden_size == 768
        assert config.num_hidden_layers == 12
        assert config.num_attention_heads == 12
        assert config.intermediate_size == 3072
        assert config.max_position_embeddings == 1024

    def test_from_hf_config_with_fallback_keys(self):
        """Test config creation with standard key names."""
        hf_config = {
            "model_type": "gpt2",
            "vocab_size": 50257,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
        }

        config = GPT2Config.from_hf_config(hf_config)

        assert config.hidden_size == 768
        assert config.num_hidden_layers == 12
        assert config.num_attention_heads == 12

    def test_from_hf_config_defaults(self):
        """Test config creation uses defaults for missing values."""
        hf_config = {"model_type": "gpt2"}

        config = GPT2Config.from_hf_config(hf_config)

        assert config.vocab_size == DefaultVocabSize.GPT2
        assert config.hidden_size == 768
        assert config.num_hidden_layers == 12
        assert config.max_position_embeddings == DefaultPositionEmbeddings.GPT2

    def test_num_key_value_heads_equals_attention_heads(self):
        """Test that GPT-2 uses MHA (num_kv_heads == num_attention_heads)."""
        config = GPT2Config.gpt2_small()
        assert config.num_key_value_heads == config.num_attention_heads


class TestGPT2ConfigValidation:
    """Tests for GPT2Config validation."""

    def test_custom_config(self):
        """Test creating custom config."""
        config = GPT2Config(
            vocab_size=10000,
            hidden_size=512,
            num_hidden_layers=8,
            num_attention_heads=8,
            num_key_value_heads=8,
            intermediate_size=2048,
            max_position_embeddings=512,
        )

        assert config.vocab_size == 10000
        assert config.hidden_size == 512
        assert config.num_hidden_layers == 8

    def test_intermediate_size_default(self):
        """Test intermediate size defaults to 4 * hidden_size."""
        # When using from_hf_config without n_inner
        hf_config = {
            "model_type": "gpt2",
            "n_embd": 256,
            "n_layer": 2,
            "n_head": 4,
        }

        config = GPT2Config.from_hf_config(hf_config)

        # Should default to 4 * hidden_size
        assert config.intermediate_size == 4 * 256
