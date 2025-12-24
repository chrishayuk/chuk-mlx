"""
Tests for models_v2.core.config.

Ensures all Pydantic configs validate correctly and derived values are computed.
"""

import json

import pytest
from pydantic import ValidationError

from chuk_lazarus.models_v2.core.config import (
    AttentionConfig,
    BackboneConfig,
    BlockConfig,
    EmbeddingConfig,
    FFNConfig,
    HeadConfig,
    ModelConfig,
    NormConfig,
    PositionConfig,
    RoPEConfig,
    SSMConfig,
)
from chuk_lazarus.models_v2.core.enums import (
    ActivationType,
    AttentionType,
    BlockType,
    FFNType,
    HeadType,
    NormType,
    PoolingType,
    PositionEmbeddingType,
    SSMType,
)


class TestRoPEConfig:
    """Tests for RoPEConfig."""

    def test_defaults(self):
        """Test default values."""
        config = RoPEConfig()
        assert config.theta == 10000.0
        assert config.traditional is False
        assert config.scaling_factor == 1.0
        assert config.scaling_type is None
        assert config.max_position_embeddings == 4096

    def test_custom_values(self):
        """Test custom values."""
        config = RoPEConfig(
            theta=500000.0,
            traditional=True,
            scaling_factor=2.0,
            scaling_type="dynamic",
            max_position_embeddings=8192,
        )
        assert config.theta == 500000.0
        assert config.traditional is True
        assert config.scaling_factor == 2.0
        assert config.scaling_type == "dynamic"
        assert config.max_position_embeddings == 8192

    def test_immutable(self):
        """Test config is frozen."""
        config = RoPEConfig()
        with pytest.raises(ValidationError):
            config.theta = 20000.0

    def test_validation_theta_positive(self):
        """Test theta must be positive."""
        with pytest.raises(ValueError):
            RoPEConfig(theta=0)
        with pytest.raises(ValueError):
            RoPEConfig(theta=-1)


class TestPositionConfig:
    """Tests for PositionConfig."""

    def test_defaults(self):
        """Test default values."""
        config = PositionConfig()
        assert config.position_type == PositionEmbeddingType.ROPE
        assert config.max_position_embeddings == 4096
        # RoPE config should be auto-created
        assert config.rope is not None

    def test_rope_auto_created(self):
        """Test RoPE config is auto-created when using ROPE."""
        config = PositionConfig(position_type=PositionEmbeddingType.ROPE)
        assert config.rope is not None
        assert config.rope.max_position_embeddings == config.max_position_embeddings

    def test_alibi_no_rope(self):
        """Test RoPE config not required for ALiBi."""
        config = PositionConfig(position_type=PositionEmbeddingType.ALIBI)
        # Should not fail, rope can be None for non-ROPE types
        assert config.position_type == PositionEmbeddingType.ALIBI


class TestEmbeddingConfig:
    """Tests for EmbeddingConfig."""

    def test_required_fields(self):
        """Test required fields must be provided."""
        with pytest.raises(ValueError):
            EmbeddingConfig()  # Missing vocab_size and hidden_size

    def test_valid_config(self):
        """Test valid configuration."""
        config = EmbeddingConfig(
            vocab_size=32000,
            hidden_size=4096,
        )
        assert config.vocab_size == 32000
        assert config.hidden_size == 4096
        assert config.padding_idx is None
        assert config.scale_factor is None
        assert config.tie_word_embeddings is True

    def test_validation_positive_values(self):
        """Test vocab_size and hidden_size must be positive."""
        with pytest.raises(ValueError):
            EmbeddingConfig(vocab_size=0, hidden_size=4096)
        with pytest.raises(ValueError):
            EmbeddingConfig(vocab_size=32000, hidden_size=0)


class TestNormConfig:
    """Tests for NormConfig."""

    def test_defaults(self):
        """Test default values."""
        config = NormConfig()
        assert config.norm_type == NormType.RMS_NORM
        assert config.eps == 1e-6
        assert config.elementwise_affine is True

    def test_layer_norm(self):
        """Test LayerNorm configuration."""
        config = NormConfig(norm_type=NormType.LAYER_NORM, eps=1e-5)
        assert config.norm_type == NormType.LAYER_NORM
        assert config.eps == 1e-5


class TestAttentionConfig:
    """Tests for AttentionConfig."""

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValueError):
            AttentionConfig()  # Missing required fields

    def test_valid_mha(self):
        """Test valid MHA configuration."""
        config = AttentionConfig(
            num_attention_heads=32,
            hidden_size=4096,
        )
        assert config.attention_type == AttentionType.GROUPED_QUERY
        assert config.num_attention_heads == 32
        assert config.num_key_value_heads == 32  # Defaults to num_attention_heads
        assert config.head_dim == 128  # 4096 // 32
        assert config.hidden_size == 4096

    def test_gqa_config(self):
        """Test GQA configuration."""
        config = AttentionConfig(
            num_attention_heads=32,
            num_key_value_heads=8,
            hidden_size=4096,
        )
        assert config.num_key_value_heads == 8
        assert config.head_dim == 128

    def test_mqa_config(self):
        """Test MQA configuration (single KV head)."""
        config = AttentionConfig(
            num_attention_heads=32,
            num_key_value_heads=1,
            hidden_size=4096,
        )
        assert config.num_key_value_heads == 1

    def test_sliding_window(self):
        """Test sliding window configuration."""
        config = AttentionConfig(
            attention_type=AttentionType.SLIDING_WINDOW,
            num_attention_heads=32,
            hidden_size=4096,
            sliding_window_size=4096,
        )
        assert config.sliding_window_size == 4096

    def test_custom_head_dim(self):
        """Test custom head dimension."""
        config = AttentionConfig(
            num_attention_heads=32,
            hidden_size=4096,
            head_dim=64,  # Override default
        )
        assert config.head_dim == 64


class TestFFNConfig:
    """Tests for FFNConfig."""

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValueError):
            FFNConfig()

    def test_swiglu_config(self):
        """Test SwiGLU configuration."""
        config = FFNConfig(
            hidden_size=4096,
            intermediate_size=11008,
        )
        assert config.ffn_type == FFNType.SWIGLU
        assert config.activation == ActivationType.SILU
        assert config.bias is False

    def test_moe_config(self):
        """Test MoE configuration."""
        config = FFNConfig(
            ffn_type=FFNType.MOE,
            hidden_size=4096,
            intermediate_size=11008,
            num_experts=8,
            num_experts_per_tok=2,
        )
        assert config.num_experts == 8
        assert config.num_experts_per_tok == 2


class TestSSMConfig:
    """Tests for SSMConfig."""

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValueError):
            SSMConfig()

    def test_mamba_config(self):
        """Test Mamba configuration."""
        config = SSMConfig(hidden_size=768)
        assert config.ssm_type == SSMType.MAMBA
        assert config.hidden_size == 768
        assert config.state_size == 16
        assert config.conv_kernel_size == 4
        assert config.expand_factor == 2


class TestBlockConfig:
    """Tests for BlockConfig."""

    def test_required_fields(self):
        """Test required fields."""
        with pytest.raises(ValueError):
            BlockConfig()

    def test_transformer_block(self):
        """Test transformer block configuration."""
        config = BlockConfig(
            block_type=BlockType.TRANSFORMER,
            hidden_size=4096,
            attention=AttentionConfig(num_attention_heads=32, hidden_size=4096),
            ffn=FFNConfig(hidden_size=4096, intermediate_size=11008),
        )
        assert config.block_type == BlockType.TRANSFORMER
        assert config.pre_norm is True
        assert config.attention is not None
        assert config.ffn is not None

    def test_mamba_block(self):
        """Test Mamba block configuration."""
        config = BlockConfig(
            block_type=BlockType.MAMBA,
            hidden_size=768,
            ssm=SSMConfig(hidden_size=768),
        )
        assert config.block_type == BlockType.MAMBA
        assert config.ssm is not None


class TestBackboneConfig:
    """Tests for BackboneConfig."""

    def test_valid_config(self):
        """Test valid backbone configuration."""
        block = BlockConfig(
            hidden_size=4096,
            attention=AttentionConfig(num_attention_heads=32, hidden_size=4096),
            ffn=FFNConfig(hidden_size=4096, intermediate_size=11008),
        )
        config = BackboneConfig(
            num_hidden_layers=32,
            hidden_size=4096,
            block=block,
        )
        assert config.num_hidden_layers == 32
        assert config.hidden_size == 4096

    def test_hybrid_with_layer_overrides(self):
        """Test hybrid backbone with per-layer overrides."""
        base_block = BlockConfig(
            block_type=BlockType.TRANSFORMER,
            hidden_size=4096,
            attention=AttentionConfig(num_attention_heads=32, hidden_size=4096),
            ffn=FFNConfig(hidden_size=4096, intermediate_size=11008),
        )
        mamba_block = BlockConfig(
            block_type=BlockType.MAMBA,
            hidden_size=4096,
            ssm=SSMConfig(hidden_size=4096),
        )
        config = BackboneConfig(
            num_hidden_layers=4,
            hidden_size=4096,
            block=base_block,
            layer_configs={1: mamba_block, 3: mamba_block},  # Alternate
        )
        assert config.layer_configs is not None
        assert 1 in config.layer_configs
        assert config.layer_configs[1].block_type == BlockType.MAMBA


class TestHeadConfig:
    """Tests for HeadConfig."""

    def test_lm_head(self):
        """Test LM head configuration."""
        config = HeadConfig(
            head_type=HeadType.LM,
            hidden_size=4096,
            vocab_size=32000,
        )
        assert config.head_type == HeadType.LM
        assert config.vocab_size == 32000
        assert config.tie_word_embeddings is True

    def test_classifier_head(self):
        """Test classifier head configuration."""
        config = HeadConfig(
            head_type=HeadType.CLASSIFIER,
            hidden_size=4096,
            num_classes=10,
            pooling=PoolingType.MEAN,
        )
        assert config.head_type == HeadType.CLASSIFIER
        assert config.num_classes == 10
        assert config.pooling == PoolingType.MEAN


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_defaults(self):
        """Test default values."""
        config = ModelConfig()
        assert config.vocab_size == 32000
        assert config.hidden_size == 4096
        assert config.intermediate_size == 11008
        assert config.num_hidden_layers == 32
        assert config.num_attention_heads == 32
        assert config.num_key_value_heads == 32  # Derived
        assert config.head_dim == 128  # Derived
        assert config.max_position_embeddings == 4096
        assert config.rope_theta == 10000.0

    def test_gqa_derived(self):
        """Test GQA settings are properly derived."""
        config = ModelConfig(
            num_attention_heads=32,
            num_key_value_heads=8,
        )
        assert config.num_key_value_heads == 8

    def test_head_dim_derived(self):
        """Test head_dim is computed correctly."""
        config = ModelConfig(
            hidden_size=2048,
            num_attention_heads=16,
        )
        assert config.head_dim == 128  # 2048 // 16

    def test_gemma_hidden_activation(self):
        """Test Gemma's hidden_activation is used."""
        config = ModelConfig(
            hidden_activation="gelu",
            hidden_act="silu",  # Should be overridden
        )
        assert config.hidden_act == "gelu"

    def test_from_file(self, tmp_path):
        """Test loading config from file."""
        config_data = {
            "model_type": "llama",
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
        }
        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        config = ModelConfig.from_file(config_path)
        assert config.model_type == "llama"
        assert config.vocab_size == 32000

    def test_from_file_directory(self, tmp_path):
        """Test loading config from directory (looks for config.json)."""
        config_data = {"model_type": "qwen2", "vocab_size": 151936}
        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        config = ModelConfig.from_file(tmp_path)  # Pass directory
        assert config.model_type == "qwen2"

    def test_save(self, tmp_path):
        """Test saving config to file."""
        config = ModelConfig(model_type="test", vocab_size=50000)
        save_path = tmp_path / "saved_config.json"
        config.save(save_path)

        assert save_path.exists()
        with open(save_path) as f:
            data = json.load(f)
        assert data["model_type"] == "test"
        assert data["vocab_size"] == 50000

    def test_to_embedding_config(self):
        """Test conversion to EmbeddingConfig."""
        config = ModelConfig(vocab_size=32000, hidden_size=4096, pad_token_id=0)
        emb_config = config.to_embedding_config()
        assert emb_config.vocab_size == 32000
        assert emb_config.hidden_size == 4096
        assert emb_config.padding_idx == 0

    def test_to_attention_config(self):
        """Test conversion to AttentionConfig."""
        config = ModelConfig(
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=8,
            rope_theta=500000.0,
        )
        attn_config = config.to_attention_config()
        assert attn_config.hidden_size == 4096
        assert attn_config.num_attention_heads == 32
        assert attn_config.num_key_value_heads == 8
        assert attn_config.attention_type == AttentionType.GROUPED_QUERY

    def test_to_attention_config_mha(self):
        """Test MHA detection in attention config."""
        config = ModelConfig(
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=32,  # Same as num_heads = MHA
        )
        attn_config = config.to_attention_config()
        assert attn_config.attention_type == AttentionType.MULTI_HEAD

    def test_to_attention_config_mqa(self):
        """Test MQA detection in attention config."""
        config = ModelConfig(
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=1,  # Single KV head = MQA
        )
        attn_config = config.to_attention_config()
        assert attn_config.attention_type == AttentionType.MULTI_QUERY

    def test_to_ffn_config(self):
        """Test conversion to FFNConfig."""
        config = ModelConfig(
            hidden_size=4096,
            intermediate_size=11008,
            hidden_act="silu",
        )
        ffn_config = config.to_ffn_config()
        assert ffn_config.hidden_size == 4096
        assert ffn_config.intermediate_size == 11008
        assert ffn_config.ffn_type == FFNType.SWIGLU
        assert ffn_config.activation == ActivationType.SILU

    def test_to_block_config(self):
        """Test conversion to BlockConfig."""
        config = ModelConfig()
        block_config = config.to_block_config()
        assert block_config.block_type == BlockType.TRANSFORMER
        assert block_config.hidden_size == config.hidden_size
        assert block_config.attention is not None
        assert block_config.ffn is not None

    def test_to_backbone_config(self):
        """Test conversion to BackboneConfig."""
        config = ModelConfig(num_hidden_layers=24)
        backbone_config = config.to_backbone_config()
        assert backbone_config.num_hidden_layers == 24
        assert backbone_config.block is not None

    def test_to_head_config(self):
        """Test conversion to HeadConfig."""
        config = ModelConfig(vocab_size=32000, hidden_size=4096)
        head_config = config.to_head_config(HeadType.LM)
        assert head_config.head_type == HeadType.LM
        assert head_config.vocab_size == 32000
        assert head_config.hidden_size == 4096

    def test_extra_fields_ignored(self):
        """Test unknown fields are ignored (HuggingFace compat)."""
        config = ModelConfig(
            model_type="llama",
            unknown_field="should_be_ignored",
            another_unknown=123,
        )
        assert config.model_type == "llama"
        # Should not raise, unknown fields are ignored


@pytest.mark.asyncio
class TestModelConfigAsync:
    """Async tests for ModelConfig."""

    async def test_from_file_async(self, tmp_path):
        """Test async loading config from file."""
        config_data = {"model_type": "mamba", "vocab_size": 50000}
        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        config = await ModelConfig.from_file_async(config_path)
        assert config.model_type == "mamba"
        assert config.vocab_size == 50000

    async def test_save_async(self, tmp_path):
        """Test async saving config to file."""
        config = ModelConfig(model_type="async_test")
        save_path = tmp_path / "async_config.json"
        await config.save_async(save_path)

        assert save_path.exists()
        with open(save_path) as f:
            data = json.load(f)
        assert data["model_type"] == "async_test"
