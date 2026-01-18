"""
GPT-OSS-Lite configuration.

GPT-OSS-Lite is a reduced-expert variant of GPT-OSS-20B with:
- 24 layers with alternating sliding/full attention
- Variable experts per layer (typically 16 for production, 4-8 for minimal)
- YaRN RoPE scaling
- Custom SwiGLU activation
"""

from __future__ import annotations

from typing import Any

from ...core.config import ModelConfig
from ..constants import ConfigField, DefaultNormEps, HFModelType


class GptOssLiteConfig(ModelConfig):
    """
    Configuration for GPT-OSS-Lite models.

    GPT-OSS-Lite is a pruned Mixture-of-Experts model derived from GPT-OSS-20B.
    Key changes from GPT-OSS:
    - Reduced experts per layer (4-16 vs 32 in original)
    - Same k=4 routing (always activate 4 experts per token)
    - Same custom SwiGLU activation and YaRN RoPE

    Production configuration (16 experts/layer):
    - 50% expert reduction
    - 8.1 GB storage (vs 10 GB original)
    - Full quality preserved (including math: 2+2=4)

    Minimal configuration (4 experts/layer):
    - 87.5% expert reduction
    - 4.5 GB storage
    - Quality degraded (language repetition, math: 2+2=3)
    """

    model_type: str = "gpt_oss_lite"

    # GPT-OSS-Lite-specific defaults
    hidden_act: str = "silu"  # Custom SwiGLU activation
    rope_theta: float = 150000.0
    rms_norm_eps: float = DefaultNormEps.LLAMA.value
    attention_bias: bool = True

    # MoE configuration - variable per layer
    num_local_experts: int = 16  # Default to 16 (production quality)
    num_experts_per_tok: int = 4  # Always 4

    # Layer types for hybrid attention
    layer_types: list[str] | None = None  # ["sliding_attention", "full_attention", ...]

    # Sliding window attention
    sliding_window: int = 128

    # RoPE scaling (YaRN)
    rope_scaling: dict[str, Any] | None = None

    # SwiGLU limit (clipping)
    swiglu_limit: float | None = 7.0

    # Lite-specific metadata
    source_model: str | None = "openai/gpt-oss-20b"
    reduction_percent: str | None = None
    original_experts: int = 768
    total_experts: int | None = None

    @property
    def is_moe(self) -> bool:
        """Check if this is a MoE model."""
        return self.num_local_experts > 1

    def get_num_experts(self, layer_idx: int) -> int:
        """Get number of experts for a specific layer.

        For uniform configurations (most common), returns num_local_experts.
        Override with experts_per_layer dict for variable configurations.
        """
        return self.num_local_experts

    def get_layer_type(self, layer_idx: int) -> str:
        """Get the attention type for a specific layer."""
        if self.layer_types is None:
            # Default: alternate sliding/full
            return "sliding_attention" if layer_idx % 2 == 0 else "full_attention"
        return self.layer_types[layer_idx % len(self.layer_types)]

    @classmethod
    def lite_16exp(cls) -> GptOssLiteConfig:
        """Create GPT-OSS-Lite with 16 experts/layer (production quality)."""
        return cls(
            vocab_size=201088,
            hidden_size=2880,
            num_hidden_layers=24,
            num_attention_heads=64,
            num_key_value_heads=8,
            intermediate_size=2880,
            max_position_embeddings=131072,
            head_dim=64,
            num_local_experts=16,
            num_experts_per_tok=4,
            sliding_window=128,
            rope_theta=150000.0,
            rope_scaling={
                "rope_type": "yarn",
                "factor": 32.0,
                "original_max_position_embeddings": 4096,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
            },
            tie_word_embeddings=False,
            reduction_percent="50.0%",
            total_experts=384,
        )

    @classmethod
    def lite_minimal(cls) -> GptOssLiteConfig:
        """Create GPT-OSS-Lite with 4 experts/layer (minimal, quality degraded)."""
        return cls(
            vocab_size=201088,
            hidden_size=2880,
            num_hidden_layers=24,
            num_attention_heads=64,
            num_key_value_heads=8,
            intermediate_size=2880,
            max_position_embeddings=131072,
            head_dim=64,
            num_local_experts=4,
            num_experts_per_tok=4,
            sliding_window=128,
            rope_theta=150000.0,
            rope_scaling={
                "rope_type": "yarn",
                "factor": 32.0,
                "original_max_position_embeddings": 4096,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
            },
            tie_word_embeddings=False,
            reduction_percent="87.5%",
            total_experts=96,
        )

    @classmethod
    def tiny(cls) -> GptOssLiteConfig:
        """Create tiny GPT-OSS-Lite for testing."""
        return cls(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=64,
            max_position_embeddings=256,
            head_dim=16,
            num_local_experts=4,
            num_experts_per_tok=2,
        )

    @classmethod
    def from_hf_config(
        cls,
        hf_config: dict[str, Any],
        weights: dict[str, Any] | None = None,
    ) -> GptOssLiteConfig:
        """
        Create config from HuggingFace config.json dict.

        Args:
            hf_config: Dict loaded from config.json
            weights: Optional weights dict (not used)

        Returns:
            GptOssLiteConfig instance
        """
        return cls(
            model_type=hf_config.get(ConfigField.MODEL_TYPE.value, "gpt_oss_lite"),
            vocab_size=hf_config[ConfigField.VOCAB_SIZE.value],
            hidden_size=hf_config[ConfigField.HIDDEN_SIZE.value],
            num_hidden_layers=hf_config[ConfigField.NUM_HIDDEN_LAYERS.value],
            num_attention_heads=hf_config[ConfigField.NUM_ATTENTION_HEADS.value],
            num_key_value_heads=hf_config.get(
                ConfigField.NUM_KEY_VALUE_HEADS.value,
                hf_config[ConfigField.NUM_ATTENTION_HEADS.value],
            ),
            intermediate_size=hf_config[ConfigField.INTERMEDIATE_SIZE.value],
            max_position_embeddings=hf_config.get(
                ConfigField.MAX_POSITION_EMBEDDINGS.value, 131072
            ),
            head_dim=hf_config.get(ConfigField.HEAD_DIM.value, 64),
            rope_theta=hf_config.get(ConfigField.ROPE_THETA.value, 150000.0),
            rms_norm_eps=hf_config.get(ConfigField.RMS_NORM_EPS.value, DefaultNormEps.LLAMA.value),
            # MoE config
            num_local_experts=hf_config.get("num_local_experts", 16),
            num_experts_per_tok=hf_config.get("num_experts_per_tok", 4),
            # Layer types
            layer_types=hf_config.get("layer_types"),
            sliding_window=hf_config.get(ConfigField.SLIDING_WINDOW.value, 128),
            # RoPE scaling
            rope_scaling=hf_config.get("rope_scaling"),
            # SwiGLU limit
            swiglu_limit=hf_config.get("swiglu_limit", 7.0),
            # Attention bias
            attention_bias=hf_config.get("attention_bias", True),
            # Token IDs
            tie_word_embeddings=hf_config.get(ConfigField.TIE_WORD_EMBEDDINGS.value, False),
            bos_token_id=hf_config.get(ConfigField.BOS_TOKEN_ID.value),
            eos_token_id=hf_config.get(ConfigField.EOS_TOKEN_ID.value, 200002),
            pad_token_id=hf_config.get(ConfigField.PAD_TOKEN_ID.value, 199999),
            # Lite-specific
            source_model=hf_config.get("source_model", "openai/gpt-oss-20b"),
            reduction_percent=hf_config.get("reduction_percent"),
            original_experts=hf_config.get("original_experts", 768),
            total_experts=hf_config.get("total_experts"),
        )
