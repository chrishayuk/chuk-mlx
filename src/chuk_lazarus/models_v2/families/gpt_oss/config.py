"""
GPT-OSS configuration.

GPT-OSS is OpenAI's open-source MoE model with:
- 24 layers with alternating sliding/full attention
- 32 experts, 4 active per token
- YaRN RoPE scaling
- SwiGLU activation
"""

from __future__ import annotations

from typing import Any

from ...core.config import ModelConfig
from ..constants import ConfigField, DefaultNormEps, HFModelType


class GptOssConfig(ModelConfig):
    """
    Configuration for GPT-OSS models.

    GPT-OSS is a Mixture-of-Experts model with:
    - Alternating sliding window and full attention layers
    - 32 local experts with 4 active per token
    - YaRN RoPE scaling for extended context

    Model specs (gpt-oss-20b):
    - 24 layers
    - hidden_size=2880
    - intermediate_size=2880 (per expert)
    - 64 attention heads, 8 KV heads
    - 32 experts, 4 active
    - 21B total params, 3.6B active
    """

    model_type: str = HFModelType.GPT_OSS.value

    # GPT-OSS-specific defaults
    hidden_act: str = "silu"  # SwiGLU activation
    rope_theta: float = 150000.0
    rms_norm_eps: float = DefaultNormEps.LLAMA.value
    attention_bias: bool = True

    # MoE configuration
    num_local_experts: int = 32
    num_experts_per_tok: int = 4

    # Layer types for hybrid attention
    layer_types: list[str] | None = None  # ["sliding_attention", "full_attention", ...]

    # Sliding window attention
    sliding_window: int = 128

    # RoPE scaling (YaRN)
    rope_scaling: dict[str, Any] | None = None

    # SwiGLU limit (clipping)
    swiglu_limit: float | None = None

    @property
    def is_moe(self) -> bool:
        """Check if this is a MoE model."""
        return self.num_local_experts > 1

    def get_layer_type(self, layer_idx: int) -> str:
        """Get the attention type for a specific layer."""
        if self.layer_types is None:
            # Default: alternate sliding/full
            return "sliding_attention" if layer_idx % 2 == 0 else "full_attention"
        return self.layer_types[layer_idx % len(self.layer_types)]

    @classmethod
    def gpt_oss_20b(cls) -> GptOssConfig:
        """Create GPT-OSS-20B configuration."""
        return cls(
            vocab_size=201088,
            hidden_size=2880,
            num_hidden_layers=24,
            num_attention_heads=64,
            num_key_value_heads=8,
            intermediate_size=2880,
            max_position_embeddings=131072,
            head_dim=64,
            num_local_experts=32,
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
            tie_word_embeddings=False,  # GPT-OSS has separate lm_head weights
        )

    @classmethod
    def tiny(cls) -> GptOssConfig:
        """Create tiny GPT-OSS for testing."""
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
    ) -> GptOssConfig:
        """
        Create config from HuggingFace config.json dict.

        Args:
            hf_config: Dict loaded from config.json
            weights: Optional weights dict (not used)

        Returns:
            GptOssConfig instance
        """
        return cls(
            model_type=hf_config.get(ConfigField.MODEL_TYPE.value, HFModelType.GPT_OSS.value),
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
            num_local_experts=hf_config.get("num_local_experts", 32),
            num_experts_per_tok=hf_config.get("num_experts_per_tok", 4),
            # Layer types
            layer_types=hf_config.get("layer_types"),
            sliding_window=hf_config.get(ConfigField.SLIDING_WINDOW.value, 128),
            # RoPE scaling
            rope_scaling=hf_config.get("rope_scaling"),
            # SwiGLU limit
            swiglu_limit=hf_config.get("swiglu_limit"),
            # Attention bias
            attention_bias=hf_config.get("attention_bias", True),
            # Token IDs
            tie_word_embeddings=hf_config.get(ConfigField.TIE_WORD_EMBEDDINGS.value, False),
            bos_token_id=hf_config.get(ConfigField.BOS_TOKEN_ID.value),
            eos_token_id=hf_config.get(ConfigField.EOS_TOKEN_ID.value, 200002),
            pad_token_id=hf_config.get(ConfigField.PAD_TOKEN_ID.value, 199999),
        )
