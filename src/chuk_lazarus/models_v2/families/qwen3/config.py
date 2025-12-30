"""
Qwen3 configuration.
"""

from __future__ import annotations

from typing import Any

from ...core.config import ModelConfig
from ..constants import ConfigField, DefaultNormEps, DefaultRoPETheta, HFModelType


class Qwen3Config(ModelConfig):
    """
    Configuration for Qwen3 models.

    Qwen3 is similar to Llama but with:
    - Bias on QKV projections
    - Different default values

    Sizes:
    - Qwen3-0.6B: 28 layers, hidden=1024
    - Qwen3-1.7B: 28 layers, hidden=2048
    - Qwen3-4B: 36 layers, hidden=3584
    - Qwen3-8B: 36 layers, hidden=4096
    """

    model_type: str = HFModelType.QWEN3.value

    # Qwen3-specific defaults
    hidden_act: str = "silu"
    rope_theta: float = DefaultRoPETheta.GEMMA3.value
    rms_norm_eps: float = DefaultNormEps.GEMMA.value
    attention_bias: bool = False  # Qwen3 does NOT have bias on QKV (older Qwen versions did)
    mlp_bias: bool = False

    # RoPE scaling
    rope_scaling: dict[str, Any] | None = None

    @classmethod
    def from_hf_config(
        cls,
        hf_config: dict[str, Any],
        weights: dict[str, Any] | None = None,
    ) -> Qwen3Config:
        """Create config from HuggingFace config.json dict."""
        return cls(
            model_type=hf_config.get(ConfigField.MODEL_TYPE.value, HFModelType.QWEN3.value),
            vocab_size=hf_config[ConfigField.VOCAB_SIZE.value],
            hidden_size=hf_config[ConfigField.HIDDEN_SIZE.value],
            num_hidden_layers=hf_config[ConfigField.NUM_HIDDEN_LAYERS.value],
            num_attention_heads=hf_config[ConfigField.NUM_ATTENTION_HEADS.value],
            num_key_value_heads=hf_config.get(
                ConfigField.NUM_KEY_VALUE_HEADS.value,
                hf_config[ConfigField.NUM_ATTENTION_HEADS.value],
            ),
            head_dim=hf_config.get(ConfigField.HEAD_DIM.value),  # Qwen3 uses explicit head_dim
            intermediate_size=hf_config[ConfigField.INTERMEDIATE_SIZE.value],
            max_position_embeddings=hf_config.get(ConfigField.MAX_POSITION_EMBEDDINGS.value, 32768),
            rope_theta=hf_config.get(ConfigField.ROPE_THETA.value, DefaultRoPETheta.GEMMA3.value),
            rms_norm_eps=hf_config.get(ConfigField.RMS_NORM_EPS.value, DefaultNormEps.GEMMA.value),
            rope_scaling=hf_config.get("rope_scaling"),
            tie_word_embeddings=hf_config.get(ConfigField.TIE_WORD_EMBEDDINGS.value, True),
            bos_token_id=hf_config.get(ConfigField.BOS_TOKEN_ID.value),
            eos_token_id=hf_config.get(ConfigField.EOS_TOKEN_ID.value),
            pad_token_id=hf_config.get(ConfigField.PAD_TOKEN_ID.value),
            attention_bias=hf_config.get("attention_bias", True),
            mlp_bias=hf_config.get("mlp_bias", False),
        )
