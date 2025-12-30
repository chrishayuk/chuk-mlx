"""
Gemma 3 configuration.

Extends base ModelConfig with Gemma-specific settings.
"""

from __future__ import annotations

from typing import Any

from ...core.config import ModelConfig
from ..constants import ConfigField, DefaultNormEps, DefaultRoPETheta, HFModelType


class GemmaConfig(ModelConfig):
    """
    Configuration for Gemma 3 models.

    Extends ModelConfig with Gemma-specific features:
    - Alternating sliding window / global attention pattern
    - Query/key pre-attention normalization
    - GELU activation (gated)
    - 4 normalization layers per block

    Supports:
    - Gemma 3 270M (FunctionGemma base)
    - Gemma 3 1B
    - Gemma 3 4B
    - Gemma 3 12B
    - Gemma 3 27B

    Example:
        >>> config = GemmaConfig.gemma3_270m()
        >>> model = GemmaForCausalLM(config)
    """

    model_type: str = HFModelType.GEMMA3_TEXT.value

    # Gemma-specific: head dimension (can differ from hidden_size / num_heads)
    head_dim: int = 256

    # Gemma-specific: query pre-attention scalar for attention scaling
    query_pre_attn_scalar: float = 256.0

    # Gemma-specific: sliding window attention pattern
    sliding_window: int = 512
    sliding_window_pattern: int = 6  # Every 6th layer is global attention

    # Gemma-specific: separate RoPE base for local (sliding) attention
    rope_local_base_freq: float = DefaultRoPETheta.LLAMA2.value

    # Gemma defaults
    hidden_act: str = "gelu_pytorch_tanh"  # Gated GELU
    rope_theta: float = DefaultRoPETheta.GEMMA3.value
    rms_norm_eps: float = DefaultNormEps.GEMMA.value

    # Optional: RoPE scaling configuration
    rope_scaling: dict[str, Any] | None = None

    @classmethod
    def gemma3_270m(cls) -> GemmaConfig:
        """
        Create Gemma 3 270M configuration.

        This is the base for FunctionGemma.
        """
        return cls(
            model_type=HFModelType.GEMMA3_TEXT.value,
            vocab_size=262144,
            hidden_size=640,
            num_hidden_layers=18,
            num_attention_heads=4,
            num_key_value_heads=1,  # MQA
            intermediate_size=2048,
            head_dim=256,
            query_pre_attn_scalar=256.0,
            sliding_window=512,
            sliding_window_pattern=6,
            max_position_embeddings=32768,
            rope_theta=DefaultRoPETheta.GEMMA3.value,
            rope_local_base_freq=DefaultRoPETheta.LLAMA2.value,
            rms_norm_eps=DefaultNormEps.GEMMA.value,
        )

    @classmethod
    def functiongemma_270m(cls) -> GemmaConfig:
        """
        Create FunctionGemma 270M configuration.

        Same architecture as Gemma 3 270M, tuned for function calling.
        """
        return cls.gemma3_270m()

    @classmethod
    def gemma3_1b(cls) -> GemmaConfig:
        """Create Gemma 3 1B configuration."""
        return cls(
            model_type=HFModelType.GEMMA3_TEXT.value,
            vocab_size=262144,
            hidden_size=1152,
            num_hidden_layers=26,
            num_attention_heads=4,
            num_key_value_heads=1,
            intermediate_size=6912,
            head_dim=256,
            query_pre_attn_scalar=256.0,
            sliding_window=512,
            sliding_window_pattern=6,
            max_position_embeddings=32768,
            rope_theta=DefaultRoPETheta.GEMMA3.value,
            rope_local_base_freq=DefaultRoPETheta.LLAMA2.value,
            rms_norm_eps=DefaultNormEps.GEMMA.value,
        )

    @classmethod
    def gemma3_4b(cls) -> GemmaConfig:
        """Create Gemma 3 4B configuration."""
        return cls(
            model_type=HFModelType.GEMMA3_TEXT.value,
            vocab_size=262144,
            hidden_size=2560,
            num_hidden_layers=34,
            num_attention_heads=8,
            num_key_value_heads=4,
            intermediate_size=10240,
            head_dim=256,
            query_pre_attn_scalar=256.0,
            sliding_window=512,
            sliding_window_pattern=6,
            max_position_embeddings=131072,
            rope_theta=DefaultRoPETheta.GEMMA3.value,
            rope_local_base_freq=DefaultRoPETheta.LLAMA2.value,
            rms_norm_eps=DefaultNormEps.GEMMA.value,
        )

    @classmethod
    def gemma3_12b(cls) -> GemmaConfig:
        """Create Gemma 3 12B configuration."""
        return cls(
            model_type=HFModelType.GEMMA3_TEXT.value,
            vocab_size=262144,
            hidden_size=3840,
            num_hidden_layers=48,
            num_attention_heads=16,
            num_key_value_heads=8,
            intermediate_size=15360,
            head_dim=256,
            query_pre_attn_scalar=256.0,
            sliding_window=512,
            sliding_window_pattern=6,
            max_position_embeddings=131072,
            rope_theta=DefaultRoPETheta.GEMMA3.value,
            rope_local_base_freq=DefaultRoPETheta.LLAMA2.value,
            rms_norm_eps=DefaultNormEps.GEMMA.value,
        )

    @classmethod
    def gemma3_27b(cls) -> GemmaConfig:
        """Create Gemma 3 27B configuration."""
        return cls(
            model_type=HFModelType.GEMMA3_TEXT.value,
            vocab_size=262144,
            hidden_size=5120,
            num_hidden_layers=62,
            num_attention_heads=24,
            num_key_value_heads=8,
            intermediate_size=20480,
            head_dim=256,
            query_pre_attn_scalar=256.0,
            sliding_window=512,
            sliding_window_pattern=6,
            max_position_embeddings=131072,
            rope_theta=DefaultRoPETheta.GEMMA3.value,
            rope_local_base_freq=DefaultRoPETheta.LLAMA2.value,
            rms_norm_eps=DefaultNormEps.GEMMA.value,
        )

    @classmethod
    def tiny(cls) -> GemmaConfig:
        """Create tiny Gemma for testing."""
        return cls(
            model_type=HFModelType.GEMMA3_TEXT.value,
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=6,  # Needs to be divisible by pattern
            num_attention_heads=2,
            num_key_value_heads=1,
            intermediate_size=128,
            head_dim=32,
            query_pre_attn_scalar=32.0,
            sliding_window=64,
            sliding_window_pattern=3,  # Every 3rd is global
            max_position_embeddings=256,
            rope_theta=DefaultRoPETheta.LLAMA2.value,
            rope_local_base_freq=DefaultRoPETheta.LLAMA2.value,
            rms_norm_eps=DefaultNormEps.GEMMA.value,
        )

    @property
    def embedding_scale(self) -> float:
        """
        Gemma scales embeddings by sqrt(hidden_size).

        This is critical for correct logit lens and introspection results.
        """
        return self.hidden_size**0.5

    def is_sliding_layer(self, layer_idx: int) -> bool:
        """Check if a layer uses sliding window attention."""
        return (layer_idx + 1) % self.sliding_window_pattern != 0

    def is_global_layer(self, layer_idx: int) -> bool:
        """Check if a layer uses global (full) attention."""
        return not self.is_sliding_layer(layer_idx)

    @classmethod
    def from_hf_config(
        cls,
        hf_config: dict[str, Any],
        weights: dict[str, Any] | None = None,
    ) -> GemmaConfig:
        """
        Create config from HuggingFace config.json dict.

        Handles both text-only (gemma3_text) and vision-language (gemma3) configs.
        For VLM configs, extracts settings from the nested 'text_config'.

        Args:
            hf_config: Dict loaded from config.json
            weights: Optional weights dict (not used for Gemma, config has all info)

        Returns:
            GemmaConfig instance

        Example:
            >>> import json
            >>> with open("config.json") as f:
            ...     hf_config = json.load(f)
            >>> config = GemmaConfig.from_hf_config(hf_config)
        """
        # Handle VLM configs with nested text_config
        # Gemma 3 VLM models (gemma-3-4b-it, etc.) have config nested under "text_config"
        if "text_config" in hf_config and ConfigField.VOCAB_SIZE.value not in hf_config:
            text_config = hf_config["text_config"]
            # Merge text_config into main config, with text_config taking precedence for model params
            merged_config = {**hf_config, **text_config}
            hf_config = merged_config

        # Get head_dim (usually 256 for Gemma 3)
        head_dim = hf_config.get(ConfigField.HEAD_DIM.value, 256)

        # Infer num_attention_heads from weights if not in config
        num_attention_heads = hf_config.get(ConfigField.NUM_ATTENTION_HEADS.value)
        num_key_value_heads = hf_config.get(ConfigField.NUM_KEY_VALUE_HEADS.value)

        if num_attention_heads is None and weights is not None:
            # Try to infer from q_proj weight shape
            # Look for q_proj in weights (may have language_model. prefix)
            for key, weight in weights.items():
                if "layers.0.self_attn.q_proj.weight" in key:
                    q_out_dim = weight.shape[0]
                    num_attention_heads = q_out_dim // head_dim
                    break

        if num_key_value_heads is None and weights is not None:
            # Try to infer from k_proj weight shape
            for key, weight in weights.items():
                if "layers.0.self_attn.k_proj.weight" in key:
                    k_out_dim = weight.shape[0]
                    num_key_value_heads = k_out_dim // head_dim
                    break

        # Fallback defaults based on hidden_size if still not found
        hidden_size = hf_config[ConfigField.HIDDEN_SIZE.value]
        if num_attention_heads is None:
            # Use known Gemma 3 configurations
            gemma3_head_config = {
                1152: (4, 4),  # 270M, 1B
                2560: (8, 4),  # 4B
                3840: (16, 8),  # 12B
                4096: (32, 16),  # 27B
            }
            if hidden_size in gemma3_head_config:
                num_attention_heads, num_key_value_heads = gemma3_head_config[hidden_size]
            else:
                # Last resort: guess based on typical ratios
                num_attention_heads = max(4, hidden_size // 320)
                num_key_value_heads = num_attention_heads // 2

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        return cls(
            model_type=hf_config.get(ConfigField.MODEL_TYPE.value, HFModelType.GEMMA3_TEXT.value),
            vocab_size=hf_config.get(ConfigField.VOCAB_SIZE.value, 262144),  # Default Gemma 3 vocab
            hidden_size=hidden_size,
            num_hidden_layers=hf_config[ConfigField.NUM_HIDDEN_LAYERS.value],
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            intermediate_size=hf_config[ConfigField.INTERMEDIATE_SIZE.value],
            head_dim=head_dim,
            query_pre_attn_scalar=hf_config.get(ConfigField.QUERY_PRE_ATTN_SCALAR.value, 256.0),
            sliding_window=hf_config.get(ConfigField.SLIDING_WINDOW.value, 512),
            sliding_window_pattern=hf_config.get(ConfigField.SLIDING_WINDOW_PATTERN.value, 6),
            max_position_embeddings=hf_config.get(ConfigField.MAX_POSITION_EMBEDDINGS.value, 32768),
            rope_theta=hf_config.get(ConfigField.ROPE_THETA.value, DefaultRoPETheta.GEMMA3.value),
            rope_local_base_freq=hf_config.get(
                "rope_local_base_freq", DefaultRoPETheta.LLAMA2.value
            ),
            rms_norm_eps=hf_config.get(ConfigField.RMS_NORM_EPS.value, DefaultNormEps.GEMMA.value),
            rope_scaling=hf_config.get("rope_scaling"),
        )
