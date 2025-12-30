"""
Llama configuration.

Extends base ModelConfig with Llama-specific settings.
"""

from __future__ import annotations

from typing import Any

from ...core.config import ModelConfig
from ..constants import ConfigField, DefaultNormEps, DefaultRoPETheta, HFModelType


class LlamaConfig(ModelConfig):
    """
    Configuration for Llama models.

    Extends ModelConfig with Llama-specific defaults and validation.

    Supports:
    - Llama 1/2/3
    - Mistral (with sliding_window)
    - Code Llama
    - Mixtral (via MoE config)

    Example:
        >>> # Llama 2 7B
        >>> config = LlamaConfig(
        ...     vocab_size=32000,
        ...     hidden_size=4096,
        ...     num_hidden_layers=32,
        ...     num_attention_heads=32,
        ...     num_key_value_heads=32,  # MHA for Llama 2 7B
        ...     intermediate_size=11008,
        ... )

        >>> # Llama 3 8B (with GQA)
        >>> config = LlamaConfig(
        ...     vocab_size=128256,
        ...     hidden_size=4096,
        ...     num_hidden_layers=32,
        ...     num_attention_heads=32,
        ...     num_key_value_heads=8,  # GQA
        ...     intermediate_size=14336,
        ... )

        >>> # Mistral 7B (with sliding window)
        >>> config = LlamaConfig(
        ...     vocab_size=32000,
        ...     hidden_size=4096,
        ...     num_hidden_layers=32,
        ...     num_attention_heads=32,
        ...     num_key_value_heads=8,
        ...     intermediate_size=14336,
        ...     sliding_window=4096,
        ... )
    """

    model_type: str = HFModelType.LLAMA.value

    # Llama-specific defaults
    hidden_act: str = "silu"  # SwiGLU activation
    rope_theta: float = DefaultRoPETheta.LLAMA2.value
    rms_norm_eps: float = DefaultNormEps.LLAMA.value

    # Optional: precomputed RoPE scaling
    rope_scaling: dict[str, Any] | None = None

    @classmethod
    def llama2_7b(cls) -> LlamaConfig:
        """Create Llama 2 7B configuration."""
        return cls(
            vocab_size=32000,
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=32,  # MHA
            intermediate_size=11008,
            max_position_embeddings=4096,
        )

    @classmethod
    def llama2_13b(cls) -> LlamaConfig:
        """Create Llama 2 13B configuration."""
        return cls(
            vocab_size=32000,
            hidden_size=5120,
            num_hidden_layers=40,
            num_attention_heads=40,
            num_key_value_heads=40,
            intermediate_size=13824,
            max_position_embeddings=4096,
        )

    @classmethod
    def llama2_70b(cls) -> LlamaConfig:
        """Create Llama 2 70B configuration."""
        return cls(
            vocab_size=32000,
            hidden_size=8192,
            num_hidden_layers=80,
            num_attention_heads=64,
            num_key_value_heads=8,  # GQA
            intermediate_size=28672,
            max_position_embeddings=4096,
        )

    @classmethod
    def llama3_8b(cls) -> LlamaConfig:
        """Create Llama 3 8B configuration."""
        return cls(
            vocab_size=128256,
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,  # GQA
            intermediate_size=14336,
            max_position_embeddings=8192,
            rope_theta=500000.0,  # Extended context
        )

    @classmethod
    def llama3_70b(cls) -> LlamaConfig:
        """Create Llama 3 70B configuration."""
        return cls(
            vocab_size=128256,
            hidden_size=8192,
            num_hidden_layers=80,
            num_attention_heads=64,
            num_key_value_heads=8,
            intermediate_size=28672,
            max_position_embeddings=8192,
            rope_theta=500000.0,
        )

    @classmethod
    def mistral_7b(cls) -> LlamaConfig:
        """Create Mistral 7B configuration."""
        return cls(
            model_type=HFModelType.MISTRAL.value,
            vocab_size=32000,
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,  # GQA
            intermediate_size=14336,
            max_position_embeddings=32768,
            sliding_window=4096,  # Sliding window attention
        )

    @classmethod
    def code_llama_7b(cls) -> LlamaConfig:
        """Create Code Llama 7B configuration."""
        return cls(
            model_type=HFModelType.CODELLAMA.value,
            vocab_size=32016,  # Extended for code tokens
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=32,
            intermediate_size=11008,
            max_position_embeddings=16384,  # Extended context
            rope_theta=1000000.0,  # RoPE scaling for long context
        )

    @classmethod
    def tiny(cls) -> LlamaConfig:
        """Create tiny Llama for testing."""
        return cls(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
            max_position_embeddings=256,
        )

    @classmethod
    def from_hf_config(
        cls,
        hf_config: dict[str, Any],
        weights: dict[str, Any] | None = None,
    ) -> LlamaConfig:
        """
        Create config from HuggingFace config.json dict.

        Handles both standard HuggingFace format and mlx-community format:
        - HF: hidden_size, num_hidden_layers, num_attention_heads, intermediate_size
        - MLX: dim, n_layers, n_heads, hidden_dim

        Args:
            hf_config: Dict loaded from config.json
            weights: Optional weights dict (not used for Llama)

        Returns:
            LlamaConfig instance

        Example:
            >>> import json
            >>> with open("config.json") as f:
            ...     hf_config = json.load(f)
            >>> config = LlamaConfig.from_hf_config(hf_config)
        """
        # Detect mlx-community format (uses n_layers, dim, etc.)
        is_mlx_format = "n_layers" in hf_config or "dim" in hf_config

        if is_mlx_format:
            # MLX community format mapping
            hidden_size = hf_config.get("dim", hf_config.get(ConfigField.HIDDEN_SIZE.value))
            num_hidden_layers = hf_config.get(
                "n_layers", hf_config.get(ConfigField.NUM_HIDDEN_LAYERS.value)
            )
            num_attention_heads = hf_config.get(
                "n_heads", hf_config.get(ConfigField.NUM_ATTENTION_HEADS.value)
            )
            num_key_value_heads = hf_config.get(
                "n_kv_heads",
                hf_config.get(ConfigField.NUM_KEY_VALUE_HEADS.value, num_attention_heads),
            )
            intermediate_size = hf_config.get(
                "hidden_dim", hf_config.get(ConfigField.INTERMEDIATE_SIZE.value)
            )
            rms_norm_eps = hf_config.get(
                "norm_eps",
                hf_config.get(ConfigField.RMS_NORM_EPS.value, DefaultNormEps.LLAMA.value),
            )
            # head_dim is extracted but currently unused (computed from hidden_size/heads)
            _ = hf_config.get("head_dim")
        else:
            # Standard HuggingFace format
            hidden_size = hf_config[ConfigField.HIDDEN_SIZE.value]
            num_hidden_layers = hf_config[ConfigField.NUM_HIDDEN_LAYERS.value]
            num_attention_heads = hf_config[ConfigField.NUM_ATTENTION_HEADS.value]
            num_key_value_heads = hf_config.get(
                ConfigField.NUM_KEY_VALUE_HEADS.value, num_attention_heads
            )
            intermediate_size = hf_config[ConfigField.INTERMEDIATE_SIZE.value]
            rms_norm_eps = hf_config.get(ConfigField.RMS_NORM_EPS.value, DefaultNormEps.LLAMA.value)

        return cls(
            model_type=hf_config.get(ConfigField.MODEL_TYPE.value, HFModelType.LLAMA.value),
            vocab_size=hf_config[ConfigField.VOCAB_SIZE.value],
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=hf_config.get(ConfigField.MAX_POSITION_EMBEDDINGS.value, 4096),
            rope_theta=hf_config.get(ConfigField.ROPE_THETA.value, DefaultRoPETheta.LLAMA2.value),
            rms_norm_eps=rms_norm_eps,
            sliding_window=hf_config.get(ConfigField.SLIDING_WINDOW.value),
            rope_scaling=hf_config.get("rope_scaling"),
            tie_word_embeddings=hf_config.get(ConfigField.TIE_WORD_EMBEDDINGS.value, False),
            bos_token_id=hf_config.get(ConfigField.BOS_TOKEN_ID.value, 1),
            eos_token_id=hf_config.get(ConfigField.EOS_TOKEN_ID.value, 2),
            pad_token_id=hf_config.get(ConfigField.PAD_TOKEN_ID.value),
        )
