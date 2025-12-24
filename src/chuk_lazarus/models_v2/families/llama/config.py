"""
Llama configuration.

Extends base ModelConfig with Llama-specific settings.
"""

from __future__ import annotations

from typing import Any

from ...core.config import ModelConfig


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

    model_type: str = "llama"

    # Llama-specific defaults
    hidden_act: str = "silu"  # SwiGLU activation
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5

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
            model_type="mistral",
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
            model_type="code_llama",
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
