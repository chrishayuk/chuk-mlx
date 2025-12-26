"""
StarCoder and StarCoder2 configuration.

Extends base ModelConfig with StarCoder-specific settings.

StarCoder (original) uses:
- LayerNorm (not RMSNorm)
- GELU activation (not SiLU/SwiGLU)
- Bias in linear layers
- Multi-Query Attention (MQA)
- Learned positional embeddings (like GPT-2)
- 8K context window

StarCoder2 uses:
- LayerNorm (not RMSNorm)
- GELU activation (not SiLU/SwiGLU)
- Bias in linear layers
- Grouped Query Attention
- Sliding window attention
- RoPE positional embeddings
- 16K context window
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from ...core.config import ModelConfig


class StarCoderConfig(ModelConfig):
    """
    Configuration for StarCoder (original) models.

    StarCoder uses GPT-2 architecture with Multi-Query Attention (MQA)
    and learned positional embeddings.

    Supports:
    - StarCoder (15.5B)
    - StarCoderBase (15.5B, not Python-finetuned)
    - SantaCoder (1.1B)

    Example:
        >>> # StarCoder 15.5B
        >>> config = StarCoderConfig(
        ...     vocab_size=49152,
        ...     hidden_size=6144,
        ...     num_hidden_layers=40,
        ...     num_attention_heads=48,
        ...     num_key_value_heads=1,  # MQA
        ...     intermediate_size=24576,
        ... )
    """

    model_type: str = "gpt_bigcode"

    # StarCoder-specific defaults
    hidden_act: str = "gelu_pytorch_tanh"  # GELU activation
    layer_norm_eps: float = 1e-5  # LayerNorm epsilon
    rms_norm_eps: float = 1e-5  # Keep for compat

    # StarCoder uses bias in projections
    attention_bias: bool = True
    mlp_bias: bool = True

    # No sliding window for original StarCoder
    sliding_window: int | None = None

    # Use learned position embeddings (GPT-2 style), not RoPE
    use_learned_position_embeddings: bool = Field(default=True)

    # MQA: all queries share single key-value head
    # For MQA, num_key_value_heads = 1
    multi_query: bool = Field(default=True)

    @classmethod
    def starcoder(cls) -> StarCoderConfig:
        """Create StarCoder 15.5B configuration."""
        return cls(
            vocab_size=49152,
            hidden_size=6144,
            num_hidden_layers=40,
            num_attention_heads=48,
            num_key_value_heads=1,  # MQA
            intermediate_size=24576,
            max_position_embeddings=8192,
            multi_query=True,
        )

    @classmethod
    def starcoderbase(cls) -> StarCoderConfig:
        """Create StarCoderBase 15.5B configuration (not Python-finetuned)."""
        return cls.starcoder()  # Same architecture

    @classmethod
    def santacoder(cls) -> StarCoderConfig:
        """Create SantaCoder 1.1B configuration."""
        return cls(
            vocab_size=49280,  # SantaCoder has slightly different vocab
            hidden_size=2048,
            num_hidden_layers=24,
            num_attention_heads=16,
            num_key_value_heads=1,  # MQA
            intermediate_size=8192,
            max_position_embeddings=2048,
            multi_query=True,
        )

    @classmethod
    def tiny(cls) -> StarCoderConfig:
        """Create tiny StarCoder for testing."""
        return cls(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=1,  # MQA
            intermediate_size=128,
            max_position_embeddings=256,
            multi_query=True,
        )


class StarCoder2Config(ModelConfig):
    """
    Configuration for StarCoder2 models.

    Extends ModelConfig with StarCoder2-specific defaults and validation.

    Supports:
    - StarCoder2 3B
    - StarCoder2 7B
    - StarCoder2 15B

    Example:
        >>> # StarCoder2 3B
        >>> config = StarCoder2Config(
        ...     vocab_size=49152,
        ...     hidden_size=3072,
        ...     num_hidden_layers=30,
        ...     num_attention_heads=24,
        ...     num_key_value_heads=2,
        ...     intermediate_size=12288,
        ... )

        >>> # StarCoder2 7B
        >>> config = StarCoder2Config(
        ...     vocab_size=49152,
        ...     hidden_size=4608,
        ...     num_hidden_layers=32,
        ...     num_attention_heads=36,
        ...     num_key_value_heads=4,
        ...     intermediate_size=18432,
        ... )

        >>> # StarCoder2 15B
        >>> config = StarCoder2Config(
        ...     vocab_size=49152,
        ...     hidden_size=6144,
        ...     num_hidden_layers=40,
        ...     num_attention_heads=48,
        ...     num_key_value_heads=4,
        ...     intermediate_size=24576,
        ... )
    """

    model_type: str = "starcoder2"

    # StarCoder2-specific defaults
    hidden_act: str = "gelu_pytorch_tanh"  # GELU activation
    rope_theta: float = 100000.0  # Extended RoPE base
    layer_norm_eps: float = 1e-5  # LayerNorm epsilon (not rms_norm_eps)
    rms_norm_eps: float = 1e-5  # Keep for compat but use layer_norm_eps

    # StarCoder2 uses bias in projections
    attention_bias: bool = True
    mlp_bias: bool = True

    # Sliding window (default for StarCoder2)
    sliding_window: int | None = 4096

    # Optional: precomputed RoPE scaling
    rope_scaling: dict[str, Any] | None = None

    @classmethod
    def starcoder2_3b(cls) -> StarCoder2Config:
        """Create StarCoder2 3B configuration."""
        return cls(
            vocab_size=49152,
            hidden_size=3072,
            num_hidden_layers=30,
            num_attention_heads=24,
            num_key_value_heads=2,  # GQA: 24 heads / 2 kv_heads = 12x ratio
            intermediate_size=12288,
            max_position_embeddings=16384,
            sliding_window=4096,
        )

    @classmethod
    def starcoder2_7b(cls) -> StarCoder2Config:
        """Create StarCoder2 7B configuration."""
        return cls(
            vocab_size=49152,
            hidden_size=4608,
            num_hidden_layers=32,
            num_attention_heads=36,
            num_key_value_heads=4,  # GQA: 36 heads / 4 kv_heads = 9x ratio
            intermediate_size=18432,
            max_position_embeddings=16384,
            sliding_window=4096,
        )

    @classmethod
    def starcoder2_15b(cls) -> StarCoder2Config:
        """Create StarCoder2 15B configuration."""
        return cls(
            vocab_size=49152,
            hidden_size=6144,
            num_hidden_layers=40,
            num_attention_heads=48,
            num_key_value_heads=4,  # GQA: 48 heads / 4 kv_heads = 12x ratio
            intermediate_size=24576,
            max_position_embeddings=16384,
            sliding_window=4096,
        )

    @classmethod
    def tiny(cls) -> StarCoder2Config:
        """Create tiny StarCoder2 for testing."""
        return cls(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
            max_position_embeddings=256,
            sliding_window=128,
        )
