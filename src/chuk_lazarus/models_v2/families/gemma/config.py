"""
Gemma 3 configuration.

Extends base ModelConfig with Gemma-specific settings.
"""

from __future__ import annotations

from typing import Any

from ...core.config import ModelConfig


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

    model_type: str = "gemma3_text"

    # Gemma-specific: head dimension (can differ from hidden_size / num_heads)
    head_dim: int = 256

    # Gemma-specific: query pre-attention scalar for attention scaling
    query_pre_attn_scalar: float = 256.0

    # Gemma-specific: sliding window attention pattern
    sliding_window: int = 512
    sliding_window_pattern: int = 6  # Every 6th layer is global attention

    # Gemma-specific: separate RoPE base for local (sliding) attention
    rope_local_base_freq: float = 10000.0

    # Gemma defaults
    hidden_act: str = "gelu_pytorch_tanh"  # Gated GELU
    rope_theta: float = 1000000.0
    rms_norm_eps: float = 1e-6

    # Optional: RoPE scaling configuration
    rope_scaling: dict[str, Any] | None = None

    @classmethod
    def gemma3_270m(cls) -> GemmaConfig:
        """
        Create Gemma 3 270M configuration.

        This is the base for FunctionGemma.
        """
        return cls(
            model_type="gemma3_text",
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
            rope_theta=1000000.0,
            rope_local_base_freq=10000.0,
            rms_norm_eps=1e-6,
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
            model_type="gemma3_text",
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
            rope_theta=1000000.0,
            rope_local_base_freq=10000.0,
            rms_norm_eps=1e-6,
        )

    @classmethod
    def gemma3_4b(cls) -> GemmaConfig:
        """Create Gemma 3 4B configuration."""
        return cls(
            model_type="gemma3_text",
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
            rope_theta=1000000.0,
            rope_local_base_freq=10000.0,
            rms_norm_eps=1e-6,
        )

    @classmethod
    def gemma3_12b(cls) -> GemmaConfig:
        """Create Gemma 3 12B configuration."""
        return cls(
            model_type="gemma3_text",
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
            rope_theta=1000000.0,
            rope_local_base_freq=10000.0,
            rms_norm_eps=1e-6,
        )

    @classmethod
    def gemma3_27b(cls) -> GemmaConfig:
        """Create Gemma 3 27B configuration."""
        return cls(
            model_type="gemma3_text",
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
            rope_theta=1000000.0,
            rope_local_base_freq=10000.0,
            rms_norm_eps=1e-6,
        )

    @classmethod
    def tiny(cls) -> GemmaConfig:
        """Create tiny Gemma for testing."""
        return cls(
            model_type="gemma3_text",
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
            rope_theta=10000.0,
            rope_local_base_freq=10000.0,
            rms_norm_eps=1e-6,
        )

    def is_sliding_layer(self, layer_idx: int) -> bool:
        """Check if a layer uses sliding window attention."""
        return (layer_idx + 1) % self.sliding_window_pattern != 0

    def is_global_layer(self, layer_idx: int) -> bool:
        """Check if a layer uses global (full) attention."""
        return not self.is_sliding_layer(layer_idx)
