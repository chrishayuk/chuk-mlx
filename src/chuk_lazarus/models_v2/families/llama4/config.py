"""
Llama 4 configuration.

Extends base ModelConfig with Llama 4-specific settings including MoE and multimodal.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from ...core.config import ModelConfig


class Llama4TextConfig(ModelConfig):
    """
    Configuration for Llama 4 text models.

    Llama 4 introduces:
    - MoE (Mixture of Experts) with shared expert
    - iRoPE (interleaved RoPE and NoPE layers)
    - QK normalization
    - Temperature scaling for attention

    Example:
        >>> # Llama 4 Scout (17B active / 109B total)
        >>> config = Llama4TextConfig(
        ...     vocab_size=202048,
        ...     hidden_size=5120,
        ...     num_hidden_layers=48,
        ...     num_attention_heads=40,
        ...     num_key_value_heads=8,
        ...     intermediate_size=8192,  # Shared expert
        ...     intermediate_size_mlp=16384,  # Routed experts
        ...     num_local_experts=16,
        ...     num_experts_per_tok=1,
        ... )
    """

    model_type: str = "llama4"

    # Llama 4 defaults
    hidden_act: str = "silu"
    rope_theta: float = 500000.0
    rms_norm_eps: float = 1e-5

    # MoE parameters
    num_local_experts: int = Field(
        default=16,
        description="Total number of routed experts",
    )
    num_experts_per_tok: int = Field(
        default=1,
        description="Number of experts activated per token (top-k)",
    )
    intermediate_size_mlp: int = Field(
        default=16384,
        description="Intermediate size for routed experts (per expert)",
    )
    moe_router_topk: int = Field(
        default=1,
        description="Top-k experts for routing",
    )

    # iRoPE parameters
    no_rope_layers: list[int] | None = Field(
        default=None,
        description="Layer indices that use NoPE (global attention without RoPE)",
    )
    attention_chunk_size: int | None = Field(
        default=8192,
        description="Chunk size for chunked attention in RoPE layers",
    )

    # Attention features
    use_qk_norm: bool = Field(
        default=True,
        description="Apply RMS normalization to Q and K after RoPE",
    )
    attn_temperature_tuning: bool = Field(
        default=False,
        description="Use learned temperature scaling for attention",
    )

    # RoPE scaling
    rope_scaling: dict[str, Any] | None = None

    @classmethod
    def scout_17b(cls) -> Llama4TextConfig:
        """
        Llama 4 Scout configuration.

        17B active parameters, 109B total with 16 experts.
        """
        return cls(
            vocab_size=202048,
            hidden_size=5120,
            num_hidden_layers=48,
            num_attention_heads=40,
            num_key_value_heads=8,
            intermediate_size=8192,  # Shared expert intermediate
            intermediate_size_mlp=16384,  # Routed expert intermediate
            num_local_experts=16,
            num_experts_per_tok=1,
            max_position_embeddings=131072,
            rope_theta=500000.0,
            use_qk_norm=True,
            tie_word_embeddings=False,
            # NoPE layers (global attention without RoPE) at 0, 4, 8, ...
            no_rope_layers=[i * 4 for i in range(12)],
        )

    @classmethod
    def maverick_17b(cls) -> Llama4TextConfig:
        """
        Llama 4 Maverick configuration.

        17B active parameters, 400B total with 128 experts.
        """
        return cls(
            vocab_size=202048,
            hidden_size=5120,
            num_hidden_layers=48,
            num_attention_heads=40,
            num_key_value_heads=8,
            intermediate_size=8192,
            intermediate_size_mlp=8192,  # Smaller per-expert
            num_local_experts=128,
            num_experts_per_tok=1,
            max_position_embeddings=131072,
            rope_theta=500000.0,
            use_qk_norm=True,
            tie_word_embeddings=False,
            no_rope_layers=[i * 4 for i in range(12)],
        )

    @classmethod
    def tiny(cls) -> Llama4TextConfig:
        """Create tiny Llama 4 for testing."""
        return cls(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,  # Shared expert
            intermediate_size_mlp=256,  # Routed experts
            num_local_experts=4,
            num_experts_per_tok=1,
            max_position_embeddings=256,
            use_qk_norm=True,
            no_rope_layers=[0],  # First layer is NoPE
        )


class Llama4VisionConfig(ModelConfig):
    """
    Configuration for Llama 4 vision encoder.

    ViT-style vision encoder with pixel shuffle for efficiency.
    """

    model_type: str = "llama4_vision"

    # Vision transformer settings
    hidden_size: int = 1280
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    intermediate_size: int = 5120

    # Image processing
    image_size: int = 560
    patch_size: int = 14
    num_channels: int = 3

    # Projector settings
    vision_output_dim: int = 5120
    pixel_shuffle_ratio: float = 0.5

    # Norm settings
    rms_norm_eps: float = 1e-5
    hidden_act: str = "gelu"

    @classmethod
    def default(cls) -> Llama4VisionConfig:
        """Default vision config for Llama 4."""
        return cls()


class Llama4Config(ModelConfig):
    """
    Full Llama 4 configuration for multimodal models.

    Combines text and vision configurations.
    """

    model_type: str = "llama4"

    text_config: Llama4TextConfig | None = None
    vision_config: Llama4VisionConfig | None = None

    # Image token settings
    image_token_index: int = 128011
    image_token: str = "<|image|>"

    @classmethod
    def scout_multimodal(cls) -> Llama4Config:
        """Llama 4 Scout with vision encoder."""
        return cls(
            text_config=Llama4TextConfig.scout_17b(),
            vision_config=Llama4VisionConfig.default(),
        )

    @classmethod
    def scout_text_only(cls) -> Llama4Config:
        """Llama 4 Scout text-only."""
        return cls(
            text_config=Llama4TextConfig.scout_17b(),
            vision_config=None,
        )

    @model_validator(mode="after")
    def set_derived_values(self) -> Llama4Config:
        """Override to skip derived value computation for wrapper config."""
        # Llama4Config is a wrapper that delegates to text_config,
        # so we don't need to compute head_dim etc. at this level
        return self

    # Forward text config attributes for convenience
    @property
    def vocab_size(self) -> int:
        return self.text_config.vocab_size if self.text_config else 0

    @property
    def hidden_size(self) -> int:
        return self.text_config.hidden_size if self.text_config else 0

    @property
    def num_hidden_layers(self) -> int:
        return self.text_config.num_hidden_layers if self.text_config else 0

    @property
    def num_attention_heads(self) -> int:
        return self.text_config.num_attention_heads if self.text_config else 0

    @property
    def num_key_value_heads(self) -> int | None:
        return self.text_config.num_key_value_heads if self.text_config else None

    @property
    def intermediate_size(self) -> int:
        return self.text_config.intermediate_size if self.text_config else 0

    @property
    def rms_norm_eps(self) -> float:
        return self.text_config.rms_norm_eps if self.text_config else 1e-5

    @property
    def tie_word_embeddings(self) -> bool:
        return self.text_config.tie_word_embeddings if self.text_config else True
