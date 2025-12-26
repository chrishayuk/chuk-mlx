"""
Granite configuration.

Supports the IBM Granite model family:
- Granite 3.0/3.1: Dense transformer with multipliers
- Granite 4.0: Hybrid Mamba-2/Transformer with optional MoE

Reference: https://huggingface.co/ibm-granite
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from ...core.config import ModelConfig


class GraniteConfig(ModelConfig):
    """
    Configuration for Granite 3.x models.

    Granite 3.x is a dense transformer similar to Llama but with:
    - Multipliers for embeddings, attention, and residuals
    - Logits scaling
    - Optional attention dropout

    Example:
        >>> # Granite 3.1 2B
        >>> config = GraniteConfig(
        ...     vocab_size=49155,
        ...     hidden_size=2048,
        ...     num_hidden_layers=40,
        ...     num_attention_heads=32,
        ...     num_key_value_heads=8,
        ...     intermediate_size=8192,
        ... )
    """

    model_type: str = "granite"

    # Granite-specific multipliers
    embedding_multiplier: float = Field(
        default=12.0,
        description="Multiplier applied to token embeddings",
    )
    attention_multiplier: float = Field(
        default=1.0,
        description="Multiplier applied to attention output (1/sqrt(num_heads) typical)",
    )
    residual_multiplier: float = Field(
        default=1.0,
        description="Multiplier applied to residual connections",
    )
    logits_scaling: float = Field(
        default=1.0,
        description="Scaling factor for output logits",
    )

    # Standard transformer settings
    hidden_act: str = "silu"
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5

    # Attention settings
    attention_dropout: float = 0.0
    attention_bias: bool = False
    mlp_bias: bool = False

    # RoPE scaling
    rope_scaling: dict[str, Any] | None = None

    @classmethod
    def granite_3_8b(cls) -> GraniteConfig:
        """Granite 3.0 8B configuration."""
        return cls(
            vocab_size=49155,
            hidden_size=4096,
            num_hidden_layers=40,
            num_attention_heads=32,
            num_key_value_heads=8,
            intermediate_size=12800,
            max_position_embeddings=4096,
            embedding_multiplier=12.0,
            attention_multiplier=0.0078125,  # 1/128
            residual_multiplier=0.22,
            logits_scaling=16.0,
            attention_dropout=0.1,
            tie_word_embeddings=True,
        )

    @classmethod
    def granite_3_1_2b(cls) -> GraniteConfig:
        """Granite 3.1 2B configuration."""
        return cls(
            vocab_size=49155,
            hidden_size=2048,
            num_hidden_layers=40,
            num_attention_heads=32,
            num_key_value_heads=8,
            intermediate_size=8192,
            max_position_embeddings=131072,
            rope_theta=5000000.0,
            embedding_multiplier=12.0,
            attention_multiplier=0.015625,  # 1/64
            residual_multiplier=0.22,
            logits_scaling=8.0,
            attention_dropout=0.1,
            tie_word_embeddings=True,
        )

    @classmethod
    def granite_3_1_8b(cls) -> GraniteConfig:
        """Granite 3.1 8B configuration."""
        return cls(
            vocab_size=49155,
            hidden_size=4096,
            num_hidden_layers=40,
            num_attention_heads=32,
            num_key_value_heads=8,
            intermediate_size=12800,
            max_position_embeddings=131072,
            rope_theta=5000000.0,
            embedding_multiplier=12.0,
            attention_multiplier=0.0078125,
            residual_multiplier=0.22,
            logits_scaling=16.0,
            attention_dropout=0.1,
            tie_word_embeddings=True,
        )

    @classmethod
    def tiny(cls) -> GraniteConfig:
        """Tiny Granite for testing."""
        return cls(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
            max_position_embeddings=256,
            embedding_multiplier=1.0,
            attention_multiplier=1.0,
            residual_multiplier=1.0,
            logits_scaling=1.0,
        )


class GraniteHybridConfig(ModelConfig):
    """
    Configuration for Granite 4.x hybrid models.

    Granite 4.0 uses a hybrid Mamba-2/Transformer architecture with:
    - Per-layer type configuration (mamba or attention)
    - Optional MoE for Tiny and Small variants
    - Shared experts for MoE variants
    - 9:1 Mamba to Transformer ratio (typical)

    Example:
        >>> # Granite 4.0 Tiny (7B total, 1B active)
        >>> config = GraniteHybridConfig(
        ...     vocab_size=49160,
        ...     hidden_size=1536,
        ...     num_hidden_layers=40,
        ...     layer_types=["mamba"]*5 + ["attention"] + ["mamba"]*9 + ["attention"] + ...,
        ...     num_local_experts=62,
        ...     num_experts_per_tok=6,
        ... )
    """

    model_type: str = "granitemoehybrid"

    # Layer configuration
    layer_types: list[Literal["mamba", "attention"]] = Field(
        default_factory=lambda: ["attention"] * 40,
        description="Type of each layer (mamba or attention)",
    )
    position_embedding_type: Literal["rope", "nope"] = Field(
        default="rope",
        description="Position embedding type (nope for Mamba-heavy models)",
    )

    # Granite-specific multipliers
    embedding_multiplier: float = 12.0
    attention_multiplier: float = 0.0078125
    residual_multiplier: float = 0.22
    logits_scaling: float = 6.0

    # Standard settings
    hidden_act: str = "silu"
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    normalization_function: str = "rmsnorm"

    # Attention settings
    attention_dropout: float = 0.0
    attention_bias: bool = False

    # Mamba-2 settings
    mamba_d_state: int = Field(default=128, description="SSM state dimension")
    mamba_d_conv: int = Field(default=4, description="Convolution kernel size")
    mamba_expand: int = Field(default=2, description="Expansion factor for Mamba")
    mamba_n_heads: int = Field(default=48, description="Number of Mamba heads")
    mamba_d_head: int = Field(default=64, description="Dimension per Mamba head")
    mamba_n_groups: int = Field(default=1, description="Number of groups for Mamba")
    mamba_chunk_size: int = Field(default=256, description="Chunk size for Mamba-2")
    mamba_conv_bias: bool = True
    mamba_proj_bias: bool = False

    # MoE settings (optional)
    num_local_experts: int = Field(default=0, description="Number of experts (0 = dense)")
    num_experts_per_tok: int = Field(default=0, description="Experts per token (0 = dense)")
    shared_intermediate_size: int = Field(
        default=0,
        description="Shared expert intermediate size (0 = no shared expert)",
    )
    router_aux_loss_coef: float = 0.0
    output_router_logits: bool = False

    # RoPE scaling
    rope_scaling: dict[str, Any] | None = None

    @property
    def is_moe(self) -> bool:
        """Whether this is a MoE model."""
        return self.num_local_experts > 0 and self.num_experts_per_tok > 0

    @property
    def num_mamba_layers(self) -> int:
        """Count of Mamba layers."""
        return sum(1 for t in self.layer_types if t == "mamba")

    @property
    def num_attention_layers(self) -> int:
        """Count of attention layers."""
        return sum(1 for t in self.layer_types if t == "attention")

    @classmethod
    def granite_4_micro(cls) -> GraniteHybridConfig:
        """
        Granite 4.0 Micro (3B dense hybrid).

        All attention layers, no MoE.
        """
        return cls(
            vocab_size=100352,
            hidden_size=2560,
            num_hidden_layers=40,
            num_attention_heads=40,
            num_key_value_heads=8,
            intermediate_size=8192,
            max_position_embeddings=131072,
            layer_types=["attention"] * 40,
            position_embedding_type="rope",
            rope_theta=10000000.0,
            embedding_multiplier=12.0,
            attention_multiplier=0.015625,
            residual_multiplier=0.22,
            logits_scaling=10.0,
            # Mamba settings (not used but present in config)
            mamba_d_state=256,
            mamba_n_heads=128,
            mamba_d_head=40,
            # Dense model
            num_local_experts=0,
            num_experts_per_tok=0,
            shared_intermediate_size=8192,
            tie_word_embeddings=True,
        )

    @classmethod
    def granite_4_tiny(cls) -> GraniteHybridConfig:
        """
        Granite 4.0 Tiny (7B total, 1B active).

        Hybrid Mamba-2/Transformer with MoE.
        - 36 Mamba layers, 4 Attention layers
        - 62 experts, 6 active per token
        """
        # Build layer_types: 9 mamba, 1 attention pattern
        layer_types: list[Literal["mamba", "attention"]] = []
        for i in range(4):  # 4 attention layers total
            layer_types.extend(["mamba"] * 5 if i == 0 else ["mamba"] * 9)
            layer_types.append("attention")

        return cls(
            vocab_size=49160,
            hidden_size=1536,
            num_hidden_layers=40,
            num_attention_heads=12,
            num_key_value_heads=4,
            intermediate_size=512,
            max_position_embeddings=131072,
            layer_types=layer_types,
            position_embedding_type="nope",
            embedding_multiplier=12.0,
            attention_multiplier=0.0078125,
            residual_multiplier=0.22,
            logits_scaling=6.0,
            # Mamba-2 settings
            mamba_d_state=128,
            mamba_n_heads=48,
            mamba_d_head=64,
            mamba_expand=2,
            mamba_chunk_size=256,
            # MoE settings
            num_local_experts=62,
            num_experts_per_tok=6,
            shared_intermediate_size=1024,
            tie_word_embeddings=True,
        )

    @classmethod
    def granite_4_small(cls) -> GraniteHybridConfig:
        """
        Granite 4.0 Small (32B total, 9B active).

        Hybrid Mamba-2/Transformer with MoE.
        """
        # Similar pattern to Tiny
        layer_types: list[Literal["mamba", "attention"]] = []
        for i in range(4):
            layer_types.extend(["mamba"] * 5 if i == 0 else ["mamba"] * 9)
            layer_types.append("attention")

        return cls(
            vocab_size=49160,
            hidden_size=3072,
            num_hidden_layers=40,
            num_attention_heads=24,
            num_key_value_heads=8,
            intermediate_size=1024,
            max_position_embeddings=131072,
            layer_types=layer_types,
            position_embedding_type="nope",
            embedding_multiplier=12.0,
            attention_multiplier=0.0078125,
            residual_multiplier=0.22,
            logits_scaling=8.0,
            # Mamba-2 settings
            mamba_d_state=128,
            mamba_n_heads=96,
            mamba_d_head=64,
            mamba_expand=2,
            mamba_chunk_size=256,
            # MoE settings
            num_local_experts=62,
            num_experts_per_tok=6,
            shared_intermediate_size=2048,
            tie_word_embeddings=True,
        )

    @classmethod
    def tiny(cls) -> GraniteHybridConfig:
        """Tiny hybrid for testing."""
        return cls(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
            max_position_embeddings=256,
            layer_types=["mamba", "mamba", "attention", "mamba"],
            position_embedding_type="rope",
            embedding_multiplier=1.0,
            attention_multiplier=1.0,
            residual_multiplier=1.0,
            logits_scaling=1.0,
            # Mamba settings
            mamba_d_state=16,
            mamba_n_heads=4,
            mamba_d_head=16,
            mamba_expand=2,
            # Dense (no MoE for tiny)
            num_local_experts=0,
            num_experts_per_tok=0,
        )

    @classmethod
    def tiny_moe(cls) -> GraniteHybridConfig:
        """Tiny hybrid with MoE for testing."""
        return cls(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=32,
            max_position_embeddings=256,
            layer_types=["mamba", "mamba", "attention", "mamba"],
            position_embedding_type="rope",
            embedding_multiplier=1.0,
            attention_multiplier=1.0,
            residual_multiplier=1.0,
            logits_scaling=1.0,
            # Mamba settings
            mamba_d_state=16,
            mamba_n_heads=4,
            mamba_d_head=16,
            mamba_expand=2,
            # MoE settings
            num_local_experts=4,
            num_experts_per_tok=2,
            shared_intermediate_size=64,
        )
