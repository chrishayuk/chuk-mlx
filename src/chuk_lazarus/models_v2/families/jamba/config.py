"""
Jamba configuration.

Jamba is a hybrid Mamba-Transformer MoE model from AI21 Labs.

Key architectural features:
- Hybrid: 1 attention layer every 8 layers (attn_layer_period=8)
- MoE: Every 2nd layer uses MoE instead of dense FFN (expert_layer_period=2)
- 16 experts with 2 active per token
- Standard: RMSNorm, GQA, SwiGLU activation
- 256K context window
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from ...core.config import ModelConfig


class JambaConfig(ModelConfig):
    """
    Configuration for Jamba models.

    Jamba is a hybrid Mamba-Transformer MoE model that combines:
    - Mamba (SSM) layers for O(n) complexity
    - Attention layers for precise recall (1 every 8 layers)
    - MoE for scaling capacity without compute (every 2 layers)

    Supports:
    - Jamba v0.1 (52B total, ~12B active)
    - Jamba 1.5 Mini (52B total, 12B active)
    - Jamba 1.5 Large (398B total, 94B active)

    Example:
        >>> config = JambaConfig(
        ...     vocab_size=65536,
        ...     hidden_size=4096,
        ...     num_hidden_layers=32,
        ...     num_attention_heads=32,
        ...     num_key_value_heads=8,
        ...     intermediate_size=14336,
        ...     attn_layer_period=8,
        ...     expert_layer_period=2,
        ...     num_experts=16,
        ...     num_experts_per_tok=2,
        ... )
    """

    model_type: str = "jamba"

    # Jamba-specific: Hybrid layer pattern
    attn_layer_period: int = Field(
        default=8,
        description="Attention layers appear every N layers (others are Mamba)",
    )
    attn_layer_offset: int = Field(
        default=4,
        description="Offset for first attention layer",
    )

    # Jamba-specific: MoE pattern
    expert_layer_period: int = Field(
        default=2,
        description="MoE layers appear every N layers (others are dense FFN)",
    )
    expert_layer_offset: int = Field(
        default=1,
        description="Offset for first MoE layer",
    )

    # MoE parameters
    num_experts: int = Field(
        default=16,
        description="Total number of experts",
    )
    num_experts_per_tok: int = Field(
        default=2,
        description="Number of experts activated per token",
    )

    # Mamba parameters
    mamba_d_state: int = Field(default=16, description="Mamba state dimension")
    mamba_d_conv: int = Field(default=4, description="Mamba convolution kernel size")
    mamba_expand: int = Field(default=2, description="Mamba expansion factor")
    mamba_dt_rank: int = Field(default=256, description="Mamba dt projection rank")
    mamba_conv_bias: bool = Field(default=True, description="Use bias in Mamba conv")
    mamba_proj_bias: bool = Field(default=False, description="Use bias in Mamba projections")

    # Standard transformer params
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 262144  # 256K context
    tie_word_embeddings: bool = False

    # Optional RoPE scaling
    rope_scaling: dict[str, Any] | None = None

    def is_attention_layer(self, layer_idx: int) -> bool:
        """Check if layer at index uses attention (vs Mamba)."""
        return (layer_idx - self.attn_layer_offset) % self.attn_layer_period == 0

    def is_moe_layer(self, layer_idx: int) -> bool:
        """Check if layer at index uses MoE (vs dense FFN)."""
        return (layer_idx - self.expert_layer_offset) % self.expert_layer_period == 0

    @classmethod
    def jamba_v0_1(cls) -> JambaConfig:
        """Create Jamba v0.1 configuration (~52B total, 12B active)."""
        return cls(
            vocab_size=65536,
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            intermediate_size=14336,
            attn_layer_period=8,
            attn_layer_offset=4,
            expert_layer_period=2,
            expert_layer_offset=1,
            num_experts=16,
            num_experts_per_tok=2,
            mamba_d_state=16,
            mamba_d_conv=4,
            mamba_expand=2,
            mamba_dt_rank=256,
            max_position_embeddings=262144,
        )

    @classmethod
    def jamba_1_5_mini(cls) -> JambaConfig:
        """Create Jamba 1.5 Mini configuration (52B total, 12B active)."""
        # Same architecture as v0.1 with updated training
        return cls.jamba_v0_1()

    @classmethod
    def jamba_1_5_large(cls) -> JambaConfig:
        """Create Jamba 1.5 Large configuration (398B total, 94B active)."""
        return cls(
            vocab_size=65536,
            hidden_size=8192,
            num_hidden_layers=64,
            num_attention_heads=64,
            num_key_value_heads=16,
            intermediate_size=28672,
            attn_layer_period=8,
            attn_layer_offset=4,
            expert_layer_period=2,
            expert_layer_offset=1,
            num_experts=16,
            num_experts_per_tok=2,
            mamba_d_state=16,
            mamba_d_conv=4,
            mamba_expand=2,
            mamba_dt_rank=512,
            max_position_embeddings=262144,
        )

    @classmethod
    def tiny(cls) -> JambaConfig:
        """Create tiny Jamba for testing."""
        return cls(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=8,  # Need at least 8 for attention pattern
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
            attn_layer_period=4,  # Every 4th layer has attention
            attn_layer_offset=2,
            expert_layer_period=2,
            expert_layer_offset=1,
            num_experts=4,
            num_experts_per_tok=2,
            mamba_d_state=8,
            mamba_d_conv=4,
            mamba_expand=2,
            mamba_dt_rank=16,
            max_position_embeddings=256,
        )
