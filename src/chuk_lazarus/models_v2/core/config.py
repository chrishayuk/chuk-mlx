"""
Pydantic configuration models for the unified architecture.

All configs are immutable (frozen) and validated.
No dictionary goop - structured types throughout.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .enums import (
    ActivationType,
    AttentionType,
    BlockType,
    FFNType,
    HeadType,
    NormType,
    PoolingType,
    PositionEmbeddingType,
    SSMType,
)


class RoPEConfig(BaseModel):
    """Configuration for Rotary Position Embeddings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    theta: float = Field(
        default=10000.0,
        gt=0,
        description="Base frequency for RoPE",
    )
    traditional: bool = Field(
        default=False,
        description="Use traditional (sin, cos) ordering vs interleaved",
    )
    scaling_factor: float = Field(
        default=1.0,
        gt=0,
        description="Scaling factor for extended context",
    )
    scaling_type: str | None = Field(
        default=None,
        description="Type of scaling: 'linear', 'dynamic', 'yarn', etc.",
    )
    max_position_embeddings: int = Field(
        default=4096,
        gt=0,
        description="Maximum sequence length for RoPE",
    )


class PositionConfig(BaseModel):
    """Configuration for position embeddings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    position_type: PositionEmbeddingType = Field(
        default=PositionEmbeddingType.ROPE,
        description="Type of position embeddings",
    )
    max_position_embeddings: int = Field(
        default=4096,
        gt=0,
        description="Maximum sequence length",
    )
    rope: RoPEConfig | None = Field(
        default=None,
        description="RoPE-specific configuration",
    )

    @model_validator(mode="after")
    def validate_rope_config(self) -> PositionConfig:
        """Ensure RoPE config exists if using RoPE."""
        if self.position_type == PositionEmbeddingType.ROPE and self.rope is None:
            # Create default RoPE config
            object.__setattr__(
                self,
                "rope",
                RoPEConfig(max_position_embeddings=self.max_position_embeddings),
            )
        return self


class EmbeddingConfig(BaseModel):
    """Configuration for token embeddings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    vocab_size: int = Field(
        gt=0,
        description="Vocabulary size",
    )
    hidden_size: int = Field(
        gt=0,
        description="Embedding dimension",
    )
    padding_idx: int | None = Field(
        default=None,
        description="Padding token index",
    )
    scale_factor: float | None = Field(
        default=None,
        description="Embedding scale factor (e.g., sqrt(hidden_size) for Gemma)",
    )
    tie_word_embeddings: bool = Field(
        default=True,
        description="Tie input and output embeddings",
    )


class NormConfig(BaseModel):
    """Configuration for normalization layers."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    norm_type: NormType = Field(
        default=NormType.RMS_NORM,
        description="Type of normalization",
    )
    eps: float = Field(
        default=1e-6,
        gt=0,
        description="Epsilon for numerical stability",
    )
    elementwise_affine: bool = Field(
        default=True,
        description="Whether to learn scale/shift parameters",
    )


class AttentionConfig(BaseModel):
    """Configuration for attention mechanisms."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    attention_type: AttentionType = Field(
        default=AttentionType.GROUPED_QUERY,
        description="Type of attention mechanism",
    )
    num_attention_heads: int = Field(
        gt=0,
        description="Number of attention heads",
    )
    num_key_value_heads: int | None = Field(
        default=None,
        description="Number of KV heads for GQA/MQA (defaults to num_attention_heads)",
    )
    head_dim: int | None = Field(
        default=None,
        description="Dimension per head (defaults to hidden_size // num_attention_heads)",
    )
    hidden_size: int = Field(
        gt=0,
        description="Hidden dimension",
    )
    attention_bias: bool = Field(
        default=False,
        description="Use bias in attention projections",
    )
    attention_dropout: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Attention dropout probability",
    )
    sliding_window_size: int | None = Field(
        default=None,
        description="Sliding window size (for sliding window attention)",
    )
    position: PositionConfig = Field(
        default_factory=PositionConfig,
        description="Position embedding configuration",
    )

    @model_validator(mode="after")
    def set_defaults(self) -> AttentionConfig:
        """Set derived defaults."""
        # Default num_key_value_heads to num_attention_heads (MHA)
        if self.num_key_value_heads is None:
            object.__setattr__(self, "num_key_value_heads", self.num_attention_heads)

        # Default head_dim
        if self.head_dim is None:
            object.__setattr__(self, "head_dim", self.hidden_size // self.num_attention_heads)

        return self


class FFNConfig(BaseModel):
    """Configuration for feed-forward networks."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ffn_type: FFNType = Field(
        default=FFNType.SWIGLU,
        description="Type of feed-forward network",
    )
    hidden_size: int = Field(
        gt=0,
        description="Input/output dimension",
    )
    intermediate_size: int = Field(
        gt=0,
        description="Intermediate dimension",
    )
    activation: ActivationType = Field(
        default=ActivationType.SILU,
        description="Activation function",
    )
    bias: bool = Field(
        default=False,
        description="Use bias in linear layers",
    )
    dropout: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Dropout probability",
    )

    # MoE-specific
    num_experts: int | None = Field(
        default=None,
        description="Number of experts (for MoE)",
    )
    num_experts_per_tok: int | None = Field(
        default=None,
        description="Number of experts activated per token (for MoE)",
    )


class SSMConfig(BaseModel):
    """Configuration for State Space Models (Mamba, etc.)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ssm_type: SSMType = Field(
        default=SSMType.MAMBA,
        description="Type of SSM",
    )
    hidden_size: int = Field(
        gt=0,
        description="Hidden dimension",
    )
    state_size: int = Field(
        default=16,
        gt=0,
        description="SSM state dimension",
    )
    conv_kernel_size: int = Field(
        default=4,
        gt=0,
        description="Convolution kernel size",
    )
    expand_factor: int = Field(
        default=2,
        gt=0,
        description="Expansion factor for inner dimension",
    )
    dt_rank: int | str = Field(
        default="auto",
        description="Rank of dt projection ('auto' = ceil(hidden_size / 16))",
    )
    use_bias: bool = Field(
        default=False,
        description="Use bias in projections",
    )
    use_conv_bias: bool = Field(
        default=True,
        description="Use bias in convolution",
    )


class BlockConfig(BaseModel):
    """Configuration for a single model block."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    block_type: BlockType = Field(
        default=BlockType.TRANSFORMER,
        description="Type of block",
    )
    hidden_size: int = Field(
        gt=0,
        description="Hidden dimension",
    )

    # Pre/post norm
    pre_norm: bool = Field(
        default=True,
        description="Use pre-normalization (vs post-norm)",
    )
    norm: NormConfig = Field(
        default_factory=NormConfig,
        description="Normalization configuration",
    )

    # For transformer blocks
    attention: AttentionConfig | None = Field(
        default=None,
        description="Attention configuration (for transformer blocks)",
    )
    ffn: FFNConfig | None = Field(
        default=None,
        description="FFN configuration (for transformer blocks)",
    )

    # For SSM blocks
    ssm: SSMConfig | None = Field(
        default=None,
        description="SSM configuration (for Mamba blocks)",
    )

    # Residual
    residual_dropout: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Dropout on residual connections",
    )


class BackboneConfig(BaseModel):
    """Configuration for the model backbone (stack of blocks)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    num_hidden_layers: int = Field(
        gt=0,
        description="Number of layers",
    )
    hidden_size: int = Field(
        gt=0,
        description="Hidden dimension",
    )

    # Block configuration (applied to all blocks unless overridden)
    block: BlockConfig = Field(
        description="Default block configuration",
    )

    # Per-layer overrides for hybrid architectures
    layer_configs: dict[int, BlockConfig] | None = Field(
        default=None,
        description="Per-layer block config overrides (for hybrid architectures)",
    )

    # Final norm
    final_norm: NormConfig = Field(
        default_factory=NormConfig,
        description="Final normalization before head",
    )


class HeadConfig(BaseModel):
    """Configuration for output head."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    head_type: HeadType = Field(
        default=HeadType.LM,
        description="Type of output head",
    )
    hidden_size: int = Field(
        gt=0,
        description="Input hidden dimension",
    )

    # For LM head
    vocab_size: int | None = Field(
        default=None,
        description="Vocabulary size (for LM head)",
    )
    tie_word_embeddings: bool = Field(
        default=True,
        description="Tie with input embeddings",
    )

    # For classifier head
    num_classes: int | None = Field(
        default=None,
        description="Number of classes (for classifier)",
    )
    pooling: PoolingType = Field(
        default=PoolingType.LAST,
        description="Pooling strategy (for classifier)",
    )

    # Common
    bias: bool = Field(
        default=False,
        description="Use bias in output projection",
    )
    dropout: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Dropout before projection",
    )


class ModelConfig(BaseModel):
    """
    Complete model configuration.

    This is the top-level config that composes all component configs.
    Compatible with HuggingFace config.json format.
    """

    model_config = ConfigDict(frozen=False, extra="ignore")  # Allow unknown fields for HF compat

    # Model identification
    model_type: str | None = Field(
        default=None,
        description="Model type identifier (e.g., 'llama', 'qwen2', 'mamba')",
    )
    architectures: list[str] | None = Field(
        default=None,
        description="Architecture class names (HuggingFace format)",
    )

    # Core dimensions (convenience fields that populate sub-configs)
    vocab_size: int = Field(
        default=32000,
        gt=0,
        description="Vocabulary size",
    )
    hidden_size: int = Field(
        default=4096,
        gt=0,
        description="Hidden dimension",
    )
    intermediate_size: int = Field(
        default=11008,
        gt=0,
        description="FFN intermediate dimension",
    )
    num_hidden_layers: int = Field(
        default=32,
        gt=0,
        description="Number of layers",
    )
    num_attention_heads: int = Field(
        default=32,
        gt=0,
        description="Number of attention heads",
    )
    num_key_value_heads: int | None = Field(
        default=None,
        description="Number of KV heads (for GQA)",
    )
    head_dim: int | None = Field(
        default=None,
        description="Dimension per attention head",
    )

    # Position embeddings
    max_position_embeddings: int = Field(
        default=4096,
        gt=0,
        description="Maximum sequence length",
    )
    rope_theta: float = Field(
        default=10000.0,
        gt=0,
        description="RoPE base frequency",
    )
    rope_scaling: dict[str, Any] | None = Field(
        default=None,
        description="RoPE scaling configuration",
    )
    rope_traditional: bool = Field(
        default=False,
        description="Use traditional RoPE ordering",
    )

    # Normalization
    rms_norm_eps: float = Field(
        default=1e-6,
        gt=0,
        description="RMSNorm epsilon",
    )

    # Activation
    hidden_act: str = Field(
        default="silu",
        description="Activation function name",
    )
    hidden_activation: str | None = Field(
        default=None,
        description="Alternative activation name (Gemma)",
    )

    # Bias settings
    attention_bias: bool = Field(
        default=False,
        description="Use bias in attention",
    )
    mlp_bias: bool = Field(
        default=False,
        description="Use bias in MLP",
    )

    # Dropout
    attention_dropout: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Attention dropout",
    )
    residual_dropout: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Residual dropout",
    )

    # Special tokens
    bos_token_id: int | None = Field(default=1, description="BOS token ID")
    eos_token_id: int | None = Field(default=2, description="EOS token ID")
    pad_token_id: int | None = Field(default=None, description="Padding token ID")

    # Embeddings
    tie_word_embeddings: bool = Field(
        default=True,
        description="Tie input/output embeddings",
    )

    # Sliding window (Mistral-style)
    sliding_window: int | None = Field(
        default=None,
        description="Sliding window size",
    )

    @model_validator(mode="after")
    def set_derived_values(self) -> ModelConfig:
        """Set derived configuration values."""
        # Default num_key_value_heads
        if self.num_key_value_heads is None:
            object.__setattr__(self, "num_key_value_heads", self.num_attention_heads)

        # Default head_dim
        if self.head_dim is None:
            object.__setattr__(self, "head_dim", self.hidden_size // self.num_attention_heads)

        # Handle Gemma's hidden_activation
        if self.hidden_activation and self.hidden_act == "silu":
            object.__setattr__(self, "hidden_act", self.hidden_activation)

        return self

    @classmethod
    def from_file(cls, path: str | Path) -> ModelConfig:
        """Load config from JSON file."""
        path = Path(path)
        if path.is_dir():
            path = path / "config.json"

        with open(path) as f:
            data = json.load(f)

        return cls.model_validate(data)

    @classmethod
    async def from_file_async(cls, path: str | Path) -> ModelConfig:
        """Async: Load config from JSON file."""
        import aiofiles

        path = Path(path)
        if path.is_dir():
            path = path / "config.json"

        async with aiofiles.open(path) as f:
            content = await f.read()
            data = json.loads(content)

        return cls.model_validate(data)

    def to_embedding_config(self) -> EmbeddingConfig:
        """Create EmbeddingConfig from this config."""
        return EmbeddingConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            padding_idx=self.pad_token_id,
            tie_word_embeddings=self.tie_word_embeddings,
        )

    def to_attention_config(self) -> AttentionConfig:
        """Create AttentionConfig from this config."""
        rope_config = RoPEConfig(
            theta=self.rope_theta,
            traditional=self.rope_traditional,
            scaling_factor=self.rope_scaling.get("factor", 1.0) if self.rope_scaling else 1.0,
            scaling_type=self.rope_scaling.get("type") if self.rope_scaling else None,
            max_position_embeddings=self.max_position_embeddings,
        )

        position_config = PositionConfig(
            position_type=PositionEmbeddingType.ROPE,
            max_position_embeddings=self.max_position_embeddings,
            rope=rope_config,
        )

        # Determine attention type
        attention_type = AttentionType.GROUPED_QUERY
        if self.num_key_value_heads == self.num_attention_heads:
            attention_type = AttentionType.MULTI_HEAD
        elif self.num_key_value_heads == 1:
            attention_type = AttentionType.MULTI_QUERY
        if self.sliding_window:
            attention_type = AttentionType.SLIDING_WINDOW

        return AttentionConfig(
            attention_type=attention_type,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            hidden_size=self.hidden_size,
            attention_bias=self.attention_bias,
            attention_dropout=self.attention_dropout,
            sliding_window_size=self.sliding_window,
            position=position_config,
        )

    def to_ffn_config(self) -> FFNConfig:
        """Create FFNConfig from this config."""
        # Map activation name to enum
        act_map = {
            "silu": ActivationType.SILU,
            "swish": ActivationType.SILU,
            "gelu": ActivationType.GELU,
            "gelu_new": ActivationType.GELU,
            "gelu_fast": ActivationType.GELU_APPROX,
            "relu": ActivationType.RELU,
            "relu2": ActivationType.RELU2,
            "tanh": ActivationType.TANH,
        }
        activation = act_map.get(self.hidden_act.lower(), ActivationType.SILU)

        # Determine FFN type
        ffn_type = FFNType.SWIGLU if activation == ActivationType.SILU else FFNType.MLP

        return FFNConfig(
            ffn_type=ffn_type,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            activation=activation,
            bias=self.mlp_bias,
        )

    def to_norm_config(self) -> NormConfig:
        """Create NormConfig from this config."""
        return NormConfig(
            norm_type=NormType.RMS_NORM,
            eps=self.rms_norm_eps,
        )

    def to_block_config(self) -> BlockConfig:
        """Create BlockConfig from this config."""
        return BlockConfig(
            block_type=BlockType.TRANSFORMER,
            hidden_size=self.hidden_size,
            pre_norm=True,
            norm=self.to_norm_config(),
            attention=self.to_attention_config(),
            ffn=self.to_ffn_config(),
            residual_dropout=self.residual_dropout,
        )

    def to_backbone_config(self) -> BackboneConfig:
        """Create BackboneConfig from this config."""
        return BackboneConfig(
            num_hidden_layers=self.num_hidden_layers,
            hidden_size=self.hidden_size,
            block=self.to_block_config(),
            final_norm=self.to_norm_config(),
        )

    def to_head_config(self, head_type: HeadType = HeadType.LM) -> HeadConfig:
        """Create HeadConfig from this config."""
        return HeadConfig(
            head_type=head_type,
            hidden_size=self.hidden_size,
            vocab_size=self.vocab_size,
            tie_word_embeddings=self.tie_word_embeddings,
        )

    def save(self, path: str | Path) -> None:
        """Save config to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=2)

    async def save_async(self, path: str | Path) -> None:
        """Async: Save config to JSON file."""
        import aiofiles

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(path, "w") as f:
            await f.write(json.dumps(self.model_dump(), indent=2))
