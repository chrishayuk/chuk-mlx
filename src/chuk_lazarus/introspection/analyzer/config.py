"""
Configuration classes and enums for model analysis.

This module contains the configuration dataclasses and enums used
to configure analysis behavior.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from ..hooks import PositionSelection


class LayerStrategy(str, Enum):
    """Strategy for selecting which layers to capture."""

    ALL = "all"
    EVENLY_SPACED = "evenly_spaced"
    FIRST_LAST = "first_last"
    CUSTOM = "custom"
    SPECIFIC = "specific"


class TrackStrategy(str, Enum):
    """Strategy for automatic token tracking."""

    MANUAL = "manual"  # Use track_tokens list explicitly
    TOP_K_FINAL = "top_k_final"  # Track top-k tokens from final layer
    EMERGENT = "emergent"  # Find tokens that spike mid-network
    TOOL_TOKENS = "tool_tokens"  # Track common tool-calling tokens


class AnalysisConfig(BaseModel):
    """Configuration for model analysis."""

    layer_strategy: LayerStrategy = Field(
        default=LayerStrategy.EVENLY_SPACED,
        description="How to select layers for capture",
    )
    layer_step: int = Field(
        default=4,
        ge=1,
        description="Step size for evenly spaced layer capture",
    )
    custom_layers: list[int] | None = Field(
        default=None,
        description="Specific layers to capture when using CUSTOM strategy",
    )
    position_strategy: PositionSelection = Field(
        default=PositionSelection.LAST,
        description="Which sequence positions to capture",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of top predictions to return",
    )
    track_tokens: list[str] = Field(
        default_factory=list,
        description="Tokens to track across layers (when using MANUAL strategy)",
    )
    track_strategy: TrackStrategy = Field(
        default=TrackStrategy.MANUAL,
        description="Strategy for automatic token tracking",
    )
    compute_entropy: bool = Field(
        default=True,
        description="Compute entropy for each layer's distribution",
    )
    compute_transitions: bool = Field(
        default=True,
        description="Compute KL/JS divergence between consecutive layers",
    )
    compute_residual_decomposition: bool = Field(
        default=False,
        description="Decompose residual stream into attention vs FFN contributions (requires ALL layers)",
    )

    model_config = ConfigDict(use_enum_values=False)
