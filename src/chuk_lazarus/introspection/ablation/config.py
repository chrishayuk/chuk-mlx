"""
Configuration classes and enums for ablation studies.

This module contains the configuration classes and enums used
to configure ablation experiment behavior.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class AblationType(str, Enum):
    """Type of ablation to perform."""

    ZERO = "zero"
    """Zero out weights."""

    MEAN = "mean"
    """Replace with mean activation."""

    NOISE = "noise"
    """Add noise to weights."""


class ComponentType(str, Enum):
    """Model component to ablate."""

    MLP = "mlp"
    ATTENTION = "attention"
    BOTH = "both"
    MLP_GATE = "mlp_gate"
    MLP_UP = "mlp_up"
    MLP_DOWN = "mlp_down"
    ATTN_Q = "attn_q"
    ATTN_K = "attn_k"
    ATTN_V = "attn_v"
    ATTN_O = "attn_o"


class AblationConfig(BaseModel):
    """Configuration for ablation experiments."""

    model_config = ConfigDict(frozen=True)

    ablation_type: AblationType = Field(
        default=AblationType.ZERO, description="Type of ablation to perform"
    )
    component: ComponentType = Field(default=ComponentType.MLP, description="Component to ablate")
    max_new_tokens: int = Field(default=60, ge=1, description="Maximum tokens to generate")
    temperature: float = Field(default=0.0, ge=0.0, description="Sampling temperature")
