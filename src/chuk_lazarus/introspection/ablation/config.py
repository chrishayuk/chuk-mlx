"""
Configuration classes and enums for ablation studies.

This module contains the configuration dataclasses and enums used
to configure ablation experiment behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AblationType(str, Enum):
    """Type of ablation to perform."""

    ZERO = "zero"  # Zero out weights
    MEAN = "mean"  # Replace with mean activation
    NOISE = "noise"  # Add noise to weights


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


@dataclass
class AblationConfig:
    """Configuration for ablation experiments."""

    ablation_type: AblationType = AblationType.ZERO
    component: ComponentType = ComponentType.MLP
    max_new_tokens: int = 60
    temperature: float = 0.0
