"""
Configuration classes for activation steering.

This module contains the configuration classes used
to configure steering behavior.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class SteeringConfig(BaseModel):
    """Configuration for activation steering."""

    model_config = ConfigDict(frozen=True)

    layers: list[int] = Field(default_factory=lambda: [24], description="Which layers to steer")
    coefficient: float = Field(
        default=1.0, description="Steering coefficient (positive = toward positive class)"
    )
    position: int | None = Field(
        default=None, description="Apply only at specific position (None = all)"
    )
    normalize_direction: bool = Field(default=True, description="Normalize direction vector")
    scale_by_activation_norm: bool = Field(
        default=False, description="Scale steering by activation norm"
    )
    max_new_tokens: int = Field(default=50, ge=1, description="Maximum tokens to generate")
    temperature: float = Field(default=0.0, ge=0.0, description="Sampling temperature")


class SteeringMode(str, Enum):
    """Steering modes for tool-calling control."""

    NORMAL = "normal"
    FORCE_TOOL = "force_tool"
    PREVENT_TOOL = "prevent_tool"
    BOOST_TOOL = "boost_tool"
    SUPPRESS_TOOL = "suppress_tool"


class LegacySteeringConfig(BaseModel):
    """Configuration for legacy tool-calling steering."""

    model_config = ConfigDict(frozen=True)

    mode: SteeringMode = Field(default=SteeringMode.NORMAL, description="Steering mode")
    steering_scale: float = Field(default=1.0, description="Scale for steering")
    neuron_boost_scale: float = Field(default=5000.0, description="Scale for neuron boost")
    use_kill_switch: bool = Field(default=False, description="Use kill switch")
    kill_switch_boost: float = Field(default=0.0, description="Kill switch boost value")
    tool_promoters: list[int] = Field(
        default_factory=lambda: [803, 2036, 831], description="Neuron indices that promote tool use"
    )
    tool_suppressors: list[int] = Field(
        default_factory=lambda: [1237, 821, 1347],
        description="Neuron indices that suppress tool use",
    )
