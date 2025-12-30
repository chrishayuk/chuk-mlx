"""
Configuration classes for activation steering.

This module contains the configuration dataclasses used
to configure steering behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


@dataclass
class SteeringConfig:
    """Configuration for activation steering."""

    # Which layers to steer
    layers: list[int] = field(default_factory=lambda: [24])

    # Steering coefficient (positive = toward positive class)
    coefficient: float = 1.0

    # Apply only at specific positions
    position: int | None = None  # None = all positions

    # Normalization
    normalize_direction: bool = True
    scale_by_activation_norm: bool = False

    # Generation settings
    max_new_tokens: int = 50
    temperature: float = 0.0


class SteeringMode(Enum):
    """Steering modes for tool-calling control (backwards compatibility)."""

    NORMAL = "normal"
    FORCE_TOOL = "force_tool"
    PREVENT_TOOL = "prevent_tool"
    BOOST_TOOL = "boost_tool"
    SUPPRESS_TOOL = "suppress_tool"


@dataclass
class LegacySteeringConfig:
    """Configuration for legacy tool-calling steering."""

    mode: SteeringMode = SteeringMode.NORMAL
    steering_scale: float = 1.0
    neuron_boost_scale: float = 5000.0
    use_kill_switch: bool = False
    kill_switch_boost: float = 0.0
    tool_promoters: list[int] | None = None
    tool_suppressors: list[int] | None = None

    def __post_init__(self):
        if self.tool_promoters is None:
            self.tool_promoters = [803, 2036, 831]
        if self.tool_suppressors is None:
            self.tool_suppressors = [1237, 821, 1347]
