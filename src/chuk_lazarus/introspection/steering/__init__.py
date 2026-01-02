"""
Steering subpackage for activation steering.

This package provides the ActivationSteering class and related utilities
for manipulating model behavior by adding learned directions to activations.
"""

from .config import LegacySteeringConfig, SteeringConfig, SteeringMode
from .core import ActivationSteering
from .hook import SteeringHook
from .legacy import SteeredGemmaMLP, ToolCallingSteering
from .utils import compare_steering_effects, format_functiongemma_prompt, steer_model

__all__ = [
    # Config
    "SteeringConfig",
    "SteeringMode",
    "LegacySteeringConfig",
    # Core
    "ActivationSteering",
    # Hook
    "SteeringHook",
    # Legacy
    "SteeredGemmaMLP",
    "ToolCallingSteering",
    # Utils
    "steer_model",
    "compare_steering_effects",
    "format_functiongemma_prompt",
]
