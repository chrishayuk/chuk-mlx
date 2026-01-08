"""Inference CLI commands.

This module provides commands for running model inference.

Commands:
    run_inference_cmd: Run inference on a model with prompts
"""

from ._types import (
    GenerationResult,
    InferenceConfig,
    InferenceResult,
    InputMode,
)
from .run import run_inference_cmd

# Alias for backwards compatibility
run_inference = run_inference_cmd

__all__ = [
    # Types
    "GenerationResult",
    "InferenceConfig",
    "InferenceResult",
    "InputMode",
    # Commands
    "run_inference",
    "run_inference_cmd",
]
