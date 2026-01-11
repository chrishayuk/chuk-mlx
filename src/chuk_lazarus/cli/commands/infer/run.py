"""Inference command handler.

This module provides the async inference command implementation.
The CLI command is a thin wrapper that delegates to InferenceService.
"""

from __future__ import annotations

import logging
from argparse import Namespace

from ._types import InferenceConfig

logger = logging.getLogger(__name__)


async def run_inference_cmd(args: Namespace) -> None:
    """CLI entry point for inference command.

    This is a thin wrapper that:
    1. Converts CLI args to InferenceConfig
    2. Calls InferenceService.run() which handles all the logic
    3. Prints the result

    Args:
        args: Parsed command-line arguments
    """
    from ....inference import InferenceService

    # Convert CLI args to config
    config = InferenceConfig.from_args(args)

    # Run inference - all logic is in the service
    result = await InferenceService.run(config)

    # Print result
    print(result.to_display())


__all__ = [
    "run_inference_cmd",
]
