"""Data generation command handler.

This module provides the async data generation implementation.
"""

from __future__ import annotations

import logging
import sys
from argparse import Namespace

from ._types import DataGenConfig, DataGenResult, DataGenType

logger = logging.getLogger(__name__)


async def generate_data(config: DataGenConfig) -> DataGenResult:
    """Generate synthetic training data.

    Args:
        config: Data generation configuration

    Returns:
        DataGenResult with generation outcomes

    Raises:
        SystemExit: If unknown data type is specified
    """
    from ....data.generators import generate_lazarus_dataset

    if config.type == DataGenType.MATH:
        logger.info(f"Generating math dataset with {config.sft_samples} SFT samples")
        generate_lazarus_dataset(
            output_dir=str(config.output),
            sft_samples=config.sft_samples,
            dpo_samples=config.dpo_samples,
            seed=config.seed,
        )
        logger.info(f"Dataset saved to {config.output}")

        return DataGenResult(
            type=config.type,
            output_dir=config.output,
            sft_samples=config.sft_samples,
            dpo_samples=config.dpo_samples,
        )
    else:
        logger.error(f"Unknown data type: {config.type}")
        sys.exit(1)


async def generate_data_cmd(args: Namespace) -> None:
    """CLI entry point for data generation command.

    Args:
        args: Parsed command-line arguments
    """
    config = DataGenConfig.from_args(args)
    result = await generate_data(config)
    print(result.to_display())


__all__ = [
    "generate_data",
    "generate_data_cmd",
]
