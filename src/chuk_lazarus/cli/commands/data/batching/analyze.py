"""Analyze batching efficiency command."""

from __future__ import annotations

import json
import logging
from argparse import Namespace

from ._types import AnalyzeConfig, AnalyzeResult

logger = logging.getLogger(__name__)


async def data_batching_analyze(config: AnalyzeConfig) -> AnalyzeResult:
    """Analyze batching efficiency for a dataset.

    Args:
        config: Analysis configuration.

    Returns:
        Analysis result with efficiency report.
    """
    from chuk_lazarus.data.batching import (
        BucketSpec,
        LengthCache,
        create_efficiency_report,
    )

    # Load length cache
    logger.info(f"Loading length cache: {config.cache}")
    cache = await LengthCache.load(config.cache)
    lengths = cache.get_all()

    # Parse bucket edges
    bucket_edges = config.get_bucket_edges()

    # Create bucket spec
    bucket_spec = BucketSpec(
        edges=bucket_edges,
        overflow_max=config.overflow_max,
    )

    # Create efficiency report
    report = create_efficiency_report(lengths, bucket_spec)

    # Save JSON if output specified
    if config.output:
        with open(config.output, "w") as f:
            json.dump(report.model_dump(), f, indent=2, default=str)

    return AnalyzeResult(
        report_ascii=report.to_ascii(),
        output_path=config.output,
    )


async def data_batching_analyze_cmd(args: Namespace) -> None:
    """CLI entry point for batching analyze command.

    Args:
        args: Parsed command line arguments.
    """
    config = AnalyzeConfig.from_args(args)
    result = await data_batching_analyze(config)
    print(result.to_display())
