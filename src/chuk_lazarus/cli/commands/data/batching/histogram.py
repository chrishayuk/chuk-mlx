"""Display length histogram command."""

from __future__ import annotations

from argparse import Namespace

from ._types import HistogramConfig, HistogramResult


async def data_batching_histogram(config: HistogramConfig) -> HistogramResult:
    """Display length histogram for a dataset.

    Args:
        config: Histogram configuration.

    Returns:
        Histogram result with percentiles.
    """
    from chuk_lazarus.data.batching import (
        LengthCache,
        compute_length_histogram,
    )

    cache = await LengthCache.load(config.cache)
    lengths = cache.get_all()

    histogram = compute_length_histogram(lengths, num_bins=config.bins)

    return HistogramResult(
        histogram_ascii=histogram.to_ascii(width=config.width),
        p25=histogram.p25,
        p50=histogram.p50,
        p75=histogram.p75,
        p90=histogram.p90,
        p95=histogram.p95,
        p99=histogram.p99,
    )


async def data_batching_histogram_cmd(args: Namespace) -> None:
    """CLI entry point for batching histogram command.

    Args:
        args: Parsed command line arguments.
    """
    config = HistogramConfig.from_args(args)
    result = await data_batching_histogram(config)
    print(result.to_display())
