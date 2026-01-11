"""Length cache statistics command."""

from __future__ import annotations

from argparse import Namespace

from ._types import EmptyStatsResult, LengthStatsConfig, LengthStatsResult


def _percentile(values: list[int], p: int) -> int:
    """Calculate percentile value.

    Args:
        values: Sorted list of values.
        p: Percentile (0-100).

    Returns:
        Value at the percentile.
    """
    idx = int(len(values) * p / 100)
    return values[min(idx, len(values) - 1)]


async def data_lengths_stats(
    config: LengthStatsConfig,
) -> LengthStatsResult | EmptyStatsResult:
    """Show statistics for a length cache.

    Args:
        config: Stats configuration.

    Returns:
        Statistics result or empty result.
    """
    from chuk_lazarus.data.batching import LengthCache

    cache = await LengthCache.load(config.cache)
    lengths = cache.get_all()

    if not lengths:
        return EmptyStatsResult()

    values = sorted(lengths.values())
    total_tokens = sum(values)

    return LengthStatsResult(
        cache_path=str(config.cache),
        tokenizer_hash=cache.tokenizer_hash,
        total_samples=len(lengths),
        total_tokens=total_tokens,
        min_length=min(values),
        max_length=max(values),
        mean_length=total_tokens / len(values),
        median_length=values[len(values) // 2],
        p10=_percentile(values, 10),
        p25=_percentile(values, 25),
        p50=_percentile(values, 50),
        p75=_percentile(values, 75),
        p90=_percentile(values, 90),
        p95=_percentile(values, 95),
        p99=_percentile(values, 99),
    )


async def data_lengths_stats_cmd(args: Namespace) -> None:
    """CLI entry point for data lengths stats command.

    Args:
        args: Parsed command line arguments.
    """
    config = LengthStatsConfig.from_args(args)
    result = await data_lengths_stats(config)
    print(result.to_display())
