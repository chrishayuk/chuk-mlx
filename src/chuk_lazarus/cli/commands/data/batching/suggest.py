"""Suggest bucket edges command."""

from __future__ import annotations

from argparse import Namespace

from ._types import OptimizationGoalType, SuggestConfig, SuggestResult


async def data_batching_suggest(config: SuggestConfig) -> SuggestResult:
    """Suggest optimal bucket edges for a dataset.

    Args:
        config: Suggestion configuration.

    Returns:
        Suggestion result with recommended edges.
    """
    from chuk_lazarus.data.batching import (
        LengthCache,
        OptimizationGoal,
        suggest_bucket_edges,
    )

    cache = await LengthCache.load(config.cache)
    lengths = cache.get_all()

    # Map to optimization goal enum
    goal_map = {
        OptimizationGoalType.WASTE: OptimizationGoal.MINIMIZE_WASTE,
        OptimizationGoalType.BALANCE: OptimizationGoal.BALANCE_BUCKETS,
        OptimizationGoalType.MEMORY: OptimizationGoal.MINIMIZE_MEMORY,
    }
    goal = goal_map.get(config.goal, OptimizationGoal.MINIMIZE_WASTE)

    suggestion = suggest_bucket_edges(
        lengths,
        num_buckets=config.num_buckets,
        goal=goal,
        max_length=config.max_length,
    )

    return SuggestResult(
        goal=suggestion.optimization_goal.value,
        num_buckets=config.num_buckets,
        edges=list(suggestion.edges),
        overflow_max=suggestion.overflow_max,
        estimated_efficiency=suggestion.estimated_efficiency,
        rationale=suggestion.rationale,
    )


async def data_batching_suggest_cmd(args: Namespace) -> None:
    """CLI entry point for batching suggest command.

    Args:
        args: Parsed command line arguments.
    """
    config = SuggestConfig.from_args(args)
    result = await data_batching_suggest(config)
    print(result.to_display())
