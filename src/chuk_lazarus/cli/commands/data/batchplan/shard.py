"""Shard batch plan command."""

from __future__ import annotations

import logging
from argparse import Namespace

from ._types import BatchPlanShardConfig, BatchPlanShardResult

logger = logging.getLogger(__name__)


async def data_batchplan_shard(config: BatchPlanShardConfig) -> BatchPlanShardResult:
    """Save sharded batch plans for distributed training.

    Args:
        config: Shard configuration.

    Returns:
        Sharding result.
    """
    from chuk_lazarus.data.batching import load_batch_plan, save_batch_plan

    # Load original plan
    logger.info(f"Loading batch plan: {config.plan}")
    plan = load_batch_plan(config.plan)

    # Create output directory
    config.output.mkdir(parents=True, exist_ok=True)

    # Create sharded plans
    shard_details = []
    for rank in range(config.world_size):
        sharded = plan.shard(rank, config.world_size)
        shard_path = config.output / f"rank_{rank}"
        save_batch_plan(sharded, shard_path)

        shard_details.append(
            {
                "rank": rank,
                "batches": sharded.total_microbatches,
                "path": str(shard_path),
            }
        )

    return BatchPlanShardResult(
        source_plan=str(config.plan),
        world_size=config.world_size,
        total_batches=plan.total_microbatches,
        shard_details=shard_details,
        output_dir=config.output,
    )


async def data_batchplan_shard_cmd(args: Namespace) -> None:
    """CLI entry point for batchplan shard command.

    Args:
        args: Parsed command line arguments.
    """
    config = BatchPlanShardConfig.from_args(args)
    result = await data_batchplan_shard(config)
    print(result.to_display())
