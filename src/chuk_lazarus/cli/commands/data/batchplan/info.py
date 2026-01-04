"""Batch plan info command."""

from __future__ import annotations

from argparse import Namespace

from ._types import (
    BatchPlanInfoConfig,
    BatchPlanInfoResult,
    InvalidRankError,
)


async def data_batchplan_info(
    config: BatchPlanInfoConfig,
) -> BatchPlanInfoResult | InvalidRankError:
    """Show information about a batch plan.

    Args:
        config: Info configuration.

    Returns:
        Plan info result or error.
    """
    from chuk_lazarus.data.batching import load_batch_plan

    plan = load_batch_plan(config.plan)

    # Apply sharding if requested
    shard_info = None
    if config.rank is not None and config.world_size is not None:
        if config.rank >= config.world_size or config.rank < 0:
            return InvalidRankError(rank=config.rank, world_size=config.world_size)
        plan = plan.shard(config.rank, config.world_size)
        shard_info = f"rank {config.rank}/{config.world_size}"

    # Collect epoch details
    epoch_details = []
    for ep in range(plan.num_epochs):
        epoch_plan = plan.get_epoch(ep)
        epoch_details.append(
            {
                "epoch": ep,
                "batches": epoch_plan.num_microbatches,
                "samples": epoch_plan.total_samples,
                "tokens": epoch_plan.total_tokens,
            }
        )

    # Collect sample batches if requested
    sample_batches = []
    if config.show_batches:
        epoch0 = plan.get_epoch(0)
        for i, mb in enumerate(epoch0.microbatches[: config.show_batches]):
            sample_batches.append(
                {
                    "index": i,
                    "size": mb.batch_size,
                    "bucket_id": mb.bucket_id,
                    "max_len": mb.max_len,
                }
            )

    return BatchPlanInfoResult(
        plan_path=str(config.plan),
        fingerprint=plan.fingerprint,
        created_at=plan.meta.created_at,
        dataset_hash=plan.meta.dataset_hash,
        tokenizer_hash=plan.meta.tokenizer_hash,
        token_budget=plan.meta.token_budget,
        bucket_edges=list(plan.meta.bucket_edges),
        epochs=plan.num_epochs,
        total_batches=plan.total_microbatches,
        shard_info=shard_info,
        epoch_details=epoch_details,
        sample_batches=sample_batches,
    )


async def data_batchplan_info_cmd(args: Namespace) -> None:
    """CLI entry point for batchplan info command.

    Args:
        args: Parsed command line arguments.
    """
    config = BatchPlanInfoConfig.from_args(args)
    result = await data_batchplan_info(config)
    print(result.to_display())
