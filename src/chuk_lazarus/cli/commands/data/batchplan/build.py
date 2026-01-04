"""Build batch plan command."""

from __future__ import annotations

import logging
from argparse import Namespace

from ._types import BatchPlanBuildConfig, BatchPlanBuildResult

logger = logging.getLogger(__name__)


async def data_batchplan_build(config: BatchPlanBuildConfig) -> BatchPlanBuildResult:
    """Build a batch plan from length cache.

    Args:
        config: Build configuration.

    Returns:
        Build result with plan details.
    """
    from chuk_lazarus.data.batching import (
        BatchingConfig,
        BatchPlanBuilder,
        LengthCache,
        save_batch_plan,
    )

    # Load length cache
    logger.info(f"Loading length cache: {config.lengths}")
    cache = await LengthCache.load(config.lengths)
    lengths = cache.get_all()

    # Parse bucket edges
    bucket_edges = config.get_bucket_edges()

    # Create batching config
    if config.predictable:
        batching_config = BatchingConfig.predictable(
            token_budget=config.token_budget,
            bucket_edges=bucket_edges,
            overflow_max=config.overflow_max,
            seed=config.seed,
        )
    else:
        batching_config = BatchingConfig.throughput(
            token_budget=config.token_budget,
            bucket_edges=bucket_edges,
            overflow_max=config.overflow_max,
        )

    # Build plan
    logger.info(f"Building batch plan for {config.epochs} epochs...")
    builder = BatchPlanBuilder(
        lengths=lengths,
        batching_config=batching_config,
        dataset_hash=config.dataset_hash or "unknown",
        tokenizer_hash=cache.tokenizer_hash,
    )

    plan = await builder.build(num_epochs=config.epochs)

    # Save plan
    save_batch_plan(plan, config.output)

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

    return BatchPlanBuildResult(
        lengths_cache=str(config.lengths),
        epochs=plan.num_epochs,
        token_budget=config.token_budget,
        mode=config.mode,
        total_batches=plan.total_microbatches,
        fingerprint=plan.fingerprint,
        output_path=config.output,
        epoch_details=epoch_details,
    )


async def data_batchplan_build_cmd(args: Namespace) -> None:
    """CLI entry point for batchplan build command.

    Args:
        args: Parsed command line arguments.
    """
    config = BatchPlanBuildConfig.from_args(args)
    result = await data_batchplan_build(config)
    print(result.to_display())
