"""
Batch plan sharding utilities.

Provides functions for sharding batch plans across distributed workers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..data.batching import BatchPlan, MicrobatchSpec

from .config import DistributedConfig


def shard_batch_plan(
    plan: BatchPlan,
    config: DistributedConfig | None = None,
) -> BatchPlan:
    """
    Shard a batch plan for the current worker.

    Shards by microbatch index: rank i gets microbatches i, i+world_size, ...

    Args:
        plan: The batch plan to shard
        config: Distributed config (uses global if not provided)

    Returns:
        Sharded batch plan for this worker
    """
    if config is None:
        config = DistributedConfig.get_global()

    if not config.is_distributed:
        return plan

    return plan.shard(config.rank, config.world_size)


def interleave_microbatches(
    microbatches: list[MicrobatchSpec],
) -> list[MicrobatchSpec]:
    """
    Interleave microbatches from different buckets.

    This ensures balanced work distribution when sharding across ranks.
    Instead of having all short-sequence batches first (from bucket 0),
    interleaving mixes batches from different buckets.

    Args:
        microbatches: List of microbatches (may be from multiple buckets)

    Returns:
        Interleaved list of microbatches
    """
    # Group by bucket
    by_bucket: dict[int, list] = {}
    for mb in microbatches:
        bucket_id = mb.bucket_id
        if bucket_id not in by_bucket:
            by_bucket[bucket_id] = []
        by_bucket[bucket_id].append(mb)

    if len(by_bucket) <= 1:
        # Only one bucket, nothing to interleave
        return microbatches

    # Round-robin interleave
    result = []
    bucket_ids = sorted(by_bucket.keys())
    indices = dict.fromkeys(bucket_ids, 0)

    total = len(microbatches)
    while len(result) < total:
        for bid in bucket_ids:
            if indices[bid] < len(by_bucket[bid]):
                result.append(by_bucket[bid][indices[bid]])
                indices[bid] += 1

    return result


def compute_shard_stats(
    plan: BatchPlan,
    world_size: int,
) -> dict[int, dict]:
    """
    Compute statistics for each shard.

    Useful for verifying balanced work distribution.

    Args:
        plan: The batch plan to analyze
        world_size: Number of workers

    Returns:
        Dict mapping rank to stats dict with:
        - num_batches: Number of microbatches for this rank
        - total_samples: Total samples for this rank
        - total_tokens: Total tokens (estimated) for this rank
    """
    stats = {}

    for rank in range(world_size):
        sharded = plan.shard(rank, world_size)
        stats[rank] = {
            "num_batches": sharded.total_microbatches,
            "total_samples": sum(ep.total_samples for ep in sharded.epochs.values()),
            "total_tokens": sum(ep.total_tokens for ep in sharded.epochs.values()),
        }

    return stats
