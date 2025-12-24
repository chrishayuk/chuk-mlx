#!/usr/bin/env python3
"""
Example 4: Distributed Training with Batch Plans

Demonstrates:
- Configuring distributed training with DistributedConfig
- Sharding batch plans across workers
- Pre-sharding plans with CLI
- Checkpoint resume with CheckpointPosition
- Balanced work distribution

Run:
    python examples/batching/04_distributed.py

CLI equivalent:
    # View sharded plan for rank 0 of 4 workers
    lazarus data batchplan info -p batch_plan/ --rank 0 --world-size 4

    # Pre-shard plan for 4 workers
    lazarus data batchplan shard -p batch_plan/ -w 4 -o shards/
"""

import asyncio
import tempfile
from pathlib import Path

from chuk_lazarus.data import (
    BatchingConfig,
    BatchPlanBuilder,
    load_batch_plan,
    save_batch_plan,
)
from chuk_lazarus.distributed import (
    CheckpointPosition,
    DistributedConfig,
    get_rank,
    get_world_size,
    interleave_microbatches,
    is_main_process,
    load_checkpoint_position,
    save_checkpoint_position,
    shard_batch_plan,
)
from chuk_lazarus.distributed.sharding import compute_shard_stats


def create_sample_lengths(num_samples: int = 500) -> dict[str, int]:
    """Create synthetic length data with realistic distribution."""
    import random

    random.seed(42)
    lengths = {}

    for i in range(num_samples):
        # Mix of short, medium, and long sequences
        if random.random() < 0.4:
            length = random.randint(50, 150)  # Short
        elif random.random() < 0.7:
            length = random.randint(150, 400)  # Medium
        else:
            length = random.randint(400, 900)  # Long

        lengths[f"sample_{i:05d}"] = length

    return lengths


async def main():
    print("=" * 70)
    print("Distributed Training Demo")
    print("=" * 70)

    # =========================================================================
    # 1. DistributedConfig Basics
    # =========================================================================
    print("\n1. DistributedConfig Basics")
    print("-" * 40)

    # Default: single worker
    config_single = DistributedConfig()
    print(f"   Single worker: rank={config_single.rank}, world_size={config_single.world_size}")
    print(f"   is_distributed: {config_single.is_distributed}")
    print(f"   is_main: {config_single.is_main}")

    # Multi-worker simulation
    config_rank0 = DistributedConfig(rank=0, world_size=4)
    config_rank1 = DistributedConfig(rank=1, world_size=4)

    print(f"\n   Rank 0 of 4: is_main={config_rank0.is_main}")
    print(f"   Rank 1 of 4: is_main={config_rank1.is_main}")

    # From environment (in real distributed setup)
    print("\n   In real distributed training, use:")
    print("   config = DistributedConfig.from_env()")
    print("   # Reads RANK, WORLD_SIZE, LOCAL_RANK, LOCAL_WORLD_SIZE")

    # =========================================================================
    # 2. Build a Batch Plan
    # =========================================================================
    print("\n2. Building Batch Plan")
    print("-" * 40)

    lengths = create_sample_lengths(num_samples=500)
    print(f"   Dataset: {len(lengths)} samples")
    print(f"   Length range: {min(lengths.values())} - {max(lengths.values())}")

    batching_config = BatchingConfig.predictable(
        token_budget=4096,
        bucket_edges=(128, 256, 512),
        overflow_max=1024,
        seed=42,
    )

    builder = BatchPlanBuilder(
        lengths=lengths,
        batching_config=batching_config,
        dataset_hash="distributed_demo_v1",
        tokenizer_hash="tokenizer_v1",
    )

    plan = await builder.build(num_epochs=3)
    print(f"   Built plan: {plan.num_epochs} epochs, {plan.total_microbatches} microbatches")

    # =========================================================================
    # 3. Shard Batch Plan
    # =========================================================================
    print("\n3. Sharding Batch Plan for Distributed Training")
    print("-" * 40)

    world_size = 4
    print(f"   Simulating {world_size} workers...\n")

    # Method 1: Using shard_batch_plan with config
    for rank in range(world_size):
        config = DistributedConfig(rank=rank, world_size=world_size)
        sharded = shard_batch_plan(plan, config)

        epoch0 = sharded.get_epoch(0)
        print(
            f"   Rank {rank}: {sharded.total_microbatches:3d} total batches, "
            f"epoch 0: {epoch0.num_microbatches:3d} batches, "
            f"{epoch0.total_samples:4d} samples"
        )

    # Method 2: Using plan.shard() directly
    print("\n   Or use plan.shard() directly:")
    shard = plan.shard(rank=0, world_size=4)
    print(f"   plan.shard(0, 4) -> {shard.total_microbatches} batches")

    # =========================================================================
    # 4. Work Distribution Analysis
    # =========================================================================
    print("\n4. Work Distribution Analysis")
    print("-" * 40)

    stats = compute_shard_stats(plan, world_size=4)

    print("   Batches per rank:")
    total_batches = sum(s["num_batches"] for s in stats.values())
    for rank, s in stats.items():
        pct = s["num_batches"] / total_batches * 100
        bar = "█" * int(pct / 5)
        print(f"   Rank {rank}: {s['num_batches']:4d} ({pct:5.1f}%) {bar}")

    print("\n   Tokens per rank:")
    total_tokens = sum(s["total_tokens"] for s in stats.values())
    for rank, s in stats.items():
        pct = s["total_tokens"] / total_tokens * 100
        print(f"   Rank {rank}: {s['total_tokens']:,} tokens ({pct:.1f}%)")

    # =========================================================================
    # 5. Interleaving for Balanced Sharding
    # =========================================================================
    print("\n5. Interleaving Microbatches for Balance")
    print("-" * 40)

    # Get epoch 0 microbatches
    epoch0 = plan.get_epoch(0)
    original_mbs = list(epoch0.microbatches)

    # Count by bucket before interleaving
    from collections import Counter
    bucket_counts = Counter(mb.bucket_id for mb in original_mbs)
    print("   Bucket distribution:")
    for bucket_id, count in sorted(bucket_counts.items()):
        print(f"   Bucket {bucket_id}: {count} batches")

    # Interleave
    interleaved = interleave_microbatches(original_mbs)

    # Show first 12 bucket IDs before and after
    print("\n   First 12 bucket IDs:")
    original_ids = [mb.bucket_id for mb in original_mbs[:12]]
    interleaved_ids = [mb.bucket_id for mb in interleaved[:12]]
    print(f"   Original:    {original_ids}")
    print(f"   Interleaved: {interleaved_ids}")

    print("\n   Interleaving spreads buckets evenly so each rank")
    print("   gets a mix of short and long sequences.")

    # =========================================================================
    # 6. Pre-sharding Plans (CLI Style)
    # =========================================================================
    print("\n6. Pre-sharding Plans to Disk")
    print("-" * 40)

    with tempfile.TemporaryDirectory() as tmpdir:
        shards_dir = Path(tmpdir) / "shards"
        shards_dir.mkdir()

        # Save original plan
        plan_path = Path(tmpdir) / "original_plan"
        save_batch_plan(plan, plan_path)
        print(f"   Saved original plan to: {plan_path.name}/")

        # Pre-shard for each rank
        for rank in range(world_size):
            shard = plan.shard(rank, world_size)
            shard_path = shards_dir / f"rank_{rank}"
            save_batch_plan(shard, shard_path)

        # List saved shards
        print(f"\n   Pre-sharded to: {shards_dir.name}/")
        for shard_dir in sorted(shards_dir.iterdir()):
            loaded = load_batch_plan(shard_dir)
            print(f"   {shard_dir.name}/: {loaded.total_microbatches} batches")

        print("\n   CLI equivalent:")
        print("   lazarus data batchplan shard -p original_plan/ -w 4 -o shards/")

    # =========================================================================
    # 7. Checkpoint Resume
    # =========================================================================
    print("\n7. Checkpoint Resume")
    print("-" * 40)

    # Simulate training progress
    checkpoint_pos = CheckpointPosition(
        epoch=1,
        microbatch_idx=15,
        global_step=47,
    )
    print(f"   Checkpoint: epoch={checkpoint_pos.epoch}, "
          f"mb_idx={checkpoint_pos.microbatch_idx}, "
          f"global_step={checkpoint_pos.global_step}")

    # Save and load checkpoint position
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "checkpoint_position.json"
        save_checkpoint_position(checkpoint_pos, ckpt_path)
        loaded_pos = load_checkpoint_position(ckpt_path)
        print(f"   Saved/loaded checkpoint position: ✓")

    # Resume iteration
    remaining = list(plan.iter_from(
        epoch=checkpoint_pos.epoch,
        microbatch_idx=checkpoint_pos.microbatch_idx,
    ))

    total_remaining = len(remaining)
    epochs_remaining = len(set(ep for ep, _, _ in remaining))
    print(f"\n   Resuming from checkpoint:")
    print(f"   - {total_remaining} microbatches remaining")
    print(f"   - Across {epochs_remaining} epoch(s)")

    # Show first few
    print("\n   Next 5 microbatches:")
    for ep, mb_idx, mb in remaining[:5]:
        print(f"   epoch={ep}, mb_idx={mb_idx}, samples={mb.batch_size}, bucket={mb.bucket_id}")

    # =========================================================================
    # 8. Distributed Training Loop Pattern
    # =========================================================================
    print("\n8. Distributed Training Loop Pattern")
    print("-" * 40)

    print("""
    # In each worker process:

    from chuk_lazarus.distributed import (
        DistributedConfig,
        shard_batch_plan,
        CheckpointPosition,
        load_checkpoint_position,
    )
    from chuk_lazarus.data import load_batch_plan

    # 1. Get distributed config from environment
    config = DistributedConfig.from_env()

    # 2. Load and shard batch plan
    plan = load_batch_plan("./batch_plan/")
    my_plan = shard_batch_plan(plan, config)

    # 3. Resume from checkpoint if exists
    checkpoint_path = f"./checkpoints/rank_{config.rank}/position.json"
    if Path(checkpoint_path).exists():
        pos = load_checkpoint_position(checkpoint_path)
        batches = my_plan.iter_from(pos.epoch, pos.microbatch_idx)
    else:
        batches = my_plan.iter_from(epoch=0, microbatch_idx=0)

    # 4. Training loop
    for epoch, mb_idx, mb in batches:
        # Load samples
        samples = load_samples(mb.sample_ids)

        # Forward/backward
        loss = train_step(model, samples)

        # Save checkpoint periodically
        if mb_idx % 100 == 0:
            save_checkpoint_position(
                CheckpointPosition(epoch, mb_idx, global_step),
                checkpoint_path
            )
    """)

    # =========================================================================
    # 9. CLI Commands Summary
    # =========================================================================
    print("\n9. CLI Commands Summary")
    print("-" * 40)

    print("""
    # View batch plan info
    lazarus data batchplan info -p batch_plan/

    # View sharded plan for a specific rank
    lazarus data batchplan info -p batch_plan/ --rank 0 --world-size 4

    # Pre-shard batch plan for all workers
    lazarus data batchplan shard -p batch_plan/ --world-size 4 -o shards/

    # Each worker loads its shard:
    # Worker 0: shards/rank_0/
    # Worker 1: shards/rank_1/
    # ...
    """)

    print("=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
