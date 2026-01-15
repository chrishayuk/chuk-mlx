#!/usr/bin/env python3
"""
Example 3: BatchPlan for Reproducible Distributed Training

Demonstrates:
- Building a complete batch plan for multiple epochs
- Saving and loading batch plans
- Sharding plans across distributed workers
- Resuming from checkpoints
- Verifying reproducibility with fingerprints

Run:
    python examples/batching/03_batch_plan.py
"""

import asyncio
import tempfile
from pathlib import Path

from chuk_lazarus.data import (
    BatchingConfig,
    BatchPlanBuilder,
    compute_batch_fingerprint,
    load_batch_plan,
    save_batch_plan,
    verify_batch_fingerprint,
)


def create_sample_lengths(num_samples: int = 200) -> dict[str, int]:
    """Create synthetic length data."""
    import random

    random.seed(42)
    lengths = {}

    for i in range(num_samples):
        # Realistic distribution
        if random.random() < 0.5:
            length = random.randint(50, 150)
        elif random.random() < 0.8:
            length = random.randint(150, 350)
        else:
            length = random.randint(350, 700)

        lengths[f"sample_{i:04d}"] = length

    return lengths


async def main():
    print("=" * 60)
    print("BatchPlan Demo")
    print("=" * 60)

    # 1. Create sample lengths
    print("\n1. Creating sample dataset...")
    lengths = create_sample_lengths(num_samples=200)
    print(f"   Created {len(lengths)} samples")
    print(f"   Length range: {min(lengths.values())} - {max(lengths.values())}")

    # 2. Configure batching
    print("\n2. Configuring batching...")
    config = BatchingConfig.predictable(
        token_budget=4096,
        bucket_edges=(128, 256, 512),
        overflow_max=1024,
        seed=42,
    )
    print(f"   Mode: {config.mode.value}")
    print(f"   Token budget: {config.token_budget}")
    print(f"   Bucket edges: {config.bucket_edges}")
    print(f"   Seed: {config.seed}")

    # 3. Build batch plan
    print("\n3. Building batch plan...")
    builder = BatchPlanBuilder(
        lengths=lengths,
        batching_config=config,
        dataset_hash="demo_dataset_v1",
        tokenizer_hash="demo_tokenizer_v1",
    )

    num_epochs = 3
    plan = await builder.build(num_epochs=num_epochs)

    print(f"   Built plan for {plan.num_epochs} epochs")
    print(f"   Total microbatches: {plan.total_microbatches}")
    print(f"   Plan fingerprint: {plan.fingerprint}")

    # Show epoch details
    print("\n   Per-epoch details:")
    for ep in range(num_epochs):
        epoch_plan = plan.get_epoch(ep)
        print(
            f"   Epoch {ep}: {epoch_plan.num_microbatches} microbatches, "
            f"{epoch_plan.total_samples} samples, "
            f"{epoch_plan.total_tokens} tokens"
        )

    # 4. Examine microbatches
    print("\n4. Sample microbatches from epoch 0:")
    epoch0 = plan.get_epoch(0)
    for i, mb in enumerate(epoch0.microbatches[:5]):
        print(f"   MB {i}: {mb.batch_size} samples, bucket={mb.bucket_id}, max_len={mb.max_len}")
        if mb.batch_size <= 4:
            print(f"         samples: {list(mb.samples)}")

    # 5. Save and load plan
    print("\n5. Saving and loading batch plan...")
    with tempfile.TemporaryDirectory() as tmpdir:
        plan_path = Path(tmpdir) / "my_plan"
        save_batch_plan(plan, plan_path)

        # Check files
        files = list(plan_path.glob("*"))
        print(f"   Saved to {plan_path}")
        print(f"   Files created: {[f.name for f in files]}")

        # Load
        loaded_plan = load_batch_plan(plan_path)
        print(f"\n   Loaded plan fingerprint: {loaded_plan.fingerprint}")

        # Verify they match
        if plan.fingerprint == loaded_plan.fingerprint:
            print("   ✓ Plans match!")
        else:
            print("   ✗ Plans differ (unexpected)")

    # 6. Distributed sharding
    print("\n6. Sharding for distributed training...")
    world_size = 4

    print(f"   Simulating {world_size} workers:")
    for rank in range(world_size):
        shard = plan.shard(rank=rank, world_size=world_size)
        epoch0_shard = shard.get_epoch(0)
        print(
            f"   Rank {rank}: {epoch0_shard.num_microbatches} microbatches, "
            f"{epoch0_shard.total_samples} samples"
        )

    # Verify shards cover all data
    print("\n   Verifying shard coverage:")
    all_samples = set()
    for rank in range(world_size):
        shard = plan.shard(rank=rank, world_size=world_size)
        for mb in shard.iter_epoch(0):
            all_samples.update(mb.samples)

    original_samples = set()
    for mb in plan.iter_epoch(0):
        original_samples.update(mb.samples)

    if all_samples == original_samples:
        print(f"   ✓ All {len(original_samples)} samples covered by shards")
    else:
        missing = original_samples - all_samples
        print(f"   ✗ Missing samples: {len(missing)}")

    # 7. Resume from checkpoint
    print("\n7. Simulating checkpoint resume...")

    # Pretend we trained through epoch 0 and 5 microbatches of epoch 1
    resume_epoch = 1
    resume_mb_idx = 5

    print(f"   Resuming from epoch {resume_epoch}, microbatch {resume_mb_idx}")

    remaining = list(plan.iter_from(epoch=resume_epoch, microbatch_idx=resume_mb_idx))
    print(f"   Remaining: {len(remaining)} (epoch, mb_idx, mb) tuples")

    # Count remaining per epoch
    from collections import Counter

    epoch_counts = Counter(ep for ep, _, _ in remaining)
    for ep, count in sorted(epoch_counts.items()):
        print(f"   Epoch {ep}: {count} microbatches remaining")

    # 8. Fingerprint verification
    print("\n8. Batch fingerprint for reproducibility...")

    # Collect batches from epoch 0
    from chuk_lazarus.data.batching import BatchSpec

    epoch0_batches = [
        BatchSpec(
            sample_ids=mb.samples,
            bucket_id=mb.bucket_id,
            max_length=mb.max_len,
            token_count=sum(lengths[sid] for sid in mb.samples),
        )
        for mb in plan.get_epoch(0).microbatches
    ]

    fp = compute_batch_fingerprint(epoch0_batches, config=config)
    print(f"   Epoch 0 fingerprint: {fp.fingerprint}")
    print(f"   Num batches: {fp.num_batches}")
    print(f"   Total samples: {fp.total_samples}")
    print(f"   Total tokens: {fp.total_tokens}")

    # Verify it matches
    matches, error = verify_batch_fingerprint(epoch0_batches, fp)
    if matches:
        print("   ✓ Fingerprint verified!")
    else:
        print(f"   ✗ Verification failed: {error}")

    # 9. Rebuild and compare
    print("\n9. Verifying reproducibility...")
    plan2 = await builder.build(num_epochs=num_epochs)

    if plan.fingerprint == plan2.fingerprint:
        print("   ✓ Rebuilding produces identical plan")
    else:
        print("   ✗ Plans differ (unexpected)")

    # Check epoch-by-epoch
    for ep in range(num_epochs):
        mbs1 = list(plan.iter_epoch(ep))
        mbs2 = list(plan2.iter_epoch(ep))
        if len(mbs1) == len(mbs2):
            all_match = all(mb1.samples == mb2.samples for mb1, mb2 in zip(mbs1, mbs2))
            if all_match:
                print(f"   ✓ Epoch {ep}: identical ordering")
            else:
                print(f"   ✗ Epoch {ep}: sample order differs")
        else:
            print(f"   ✗ Epoch {ep}: different batch counts")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
