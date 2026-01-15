#!/usr/bin/env python3
"""
Example 1: Basic Token-Budget Batching

Demonstrates:
- Creating samples with the canonical schema
- Building a length cache
- Using TokenBudgetBatchSampler to form efficient batches
- Computing and displaying batching metrics

Run:
    python examples/batching/01_basic_batching.py
"""

import asyncio
import tempfile
from pathlib import Path

from chuk_lazarus.data import (
    BucketSpec,
    LengthCache,
    Sample,
    SampleMeta,
    TokenBudgetBatchSampler,
)


def create_sample_dataset(num_samples: int = 100) -> list[Sample]:
    """Create a synthetic dataset with varying sequence lengths."""
    import random

    random.seed(42)
    samples = []

    for i in range(num_samples):
        # Simulate realistic length distribution (many short, few long)
        if random.random() < 0.6:
            length = random.randint(50, 150)  # Short sequences
        elif random.random() < 0.8:
            length = random.randint(150, 400)  # Medium sequences
        else:
            length = random.randint(400, 800)  # Long sequences

        # Create sample with random tokens
        input_ids = list(range(length))
        # SFT-style: first 30% is prompt (no loss), rest is completion
        prompt_len = int(length * 0.3)
        loss_mask = [0] * prompt_len + [1] * (length - prompt_len)

        sample = Sample(
            input_ids=input_ids,
            loss_mask=loss_mask,
            meta=SampleMeta(
                sample_id=f"sample_{i:04d}",
                dataset_id="synthetic_demo",
            ),
        )
        samples.append(sample)

    return samples


async def build_length_cache(
    samples: list[Sample],
    cache_path: Path,
    tokenizer_hash: str = "demo_tokenizer_v1",
) -> LengthCache:
    """Build a length cache from samples."""
    async with LengthCache.create(cache_path, tokenizer_hash) as cache:
        for sample in samples:
            await cache.add(sample.meta.sample_id, sample.length)

    # Load the cache back
    return await LengthCache.load(cache_path)


async def main():
    print("=" * 60)
    print("Token-Budget Batching Demo")
    print("=" * 60)

    # 1. Create synthetic dataset
    print("\n1. Creating synthetic dataset...")
    samples = create_sample_dataset(num_samples=100)
    print(f"   Created {len(samples)} samples")

    lengths = [s.length for s in samples]
    print(f"   Length range: {min(lengths)} - {max(lengths)}")
    print(f"   Average length: {sum(lengths) / len(lengths):.1f}")

    # 2. Build length cache
    print("\n2. Building length cache...")
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "lengths.jsonl"
        cache = await build_length_cache(samples, cache_path)
        print(f"   Cached {len(cache)} sample lengths")

        # 3. Configure bucketing
        print("\n3. Configuring buckets...")
        bucket_spec = BucketSpec(
            edges=(128, 256, 512),
            overflow_max=1024,
        )
        print(f"   Bucket edges: {bucket_spec.edges}")
        print(f"   Overflow max: {bucket_spec.overflow_max}")
        print(f"   Number of buckets: {bucket_spec.num_buckets}")

        # Show bucket ranges
        for i in range(bucket_spec.num_buckets):
            min_len, max_len = bucket_spec.get_bucket_range(i)
            is_overflow = " (overflow)" if bucket_spec.is_overflow(i) else ""
            print(f"   Bucket {i}: {min_len}-{max_len}{is_overflow}")

        # 4. Create sampler
        print("\n4. Creating TokenBudgetBatchSampler...")
        token_budget = 2048
        sampler = TokenBudgetBatchSampler(
            lengths=cache.get_all(),
            bucket_spec=bucket_spec,
            token_budget=token_budget,
            seed=42,
        )
        print(f"   Token budget: {token_budget}")
        print(f"   Samples to batch: {sampler.num_samples}")
        print(f"   Skipped (out of range): {sampler.num_skipped}")
        print(f"   Estimated batches/epoch: {sampler.estimate_batches_per_epoch()}")

        # Show bucket distribution
        print("\n   Bucket distribution:")
        for bucket_id, count in sorted(sampler.bucket_sizes().items()):
            pct = count / sampler.num_samples * 100
            print(f"   Bucket {bucket_id}: {count} samples ({pct:.1f}%)")

        # 5. Iterate one epoch
        print("\n5. Iterating epoch 0...")
        batches = []
        async for batch_spec in sampler.iter_epoch(epoch=0):
            batches.append(batch_spec)

        print(f"   Generated {len(batches)} batches")

        # Show first few batches
        print("\n   First 5 batches:")
        for i, batch in enumerate(batches[:5]):
            print(
                f"   Batch {i}: {batch.batch_size} samples, "
                f"bucket={batch.bucket_id}, "
                f"max_len={batch.max_length}, "
                f"tokens={batch.token_count}"
            )

        # 6. Compute metrics
        print("\n6. Computing efficiency metrics...")
        # Build loss token counts
        loss_tokens = {s.meta.sample_id: s.num_loss_tokens for s in samples}
        metrics = sampler.compute_metrics(loss_tokens_per_sample=loss_tokens)

        print("\n   Summary:")
        for key, value in metrics.summary().items():
            print(f"   {key}: {value}")

        print("\n   Per-bucket efficiency:")
        for bucket_info in metrics.bucket_summary():
            print(
                f"   Bucket {bucket_info['bucket_id']}: "
                f"{bucket_info['samples']} samples, "
                f"efficiency={bucket_info['efficiency']}, "
                f"avg_len={bucket_info['avg_length']}"
            )

        # 7. Verify determinism
        print("\n7. Verifying deterministic batching...")
        batches_run2 = [b async for b in sampler.iter_epoch(epoch=0)]

        if len(batches) == len(batches_run2):
            all_match = all(b1.sample_ids == b2.sample_ids for b1, b2 in zip(batches, batches_run2))
            if all_match:
                print("   ✓ Same epoch produces identical batches")
            else:
                print("   ✗ Batches differ (unexpected)")
        else:
            print("   ✗ Different batch counts (unexpected)")

        # Different epoch should have different order
        batches_epoch1 = [b async for b in sampler.iter_epoch(epoch=1)]
        samples_e0 = set(batches[0].sample_ids)
        samples_e1 = set(batches_epoch1[0].sample_ids)
        if samples_e0 != samples_e1:
            print("   ✓ Different epochs have different orderings")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
