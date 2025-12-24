#!/usr/bin/env python3
"""
Example 7: Streaming Sample Sources

Demonstrates the streaming module for online learning:
- Using OfflineDatasetStream for static datasets
- Using AsyncOfflineDatasetStream for async I/O
- StreamMetrics for tracking sample statistics
- Sample filtering and transformation

Run:
    python examples/batching/07_streaming.py
"""

import asyncio
import json
import tempfile
from pathlib import Path

from chuk_lazarus.data.batching.streaming import (
    AsyncOfflineDatasetStream,
    OfflineDatasetStream,
    SampleSource,
    StreamMetrics,
    StreamSample,
    StreamState,
)


def create_sample_dataset(path: Path, num_samples: int = 50) -> None:
    """Create a synthetic JSONL dataset."""
    import random

    random.seed(42)

    with open(path, "w") as f:
        for i in range(num_samples):
            # Vary sequence lengths
            length = random.randint(50, 200)
            difficulty = random.random()
            success = random.random() > 0.3

            sample = {
                "sample_id": f"sample_{i:04d}",
                "dataset_id": "demo_stream",
                "input_ids": list(range(length)),
                "loss_mask": [0] * (length // 3) + [1] * (length - length // 3),
                "difficulty": difficulty,
                "success": success,
            }
            f.write(json.dumps(sample) + "\n")


def demo_sync_stream():
    """Demonstrate synchronous offline stream."""
    print("\n" + "=" * 60)
    print("1. Synchronous Offline Stream")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dataset
        dataset_path = Path(tmpdir) / "train.jsonl"
        create_sample_dataset(dataset_path)
        print(f"   Created dataset: {dataset_path}")

        # Create stream
        stream = OfflineDatasetStream(
            path=dataset_path,
            dataset_id="demo",
        )

        print(f"\n   Initial state: {stream.state}")
        assert stream.state == StreamState.IDLE

        # Iterate samples
        samples = []
        for sample in stream:
            samples.append(sample)
            if len(samples) == 5:
                break

        print(f"   After partial iteration: {stream.state}")
        print(f"   Collected {len(samples)} samples")

        # Show sample properties
        print("\n   First sample:")
        s = samples[0]
        print(f"     sample_id:  {s.sample_id}")
        print(f"     length:     {s.length}")
        print(f"     source:     {s.source}")
        print(f"     is_replay:  {s.is_replay}")
        print(f"     difficulty: {s.difficulty:.2f}")

        # Continue to end
        for sample in stream:
            samples.append(sample)

        print(f"\n   Final state: {stream.state}")
        print(f"   Total samples: {len(samples)}")
        assert stream.state == StreamState.EXHAUSTED


async def demo_async_stream():
    """Demonstrate async offline stream."""
    print("\n" + "=" * 60)
    print("2. Async Offline Stream")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dataset
        dataset_path = Path(tmpdir) / "train.jsonl"
        create_sample_dataset(dataset_path, num_samples=100)
        print(f"   Created dataset: {dataset_path}")

        # Create async stream
        stream = AsyncOfflineDatasetStream(
            path=dataset_path,
            dataset_id="demo_async",
        )

        # Use context manager
        async with stream:
            samples = []
            async for sample in stream:
                samples.append(sample)

        print(f"   Collected {len(samples)} samples")
        print(f"   Final state: {stream.state}")

        # Check metrics
        metrics = stream.metrics
        print("\n   Stream metrics:")
        print(f"     samples_produced: {metrics.samples_produced}")
        print(f"     total_tokens:     {metrics.total_tokens}")
        print(f"     mean_length:      {metrics.mean_length:.1f}")
        print(f"     min_length:       {metrics.min_length_seen}")
        print(f"     max_length:       {metrics.max_length_seen}")

        # Reset and iterate again
        await stream.reset()
        print(f"\n   After reset: {stream.state}")
        assert stream.state == StreamState.IDLE


def demo_stream_metrics():
    """Demonstrate StreamMetrics tracking."""
    print("\n" + "=" * 60)
    print("3. Stream Metrics Tracking")
    print("=" * 60)

    metrics = StreamMetrics()

    # Create some samples
    samples = [
        StreamSample(
            input_ids=tuple(range(100)),
            loss_mask=tuple([1] * 100),
            sample_id=f"sample_{i}",
            dataset_id="test",
            source=SampleSource.OFFLINE,
            difficulty=i / 10,
            reward=0.5 + i * 0.1,
            success=i % 2 == 0,
        )
        for i in range(10)
    ]

    # Record samples
    for sample in samples:
        metrics.record_sample(sample)

    # Record some filtered samples
    metrics.record_filtered()
    metrics.record_filtered()
    metrics.record_filtered()

    print(f"   Samples produced: {metrics.samples_produced}")
    print(f"   Samples filtered: {metrics.samples_filtered}")
    print(f"   Filter rate:      {metrics.filter_rate:.1%}")
    print(f"   Total tokens:     {metrics.total_tokens}")
    print(f"   Mean length:      {metrics.mean_length:.1f}")
    print(f"   Mean difficulty:  {metrics.mean_difficulty:.2f}")
    print(f"   Mean reward:      {metrics.mean_reward:.2f}")

    # Episode tracking
    metrics.record_episode(success=True)
    metrics.record_episode(success=True)
    metrics.record_episode(success=False)

    print(f"\n   Episodes completed: {metrics.episodes_completed}")
    print(f"   Episodes successful: {metrics.episodes_successful}")
    print(f"   Success rate:        {metrics.success_rate:.1%}")

    # Export to dict
    d = metrics.to_dict()
    print(f"\n   Metrics dict keys: {list(d.keys())[:5]}...")


def demo_sample_filtering():
    """Demonstrate sample filtering."""
    print("\n" + "=" * 60)
    print("4. Sample Filtering and Transformation")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dataset
        dataset_path = Path(tmpdir) / "train.jsonl"
        create_sample_dataset(dataset_path, num_samples=100)

        # Create stream
        stream = OfflineDatasetStream(
            path=dataset_path,
            dataset_id="demo_filter",
        )

        # Filter by difficulty
        easy_samples = []
        hard_samples = []

        for sample in stream:
            if sample.difficulty is not None:
                if sample.difficulty < 0.3:
                    easy_samples.append(sample)
                elif sample.difficulty > 0.7:
                    hard_samples.append(sample)

        print(f"   Easy samples (difficulty < 0.3): {len(easy_samples)}")
        print(f"   Hard samples (difficulty > 0.7): {len(hard_samples)}")

        # Filter by length
        stream = OfflineDatasetStream(
            path=dataset_path,
            dataset_id="demo_filter",
        )

        short_samples = [s for s in stream if s.length < 100]
        print(f"   Short samples (length < 100): {len(short_samples)}")

        # Filter by success
        stream = OfflineDatasetStream(
            path=dataset_path,
            dataset_id="demo_filter",
        )

        successful = [s for s in stream if s.success is True]
        print(f"   Successful samples: {len(successful)}")


def demo_replay_marking():
    """Demonstrate replay sample marking."""
    print("\n" + "=" * 60)
    print("5. Replay Sample Marking")
    print("=" * 60)

    # Create original sample
    original = StreamSample(
        input_ids=(1, 2, 3, 4, 5),
        loss_mask=(0, 0, 1, 1, 1),
        sample_id="original_001",
        dataset_id="demo",
        source=SampleSource.OFFLINE,
        replay_count=0,
    )

    print("   Original sample:")
    print(f"     sample_id:    {original.sample_id}")
    print(f"     source:       {original.source}")
    print(f"     is_replay:    {original.is_replay}")
    print(f"     replay_count: {original.replay_count}")

    # Create replay
    replay1 = original.with_replay()

    print("\n   First replay:")
    print(f"     sample_id:    {replay1.sample_id}")
    print(f"     source:       {replay1.source}")
    print(f"     is_replay:    {replay1.is_replay}")
    print(f"     replay_count: {replay1.replay_count}")

    # Create second replay
    replay2 = replay1.with_replay()

    print("\n   Second replay:")
    print(f"     replay_count: {replay2.replay_count}")

    # Original is unchanged (immutable)
    print("\n   Original unchanged:")
    print(f"     replay_count: {original.replay_count}")
    print(f"     is_replay:    {original.is_replay}")


async def main():
    print("=" * 60)
    print("Streaming Sample Sources Demo")
    print("=" * 60)

    # Synchronous stream
    demo_sync_stream()

    # Async stream
    await demo_async_stream()

    # Metrics tracking
    demo_stream_metrics()

    # Filtering
    demo_sample_filtering()

    # Replay marking
    demo_replay_marking()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
