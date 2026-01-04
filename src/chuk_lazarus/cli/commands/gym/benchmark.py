"""Benchmark command handler.

This module provides the async benchmark implementation.
"""

from __future__ import annotations

import json
import logging
import random
import statistics
import time
from argparse import Namespace

from ._types import BenchmarkConfig, BenchmarkResult

logger = logging.getLogger(__name__)


async def bench_pipeline(config: BenchmarkConfig) -> BenchmarkResult:
    """Run comprehensive batching pipeline benchmark.

    Args:
        config: Benchmark configuration

    Returns:
        BenchmarkResult with benchmark outcomes
    """
    from ....data.batching import (
        BatchingConfig,
        BatchPlanBuilder,
        BucketSpec,
        PackingConfig,
        PackingMode,
        SequenceToPack,
        analyze_bucket_efficiency,
        compute_length_histogram,
        compute_packing_metrics,
        create_efficiency_report,
        pack_sequences,
    )
    from ....utils.tokenizer_loader import load_tokenizer

    print(f"\n{'=' * 70}")
    print("LAZARUS PIPELINE BENCHMARK")
    print(f"{'=' * 70}")

    bucket_edges = config.get_bucket_edges()

    # Load dataset or use synthetic data
    if config.dataset:
        print(f"\nDataset: {config.dataset}")
        print(f"Tokenizer: {config.tokenizer}")

        tokenizer = load_tokenizer(config.tokenizer)

        print("\n[1/7] Tokenizing dataset...")
        start = time.time()
        lengths: dict[str, int] = {}
        samples: dict[str, list[int]] = {}

        with open(config.dataset) as f:
            for i, line in enumerate(f):
                if config.max_samples and i >= config.max_samples:
                    break
                data = json.loads(line)
                text = data.get("text", data.get("content", data.get("instruction", "")))
                if text:
                    ids = tokenizer.encode(text)
                    sample_id = data.get("id", f"sample_{i}")
                    lengths[sample_id] = len(ids)
                    samples[sample_id] = ids

        tokenize_time = time.time() - start
        tokenize_throughput = len(lengths) / tokenize_time if tokenize_time > 0 else 0
        print(f"    Tokenized {len(lengths)} samples in {tokenize_time:.2f}s")
        print(f"    Throughput: {tokenize_throughput:.0f} samples/sec")
    else:
        print("\nUsing synthetic data (no --dataset provided)")
        print(f"Samples: {config.num_samples}")

        random.seed(config.seed)
        lengths = {
            f"s{i}": random.randint(32, config.max_length) for i in range(config.num_samples)
        }
        samples = {sid: list(range(length)) for sid, length in lengths.items()}
        tokenize_time = 0.0

    total_tokens = sum(lengths.values())
    length_values = list(lengths.values())
    length_stddev = statistics.stdev(length_values) if len(length_values) > 1 else 0

    # Length histogram
    print("\n[2/7] Computing length histogram...")
    histogram = compute_length_histogram(lengths, num_bins=15)
    print(f"\n{histogram.to_ascii(width=50)}")
    print(f"    Min: {histogram.min_length}, Max: {histogram.max_length}")
    print(f"    Mean: {histogram.mean_length:.1f}, Median: {histogram.median_length}")
    print(f"    StdDev: {length_stddev:.1f}")
    print(f"    P90: {histogram.p90}, P99: {histogram.p99}")

    # Bucket efficiency analysis
    print("\n[3/7] Analyzing bucket efficiency...")
    bucket_spec = BucketSpec(edges=bucket_edges, overflow_max=config.max_length)
    bucket_analysis = analyze_bucket_efficiency(lengths, bucket_spec)
    print(f"\n{bucket_analysis.to_ascii()}")
    print(f"    Overall efficiency: {bucket_analysis.overall_efficiency:.1%}")

    # Batch plan building
    print("\n[4/7] Building batch plan...")
    batching_config = BatchingConfig.predictable(
        token_budget=config.token_budget,
        bucket_edges=bucket_edges,
        overflow_max=config.max_length,
        seed=config.seed,
    )

    start = time.time()
    builder = BatchPlanBuilder(
        lengths=lengths,
        batching_config=batching_config,
        dataset_hash="benchmark",
        tokenizer_hash="benchmark",
    )

    import asyncio

    plan = await asyncio.to_thread(lambda: asyncio.run(builder.build(num_epochs=1)))
    plan_time = time.time() - start

    total_batches = plan.total_microbatches
    epoch = plan.get_epoch(0)
    epoch_tokens = epoch.total_tokens

    print(f"    Built plan in {plan_time:.3f}s")
    print(f"    Total microbatches: {total_batches}")
    print(f"    Total tokens: {epoch_tokens:,}")
    print(f"    Fingerprint: {plan.fingerprint}")

    avg_tokens_per_batch = epoch_tokens / total_batches if total_batches > 0 else 0

    # Padding waste
    print("\n[5/7] Computing padding waste...")
    padded_tokens = 0
    for length in lengths.values():
        bucket_id = bucket_spec.get_bucket_id(length)
        _, max_len = bucket_spec.get_bucket_range(bucket_id)
        padded_tokens += max_len
    padding_waste = 1.0 - (total_tokens / padded_tokens) if padded_tokens > 0 else 0
    print(f"    Padding waste: {padding_waste:.1%}")

    # Packing analysis
    print("\n[6/7] Packing analysis...")
    sample_seqs = [
        SequenceToPack(
            sample_id=sid,
            input_ids=tuple(samples[sid][: lengths[sid]]),
            loss_mask=tuple([1] * lengths[sid]),
        )
        for sid in list(lengths.keys())[: min(500, len(lengths))]
    ]

    pack_config = PackingConfig(
        mode=PackingMode.GREEDY,
        max_length=config.max_length,
        pad_to_max=True,
    )

    start = time.time()
    packed = pack_sequences(sample_seqs, pack_config, pad_token_id=0)
    pack_time = time.time() - start
    pack_metrics = compute_packing_metrics(packed)

    print(f"    Packed {len(sample_seqs)} -> {len(packed)} in {pack_time:.3f}s")
    print(f"    Packing ratio: {pack_metrics.packing_ratio:.2f}x")
    print(f"    Efficiency: {pack_metrics.efficiency:.1%}")

    # Report
    print("\n[7/7] Creating efficiency report...")
    report = create_efficiency_report(lengths, bucket_spec)
    print(f"\n{report.to_ascii()}")

    # Summary
    print(f"\n{'=' * 70}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 70}")

    token_budget_utilization = avg_tokens_per_batch / config.token_budget

    return BenchmarkResult(
        samples=len(lengths),
        total_tokens=total_tokens,
        plan_fingerprint=plan.fingerprint,
        bucket_efficiency=bucket_analysis.overall_efficiency,
        packing_ratio=pack_metrics.packing_ratio,
        packing_efficiency=pack_metrics.efficiency,
        token_budget_utilization=token_budget_utilization,
        microbatches=total_batches,
    )


async def bench_pipeline_cmd(args: Namespace) -> None:
    """CLI entry point for benchmark command.

    Args:
        args: Parsed command-line arguments
    """
    config = BenchmarkConfig.from_args(args)
    result = await bench_pipeline(config)
    print(result.to_display())


__all__ = [
    "bench_pipeline",
    "bench_pipeline_cmd",
]
