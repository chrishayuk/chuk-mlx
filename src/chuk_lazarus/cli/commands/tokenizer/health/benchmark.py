"""Tokenizer benchmark command handler."""

import logging

from .._types import BenchmarkConfig, BenchmarkResult

logger = logging.getLogger(__name__)


def tokenizer_benchmark(config: BenchmarkConfig) -> BenchmarkResult | None:
    """Benchmark tokenizer throughput.

    Args:
        config: Benchmark configuration.

    Returns:
        Benchmark result with throughput metrics, or None for comparison mode.
    """
    from .....data.tokenizers.backends.benchmark import (
        benchmark_tokenizer,
        compare_backends,
        generate_benchmark_corpus,
    )
    from .....utils.tokenizer_loader import load_tokenizer

    logger.info(f"Loading tokenizer: {config.tokenizer}")
    tokenizer = load_tokenizer(config.tokenizer)

    # Generate or load corpus
    if config.file:
        logger.info(f"Loading corpus from: {config.file}")
        with open(config.file) as f:
            corpus = [line.strip() for line in f if line.strip()]
        if config.samples and len(corpus) > config.samples:
            corpus = corpus[: config.samples]
    else:
        logger.info(f"Generating synthetic corpus ({config.samples} samples)...")
        corpus = generate_benchmark_corpus(
            num_samples=config.samples,
            avg_length=config.avg_length,
            seed=config.seed,
        )

    print(f"\n{'=' * 60}")
    print("Tokenizer Benchmark")
    print(f"{'=' * 60}")
    print(f"  Tokenizer:  {config.tokenizer}")
    print(f"  Samples:    {len(corpus):,}")
    print(f"  Avg length: ~{sum(len(t.split()) for t in corpus) // len(corpus)} words")
    print(f"  Workers:    {config.workers}")
    print()

    if config.compare:
        # Compare HuggingFace vs Fast backend
        logger.info("Running backend comparison...")
        comparison = compare_backends(
            tokenizer,
            corpus,
            num_workers=config.workers,
            add_special_tokens=config.special_tokens,
        )
        print(comparison.summary())
        return None
    else:
        # Single backend benchmark
        logger.info("Running benchmark...")
        result = benchmark_tokenizer(
            tokenizer,
            corpus,
            num_workers=config.workers,
            add_special_tokens=config.special_tokens,
            warmup_samples=min(config.warmup, len(corpus)),
        )

        print("Results:")
        print(f"  Backend:      {result.backend_type}")
        print(f"  Total tokens: {result.total_tokens:,}")
        print(f"  Time:         {result.elapsed_seconds:.2f}s")
        print(f"  Throughput:   {result.tokens_per_second:,.0f} tokens/sec")
        print(f"  Samples/sec:  {result.samples_per_second:,.1f}")
        print(f"  Avg tok/sample: {result.avg_tokens_per_sample:.1f}")
        print(f"{'=' * 60}")

        return BenchmarkResult(
            backend_type=result.backend_type,
            total_tokens=result.total_tokens,
            elapsed_seconds=result.elapsed_seconds,
            tokens_per_second=result.tokens_per_second,
            samples_per_second=result.samples_per_second,
            avg_tokens_per_sample=result.avg_tokens_per_sample,
        )
