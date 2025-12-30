"""Gym and benchmarking command handlers for chuk-lazarus CLI."""

import logging

logger = logging.getLogger(__name__)


def gym_run(args):
    """Run gym episode streaming and collect samples."""
    import asyncio

    from ...data.batching.streaming import (
        GymConfig,
        GymEpisodeStream,
        GymOutputMode,
        GymTransport,
        MockGymStream,
        ReplayBuffer,
        ReplayBufferConfig,
    )
    from ...utils.tokenizer_loader import load_tokenizer

    async def run():
        # Load tokenizer
        logger.info(f"Loading tokenizer: {args.tokenizer}")
        tokenizer = load_tokenizer(args.tokenizer)

        # Configure replay buffer
        buffer_config = ReplayBufferConfig(
            max_size=args.buffer_size,
            seed=args.seed,
        )
        buffer = ReplayBuffer(buffer_config)

        # Configure gym stream
        if args.mock:
            logger.info("Using mock gym stream for testing")
            stream = MockGymStream(
                tokenizer=tokenizer,
                num_episodes=args.num_episodes,
                steps_per_episode=args.steps_per_episode,
                difficulty_range=(args.difficulty_min, args.difficulty_max),
                success_rate=args.success_rate,
                seed=args.seed,
            )
        else:
            # Parse transport
            transport = GymTransport(args.transport)
            output_mode = GymOutputMode(args.output_mode)

            config = GymConfig(
                host=args.host,
                port=args.port,
                transport=transport,
                output_mode=output_mode,
                connect_timeout=args.timeout,
                max_retries=args.retries,
                difficulty_range=(args.difficulty_min, args.difficulty_max),
            )

            stream = GymEpisodeStream(
                config=config,
                tokenizer=tokenizer,
            )

        # Run streaming
        logger.info(f"Starting gym stream to {args.host}:{args.port}")
        print(f"\n{'=' * 60}")
        print("Gym Episode Streaming")
        print(f"{'=' * 60}")

        sample_count = 0
        episode_ids = set()

        async with stream:
            async for sample in stream:
                buffer.add(sample)
                sample_count += 1
                if sample.episode_id:
                    episode_ids.add(sample.episode_id)

                if sample_count % 100 == 0:
                    print(
                        f"  Samples: {sample_count}, "
                        f"Episodes: {len(episode_ids)}, "
                        f"Buffer: {buffer.size}"
                    )

                if args.max_samples and sample_count >= args.max_samples:
                    logger.info(f"Reached max samples: {args.max_samples}")
                    break

        # Print summary
        print(f"\n{'=' * 60}")
        print("Summary")
        print(f"{'=' * 60}")
        print(f"  Total samples:    {sample_count}")
        print(f"  Total episodes:   {len(episode_ids)}")
        print(f"  Buffer size:      {buffer.size}")
        print(f"  Success rate:     {buffer.success_rate:.1%}")
        print(f"  Mean difficulty:  {buffer.mean_difficulty:.2f}")
        print(f"  Mean reward:      {buffer.mean_reward:.2f}")

        # Save buffer if output specified
        if args.output:
            import json
            from pathlib import Path

            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            buffer_data = buffer.to_dict()
            with open(output_path, "w") as f:
                json.dump(buffer_data, f, indent=2, default=str)

            print(f"\n  Buffer saved to: {output_path}")

        return buffer

    asyncio.run(run())


def bench_pipeline(args):
    """Run comprehensive batching pipeline benchmark."""
    import asyncio
    import statistics
    import time

    from ...data.batching import (
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
    from ...utils.tokenizer_loader import load_tokenizer

    print(f"\n{'=' * 70}")
    print("LAZARUS PIPELINE BENCHMARK")
    print(f"{'=' * 70}")

    # Load tokenizer if provided, else use mock lengths
    if args.dataset:
        print(f"\nDataset: {args.dataset}")
        print(f"Tokenizer: {args.tokenizer}")

        tokenizer = load_tokenizer(args.tokenizer)

        # Tokenize and build lengths
        print("\n[1/7] Tokenizing dataset...")
        start = time.time()
        lengths = {}
        samples = {}
        import json

        with open(args.dataset) as f:
            for i, line in enumerate(f):
                if args.max_samples and i >= args.max_samples:
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
        print(f"Samples: {args.num_samples}")

        # Generate synthetic lengths
        import random

        random.seed(args.seed)
        lengths = {f"s{i}": random.randint(32, args.max_length) for i in range(args.num_samples)}
        samples = {sid: list(range(length)) for sid, length in lengths.items()}
        tokenize_time = 0.0
        tokenize_throughput = 0.0

    # Parse bucket edges
    bucket_edges = tuple(int(x) for x in args.bucket_edges.split(","))
    total_tokens = sum(lengths.values())
    length_values = list(lengths.values())
    length_variance = statistics.variance(length_values) if len(length_values) > 1 else 0
    length_stddev = statistics.stdev(length_values) if len(length_values) > 1 else 0

    # Length histogram
    print("\n[2/7] Computing length histogram...")
    histogram = compute_length_histogram(lengths, num_bins=15)
    print(f"\n{histogram.to_ascii(width=50)}")
    print(f"    Min: {histogram.min_length}, Max: {histogram.max_length}")
    print(f"    Mean: {histogram.mean_length:.1f}, Median: {histogram.median_length}")
    print(f"    StdDev: {length_stddev:.1f}, Variance: {length_variance:.1f}")
    print(f"    P90: {histogram.p90}, P99: {histogram.p99}")

    # Bucket efficiency analysis
    print("\n[3/7] Analyzing bucket efficiency...")
    bucket_spec = BucketSpec(edges=bucket_edges, overflow_max=args.max_length)
    bucket_analysis = analyze_bucket_efficiency(lengths, bucket_spec)
    print(f"\n{bucket_analysis.to_ascii()}")
    print(f"    Overall efficiency: {bucket_analysis.overall_efficiency:.1%}")

    # Batch plan building
    print("\n[4/7] Building batch plan...")
    config = BatchingConfig.predictable(
        token_budget=args.token_budget,
        bucket_edges=bucket_edges,
        overflow_max=args.max_length,
        seed=args.seed,
    )

    start = time.time()
    builder = BatchPlanBuilder(
        lengths=lengths,
        batching_config=config,
        dataset_hash="benchmark",
        tokenizer_hash="benchmark",
    )
    plan = asyncio.run(builder.build(num_epochs=1))
    plan_time = time.time() - start

    total_batches = plan.total_microbatches
    epoch = plan.get_epoch(0)
    epoch_tokens = epoch.total_tokens

    print(f"    Built plan in {plan_time:.3f}s")
    print(f"    Total microbatches: {total_batches}")
    print(f"    Total tokens: {epoch_tokens:,}")
    print(f"    Fingerprint: {plan.fingerprint}")

    # Compute batch metrics
    avg_batch_size = epoch.total_samples / total_batches if total_batches > 0 else 0
    avg_tokens_per_batch = epoch_tokens / total_batches if total_batches > 0 else 0

    # Compute padding waste for pad-to-bucket strategy
    print("\n[5/7] Computing padding waste (pad-to-bucket)...")
    padded_tokens_bucket = 0
    for sid, length in lengths.items():
        bucket_id = bucket_spec.get_bucket_id(length)
        _, max_len = bucket_spec.get_bucket_range(bucket_id)
        padded_tokens_bucket += max_len
    padding_waste_bucket = (
        1.0 - (total_tokens / padded_tokens_bucket) if padded_tokens_bucket > 0 else 0
    )
    print(f"    Total tokens (raw): {total_tokens:,}")
    print(f"    Total tokens (padded to bucket): {padded_tokens_bucket:,}")
    print(f"    Padding waste: {padding_waste_bucket:.1%}")

    # Packing analysis
    print("\n[6/7] Packing analysis...")
    # Take a sample of sequences for packing demo
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
        max_length=args.max_length,
        pad_to_max=True,
    )

    start = time.time()
    packed = pack_sequences(sample_seqs, pack_config, pad_token_id=0)
    pack_time = time.time() - start
    pack_metrics = compute_packing_metrics(packed)

    print(f"    Packed {len(sample_seqs)} → {len(packed)} sequences in {pack_time:.3f}s")
    print(f"    Packing ratio: {pack_metrics.packing_ratio:.2f}x")
    print(f"    Efficiency: {pack_metrics.efficiency:.1%}")
    if pack_metrics.packing_ratio > 1:
        print(f"    Token reduction: {1 - 1 / pack_metrics.packing_ratio:.0%}")

    # Memory footprint estimation
    print("\n[7/7] Memory footprint estimation...")
    # Estimate memory for different strategies
    bytes_per_token = 4  # int32
    mem_raw = total_tokens * bytes_per_token
    mem_padded_bucket = padded_tokens_bucket * bytes_per_token
    mem_packed = (
        sum(len(p.input_ids) for p in packed) * bytes_per_token * (len(lengths) / len(sample_seqs))
    )

    print(f"    Raw tokens: {mem_raw / 1024 / 1024:.1f} MB")
    print(f"    Padded (bucket): {mem_padded_bucket / 1024 / 1024:.1f} MB")
    print(f"    Packed (estimated): {mem_packed / 1024 / 1024:.1f} MB")

    # Efficiency report
    print("\n[8/8] Creating efficiency report...")
    report = create_efficiency_report(lengths, bucket_spec)
    print(f"\n{report.to_ascii()}")

    # ═══════════════════════════════════════════════════════════════════════════
    # PACK VS PAD COMPARISON
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("PACK VS PAD COMPARISON")
    print(f"{'=' * 70}")

    print(f"\n{'Strategy':<25} {'Tokens':>15} {'Waste %':>12} {'Memory':>12}")
    print("-" * 66)
    print(
        f"{'Raw (no padding)':<25} {total_tokens:>15,} {'0.0%':>12} {mem_raw / 1024 / 1024:>10.1f} MB"
    )
    print(
        f"{'Pad-to-bucket':<25} {padded_tokens_bucket:>15,} {padding_waste_bucket:>11.1%} {mem_padded_bucket / 1024 / 1024:>10.1f} MB"
    )

    # Estimate packed total tokens
    packed_total_tokens = (
        int(total_tokens / pack_metrics.efficiency) if pack_metrics.efficiency > 0 else total_tokens
    )
    packed_waste = 1.0 - pack_metrics.efficiency
    print(
        f"{'Packed (greedy)':<25} {packed_total_tokens:>15,} {packed_waste:>11.1%} {mem_packed / 1024 / 1024:>10.1f} MB"
    )

    if padding_waste_bucket > packed_waste:
        savings = padding_waste_bucket - packed_waste
        print(f"\n    → Packing saves {savings:.1%} waste vs pad-to-bucket")
    else:
        print("\n    → Pad-to-bucket is more efficient for this distribution")

    # ═══════════════════════════════════════════════════════════════════════════
    # THROUGHPUT METRICS
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("THROUGHPUT METRICS")
    print(f"{'=' * 70}")

    print(f"\n{'Metric':<35} {'Value':>20}")
    print("-" * 57)
    print(f"{'Tokenization throughput':<35} {tokenize_throughput:>15.0f} samp/s")
    print(f"{'Plan build throughput':<35} {len(lengths) / plan_time:>15.0f} samp/s")
    print(f"{'Effective tokens/batch':<35} {avg_tokens_per_batch:>20.0f}")
    print(f"{'Tokens/batch (theoretical max)':<35} {args.token_budget:>20}")
    print(f"{'Token budget utilization':<35} {avg_tokens_per_batch / args.token_budget:>19.1%}")

    # Batch size variance
    batch_sizes = [len(mb.samples) for mb in epoch.microbatches]
    batch_size_variance = statistics.variance(batch_sizes) if len(batch_sizes) > 1 else 0
    batch_size_stddev = statistics.stdev(batch_sizes) if len(batch_sizes) > 1 else 0

    print(f"{'Batch size mean':<35} {statistics.mean(batch_sizes):>20.1f}")
    print(f"{'Batch size stddev':<35} {batch_size_stddev:>20.1f}")
    print(f"{'Batch size variance':<35} {batch_size_variance:>20.1f}")

    # ═══════════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n{'Metric':<35} {'Value':>20}")
    print("-" * 57)
    print(f"{'Samples':<35} {len(lengths):>20,}")
    print(f"{'Total tokens':<35} {total_tokens:>20,}")
    print(f"{'Length stddev':<35} {length_stddev:>20.1f}")
    print(f"{'Tokenization time':<35} {tokenize_time:>19.2f}s")
    print(f"{'Plan build time':<35} {plan_time:>19.3f}s")
    print(f"{'Pack time (500 samples)':<35} {pack_time:>19.3f}s")
    print(f"{'Microbatches per epoch':<35} {total_batches:>20,}")
    print(f"{'Avg batch size':<35} {avg_batch_size:>20.1f}")
    print(f"{'Avg tokens/batch':<35} {avg_tokens_per_batch:>20.0f}")
    print(f"{'Token budget utilization':<35} {avg_tokens_per_batch / args.token_budget:>19.1%}")
    print(f"{'Bucket efficiency':<35} {bucket_analysis.overall_efficiency:>19.1%}")
    print(f"{'Padding waste (bucket)':<35} {padding_waste_bucket:>19.1%}")
    print(f"{'Packing ratio':<35} {pack_metrics.packing_ratio:>19.2f}x")
    print(f"{'Packing efficiency':<35} {pack_metrics.efficiency:>19.1%}")
    print(f"{'Plan fingerprint':<35} {plan.fingerprint:>20}")

    if report.recommendations:
        print(f"\n{'Recommendations:':<35}")
        for rec in report.recommendations[:3]:
            print(f"  • {rec}")

    # Key insight
    print(f"\n{'=' * 70}")
    print("KEY INSIGHT")
    print(f"{'=' * 70}")
    if pack_metrics.packing_ratio > 1.3:
        print(f"\n  Packing recommended: {pack_metrics.packing_ratio:.1f}x compression saves")
        print(f"  {1 - 1 / pack_metrics.packing_ratio:.0%} tokens per epoch.")
    elif bucket_analysis.overall_efficiency > 0.85:
        print(f"\n  Bucket efficiency is high ({bucket_analysis.overall_efficiency:.0%}).")
        print("  Pad-to-bucket is sufficient for this distribution.")
    else:
        print(
            f"\n  Consider adjusting bucket edges. Current efficiency: {bucket_analysis.overall_efficiency:.0%}"
        )
        print("  Suggested edges from report may improve utilization.")

    print(f"\n{'=' * 70}")
    print("Benchmark complete. Plan fingerprint can be used for CI/CD verification.")
    print(f"{'=' * 70}\n")


def gym_info(args):
    """Display gym stream configuration info."""
    from ...data.batching.streaming import (
        GymOutputMode,
        GymTransport,
    )

    print(f"\n{'=' * 60}")
    print("Gym Stream Configuration")
    print(f"{'=' * 60}")

    print("\nSupported Transports:")
    for transport in GymTransport:
        print(f"  - {transport.value}")

    print("\nSupported Output Modes:")
    for mode in GymOutputMode:
        print(f"  - {mode.value}")

    print("\nDefault Configuration:")
    print("  Host:             localhost")
    print("  Port:             8023")
    print("  Transport:        telnet")
    print("  Output Mode:      json")
    print("  Connect Timeout:  10.0s")
    print("  Max Retries:      3")

    print("\nExample Usage:")
    print("  # Run mock stream for testing")
    print("  lazarus gym run --tokenizer gpt2 --mock --num-episodes 10")
    print()
    print("  # Connect to puzzle arcade server")
    print("  lazarus gym run --tokenizer gpt2 --host localhost --port 8023")
    print()
    print("  # Save samples to buffer file")
    print("  lazarus gym run --tokenizer gpt2 --mock --output buffer.json")
