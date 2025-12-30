"""Data processing command handlers for chuk-lazarus CLI."""

import logging
import sys

logger = logging.getLogger(__name__)


def data_lengths_build(args):
    """Build a length cache from a dataset."""
    import asyncio
    import json
    from pathlib import Path

    from ...data.batching import LengthCache
    from ...utils.tokenizer_loader import load_tokenizer

    logger.info(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = load_tokenizer(args.tokenizer)

    # Compute tokenizer hash for cache invalidation
    try:
        from ...data.tokenizers.fingerprint import compute_fingerprint

        fp = compute_fingerprint(tokenizer)
        tokenizer_hash = fp.fingerprint
    except Exception:
        tokenizer_hash = "unknown"

    logger.info(f"Loading dataset: {args.dataset}")
    with open(args.dataset) as f:
        if args.dataset.endswith(".jsonl"):
            samples = [json.loads(line) for line in f if line.strip()]
        else:
            samples = json.load(f)

    async def build_cache():
        output_path = Path(args.output)
        async with LengthCache.create(output_path, tokenizer_hash) as cache:
            for i, sample in enumerate(samples):
                # Get sample ID
                sample_id = sample.get("id") or sample.get("sample_id") or f"sample_{i:06d}"

                # Get text to tokenize
                text = sample.get("text") or sample.get("content") or sample.get("input")
                if text is None and "messages" in sample:
                    # Chat format - concatenate messages
                    text = " ".join(m.get("content", "") for m in sample["messages"])

                if text:
                    token_ids = tokenizer.encode(text, add_special_tokens=True)
                    await cache.add(sample_id, len(token_ids))

                if (i + 1) % 1000 == 0:
                    logger.info(f"Processed {i + 1}/{len(samples)} samples")

        return cache

    cache = asyncio.run(build_cache())

    print(f"\n{'=' * 60}")
    print("Length Cache Built")
    print(f"{'=' * 60}")
    print(f"  Dataset:       {args.dataset}")
    print(f"  Tokenizer:     {args.tokenizer}")
    print(f"  Samples:       {len(cache):,}")
    print(f"  Output:        {args.output}")
    print(f"  Tokenizer hash: {tokenizer_hash}")


def data_lengths_stats(args):
    """Show statistics for a length cache."""
    import asyncio
    from pathlib import Path

    from ...data.batching import LengthCache

    async def load_and_stats():
        cache = await LengthCache.load(Path(args.cache))
        return cache

    cache = asyncio.run(load_and_stats())
    lengths = cache.get_all()

    if not lengths:
        print("Cache is empty")
        return

    values = list(lengths.values())
    values.sort()

    print(f"\n{'=' * 60}")
    print("Length Cache Statistics")
    print(f"{'=' * 60}")
    print(f"  Cache file:    {args.cache}")
    print(f"  Tokenizer:     {cache.tokenizer_hash}")
    print(f"  Total samples: {len(lengths):,}")
    print(f"  Total tokens:  {sum(values):,}")
    print()
    print(f"  Min length:    {min(values)}")
    print(f"  Max length:    {max(values)}")
    print(f"  Mean length:   {sum(values) / len(values):.1f}")
    print(f"  Median:        {values[len(values) // 2]}")

    # Percentiles
    def percentile(p):
        idx = int(len(values) * p / 100)
        return values[min(idx, len(values) - 1)]

    print()
    print(f"  P10:           {percentile(10)}")
    print(f"  P25:           {percentile(25)}")
    print(f"  P50:           {percentile(50)}")
    print(f"  P75:           {percentile(75)}")
    print(f"  P90:           {percentile(90)}")
    print(f"  P95:           {percentile(95)}")
    print(f"  P99:           {percentile(99)}")


def data_batchplan_build(args):
    """Build a batch plan from length cache."""
    import asyncio
    from pathlib import Path

    from ...data.batching import (
        BatchingConfig,
        BatchPlanBuilder,
        LengthCache,
        save_batch_plan,
    )

    async def build_plan():
        # Load length cache
        logger.info(f"Loading length cache: {args.lengths}")
        cache = await LengthCache.load(Path(args.lengths))
        lengths = cache.get_all()

        # Parse bucket edges
        bucket_edges = tuple(int(x.strip()) for x in args.bucket_edges.split(","))

        # Create config
        if args.predictable:
            config = BatchingConfig.predictable(
                token_budget=args.token_budget,
                bucket_edges=bucket_edges,
                overflow_max=args.overflow_max,
                seed=args.seed,
            )
        else:
            config = BatchingConfig.throughput(
                token_budget=args.token_budget,
                bucket_edges=bucket_edges,
                overflow_max=args.overflow_max,
            )

        # Build plan
        logger.info(f"Building batch plan for {args.epochs} epochs...")
        builder = BatchPlanBuilder(
            lengths=lengths,
            batching_config=config,
            dataset_hash=args.dataset_hash or "unknown",
            tokenizer_hash=cache.tokenizer_hash,
        )

        plan = await builder.build(num_epochs=args.epochs)

        # Save plan
        output_path = Path(args.output)
        save_batch_plan(plan, output_path)

        return plan, output_path

    plan, output_path = asyncio.run(build_plan())

    print(f"\n{'=' * 60}")
    print("Batch Plan Built")
    print(f"{'=' * 60}")
    print(f"  Lengths cache: {args.lengths}")
    print(f"  Epochs:        {plan.num_epochs}")
    print(f"  Token budget:  {args.token_budget}")
    print(f"  Mode:          {'predictable' if args.predictable else 'throughput'}")
    print()
    print(f"  Total batches: {plan.total_microbatches}")
    print(f"  Fingerprint:   {plan.fingerprint}")
    print()
    print(f"  Output:        {output_path}")

    # Per-epoch summary
    print("\n  Per-epoch details:")
    for ep in range(plan.num_epochs):
        epoch_plan = plan.get_epoch(ep)
        print(
            f"    Epoch {ep}: {epoch_plan.num_microbatches} batches, "
            f"{epoch_plan.total_samples} samples, {epoch_plan.total_tokens:,} tokens"
        )


def data_batchplan_info(args):
    """Show information about a batch plan."""
    from pathlib import Path

    from ...data.batching import load_batch_plan

    plan = load_batch_plan(Path(args.plan))

    # Apply sharding if requested
    if args.rank is not None and args.world_size is not None:
        if args.rank >= args.world_size or args.rank < 0:
            print(f"Error: rank must be in range [0, {args.world_size})")
            return
        plan = plan.shard(args.rank, args.world_size)
        shard_info = f" (rank {args.rank}/{args.world_size})"
    else:
        shard_info = ""

    print(f"\n{'=' * 60}")
    print(f"Batch Plan Info{shard_info}")
    print(f"{'=' * 60}")
    print(f"  Plan path:     {args.plan}")
    print(f"  Fingerprint:   {plan.fingerprint}")
    print(f"  Created:       {plan.meta.created_at}")
    print()
    print(f"  Dataset hash:  {plan.meta.dataset_hash}")
    print(f"  Tokenizer:     {plan.meta.tokenizer_hash}")
    print(f"  Token budget:  {plan.meta.token_budget}")
    print(f"  Bucket edges:  {plan.meta.bucket_edges}")
    print()
    print(f"  Epochs:        {plan.num_epochs}")
    print(f"  Total batches: {plan.total_microbatches}")

    # Per-epoch summary
    print("\n  Per-epoch details:")
    for ep in range(plan.num_epochs):
        epoch_plan = plan.get_epoch(ep)
        print(
            f"    Epoch {ep}: {epoch_plan.num_microbatches} batches, "
            f"{epoch_plan.total_samples} samples, {epoch_plan.total_tokens:,} tokens"
        )

    # Sample batches
    if args.show_batches:
        print("\n  Sample batches from epoch 0:")
        epoch0 = plan.get_epoch(0)
        for i, mb in enumerate(epoch0.microbatches[: args.show_batches]):
            print(
                f"    Batch {i}: {mb.batch_size} samples, bucket={mb.bucket_id}, max_len={mb.max_len}"
            )


def data_batchplan_verify(args):
    """Verify a batch plan can be reproduced."""
    import asyncio
    from pathlib import Path

    from ...data.batching import (
        BatchPlanBuilder,
        LengthCache,
        load_batch_plan,
    )

    async def verify():
        # Load original plan
        logger.info(f"Loading batch plan: {args.plan}")
        original = load_batch_plan(Path(args.plan))

        # Rebuild from lengths
        logger.info(f"Rebuilding from lengths: {args.lengths}")
        cache = await LengthCache.load(Path(args.lengths))
        lengths = cache.get_all()

        # Recreate config from plan meta
        from ...data.batching import BatchingConfig, BatchingMode, PadPolicy

        config = BatchingConfig(
            mode=BatchingMode(original.meta.mode),
            pad_policy=PadPolicy(original.meta.pad_policy),
            token_budget=original.meta.token_budget,
            bucket_edges=tuple(original.meta.bucket_edges),
            overflow_max=original.meta.overflow_max,
            seed=original.meta.seed,
        )

        builder = BatchPlanBuilder(
            lengths=lengths,
            batching_config=config,
            dataset_hash=original.meta.dataset_hash,
            tokenizer_hash=original.meta.tokenizer_hash,
        )

        rebuilt = await builder.build(num_epochs=original.num_epochs)

        return original, rebuilt

    original, rebuilt = asyncio.run(verify())

    print(f"\n{'=' * 60}")
    print("Batch Plan Verification")
    print(f"{'=' * 60}")
    print(f"  Original fingerprint: {original.fingerprint}")
    print(f"  Rebuilt fingerprint:  {rebuilt.fingerprint}")

    if original.fingerprint == rebuilt.fingerprint:
        print("\n  Result: MATCH")
        print("  The batch plan is reproducible.")
    else:
        print("\n  Result: MISMATCH")
        print("  Warning: Rebuilt plan differs from original!")

        # Check epoch-by-epoch
        for ep in range(original.num_epochs):
            orig_mbs = list(original.iter_epoch(ep))
            rebuilt_mbs = list(rebuilt.iter_epoch(ep))

            if len(orig_mbs) != len(rebuilt_mbs):
                print(
                    f"    Epoch {ep}: batch count differs ({len(orig_mbs)} vs {len(rebuilt_mbs)})"
                )
            else:
                matches = sum(1 for o, r in zip(orig_mbs, rebuilt_mbs) if o.samples == r.samples)
                print(f"    Epoch {ep}: {matches}/{len(orig_mbs)} batches match")

        sys.exit(1)


def data_batchplan_shard(args):
    """Save sharded batch plans for distributed training."""
    from pathlib import Path

    from ...data.batching import load_batch_plan, save_batch_plan

    # Load original plan
    logger.info(f"Loading batch plan: {args.plan}")
    plan = load_batch_plan(Path(args.plan))

    output_base = Path(args.output)
    output_base.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("Batch Plan Sharding")
    print(f"{'=' * 60}")
    print(f"  Source plan:   {args.plan}")
    print(f"  World size:    {args.world_size}")
    print(f"  Total batches: {plan.total_microbatches}")
    print()

    # Create sharded plans
    for rank in range(args.world_size):
        sharded = plan.shard(rank, args.world_size)
        shard_path = output_base / f"rank_{rank}"
        save_batch_plan(sharded, shard_path)

        print(f"  Rank {rank}: {sharded.total_microbatches} batches -> {shard_path}")

    print()
    print(f"  Output:        {output_base}")


def data_batching_analyze(args):
    """Analyze batching efficiency for a dataset."""
    import asyncio
    from pathlib import Path

    from ...data.batching import (
        BucketSpec,
        LengthCache,
        create_efficiency_report,
    )

    async def analyze():
        # Load length cache
        logger.info(f"Loading length cache: {args.cache}")
        cache = await LengthCache.load(Path(args.cache))
        lengths = cache.get_all()

        # Parse bucket edges
        bucket_edges = tuple(int(x.strip()) for x in args.bucket_edges.split(","))

        # Create bucket spec
        bucket_spec = BucketSpec(
            edges=bucket_edges,
            overflow_max=args.overflow_max,
        )

        # Create efficiency report
        report = create_efficiency_report(lengths, bucket_spec)
        return report

    report = asyncio.run(analyze())

    # Print report
    print(report.to_ascii())

    if args.output:
        # Save JSON report
        import json

        with open(args.output, "w") as f:
            json.dump(report.model_dump(), f, indent=2, default=str)
        print(f"\nReport saved to: {args.output}")


def data_batching_histogram(args):
    """Display length histogram for a dataset."""
    import asyncio
    from pathlib import Path

    from ...data.batching import (
        LengthCache,
        compute_length_histogram,
    )

    async def load():
        cache = await LengthCache.load(Path(args.cache))
        return cache.get_all()

    lengths = asyncio.run(load())
    histogram = compute_length_histogram(lengths, num_bins=args.bins)

    print(histogram.to_ascii(width=args.width))

    print("\n--- Percentiles ---")
    print(f"  P25: {histogram.p25}")
    print(f"  P50: {histogram.p50}")
    print(f"  P75: {histogram.p75}")
    print(f"  P90: {histogram.p90}")
    print(f"  P95: {histogram.p95}")
    print(f"  P99: {histogram.p99}")


def data_batching_suggest(args):
    """Suggest optimal bucket edges for a dataset."""
    import asyncio
    from pathlib import Path

    from ...data.batching import (
        LengthCache,
        OptimizationGoal,
        suggest_bucket_edges,
    )

    async def load():
        cache = await LengthCache.load(Path(args.cache))
        return cache.get_all()

    lengths = asyncio.run(load())

    # Get goal
    goal_map = {
        "waste": OptimizationGoal.MINIMIZE_WASTE,
        "balance": OptimizationGoal.BALANCE_BUCKETS,
        "memory": OptimizationGoal.MINIMIZE_MEMORY,
    }
    goal = goal_map.get(args.goal, OptimizationGoal.MINIMIZE_WASTE)

    suggestion = suggest_bucket_edges(
        lengths,
        num_buckets=args.num_buckets,
        goal=goal,
        max_length=args.max_length,
    )

    print(f"\n{'=' * 60}")
    print("Bucket Edge Suggestions")
    print(f"{'=' * 60}")
    print(f"  Goal:           {suggestion.optimization_goal.value}")
    print(f"  Num buckets:    {args.num_buckets}")
    print()
    print(f"  Suggested edges:  {suggestion.edges}")
    print(f"  Overflow max:     {suggestion.overflow_max}")
    print(f"  Est. efficiency:  {suggestion.estimated_efficiency:.1%}")
    print()
    print(f"  Rationale: {suggestion.rationale}")

    # Show CLI command to use
    edges_str = ",".join(str(e) for e in suggestion.edges)
    print("\n  Use with:")
    print(
        f"    lazarus data batchplan build --bucket-edges {edges_str} --overflow-max {suggestion.overflow_max} ..."
    )


def data_batch_generate(args):
    """Generate NPZ batch files from a BatchPlan."""
    import asyncio
    import json
    from pathlib import Path

    from ...data.batching import (
        BatchReader,
        BatchWriter,
        load_batch_plan,
    )
    from ...utils.tokenizer_loader import load_tokenizer

    async def generate():
        # Load batch plan
        logger.info(f"Loading batch plan: {args.plan}")
        plan = load_batch_plan(Path(args.plan))

        # Load tokenizer
        logger.info(f"Loading tokenizer: {args.tokenizer}")
        tokenizer = load_tokenizer(args.tokenizer)

        # Load dataset
        logger.info(f"Loading dataset: {args.dataset}")
        with open(args.dataset) as f:
            if args.dataset.endswith(".jsonl"):
                raw_samples = [json.loads(line) for line in f if line.strip()]
            else:
                raw_samples = json.load(f)

        # Tokenize samples
        logger.info("Tokenizing samples...")
        samples = {}
        for i, sample in enumerate(raw_samples):
            sample_id = sample.get("id") or sample.get("sample_id") or f"sample_{i:06d}"

            # Get text
            text = sample.get("text") or sample.get("content") or sample.get("input")
            if text is None and "messages" in sample:
                text = " ".join(m.get("content", "") for m in sample["messages"])

            if text:
                input_ids = tokenizer.encode(text, add_special_tokens=True)
                # Create simple loss mask (all 1s for now)
                loss_mask = [1] * len(input_ids)
                samples[sample_id] = {
                    "input_ids": input_ids,
                    "loss_mask": loss_mask,
                }

            if (i + 1) % 1000 == 0:
                logger.info(f"Tokenized {i + 1}/{len(raw_samples)} samples")

        # Create writer
        output_dir = Path(args.output)
        logger.info(f"Writing batches to: {output_dir}")

        writer = BatchWriter(
            plan=plan,
            samples=samples,
            output_dir=output_dir,
            pad_id=tokenizer.pad_token_id or 0,
        )

        # Write batches
        files = writer.write_all()

        return len(files), output_dir

    num_files, output_dir = asyncio.run(generate())

    print(f"\n{'=' * 60}")
    print("Batch Generation Complete")
    print(f"{'=' * 60}")
    print(f"  Batch plan:   {args.plan}")
    print(f"  Dataset:      {args.dataset}")
    print(f"  Output:       {output_dir}")
    print(f"  Files:        {num_files}")

    # Verify
    reader = BatchReader(output_dir)
    print(f"  Epochs:       {reader.num_epochs}")
    if reader.fingerprint:
        print(f"  Fingerprint:  {reader.fingerprint}")
