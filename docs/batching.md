# Batching Module

A comprehensive, Pydantic-native batching infrastructure for efficient and reproducible ML training. Implements token-budget batching, sequence packing, and distributed batch planning.

## Overview

This module provides:
- **Token-budget batching** - Form batches by token count rather than sample count for optimal GPU utilization
- **Length-based bucketing** - Group similar-length sequences to minimize padding waste
- **Sequence packing** - Pack multiple short sequences into single training examples
- **Segment-aware attention** - Block-diagonal attention masks for packed sequences
- **BatchPlan artifacts** - Precomputed batch schedules for reproducibility and distributed training
- **Fingerprinting** - Verify batch ordering hasn't changed across runs

## Design Principles

- **Pydantic-native**: All data structures use Pydantic BaseModel for validation
- **Async-first**: Async I/O for length caching and batch plan persistence
- **Deterministic**: Seed control ensures identical batching across runs
- **Distributed-ready**: Built-in sharding and checkpoint resume support

## Quick Start

```python
import asyncio
from chuk_lazarus.data import (
    BucketSpec,
    TokenBudgetBatchSampler,
    BatchingConfig,
    BatchPlanBuilder,
)

async def main():
    # Sample lengths (normally from tokenized data)
    lengths = {f"s{i}": 100 + i * 10 for i in range(50)}

    # Configure bucketing
    bucket_spec = BucketSpec(
        edges=(128, 256, 512),
        overflow_max=1024,
    )

    # Create sampler
    sampler = TokenBudgetBatchSampler(
        lengths=lengths,
        bucket_spec=bucket_spec,
        token_budget=2048,
        seed=42,
    )

    # Iterate batches
    async for batch in sampler.iter_epoch(epoch=0):
        print(f"Batch: {batch.batch_size} samples, {batch.max_length} max len")

asyncio.run(main())
```

## Architecture

```
batching/
├── __init__.py              # Public API (re-exports from submodules)
├── README.md
│
├── core/                    # Core batching primitives
│   ├── buckets.py          # BucketSpec, BucketStats, BucketId
│   ├── sampler.py          # TokenBudgetBatchSampler, BatchSpec
│   └── metrics.py          # BatchMetrics, BatchShapeHistogram
│
├── planning/               # Batch plan artifacts & reproducibility
│   ├── batch_plan.py       # BatchPlan, BatchPlanBuilder, EpochPlan
│   ├── predictability.py   # BatchingConfig, PadPolicy, fingerprints
│   └── packing.py          # PackingConfig, pack_sequences, segment masks
│
├── generation/             # Batch file I/O (unified batch generation)
│   ├── io.py               # BatchWriter, BatchReader, CollatedBatch
│   └── length_cache.py     # LengthCache, LengthEntry
│
├── streaming/              # Online/RL data sources
│   ├── telnet_client.py    # TelnetGymClient for puzzle arcade
│   ├── gym_stream.py       # GymEpisodeStream, MockGymStream
│   ├── replay_buffer.py    # ReplayBuffer with priority sampling
│   ├── rolling_window.py   # RollingBatchPlanWindow
│   ├── protocols.py        # SampleStream, AsyncSampleStream protocols
│   └── types.py            # StreamSample, StreamMetrics
│
└── analyze/                # Analysis & instrumentation
    └── efficiency.py       # Histograms, bucket suggestions, reports
```

## Core Modules

### `core/buckets.py` - Length-Based Bucketing

Configure how sequences are grouped by length.

```python
from chuk_lazarus.data.batching import BucketSpec, BucketStats

# Define bucket boundaries
bucket_spec = BucketSpec(
    edges=(128, 256, 512),  # Bucket boundaries
    overflow_max=1024,       # Max for overflow bucket
)

# Get bucket for a length
bucket_id = bucket_spec.get_bucket(150)  # Returns 1 (129-256)

# Get bucket range
min_len, max_len = bucket_spec.get_bucket_range(bucket_id)

# Check if overflow
is_overflow = bucket_spec.is_overflow(bucket_id)
```

### `generation/length_cache.py` - Async Length Caching

Cache sequence lengths for efficient batching without re-tokenizing.

```python
from chuk_lazarus.data.batching import LengthCache

# Build cache
async with LengthCache.create("lengths.jsonl", "tokenizer_v1") as cache:
    for sample in samples:
        await cache.add(sample.id, sample.length)

# Load cache
cache = await LengthCache.load("lengths.jsonl")

# Get all lengths
lengths = cache.get_all()  # Dict[str, int]
```

### `core/sampler.py` - Token-Budget Batch Sampler

Form batches that maximize token utilization.

```python
from chuk_lazarus.data.batching import TokenBudgetBatchSampler, BatchSpec

sampler = TokenBudgetBatchSampler(
    lengths=lengths,           # Dict[str, int] from LengthCache
    bucket_spec=bucket_spec,
    token_budget=4096,         # Max tokens per batch
    seed=42,                   # For reproducibility
)

# Get info
print(f"Samples: {sampler.num_samples}")
print(f"Est. batches/epoch: {sampler.estimate_batches_per_epoch()}")

# Iterate epoch
async for batch_spec in sampler.iter_epoch(epoch=0):
    print(f"Batch: {batch_spec.batch_size} samples")
    print(f"Max length: {batch_spec.max_length}")
    print(f"Sample IDs: {batch_spec.sample_ids}")
```

### `core/metrics.py` - Batch Metrics

Track batching efficiency and waste.

```python
from chuk_lazarus.data.batching import BatchMetrics

# Compute metrics
loss_tokens = {sample_id: sample.num_loss_tokens for sample_id, sample in samples.items()}
metrics = sampler.compute_metrics(loss_tokens_per_sample=loss_tokens)

# Summary
print(metrics.summary())
# {
#     'total_samples': 100,
#     'total_batches': 15,
#     'efficiency': 0.66,
#     'padding_waste': 0.34,
#     ...
# }

# Per-bucket breakdown
for bucket in metrics.bucket_summary():
    print(f"Bucket {bucket['bucket_id']}: {bucket['efficiency']}")
```

## Planning Modules

### `planning/predictability.py` - Predictability Mode

Ensure deterministic, reproducible batching.

```python
from chuk_lazarus.data.batching import (
    BatchingConfig,
    BatchingMode,
    PadPolicy,
    compute_batch_fingerprint,
    verify_batch_fingerprint,
)

# Create predictable config
config = BatchingConfig.predictable(
    token_budget=4096,
    bucket_edges=(128, 256, 512),
    seed=42,
)

# Or throughput-optimized config
config = BatchingConfig.throughput(
    token_budget=4096,
    bucket_edges=(128, 256, 512),
)

# Compute fingerprint for verification
fingerprint = compute_batch_fingerprint(batches, config)
print(f"Fingerprint: {fingerprint.fingerprint}")

# Verify later runs match
matches, error = verify_batch_fingerprint(batches, fingerprint)
if not matches:
    print(f"Batching changed: {error}")
```

### `planning/packing.py` - Sequence Packing

Pack multiple short sequences into single training examples.

```python
from chuk_lazarus.data.batching import (
    PackingConfig,
    PackingMode,
    SequenceToPack,
    pack_sequences,
    create_segment_attention_mask,
    compute_packing_metrics,
)

# Prepare sequences
sequences = [
    SequenceToPack(
        sample_id="s1",
        input_ids=[1, 2, 3, 4],
        loss_mask=[0, 0, 1, 1],
    ),
    # ... more sequences
]

# Configure packing
config = PackingConfig(
    mode=PackingMode.FIRST_FIT,
    max_length=512,
    pad_to_max=True,
)

# Pack sequences
packed = pack_sequences(sequences, config, pad_token_id=0)

# Each packed sequence has:
for p in packed:
    print(f"Samples: {p.sample_ids}")
    print(f"Segments: {p.num_segments}")
    print(f"Efficiency: {p.efficiency:.1%}")

# Create attention mask (block-diagonal)
mask = create_segment_attention_mask(packed[0].segment_ids)
# Shape: (seq_len, seq_len), blocks attention across segments

# Compute packing metrics
metrics = compute_packing_metrics(packed)
print(f"Packing ratio: {metrics.packing_ratio:.2f}x")
print(f"Token reduction: {1 - 1/metrics.packing_ratio:.0%}")
```

### `planning/batch_plan.py` - Batch Plan Artifacts

Precompute batch schedules for reproducibility and distributed training.

```python
from chuk_lazarus.data.batching import (
    BatchPlan,
    BatchPlanBuilder,
    save_batch_plan,
    load_batch_plan,
)

# Build plan
builder = BatchPlanBuilder(
    lengths=lengths,
    batching_config=config,
    dataset_hash="my_dataset_v1",
    tokenizer_hash="tokenizer_v1",
)

plan = await builder.build(num_epochs=3)

print(f"Total microbatches: {plan.total_microbatches}")
print(f"Fingerprint: {plan.fingerprint}")

# Save/load
save_batch_plan(plan, "batch_plan/")
loaded = load_batch_plan("batch_plan/")

# Shard for distributed training
for rank in range(world_size):
    shard = plan.shard(rank=rank, world_size=world_size)
    # Each worker gets non-overlapping batches

# Resume from checkpoint
for epoch, mb_idx, mb in plan.iter_from(epoch=1, microbatch_idx=5):
    # Continues from where training left off
    pass
```

## Generation Modules

### `generation/io.py` - Batch I/O

Read and write NPZ batch files from BatchPlan.

```python
from chuk_lazarus.data.batching import BatchWriter, BatchReader

# Write batches to disk
writer = BatchWriter(plan, samples, output_dir="./batches")
writer.write_all()

# Read batches back
reader = BatchReader("./batches")
for batch in reader.iter_epoch(0):
    model(batch["input_ids"])
```

## Analysis Modules

### `analyze/efficiency.py` - Batching Analysis

Analyze and optimize batching configuration.

```python
from chuk_lazarus.data.batching import (
    compute_length_histogram,
    analyze_bucket_efficiency,
    suggest_bucket_edges,
    create_efficiency_report,
    OptimizationGoal,
)

# Compute length histogram
histogram = compute_length_histogram(lengths, num_bins=15)
print(histogram.to_ascii(width=50))
print(f"P50: {histogram.p50}, P90: {histogram.p90}")

# Analyze bucket efficiency
analysis = analyze_bucket_efficiency(lengths, bucket_spec)
print(f"Overall efficiency: {analysis.overall_efficiency:.1%}")
print(analysis.to_ascii())

# Get bucket edge suggestions
suggestion = suggest_bucket_edges(
    lengths,
    num_buckets=4,
    goal=OptimizationGoal.MINIMIZE_WASTE,
)
print(f"Suggested edges: {suggestion.edges}")
print(f"Estimated efficiency: {suggestion.estimated_efficiency:.1%}")

# Create complete efficiency report
report = create_efficiency_report(lengths, bucket_spec)
print(report.to_ascii())
for rec in report.recommendations:
    print(f"  - {rec}")
```

## Streaming Modules

### `streaming/telnet_client.py` - Puzzle Arcade Client

Connect to the puzzle arcade server for online training data.

```python
from chuk_lazarus.data.batching.streaming import (
    TelnetGymClient,
    TelnetClientConfig,
    PuzzleGame,
    PuzzleDifficulty,
)

# Configure connection
config = TelnetClientConfig(
    host="localhost",
    port=8023,
    connect_timeout=10.0,
    read_timeout=30.0,
)

# Connect and play puzzles
async with TelnetGymClient(config) as client:
    # Start a puzzle
    obs = await client.start_puzzle(PuzzleGame.SUDOKU, PuzzleDifficulty.EASY)
    print(f"Game: {obs.game}, Seed: {obs.seed}")
    print(f"Optimal steps: {obs.optimal_steps}")

    # Get hints (optimal moves)
    hint = await client.get_hint()
    print(f"Next move: {hint.message}")

    # Get current state
    state = await client.show_state()

    # Quit puzzle
    await client.quit_puzzle()
```

**Supported puzzles:** Sudoku, KenKen, Kakuro, Binary, Futoshiki, Nonogram, Logic Grid, Killer Sudoku, Lights Out, Mastermind, Slitherlink, Bridges, Hitori, Shikaku, Hidato, Tents, Fillomino, Star Battle, Sokoban, Knapsack, Nurikabe, Minesweeper

### `streaming/replay_buffer.py` - Replay Buffer

Bounded buffer for online learning with priority sampling.

```python
from chuk_lazarus.data.batching.streaming import (
    ReplayBuffer,
    ReplayBufferConfig,
    BufferEvictionPolicy,
    StreamSample,
    SampleSource,
)

# Create buffer with difficulty tracking
buffer = ReplayBuffer(
    ReplayBufferConfig(
        max_size=10000,
        eviction_policy=BufferEvictionPolicy.FIFO,
        track_difficulty=True,
        track_success=True,
    )
)

# Add samples
sample = StreamSample(
    input_ids=(1, 2, 3, 4, 5),
    loss_mask=(0, 0, 1, 1, 1),
    sample_id="sudoku_42_step0",
    dataset_id="puzzle_arcade",
    source=SampleSource.GYM,
    episode_id="sudoku_42",
    step_index=0,
    difficulty=0.3,
    success=True,
)
buffer.add(sample)

# Sample from buffer
samples = buffer.sample(n=32)

# Get statistics
print(f"Buffer size: {buffer.size}")
print(f"Mean difficulty: {buffer.mean_difficulty:.2f}")

# Priority sampling by difficulty
hard_samples = buffer.sample(n=16, min_difficulty=0.7)
```

### `streaming/rolling_window.py` - Rolling Batch Windows

Build batch plans over rolling buffer snapshots for online learning.

```python
from chuk_lazarus.data.batching.streaming import (
    RollingBatchPlanWindow,
    WindowConfig,
)

# Configure rolling window
window = RollingBatchPlanWindow(
    buffer=buffer,
    config=WindowConfig(
        window_size=1000,
        overlap=100,
    ),
    batching_config=batching_config,
)

# Get next window's batch plan
plan = await window.next_window()

# Iterate batches
for mb in plan.iter_epoch(0):
    batch = collate([samples[sid] for sid in mb.samples])
    train_step(batch)
```

### Complete Online Training Example

```python
import asyncio
from chuk_lazarus.data.batching.streaming import (
    TelnetGymClient, TelnetClientConfig, PuzzleGame, PuzzleDifficulty,
    ReplayBuffer, ReplayBufferConfig, StreamSample, SampleSource,
)

async def collect_training_data():
    config = TelnetClientConfig(host="localhost", port=8023)
    buffer = ReplayBuffer(ReplayBufferConfig(max_size=10000))

    puzzles = [
        (PuzzleGame.SUDOKU, PuzzleDifficulty.EASY),
        (PuzzleGame.SUDOKU, PuzzleDifficulty.MEDIUM),
        (PuzzleGame.BINARY, PuzzleDifficulty.EASY),
    ]

    for game, difficulty in puzzles:
        async with TelnetGymClient(config) as client:
            obs = await client.start_puzzle(game, difficulty)
            episode_id = f"{game.value}_{obs.seed}"

            for step in range(5):
                hint = await client.get_hint()
                if not hint.success:
                    break

                # Create training sample
                sample = StreamSample(
                    input_ids=tokenize(f"Puzzle: {game.value}\nMove?"),
                    loss_mask=create_loss_mask(hint.message),
                    sample_id=f"{episode_id}_step{step}",
                    dataset_id="puzzle_arcade",
                    source=SampleSource.GYM,
                    episode_id=episode_id,
                    step_index=step,
                    difficulty={"easy": 0.3, "medium": 0.6}[difficulty.value],
                )
                buffer.add(sample)

            await client.quit_puzzle()

    print(f"Collected {buffer.size} samples")
    return buffer

asyncio.run(collect_training_data())
```

## CLI Commands

### Length Cache
```bash
# Build length cache
lazarus data lengths build --dataset train.jsonl --tokenizer gpt2 --output lengths.jsonl

# Show length cache stats
lazarus data lengths stats --cache lengths.jsonl
```

### Batch Plan
```bash
# Build batch plan
lazarus data batchplan build \
    --lengths lengths.jsonl \
    --epochs 3 \
    --token-budget 4096 \
    --bucket-edges 128,256,512 \
    --output batch_plan/

# Show batch plan info
lazarus data batchplan info --plan batch_plan/

# Show sharded view for distributed training
lazarus data batchplan info --plan batch_plan/ --rank 0 --world-size 4

# Verify batch plan reproducibility
lazarus data batchplan verify --plan batch_plan/ --lengths lengths.jsonl

# Pre-shard for distributed training
lazarus data batchplan shard --plan batch_plan/ --world-size 4 --output shards/
```

### Batching Analysis
```bash
# Analyze batching efficiency
lazarus data batching analyze --cache lengths.jsonl --bucket-edges 128,256,512

# Display length histogram
lazarus data batching histogram --cache lengths.jsonl --bins 20 --width 50

# Get bucket edge suggestions
lazarus data batching suggest --cache lengths.jsonl --num-buckets 4 --goal waste
```

### Batch File Generation
```bash
# Generate NPZ batch files from BatchPlan
lazarus data batch generate \
    --plan batch_plan/ \
    --dataset train.jsonl \
    --tokenizer gpt2 \
    --output batches/
```

## Performance Tips

1. **Choose appropriate bucket edges**: Match your data's length distribution
2. **Tune token budget**: Balance between GPU memory and batch diversity
3. **Use packing for short sequences**: Especially effective when many sequences are < 50% of max length
4. **Precompute BatchPlans**: Avoids batching overhead during training
5. **Use predictable mode for debugging**: Ensures identical batches across runs

## Examples

See `examples/batching/` for complete working examples:

- `01_basic_batching.py` - Token-budget batching, bucketing, metrics
- `02_sequence_packing.py` - Packing algorithms, segment attention masks
- `03_batch_plan.py` - BatchPlan building, saving/loading, sharding, resume
- `04_distributed.py` - Distributed training, sharding, checkpoints
- `05_e2e_pipeline.py` - Complete end-to-end data pipeline
- `06_analyze.py` - Length histograms, bucket analysis, optimization
- `08_online_learning.py` - Online learning with gym streams
- `09_puzzle_arcade_integration.py` - Complete puzzle arcade integration

## Testing

```bash
pytest tests/data/batching/ -v --cov=src/chuk_lazarus/data/batching
```
