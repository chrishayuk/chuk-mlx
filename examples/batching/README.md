# Batching Examples

This directory contains examples demonstrating the batching infrastructure for efficient and reproducible training.

## Examples

### 1. Basic Token-Budget Batching (`01_basic_batching.py`)

Demonstrates the core batching functionality:
- Creating samples with the canonical `Sample` schema
- Building and using `LengthCache` for sequence lengths
- Configuring `BucketSpec` for length-based bucketing
- Using `TokenBudgetBatchSampler` to form efficient batches
- Computing and interpreting `BatchMetrics`

```bash
python examples/batching/01_basic_batching.py
```

**Key concepts:**
- Token-budget batching maximizes GPU utilization by filling batches to a token limit rather than a fixed sample count
- Bucketing groups similar-length sequences to reduce padding waste
- Deterministic batching with seed control enables reproducibility

### 2. Sequence Packing (`02_sequence_packing.py`)

Demonstrates packing multiple sequences into single training examples:
- Using `PackingConfig` to configure packing behavior
- Comparing packing algorithms (FIRST_FIT, BEST_FIT, GREEDY)
- Creating `PackedSequence` objects with segment tracking
- Generating segment-aware attention masks with `create_segment_attention_mask()`
- Measuring packing efficiency with `PackingMetrics`

```bash
python examples/batching/02_sequence_packing.py
```

**Key concepts:**
- Packing concatenates short sequences to reduce padding waste (often 50-70% token reduction)
- Segment-aware attention masks prevent cross-sample information leakage
- Different packing algorithms offer trade-offs between speed and efficiency

### 3. BatchPlan for Distributed Training (`03_batch_plan.py`)

Demonstrates precomputed batch plans for reproducibility and distributed training:
- Building `BatchPlan` artifacts with `BatchPlanBuilder`
- Saving and loading plans to disk
- Sharding plans across distributed workers with `.shard()`
- Resuming from checkpoints with `.iter_from()`
- Verifying reproducibility with `compute_batch_fingerprint()`

```bash
python examples/batching/03_batch_plan.py
```

**Key concepts:**
- Batch plans are precomputed schedules that ensure identical training order across runs
- Plans can be sharded across ranks for data-parallel distributed training
- Fingerprints enable CI/CD verification that batching hasn't changed

### 4. Distributed Training (`04_distributed.py`)

Demonstrates the complete distributed training infrastructure:
- Using `DistributedConfig` for rank/world_size management
- Sharding batch plans with `shard_batch_plan()`
- Interleaving microbatches for balanced work distribution
- Checkpoint resume with `CheckpointPosition`
- Pre-sharding plans via CLI

```bash
python examples/batching/04_distributed.py
```

**Key concepts:**
- `DistributedConfig.from_env()` reads standard distributed environment variables
- `shard_batch_plan()` divides work across workers (rank i gets microbatches i, i+world_size, ...)
- `interleave_microbatches()` ensures each rank gets a mix of bucket sizes
- `CheckpointPosition` tracks epoch and microbatch index for resuming

### 5. End-to-End Pipeline (`05_e2e_pipeline.py`)

**The complete data pipeline from raw dataset to training-ready batches.** This is the recommended starting point for understanding how all components work together.

```bash
python examples/batching/05_e2e_pipeline.py
```

Demonstrates all stages:

1. **Dataset Preparation** - Create/load JSONL training data
2. **Tokenization** - Tokenize with instruction/response formatting
3. **Length Caching** - Compute token lengths for efficient batching
4. **Batch Planning** - Create reproducible `BatchPlan` with bucketing
5. **Training Loop** - Two paths:
   - **Runtime streaming**: `BatchPlan.iter_epoch()` (recommended)
   - **Offline loading**: Pre-generated NPZ batch files
6. **Distributed Training** - Shard plans across workers

**Pipeline diagram:**
```
Dataset (JSONL) → Tokenize → Length Cache → BatchPlan → Training Loop
                                              ↓
                              Optional: BatchWriter → NPZ files
```

### 6. Batching Analysis (`06_analyze.py`)

Demonstrates analysis and optimization tools:
- Computing length histograms with percentiles
- Analyzing bucket efficiency metrics
- Getting optimal bucket edge suggestions
- Creating complete efficiency reports
- Comparing original vs optimized configurations

```bash
python examples/batching/06_analyze.py
```

**Key concepts:**
- Length histograms reveal your data distribution for choosing bucket edges
- Three optimization goals: minimize_waste, balance_buckets, minimize_memory
- Efficiency reports combine histogram, bucket analysis, and recommendations

**CLI equivalents:**
```bash
# View length histogram
lazarus data batching histogram -c lengths.jsonl --bins 15

# Analyze bucket efficiency
lazarus data batching analyze -c lengths.jsonl --bucket-edges 128,256,512

# Get bucket suggestions
lazarus data batching suggest -c lengths.jsonl --num-buckets 4 --goal waste
```

### 7. Streaming Sample Sources (`07_streaming.py`)

Demonstrates the streaming module for online learning:
- Using `OfflineDatasetStream` for static datasets
- Using `AsyncOfflineDatasetStream` for async I/O
- `StreamMetrics` for tracking sample statistics
- Sample filtering and transformation
- Replay sample marking with `with_replay()`

```bash
python examples/batching/07_streaming.py
```

**Key concepts:**
- Streams provide a unified interface for sample sources (offline, gym, synthetic)
- `StreamSample` is the canonical streaming data format (Pydantic model)
- Metrics track sample counts, lengths, difficulty, and success rates
- Samples are immutable; `with_replay()` creates a marked copy

### 8. Online Learning with Gym Streams (`08_online_learning.py`)

Demonstrates the complete online learning infrastructure:
- `MockGymStream` for testing (simulates puzzle arcade)
- `ReplayBuffer` for experience storage with priority sampling
- `RollingBatchPlanWindow` for streaming batch plans
- Curriculum-aware difficulty tracking
- `GymConfig` for real server connections

```bash
python examples/batching/08_online_learning.py
```

**Key concepts:**
- Replay buffers store experiences with configurable eviction (FIFO, priority, random)
- Rolling windows build batch plans from buffer snapshots
- Priority-weighted sampling focuses training on important experiences
- Gym streams connect to puzzle arcade servers for RL training

**CLI equivalents:**
```bash
# Run mock gym stream for testing
lazarus gym run -t gpt2 --mock --num-episodes 10

# Connect to puzzle arcade server
lazarus gym run -t gpt2 --host localhost --port 8023

# Save samples to buffer file
lazarus gym run -t gpt2 --mock --output buffer.json

# Display gym configuration info
lazarus gym info
```

## Module Architecture

```
batching/
├── __init__.py              # Public API (re-exports from submodules)
│
├── core/                    # Core batching primitives
│   ├── buckets.py          # BucketSpec, BucketStats
│   ├── sampler.py          # TokenBudgetBatchSampler, BatchSpec
│   └── metrics.py          # BatchMetrics, BatchShapeHistogram
│
├── planning/               # Batch plan artifacts & reproducibility
│   ├── batch_plan.py       # BatchPlan, BatchPlanBuilder, EpochPlan
│   ├── predictability.py   # BatchingConfig, PadPolicy, fingerprints
│   └── packing.py          # PackingConfig, pack_sequences, segment masks
│
├── generation/             # Batch file I/O
│   ├── io.py               # BatchWriter, BatchReader, CollatedBatch
│   └── length_cache.py     # LengthCache, LengthEntry
│
├── analyze/                # Analysis & instrumentation
│   └── efficiency.py       # Histograms, bucket suggestions, reports
│
└── streaming/              # Online learning infrastructure
    ├── types.py            # StreamSample, StreamMetrics, enums
    ├── protocols.py        # SampleStream, AsyncSampleStream protocols
    ├── offline_stream.py   # OfflineDatasetStream, AsyncOfflineDatasetStream
    ├── gym_stream.py       # GymEpisodeStream, MockGymStream, GymConfig
    ├── replay_buffer.py    # ReplayBuffer, BufferEvictionPolicy
    └── rolling_window.py   # RollingBatchPlanWindow, WindowConfig
```

## CLI Commands

```bash
# Length cache
lazarus data lengths build -d train.jsonl -t gpt2 -o lengths.jsonl
lazarus data lengths stats -c lengths.jsonl

# Batch plan
lazarus data batchplan build -l lengths.jsonl -e 3 -b 4096 -o batch_plan/
lazarus data batchplan info -p batch_plan/
lazarus data batchplan verify -p batch_plan/ -l lengths.jsonl
lazarus data batchplan shard -p batch_plan/ -w 4 -o shards/

# Analysis
lazarus data batching histogram -c lengths.jsonl --bins 20
lazarus data batching analyze -c lengths.jsonl --bucket-edges 128,256,512
lazarus data batching suggest -c lengths.jsonl --num-buckets 4 --goal waste

# Batch generation
lazarus data batch generate -p batch_plan/ -d train.jsonl -t gpt2 -o batches/

# Gym streaming (online learning)
lazarus gym run -t gpt2 --mock --num-episodes 10        # Test with mock stream
lazarus gym run -t gpt2 --host localhost --port 8023    # Connect to server
lazarus gym run -t gpt2 --mock --output buffer.json     # Save to buffer file
lazarus gym info                                         # Show configuration

# Training with batching options
lazarus train sft --model model --data train.jsonl --batchplan batch_plan/
lazarus train sft --model model --data train.jsonl --bucket-edges 128,256,512 --token-budget 4096
lazarus train sft --model model --data train.jsonl --pack --pack-max-len 2048
lazarus train sft --model model --data train.jsonl --online --gym-host localhost --gym-port 8023
```

## Quick Start

```python
import asyncio
from chuk_lazarus.data.batching import (
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

## Performance Tips

1. **Choose appropriate bucket edges**: Use `lazarus data batching suggest` to find optimal edges for your data
2. **Tune token budget**: Balance between GPU memory and batch diversity
3. **Use packing for short sequences**: Especially effective when many sequences are < 50% of max length
4. **Precompute BatchPlans**: Avoids batching overhead during training
5. **Use predictable mode for debugging**: Ensures identical batches across runs
