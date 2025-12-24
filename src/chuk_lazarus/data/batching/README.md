# Batching Module

A comprehensive, Pydantic-native batching infrastructure for efficient and reproducible ML training.

**Full documentation**: [docs/batching.md](../../../../docs/batching.md)

## Quick Start

```python
import asyncio
from chuk_lazarus.data.batching import (
    BucketSpec,
    TokenBudgetBatchSampler,
    BatchingConfig,
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

## Features

- **Token-budget batching** - Form batches by token count for optimal GPU utilization
- **Length-based bucketing** - Group similar-length sequences to minimize padding
- **Sequence packing** - Pack multiple short sequences into single examples
- **Segment-aware attention** - Block-diagonal masks for packed sequences
- **BatchPlan artifacts** - Precomputed schedules for reproducibility
- **Fingerprinting** - Verify batch ordering across runs

## Architecture

```
batching/
├── core/           # BucketSpec, TokenBudgetBatchSampler, BatchMetrics
├── planning/       # BatchPlan, BatchingConfig, PackingConfig
├── generation/     # LengthCache, BatchWriter, BatchReader
└── analyze/        # Histograms, bucket suggestions, efficiency reports
```

## Design Principles

- **Pydantic-native**: All data structures use Pydantic BaseModel
- **Async-first**: Async I/O for caching and persistence
- **Deterministic**: Seed control for identical batching across runs
- **Distributed-ready**: Built-in sharding and checkpoint resume

## CLI Commands

```bash
# Build length cache
lazarus data lengths build -d train.jsonl -t gpt2 -o lengths.jsonl

# Build batch plan
lazarus data batchplan build -l lengths.jsonl -e 3 -b 4096 -o batch_plan/

# Analyze efficiency
lazarus data batching analyze --cache lengths.jsonl --bucket-edges 128,256,512
```

## Testing

```bash
pytest tests/data/batching/ -v
```
