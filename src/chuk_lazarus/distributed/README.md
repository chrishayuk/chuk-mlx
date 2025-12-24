# Distributed Training Module

Utilities for distributed training with batch plan sharding, checkpoint resume, and rank management.

## Overview

The distributed module provides:
- **Rank/World Size Management**: `DistributedConfig` reads from environment variables
- **Batch Plan Sharding**: Divide work evenly across workers
- **Checkpoint Resume**: Save and restore training position
- **Work Balancing**: Interleave microbatches for even distribution

## Quick Start

```python
from chuk_lazarus.distributed import (
    DistributedConfig,
    shard_batch_plan,
    CheckpointPosition,
    is_main_process,
)
from chuk_lazarus.data import load_batch_plan

# 1. Get config from environment (RANK, WORLD_SIZE, etc.)
config = DistributedConfig.from_env()

# 2. Load and shard batch plan
plan = load_batch_plan("./batch_plan/")
my_plan = shard_batch_plan(plan, config)

# 3. Only main process prints
if is_main_process():
    print(f"Training with {config.world_size} workers")
    print(f"Total batches: {plan.total_microbatches}")
    print(f"My batches: {my_plan.total_microbatches}")

# 4. Training loop with resume support
for epoch, mb_idx, mb in my_plan.iter_from(epoch=0, microbatch_idx=0):
    samples = load_samples(mb.sample_ids)
    loss = train_step(model, samples)
```

## Components

### DistributedConfig

Configuration for distributed training with environment variable support.

```python
from chuk_lazarus.distributed import DistributedConfig

# Single worker (default)
config = DistributedConfig()
assert config.rank == 0
assert config.world_size == 1

# Explicit configuration
config = DistributedConfig(rank=2, world_size=8)

# From environment variables
# Reads: RANK, WORLD_SIZE, LOCAL_RANK, LOCAL_WORLD_SIZE
config = DistributedConfig.from_env()

# Properties
config.is_main        # True if rank == 0
config.is_distributed # True if world_size > 1
```

**Environment Variables:**
| Variable | Description |
|----------|-------------|
| `RANK` or `WORLD_RANK` | Global worker rank |
| `WORLD_SIZE` | Total number of workers |
| `LOCAL_RANK` | Rank on current node |
| `LOCAL_WORLD_SIZE` | Workers per node |

### Convenience Functions

```python
from chuk_lazarus.distributed import (
    get_rank,
    get_world_size,
    is_main_process,
)

rank = get_rank()           # Current worker rank
world = get_world_size()    # Total workers
if is_main_process():
    print("I am the main process")
```

### Sharding

Divide batch plans across workers.

```python
from chuk_lazarus.distributed import shard_batch_plan, interleave_microbatches
from chuk_lazarus.distributed.sharding import compute_shard_stats

# Shard using current config
my_plan = shard_batch_plan(plan, config)

# Or use plan.shard() directly
my_plan = plan.shard(rank=0, world_size=4)

# Interleave microbatches for balanced bucket distribution
from chuk_lazarus.distributed import interleave_microbatches
balanced = interleave_microbatches(microbatches)

# Analyze work distribution
stats = compute_shard_stats(plan, world_size=4)
for rank, s in stats.items():
    print(f"Rank {rank}: {s['num_batches']} batches, {s['total_tokens']} tokens")
```

**Sharding Strategy:**
- Worker `i` gets microbatches `i, i+world_size, i+2*world_size, ...`
- Interleaving ensures each worker gets a mix of bucket sizes
- Deterministic: same plan + same rank = same batches

### Checkpoint Resume

Save and restore training position within a batch plan.

```python
from chuk_lazarus.distributed import (
    CheckpointPosition,
    save_checkpoint_position,
    load_checkpoint_position,
)
from chuk_lazarus.distributed.checkpoint import iter_from_checkpoint

# Create checkpoint position
pos = CheckpointPosition(
    epoch=1,
    microbatch_idx=50,
    global_step=250,
)

# Save to disk
save_checkpoint_position(pos, "checkpoint_pos.json")

# Load from disk
pos = load_checkpoint_position("checkpoint_pos.json")

# Resume iteration
for epoch, mb_idx, mb in plan.iter_from(pos.epoch, pos.microbatch_idx):
    # Continue training from checkpoint
    pass

# Or use helper function
from chuk_lazarus.distributed.checkpoint import iter_from_checkpoint
for epoch, mb_idx, mb in iter_from_checkpoint(plan, pos):
    pass
```

## CLI Commands

```bash
# View batch plan info
lazarus data batchplan info -p batch_plan/

# View sharded plan for specific rank
lazarus data batchplan info -p batch_plan/ --rank 0 --world-size 4

# Pre-shard plan for all workers
lazarus data batchplan shard -p batch_plan/ --world-size 4 -o shards/
```

## Distributed Training Pattern

Complete example for a distributed training setup:

```python
#!/usr/bin/env python3
"""Distributed training script."""

import os
from pathlib import Path

from chuk_lazarus.distributed import (
    DistributedConfig,
    CheckpointPosition,
    shard_batch_plan,
    save_checkpoint_position,
    load_checkpoint_position,
    is_main_process,
)
from chuk_lazarus.data import load_batch_plan


def main():
    # 1. Initialize distributed config from environment
    config = DistributedConfig.from_env()

    if is_main_process():
        print(f"Starting distributed training with {config.world_size} workers")

    # 2. Load and shard batch plan
    plan = load_batch_plan("./batch_plan/")
    my_plan = shard_batch_plan(plan, config)

    print(f"Rank {config.rank}: {my_plan.total_microbatches} batches assigned")

    # 3. Load checkpoint if resuming
    checkpoint_dir = Path(f"./checkpoints/rank_{config.rank}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "position.json"

    if checkpoint_path.exists():
        pos = load_checkpoint_position(checkpoint_path)
        print(f"Rank {config.rank}: Resuming from epoch {pos.epoch}, step {pos.global_step}")
    else:
        pos = CheckpointPosition(epoch=0, microbatch_idx=0, global_step=0)

    # 4. Training loop
    global_step = pos.global_step

    for epoch, mb_idx, mb in my_plan.iter_from(pos.epoch, pos.microbatch_idx):
        # Load samples for this microbatch
        samples = load_samples(mb.sample_ids)

        # Training step
        loss = train_step(model, samples, mb.max_len)
        global_step += 1

        # Checkpoint every 100 steps
        if global_step % 100 == 0:
            save_checkpoint_position(
                CheckpointPosition(epoch, mb_idx + 1, global_step),
                checkpoint_path,
            )
            if is_main_process():
                save_model_checkpoint(model, f"./checkpoints/step_{global_step}")

    if is_main_process():
        print("Training complete!")


if __name__ == "__main__":
    main()
```

## Launch Scripts

### Single Machine, Multiple Processes

```bash
# Using environment variables
RANK=0 WORLD_SIZE=4 python train.py &
RANK=1 WORLD_SIZE=4 python train.py &
RANK=2 WORLD_SIZE=4 python train.py &
RANK=3 WORLD_SIZE=4 python train.py &
wait
```

### With torchrun (PyTorch Distributed)

```bash
torchrun --nproc_per_node=4 train.py
```

### Pre-sharding for Simple Parallelism

```bash
# Pre-shard batch plans
lazarus data batchplan shard -p batch_plan/ -w 4 -o shards/

# Each worker loads its own shard
RANK=0 python train.py --plan shards/rank_0/ &
RANK=1 python train.py --plan shards/rank_1/ &
RANK=2 python train.py --plan shards/rank_2/ &
RANK=3 python train.py --plan shards/rank_3/ &
wait
```
