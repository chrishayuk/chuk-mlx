"""
Distributed training infrastructure.

This module provides utilities for distributed training:
- Rank and world size management
- Batch plan sharding
- Checkpoint resume support
- Worker synchronization

Usage:
    from chuk_lazarus.distributed import (
        DistributedConfig,
        get_rank,
        get_world_size,
        is_main_process,
        shard_batch_plan,
    )

    # From environment variables
    config = DistributedConfig.from_env()

    # Or explicit configuration
    config = DistributedConfig(rank=0, world_size=4)

    # Check if main process
    if is_main_process():
        print("This is the main process")

    # Shard a batch plan
    my_plan = shard_batch_plan(plan, config)
"""

from .checkpoint import (
    CheckpointPosition,
    load_checkpoint_position,
    save_checkpoint_position,
)
from .config import (
    DistributedConfig,
    get_rank,
    get_world_size,
    is_main_process,
)
from .sharding import (
    interleave_microbatches,
    shard_batch_plan,
)

__all__ = [
    # Config
    "DistributedConfig",
    "get_rank",
    "get_world_size",
    "is_main_process",
    # Sharding
    "shard_batch_plan",
    "interleave_microbatches",
    # Checkpointing
    "CheckpointPosition",
    "save_checkpoint_position",
    "load_checkpoint_position",
]
