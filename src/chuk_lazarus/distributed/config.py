"""
Distributed configuration utilities.

Provides rank and world size management for distributed training.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Global config instance for convenience functions
_global_config: DistributedConfig | None = None


@dataclass
class DistributedConfig:
    """
    Configuration for distributed training.

    Attributes:
        rank: Current worker rank (0-indexed)
        world_size: Total number of workers
        local_rank: Local rank on this node (for multi-GPU per node)
        local_world_size: Number of workers on this node
    """

    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    local_world_size: int = 1

    def __post_init__(self):
        if self.rank < 0 or self.rank >= self.world_size:
            raise ValueError(f"Invalid rank {self.rank} for world_size {self.world_size}")
        if self.local_rank < 0 or self.local_rank >= self.local_world_size:
            raise ValueError(
                f"Invalid local_rank {self.local_rank} for local_world_size {self.local_world_size}"
            )

    @property
    def is_main(self) -> bool:
        """Check if this is the main (rank 0) process."""
        return self.rank == 0

    @property
    def is_distributed(self) -> bool:
        """Check if running in distributed mode."""
        return self.world_size > 1

    @classmethod
    def from_env(cls) -> DistributedConfig:
        """
        Create config from environment variables.

        Supports common distributed training environment variables:
        - RANK / WORLD_RANK: Global rank
        - WORLD_SIZE: Total workers
        - LOCAL_RANK: Local rank on node
        - LOCAL_WORLD_SIZE: Workers per node

        Returns:
            DistributedConfig with values from environment
        """
        rank = int(os.environ.get("RANK", os.environ.get("WORLD_RANK", "0")))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))

        return cls(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            local_world_size=local_world_size,
        )

    def set_global(self) -> None:
        """Set this config as the global config."""
        global _global_config
        _global_config = self

    @classmethod
    def get_global(cls) -> DistributedConfig:
        """Get the global config, creating from env if not set."""
        global _global_config
        if _global_config is None:
            _global_config = cls.from_env()
        return _global_config


def get_rank() -> int:
    """Get current worker rank."""
    return DistributedConfig.get_global().rank


def get_world_size() -> int:
    """Get total number of workers."""
    return DistributedConfig.get_global().world_size


def is_main_process() -> bool:
    """Check if this is the main (rank 0) process."""
    return DistributedConfig.get_global().is_main


def is_distributed() -> bool:
    """Check if running in distributed mode."""
    return DistributedConfig.get_global().is_distributed
