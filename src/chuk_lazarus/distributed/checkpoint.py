"""
Checkpoint position utilities for distributed training.

Provides utilities for saving and resuming training position within a batch plan.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..data.batching import BatchPlan, MicrobatchSpec


@dataclass
class CheckpointPosition:
    """
    Position within a batch plan for checkpoint resume.

    Attributes:
        epoch: Current epoch number
        microbatch_idx: Index of next microbatch to process
        global_step: Total training steps completed
    """

    epoch: int
    microbatch_idx: int
    global_step: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> CheckpointPosition:
        """Create from dictionary."""
        return cls(
            epoch=data["epoch"],
            microbatch_idx=data["microbatch_idx"],
            global_step=data.get("global_step", 0),
        )


def save_checkpoint_position(
    position: CheckpointPosition,
    path: str | Path,
) -> None:
    """
    Save checkpoint position to file.

    Args:
        position: The position to save
        path: Path to save to (JSON file)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(position.to_dict(), f, indent=2)


def load_checkpoint_position(path: str | Path) -> CheckpointPosition:
    """
    Load checkpoint position from file.

    Args:
        path: Path to load from (JSON file)

    Returns:
        CheckpointPosition
    """
    with open(path) as f:
        data = json.load(f)
    return CheckpointPosition.from_dict(data)


def iter_from_checkpoint(
    plan: BatchPlan,
    position: CheckpointPosition | None = None,
) -> Iterator[tuple[int, int, MicrobatchSpec]]:
    """
    Iterate over a batch plan from a checkpoint position.

    If no position is provided, starts from the beginning.

    Args:
        plan: The batch plan to iterate
        position: Starting position (None = start from beginning)

    Yields:
        Tuples of (epoch, microbatch_idx, microbatch_spec)
    """
    if position is None:
        epoch = 0
        microbatch_idx = 0
    else:
        epoch = position.epoch
        microbatch_idx = position.microbatch_idx

    yield from plan.iter_from(epoch, microbatch_idx)
