"""
Batch I/O: Write and read batch files from BatchPlan.

This module provides:
- BatchWriter: Write BatchPlan epochs to NPZ files
- BatchReader: Read NPZ batch files back for training
- pad_sequences: Utility to pad sequences to uniform length

Design principles:
- BatchPlan is the universal IR (intermediate representation)
- Writing is just materialization of the plan to disk
- Reading provides same iteration interface as plan.iter_epoch()
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..planning.batch_plan import BatchPlan, MicrobatchSpec, load_batch_plan, save_batch_plan


def pad_sequences(
    sequences: Sequence[Sequence[int]],
    pad_value: int,
    max_length: int | None = None,
    dtype: np.dtype = np.int32,
) -> np.ndarray:
    """
    Pad sequences to uniform length.

    Args:
        sequences: List of sequences (lists of ints)
        pad_value: Value to use for padding
        max_length: Target length. If None, uses longest sequence.
        dtype: NumPy dtype for output array

    Returns:
        NumPy array of shape (num_sequences, max_length)
    """
    if not sequences:
        return np.array([], dtype=dtype)

    if max_length is None:
        max_length = max(len(seq) for seq in sequences)

    padded = []
    for seq in sequences:
        seq_list = list(seq)
        if len(seq_list) >= max_length:
            padded.append(seq_list[:max_length])
        else:
            padded.append(seq_list + [pad_value] * (max_length - len(seq_list)))

    return np.array(padded, dtype=dtype)


class CollatedBatch(BaseModel):
    """
    A collated batch ready for training.

    Contains numpy arrays and metadata.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    input_ids: Any = Field(description="Input token IDs (batch_size, seq_len)")
    loss_mask: Any = Field(description="Loss mask (batch_size, seq_len)")
    sample_ids: tuple[str, ...] = Field(description="Sample IDs in batch")
    bucket_id: int = Field(description="Bucket ID")
    max_len: int = Field(description="Padded sequence length")
    index: int = Field(description="Batch index within epoch")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for np.savez."""
        return {
            "input_ids": self.input_ids,
            "loss_mask": self.loss_mask,
            "sample_ids": list(self.sample_ids),
            "bucket_id": self.bucket_id,
            "max_len": self.max_len,
            "index": self.index,
        }

    @classmethod
    def from_npz(cls, data: dict[str, Any]) -> CollatedBatch:
        """Load from np.load() result."""
        return cls(
            input_ids=data["input_ids"],
            loss_mask=data["loss_mask"],
            sample_ids=tuple(data["sample_ids"]),
            bucket_id=int(data["bucket_id"]),
            max_len=int(data["max_len"]),
            index=int(data["index"]),
        )


def default_collate(
    samples: list[dict[str, list[int]]],
    max_len: int,
    pad_id: int = 0,
) -> dict[str, np.ndarray]:
    """
    Default collate function.

    Expects samples with 'input_ids' and 'loss_mask' keys.
    Pads to max_len and stacks into arrays.
    """

    def pad(seq: list[int], pad_val: int) -> list[int]:
        if len(seq) >= max_len:
            return seq[:max_len]
        return seq + [pad_val] * (max_len - len(seq))

    input_ids = np.array(
        [pad(s["input_ids"], pad_id) for s in samples],
        dtype=np.int32,
    )
    loss_mask = np.array(
        [pad(s.get("loss_mask", [1] * len(s["input_ids"])), 0) for s in samples],
        dtype=np.int32,
    )

    return {"input_ids": input_ids, "loss_mask": loss_mask}


class BatchWriter:
    """
    Write BatchPlan epochs to NPZ files.

    Usage:
        writer = BatchWriter(plan, samples, output_dir)
        writer.write_all()  # Write all epochs

        # Or write specific epoch
        writer.write_epoch(0)
    """

    def __init__(
        self,
        plan: BatchPlan,
        samples: dict[str, dict[str, list[int]]],
        output_dir: str | Path,
        collate_fn: Callable[[list[dict], int, int], dict[str, np.ndarray]] | None = None,
        pad_id: int = 0,
    ):
        """
        Initialize BatchWriter.

        Args:
            plan: BatchPlan to materialize
            samples: Dict mapping sample_id -> {"input_ids": [...], "loss_mask": [...]}
            output_dir: Directory to write batch files
            collate_fn: Optional custom collate function(samples, max_len, pad_id) -> arrays
            pad_id: Padding token ID
        """
        self.plan = plan
        self.samples = samples
        self.output_dir = Path(output_dir)
        self.collate_fn = collate_fn or default_collate
        self.pad_id = pad_id

    def write_all(self) -> list[Path]:
        """
        Write all epochs to disk.

        Returns list of all written file paths.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save the plan itself for later loading
        save_batch_plan(self.plan, self.output_dir / "plan")

        all_files = []
        for epoch in range(self.plan.num_epochs):
            files = self.write_epoch(epoch)
            all_files.extend(files)

        # Write manifest
        manifest = {
            "num_epochs": self.plan.num_epochs,
            "total_batches": len(all_files),
            "fingerprint": self.plan.fingerprint,
        }
        with open(self.output_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        return all_files

    def write_epoch(self, epoch: int) -> list[Path]:
        """
        Write a single epoch to disk.

        Returns list of written file paths.
        """
        epoch_dir = self.output_dir / f"epoch_{epoch}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        files = []
        for mb in self.plan.iter_epoch(epoch):
            path = self._write_microbatch(epoch_dir, mb)
            files.append(path)

        return files

    def _write_microbatch(self, epoch_dir: Path, mb: MicrobatchSpec) -> Path:
        """Write a single microbatch to NPZ file."""
        # Gather samples
        batch_samples = [self.samples[sid] for sid in mb.samples]

        # Collate
        arrays = self.collate_fn(batch_samples, mb.max_len, self.pad_id)

        # Add metadata
        arrays["sample_ids"] = np.array(list(mb.samples), dtype=object)
        arrays["bucket_id"] = mb.bucket_id
        arrays["max_len"] = mb.max_len
        arrays["index"] = mb.index

        # Write
        path = epoch_dir / f"batch_{mb.index:06d}.npz"
        np.savez(path, **arrays)

        return path


class BatchReader:
    """
    Read NPZ batch files written by BatchWriter.

    Provides same iteration interface as BatchPlan.iter_epoch().

    Usage:
        reader = BatchReader("./batches")
        for batch in reader.iter_epoch(0):
            model(batch["input_ids"])
    """

    def __init__(self, batch_dir: str | Path):
        """
        Initialize BatchReader.

        Args:
            batch_dir: Directory containing batch files (created by BatchWriter)
        """
        self.batch_dir = Path(batch_dir)

        # Load plan
        plan_dir = self.batch_dir / "plan"
        if plan_dir.exists():
            self.plan = load_batch_plan(plan_dir)
        else:
            self.plan = None

        # Load manifest
        manifest_path = self.batch_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                self.manifest = json.load(f)
        else:
            self.manifest = None

    @property
    def num_epochs(self) -> int:
        """Number of epochs available."""
        if self.plan:
            return self.plan.num_epochs
        if self.manifest:
            return self.manifest["num_epochs"]
        # Count epoch directories
        return len(list(self.batch_dir.glob("epoch_*")))

    @property
    def fingerprint(self) -> str | None:
        """Plan fingerprint for verification."""
        if self.plan:
            return self.plan.fingerprint
        if self.manifest:
            return self.manifest.get("fingerprint")
        return None

    def iter_epoch(self, epoch: int) -> Iterator[dict[str, Any]]:
        """
        Iterate over batches in an epoch.

        Yields dicts with numpy arrays and metadata.
        """
        epoch_dir = self.batch_dir / f"epoch_{epoch}"
        if not epoch_dir.exists():
            raise FileNotFoundError(f"Epoch directory not found: {epoch_dir}")

        # Get batch files in order
        batch_files = sorted(epoch_dir.glob("batch_*.npz"))

        for path in batch_files:
            data = dict(np.load(path, allow_pickle=True))
            yield data

    def iter_epoch_specs(self, epoch: int) -> Iterator[tuple[MicrobatchSpec, dict[str, Any]]]:
        """
        Iterate over batches with their MicrobatchSpec.

        Yields (MicrobatchSpec, batch_data) tuples.
        Requires that the plan was saved with the batches.
        """
        if self.plan is None:
            raise ValueError("Plan not found. Cannot iterate with specs.")

        epoch_dir = self.batch_dir / f"epoch_{epoch}"
        if not epoch_dir.exists():
            raise FileNotFoundError(f"Epoch directory not found: {epoch_dir}")

        for mb in self.plan.iter_epoch(epoch):
            path = epoch_dir / f"batch_{mb.index:06d}.npz"
            data = dict(np.load(path, allow_pickle=True))
            yield mb, data

    def get_batch(self, epoch: int, index: int) -> dict[str, Any]:
        """Load a specific batch by epoch and index."""
        path = self.batch_dir / f"epoch_{epoch}" / f"batch_{index:06d}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Batch file not found: {path}")
        return dict(np.load(path, allow_pickle=True))

    def verify_fingerprint(self, expected: str) -> bool:
        """Verify that the batch fingerprint matches expected."""
        return self.fingerprint == expected
