"""
BatchPlan artifacts for reproducible, distributed training.

A BatchPlan is a precomputed schedule of microbatches that can be:
- Saved to disk for reproducibility
- Shared across ranks for distributed training
- Resumed from checkpoints

Design principles:
- Deterministic: Same plan produces identical training order
- Portable: Plans are self-contained with all metadata
- Shardable: Easy to divide across distributed workers
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator, Iterator
from datetime import datetime, timezone
from pathlib import Path

import aiofiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .packing import PackingConfig
from .predictability import BatchingConfig, PadPolicy


class MicrobatchSpec(BaseModel):
    """
    Specification for a single microbatch.

    Contains all information needed to construct the batch at runtime.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Sample IDs in this microbatch
    samples: tuple[str, ...] = Field(
        description="Sample IDs in this microbatch",
    )

    # Packing info (if packing enabled)
    packs: tuple[tuple[str, ...], ...] | None = Field(
        default=None,
        description="Packed sample groups (if packing enabled)",
    )

    # Bucket info
    bucket_id: int = Field(
        ge=0,
        description="Bucket ID for this microbatch",
    )
    max_len: int = Field(
        gt=0,
        description="Maximum sequence length for padding",
    )

    # Index
    index: int = Field(
        ge=0,
        description="Microbatch index within epoch",
    )

    @field_validator("samples", "packs", mode="before")
    @classmethod
    def convert_to_tuple(cls, v):
        if v is None:
            return v
        if isinstance(v, (list, tuple)):
            # Handle nested lists for packs
            if v and isinstance(v[0], (list, tuple)):
                return tuple(tuple(inner) for inner in v)
            return tuple(v)
        raise ValueError(f"Expected list or tuple, got {type(v)}")

    @property
    def batch_size(self) -> int:
        """Number of samples in this microbatch."""
        return len(self.samples)

    @property
    def num_packs(self) -> int:
        """Number of packed sequences (or batch_size if not packing)."""
        if self.packs is None:
            return self.batch_size
        return len(self.packs)


class EpochPlan(BaseModel):
    """
    Plan for a single epoch.

    Contains ordered list of microbatches with statistics.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    epoch: int = Field(
        ge=0,
        description="Epoch number",
    )
    microbatches: tuple[MicrobatchSpec, ...] = Field(
        description="Ordered list of microbatches",
    )
    seed: int = Field(
        description="Random seed used for this epoch",
    )

    # Stats
    total_samples: int = Field(
        ge=0,
        description="Total samples in epoch",
    )
    total_tokens: int = Field(
        ge=0,
        description="Total tokens in epoch (estimated)",
    )

    @field_validator("microbatches", mode="before")
    @classmethod
    def convert_microbatches(cls, v):
        if isinstance(v, (list, tuple)):
            return tuple(mb if isinstance(mb, MicrobatchSpec) else MicrobatchSpec(**mb) for mb in v)
        raise ValueError(f"Expected list or tuple, got {type(v)}")

    @property
    def num_microbatches(self) -> int:
        """Number of microbatches in epoch."""
        return len(self.microbatches)

    def __iter__(self) -> Iterator[MicrobatchSpec]:
        """Iterate over microbatches."""
        return iter(self.microbatches)

    def __getitem__(self, idx: int) -> MicrobatchSpec:
        """Get microbatch by index."""
        return self.microbatches[idx]


class BatchPlanMeta(BaseModel):
    """
    Metadata for a batch plan.

    Contains all configuration needed to validate and reproduce the plan.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Fingerprints for cache invalidation
    dataset_hash: str = Field(
        description="Hash of dataset content",
    )
    tokenizer_hash: str = Field(
        description="Hash of tokenizer configuration",
    )

    # Batching configuration
    bucket_edges: tuple[int, ...] = Field(
        description="Bucket edge lengths",
    )
    overflow_max: int = Field(
        description="Maximum sequence length",
    )
    token_budget: int = Field(
        description="Token budget per microbatch",
    )
    pad_policy: PadPolicy = Field(
        description="Padding policy",
    )

    # Packing configuration (optional)
    pack_config: PackingConfig | None = Field(
        default=None,
        description="Packing configuration (if packing enabled)",
    )

    # Plan metadata
    num_epochs: int = Field(
        ge=1,
        description="Number of epochs in plan",
    )
    base_seed: int = Field(
        description="Base random seed",
    )

    # Timestamps
    created_at: str = Field(
        description="ISO timestamp of plan creation",
    )
    version: str = Field(
        default="1.0",
        description="Plan format version",
    )

    @field_validator("bucket_edges", mode="before")
    @classmethod
    def convert_edges(cls, v):
        if isinstance(v, (list, tuple)):
            return tuple(v)
        raise ValueError(f"Expected list or tuple, got {type(v)}")

    @classmethod
    def create(
        cls,
        dataset_hash: str,
        tokenizer_hash: str,
        batching_config: BatchingConfig,
        num_epochs: int,
        pack_config: PackingConfig | None = None,
    ) -> BatchPlanMeta:
        """Create metadata from configuration."""
        return cls(
            dataset_hash=dataset_hash,
            tokenizer_hash=tokenizer_hash,
            bucket_edges=batching_config.bucket_edges,
            overflow_max=batching_config.overflow_max,
            token_budget=batching_config.token_budget,
            pad_policy=batching_config.pad_policy,
            pack_config=pack_config,
            num_epochs=num_epochs,
            base_seed=batching_config.seed,
            created_at=datetime.now(timezone.utc).isoformat(),
        )


class BatchPlan(BaseModel):
    """
    Complete batch plan for training.

    A BatchPlan contains:
    - Metadata for validation and reproducibility
    - Epoch plans with microbatch specifications
    - Support for sharding across distributed workers
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    meta: BatchPlanMeta = Field(
        description="Plan metadata",
    )
    epochs: dict[int, EpochPlan] = Field(
        default_factory=dict,
        description="Epoch plans by epoch number",
    )

    # Computed fingerprint
    _fingerprint: str | None = None

    @property
    def num_epochs(self) -> int:
        """Number of epochs in plan."""
        return len(self.epochs)

    @property
    def total_microbatches(self) -> int:
        """Total microbatches across all epochs."""
        return sum(ep.num_microbatches for ep in self.epochs.values())

    @property
    def fingerprint(self) -> str:
        """Compute fingerprint of plan contents."""
        if self._fingerprint is None:
            hasher = hashlib.sha256()
            hasher.update(self.meta.dataset_hash.encode())
            hasher.update(self.meta.tokenizer_hash.encode())
            hasher.update(str(self.meta.base_seed).encode())

            for epoch_num in sorted(self.epochs.keys()):
                epoch = self.epochs[epoch_num]
                for mb in epoch.microbatches:
                    hasher.update(",".join(mb.samples).encode())

            self._fingerprint = hasher.hexdigest()[:16]

        return self._fingerprint

    def get_epoch(self, epoch: int) -> EpochPlan:
        """Get plan for a specific epoch."""
        if epoch not in self.epochs:
            raise KeyError(f"Epoch {epoch} not in plan (available: {list(self.epochs.keys())})")
        return self.epochs[epoch]

    def add_epoch(self, epoch_plan: EpochPlan) -> None:
        """Add an epoch plan."""
        self.epochs[epoch_plan.epoch] = epoch_plan
        self._fingerprint = None  # Invalidate cache

    # =========================================================================
    # Sharding for Distributed Training
    # =========================================================================

    def shard(self, rank: int, world_size: int) -> BatchPlan:
        """
        Create a sharded view of this plan for distributed training.

        Shards by microbatch index: rank i gets microbatches i, i+world_size, ...

        Args:
            rank: Current worker rank (0-indexed)
            world_size: Total number of workers

        Returns:
            New BatchPlan containing only this rank's microbatches
        """
        if rank < 0 or rank >= world_size:
            raise ValueError(f"Invalid rank {rank} for world_size {world_size}")

        sharded_epochs = {}
        for epoch_num, epoch in self.epochs.items():
            sharded_mbs = [mb for i, mb in enumerate(epoch.microbatches) if i % world_size == rank]
            if sharded_mbs:
                # Reindex microbatches
                reindexed = [
                    MicrobatchSpec(
                        samples=mb.samples,
                        packs=mb.packs,
                        bucket_id=mb.bucket_id,
                        max_len=mb.max_len,
                        index=i,
                    )
                    for i, mb in enumerate(sharded_mbs)
                ]
                sharded_epochs[epoch_num] = EpochPlan(
                    epoch=epoch_num,
                    microbatches=tuple(reindexed),
                    seed=epoch.seed,
                    total_samples=sum(mb.batch_size for mb in reindexed),
                    total_tokens=epoch.total_tokens // world_size,  # Approximate
                )

        return BatchPlan(meta=self.meta, epochs=sharded_epochs)

    def iter_epoch(self, epoch: int) -> Iterator[MicrobatchSpec]:
        """Iterate over microbatches for an epoch."""
        return iter(self.get_epoch(epoch))

    async def iter_epoch_async(self, epoch: int) -> AsyncIterator[MicrobatchSpec]:
        """Async iterate over microbatches for an epoch."""
        for mb in self.get_epoch(epoch):
            yield mb

    # =========================================================================
    # Resume Support
    # =========================================================================

    def iter_from(
        self,
        epoch: int,
        microbatch_idx: int,
    ) -> Iterator[tuple[int, int, MicrobatchSpec]]:
        """
        Iterate from a specific position (for resuming).

        Yields:
            Tuples of (epoch, microbatch_idx, microbatch_spec)
        """
        for ep_num in sorted(self.epochs.keys()):
            if ep_num < epoch:
                continue

            epoch_plan = self.epochs[ep_num]
            start_idx = microbatch_idx if ep_num == epoch else 0

            for mb in epoch_plan.microbatches[start_idx:]:
                yield ep_num, mb.index, mb


# =============================================================================
# I/O
# =============================================================================


def save_batch_plan(plan: BatchPlan, path: str | Path) -> None:
    """
    Save batch plan to directory.

    Creates structure:
        path/
            meta.json
            epoch_0.jsonl
            epoch_1.jsonl
            ...
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Save metadata
    meta_path = path / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(plan.meta.model_dump(), f, indent=2, default=str)

    # Save each epoch
    for epoch_num, epoch in plan.epochs.items():
        epoch_path = path / f"epoch_{epoch_num}.jsonl"
        with open(epoch_path, "w") as f:
            # Write epoch header
            header = {
                "epoch": epoch.epoch,
                "seed": epoch.seed,
                "total_samples": epoch.total_samples,
                "total_tokens": epoch.total_tokens,
                "num_microbatches": epoch.num_microbatches,
            }
            f.write(json.dumps(header) + "\n")

            # Write microbatches
            for mb in epoch.microbatches:
                f.write(json.dumps(mb.model_dump()) + "\n")


def load_batch_plan(path: str | Path) -> BatchPlan:
    """Load batch plan from directory."""
    path = Path(path)

    # Load metadata
    meta_path = path / "meta.json"
    with open(meta_path) as f:
        meta_data = json.load(f)
    meta = BatchPlanMeta(**meta_data)

    # Find and load epoch files
    epochs = {}
    for epoch_file in sorted(path.glob("epoch_*.jsonl")):
        epoch_num = int(epoch_file.stem.split("_")[1])

        with open(epoch_file) as f:
            lines = f.readlines()

        # First line is header
        header = json.loads(lines[0])

        # Rest are microbatches
        microbatches = [MicrobatchSpec(**json.loads(line)) for line in lines[1:] if line.strip()]

        epochs[epoch_num] = EpochPlan(
            epoch=header["epoch"],
            microbatches=tuple(microbatches),
            seed=header["seed"],
            total_samples=header["total_samples"],
            total_tokens=header["total_tokens"],
        )

    return BatchPlan(meta=meta, epochs=epochs)


async def save_batch_plan_async(plan: BatchPlan, path: str | Path) -> None:
    """Async: Save batch plan to directory."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Save metadata
    meta_path = path / "meta.json"
    async with aiofiles.open(meta_path, "w") as f:
        await f.write(json.dumps(plan.meta.model_dump(), indent=2, default=str))

    # Save each epoch
    for epoch_num, epoch in plan.epochs.items():
        epoch_path = path / f"epoch_{epoch_num}.jsonl"
        async with aiofiles.open(epoch_path, "w") as f:
            # Write epoch header
            header = {
                "epoch": epoch.epoch,
                "seed": epoch.seed,
                "total_samples": epoch.total_samples,
                "total_tokens": epoch.total_tokens,
                "num_microbatches": epoch.num_microbatches,
            }
            await f.write(json.dumps(header) + "\n")

            # Write microbatches
            for mb in epoch.microbatches:
                await f.write(json.dumps(mb.model_dump()) + "\n")


async def load_batch_plan_async(path: str | Path) -> BatchPlan:
    """Async: Load batch plan from directory."""
    path = Path(path)

    # Load metadata
    meta_path = path / "meta.json"
    async with aiofiles.open(meta_path) as f:
        content = await f.read()
        meta_data = json.loads(content)
    meta = BatchPlanMeta(**meta_data)

    # Find and load epoch files
    epochs = {}
    for epoch_file in sorted(path.glob("epoch_*.jsonl")):
        epoch_num = int(epoch_file.stem.split("_")[1])

        async with aiofiles.open(epoch_file) as f:
            content = await f.read()
            lines = content.strip().split("\n")

        # First line is header
        header = json.loads(lines[0])

        # Rest are microbatches
        microbatches = [MicrobatchSpec(**json.loads(line)) for line in lines[1:] if line.strip()]

        epochs[epoch_num] = EpochPlan(
            epoch=header["epoch"],
            microbatches=tuple(microbatches),
            seed=header["seed"],
            total_samples=header["total_samples"],
            total_tokens=header["total_tokens"],
        )

    return BatchPlan(meta=meta, epochs=epochs)


# =============================================================================
# Plan Builder
# =============================================================================


class BatchPlanBuilder:
    """
    Builder for constructing batch plans from a sampler.

    Usage:
        builder = BatchPlanBuilder(
            sampler=sampler,
            batching_config=config,
            dataset_hash="abc123",
            tokenizer_hash="def456",
        )
        plan = await builder.build(num_epochs=3)
    """

    def __init__(
        self,
        lengths: dict[str, int],
        batching_config: BatchingConfig,
        dataset_hash: str,
        tokenizer_hash: str,
        pack_config: PackingConfig | None = None,
    ):
        self.lengths = lengths
        self.batching_config = batching_config
        self.dataset_hash = dataset_hash
        self.tokenizer_hash = tokenizer_hash
        self.pack_config = pack_config

    async def build(self, num_epochs: int) -> BatchPlan:
        """Build a complete batch plan."""
        from ..core.buckets import BucketSpec
        from ..core.sampler import TokenBudgetBatchSampler

        # Create sampler
        bucket_spec = BucketSpec(
            edges=self.batching_config.bucket_edges,
            overflow_max=self.batching_config.overflow_max,
        )
        sampler = TokenBudgetBatchSampler(
            lengths=self.lengths,
            bucket_spec=bucket_spec,
            token_budget=self.batching_config.token_budget,
            seed=self.batching_config.seed,
            drop_last=self.batching_config.drop_last,
        )

        # Create metadata
        meta = BatchPlanMeta.create(
            dataset_hash=self.dataset_hash,
            tokenizer_hash=self.tokenizer_hash,
            batching_config=self.batching_config,
            num_epochs=num_epochs,
            pack_config=self.pack_config,
        )

        plan = BatchPlan(meta=meta)

        # Build each epoch
        for epoch in range(num_epochs):
            microbatches = []
            total_samples = 0
            total_tokens = 0

            idx = 0
            async for batch_spec in sampler.iter_epoch(
                epoch=epoch,
                interleave_buckets=self.batching_config.interleave_buckets,
            ):
                mb = MicrobatchSpec(
                    samples=batch_spec.sample_ids,
                    bucket_id=int(batch_spec.bucket_id),
                    max_len=batch_spec.max_length,
                    index=idx,
                )
                microbatches.append(mb)
                total_samples += batch_spec.batch_size
                total_tokens += batch_spec.token_count
                idx += 1

            epoch_plan = EpochPlan(
                epoch=epoch,
                microbatches=tuple(microbatches),
                seed=self.batching_config.seed + epoch,
                total_samples=total_samples,
                total_tokens=total_tokens,
            )
            plan.add_epoch(epoch_plan)

        return plan
