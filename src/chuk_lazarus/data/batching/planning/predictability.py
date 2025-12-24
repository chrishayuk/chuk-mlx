"""
Predictability mode for reproducible batching.

Provides:
- PadPolicy: Control padding behavior (pad to bucket vs pad to max in batch)
- BatchingConfig: Complete batching configuration
- compute_batch_fingerprint: Hash batch contents for replay verification

Design principles:
- Deterministic: Same inputs produce identical batches
- Verifiable: Fingerprints allow CI/CD validation
- Configurable: Trade throughput for predictability when needed
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from ..core.sampler import BatchSpec


class PadPolicy(str, Enum):
    """
    Padding strategy for batches.

    PAD_TO_BUCKET: Pad all sequences to bucket max length (predictable shapes)
    PAD_TO_MAX_IN_BATCH: Pad to max length in batch (better throughput)
    """

    PAD_TO_BUCKET = "pad_to_bucket"
    PAD_TO_MAX_IN_BATCH = "pad_to_max"


class BatchingMode(str, Enum):
    """
    Batching mode preset.

    PREDICTABLE: Fixed shapes, reproducible across runs
    THROUGHPUT: Dynamic shapes, maximizes efficiency
    """

    PREDICTABLE = "predictable"
    THROUGHPUT = "throughput"


class BatchingConfig(BaseModel):
    """
    Complete configuration for batching behavior.

    Controls bucketing, padding, and reproducibility settings.
    Use `predictable()` or `throughput()` factory methods for presets.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Mode preset
    mode: BatchingMode = Field(
        default=BatchingMode.THROUGHPUT,
        description="Batching mode preset",
    )

    # Padding
    pad_policy: PadPolicy = Field(
        default=PadPolicy.PAD_TO_MAX_IN_BATCH,
        description="How to pad sequences within batches",
    )

    # Token budget
    token_budget: int = Field(
        default=4096,
        gt=0,
        description="Maximum tokens per batch",
    )

    # Bucket configuration
    bucket_edges: tuple[int, ...] = Field(
        default=(128, 256, 512, 1024),
        description="Bucket edge lengths",
    )
    overflow_max: int = Field(
        default=2048,
        gt=0,
        description="Maximum sequence length (overflow bucket)",
    )

    # Reproducibility
    seed: int = Field(
        default=42,
        description="Base random seed for shuffling",
    )
    drop_last: bool = Field(
        default=False,
        description="Drop last incomplete batch per bucket",
    )
    interleave_buckets: bool = Field(
        default=True,
        description="Interleave batches from different buckets",
    )

    # Predictability mode specifics
    bucket_max_lengths: dict[int, int] | None = Field(
        default=None,
        description="Override max length per bucket (predictable mode)",
    )

    @classmethod
    def predictable(
        cls,
        token_budget: int = 4096,
        bucket_edges: tuple[int, ...] = (128, 256, 512, 1024),
        overflow_max: int = 2048,
        seed: int = 42,
    ) -> BatchingConfig:
        """
        Create predictable mode configuration.

        Uses fixed bucket shapes for reproducibility.
        """
        return cls(
            mode=BatchingMode.PREDICTABLE,
            pad_policy=PadPolicy.PAD_TO_BUCKET,
            token_budget=token_budget,
            bucket_edges=bucket_edges,
            overflow_max=overflow_max,
            seed=seed,
            drop_last=True,  # Predictable mode drops incomplete batches
            interleave_buckets=False,  # Sequential for reproducibility
        )

    @classmethod
    def throughput(
        cls,
        token_budget: int = 4096,
        bucket_edges: tuple[int, ...] = (128, 256, 512, 1024),
        overflow_max: int = 2048,
        seed: int = 42,
    ) -> BatchingConfig:
        """
        Create throughput mode configuration.

        Uses dynamic padding for maximum efficiency.
        """
        return cls(
            mode=BatchingMode.THROUGHPUT,
            pad_policy=PadPolicy.PAD_TO_MAX_IN_BATCH,
            token_budget=token_budget,
            bucket_edges=bucket_edges,
            overflow_max=overflow_max,
            seed=seed,
            drop_last=False,
            interleave_buckets=True,
        )

    @property
    def is_predictable(self) -> bool:
        """Check if using predictable mode."""
        return self.mode == BatchingMode.PREDICTABLE

    def get_pad_length(self, bucket_id: int, max_in_batch: int) -> int:
        """
        Get the padding target length for a batch.

        Args:
            bucket_id: The bucket ID
            max_in_batch: Maximum sequence length in the batch

        Returns:
            Target length to pad sequences to
        """
        if self.pad_policy == PadPolicy.PAD_TO_BUCKET:
            # Use bucket max or override
            if self.bucket_max_lengths and bucket_id in self.bucket_max_lengths:
                return self.bucket_max_lengths[bucket_id]
            # Calculate from edges
            if bucket_id < len(self.bucket_edges):
                return self.bucket_edges[bucket_id]
            return self.overflow_max
        else:
            # Pad to max in batch (throughput mode)
            return max_in_batch


# =============================================================================
# Batch Fingerprinting
# =============================================================================


class BatchFingerprint(BaseModel):
    """
    Fingerprint of batch contents for reproducibility verification.

    Captures the essential properties of a batch sequence for
    comparing across runs without storing full data.
    """

    model_config = ConfigDict(frozen=True)

    # Primary fingerprint
    fingerprint: str = Field(description="Short fingerprint (16 chars)")
    full_hash: str = Field(description="Full SHA-256 hash")

    # Summary stats
    num_batches: int = Field(ge=0, description="Number of batches fingerprinted")
    total_samples: int = Field(ge=0, description="Total samples across batches")
    total_tokens: int = Field(ge=0, description="Total tokens (before padding)")

    # Configuration hash
    config_hash: str = Field(description="Hash of batching configuration")

    # Version
    version: int = Field(default=1, description="Fingerprint format version")

    def matches(self, other: BatchFingerprint) -> bool:
        """Check if fingerprints match."""
        return self.full_hash == other.full_hash

    def matches_config(self, other: BatchFingerprint) -> bool:
        """Check if configurations match (ignoring data)."""
        return self.config_hash == other.config_hash


def _hash_batch_spec(batch: BatchSpec) -> str:
    """Create a deterministic hash of a batch spec."""
    data = {
        "sample_ids": sorted(batch.sample_ids),  # Sort for determinism
        "bucket_id": int(batch.bucket_id),
        "max_length": batch.max_length,
        "token_count": batch.token_count,
    }
    serialized = json.dumps(data, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()


def compute_batch_fingerprint(
    batches: list[BatchSpec],
    config: BatchingConfig | None = None,
    n_batches: int | None = None,
) -> BatchFingerprint:
    """
    Compute a fingerprint for a sequence of batches.

    Used to verify that batching is reproducible across runs.

    Args:
        batches: List of batch specifications
        config: Optional batching configuration to include in hash
        n_batches: Number of batches to fingerprint (None = all)

    Returns:
        BatchFingerprint capturing batch sequence identity
    """
    if n_batches is not None:
        batches = batches[:n_batches]

    # Hash each batch
    hasher = hashlib.sha256()
    total_samples = 0
    total_tokens = 0

    for batch in batches:
        batch_hash = _hash_batch_spec(batch)
        hasher.update(batch_hash.encode())
        total_samples += batch.batch_size
        total_tokens += batch.token_count

    # Hash config if provided
    if config is not None:
        config_data = config.model_dump()
        config_serialized = json.dumps(config_data, sort_keys=True, default=str)
        config_hash = hashlib.sha256(config_serialized.encode()).hexdigest()[:16]
    else:
        config_hash = "none"

    full_hash = hasher.hexdigest()

    return BatchFingerprint(
        fingerprint=full_hash[:16],
        full_hash=full_hash,
        num_batches=len(batches),
        total_samples=total_samples,
        total_tokens=total_tokens,
        config_hash=config_hash,
    )


async def compute_batch_fingerprint_async(
    batches,  # AsyncIterator[BatchSpec]
    config: BatchingConfig | None = None,
    n_batches: int | None = None,
) -> BatchFingerprint:
    """
    Async version of compute_batch_fingerprint.

    Args:
        batches: Async iterator of batch specifications
        config: Optional batching configuration
        n_batches: Number of batches to fingerprint (None = all)

    Returns:
        BatchFingerprint capturing batch sequence identity
    """
    hasher = hashlib.sha256()
    total_samples = 0
    total_tokens = 0
    count = 0

    async for batch in batches:
        if n_batches is not None and count >= n_batches:
            break

        batch_hash = _hash_batch_spec(batch)
        hasher.update(batch_hash.encode())
        total_samples += batch.batch_size
        total_tokens += batch.token_count
        count += 1

    # Hash config if provided
    if config is not None:
        config_data = config.model_dump()
        config_serialized = json.dumps(config_data, sort_keys=True, default=str)
        config_hash = hashlib.sha256(config_serialized.encode()).hexdigest()[:16]
    else:
        config_hash = "none"

    full_hash = hasher.hexdigest()

    return BatchFingerprint(
        fingerprint=full_hash[:16],
        full_hash=full_hash,
        num_batches=count,
        total_samples=total_samples,
        total_tokens=total_tokens,
        config_hash=config_hash,
    )


def verify_batch_fingerprint(
    batches: list[BatchSpec],
    expected: BatchFingerprint | str,
    config: BatchingConfig | None = None,
) -> tuple[bool, str | None]:
    """
    Verify batches match an expected fingerprint.

    Args:
        batches: List of batch specifications
        expected: Expected fingerprint (or fingerprint string)
        config: Optional batching configuration

    Returns:
        Tuple of (matches, error_message)
    """
    actual = compute_batch_fingerprint(
        batches,
        config=config,
        n_batches=expected.num_batches if isinstance(expected, BatchFingerprint) else None,
    )

    if isinstance(expected, str):
        # Short fingerprint comparison
        if actual.fingerprint == expected or actual.full_hash.startswith(expected):
            return True, None
        return False, f"Fingerprint mismatch: expected {expected}, got {actual.fingerprint}"

    # Full fingerprint comparison
    if actual.matches(expected):
        return True, None

    # Detailed error message
    issues = []
    if actual.num_batches != expected.num_batches:
        issues.append(f"batch count: {expected.num_batches} vs {actual.num_batches}")
    if actual.total_samples != expected.total_samples:
        issues.append(f"sample count: {expected.total_samples} vs {actual.total_samples}")
    if actual.total_tokens != expected.total_tokens:
        issues.append(f"token count: {expected.total_tokens} vs {actual.total_tokens}")

    return False, f"Fingerprint mismatch ({', '.join(issues) or 'content differs'})"


# =============================================================================
# I/O
# =============================================================================


def save_fingerprint(fingerprint: BatchFingerprint, path: str) -> None:
    """Save batch fingerprint to file."""
    from pathlib import Path

    Path(path).write_text(json.dumps(fingerprint.model_dump(), indent=2))


def load_fingerprint(path: str) -> BatchFingerprint:
    """Load batch fingerprint from file."""
    from pathlib import Path

    data = json.loads(Path(path).read_text())
    return BatchFingerprint(**data)


async def save_fingerprint_async(fingerprint: BatchFingerprint, path: str) -> None:
    """Async: Save batch fingerprint to file."""

    import aiofiles

    async with aiofiles.open(path, "w") as f:
        await f.write(json.dumps(fingerprint.model_dump(), indent=2))


async def load_fingerprint_async(path: str) -> BatchFingerprint:
    """Async: Load batch fingerprint from file."""
    import aiofiles

    async with aiofiles.open(path) as f:
        content = await f.read()
    data = json.loads(content)
    return BatchFingerprint(**data)
