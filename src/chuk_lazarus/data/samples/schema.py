"""
Canonical sample schema for training.

This module defines the standard sample format for the batching and training
pipeline. Pydantic-native with no magic strings.

Design principles:
- Pydantic-native: All structures use BaseModel for validation
- No magic strings: Enums and constants for type safety
- No dictionary goop: Structured fields with clear semantics
- Training-aware: Metadata supports curriculum, reproducibility, and tracking
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# =============================================================================
# Enums - Eliminate magic strings
# =============================================================================


class SampleType(str, Enum):
    """Type of training sample."""

    SFT = "sft"  # Supervised fine-tuning
    DPO = "dpo"  # Direct preference optimization
    PRETRAIN = "pretrain"  # Pretraining (no masking)
    RL = "rl"  # Reinforcement learning (with rewards)
    GRPO = "grpo"  # Group relative policy optimization
    PPO = "ppo"  # Proximal policy optimization


class DatasetSource(str, Enum):
    """Source of the dataset."""

    LOCAL = "local"  # Local file
    HUGGINGFACE = "huggingface"  # HuggingFace Hub
    GYM = "gym"  # Gym environment (online)
    SYNTHETIC = "synthetic"  # Generated data
    REMOTE = "remote"  # Remote URL


class DifficultyLevel(str, Enum):
    """Difficulty level for curriculum learning."""

    TRIVIAL = "trivial"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


# =============================================================================
# Validation Errors
# =============================================================================


class SampleValidationError(ValueError):
    """Raised when sample validation fails."""

    pass


# =============================================================================
# Metadata Models
# =============================================================================


class SampleMeta(BaseModel):
    """
    Metadata for a training sample.

    Supports tracking, curriculum learning, and reproducibility.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Identity
    sample_id: str = Field(description="Unique identifier for this sample")
    dataset_id: str = Field(description="Identifier for the source dataset")

    # Optional tracking
    episode_id: str | None = Field(
        default=None,
        description="Episode ID for RL/gym samples",
    )
    source: DatasetSource = Field(
        default=DatasetSource.LOCAL,
        description="Where this sample came from",
    )

    # Curriculum learning
    difficulty: DifficultyLevel | None = Field(
        default=None,
        description="Difficulty level for curriculum ordering",
    )
    difficulty_score: Annotated[float, Field(ge=0.0, le=1.0)] | None = Field(
        default=None,
        description="Numeric difficulty score (0.0 = easiest, 1.0 = hardest)",
    )

    # RL-specific
    reward: float | None = Field(
        default=None,
        description="Reward signal for RL samples",
    )
    success: bool | None = Field(
        default=None,
        description="Whether the episode succeeded",
    )

    # Provenance
    original_index: int | None = Field(
        default=None,
        description="Original index in source dataset",
    )
    split: str | None = Field(
        default=None,
        description="Dataset split (train, val, test)",
    )

    @classmethod
    def create(
        cls,
        sample_id: str,
        dataset_id: str,
        **kwargs,
    ) -> SampleMeta:
        """Create metadata with required fields."""
        return cls(sample_id=sample_id, dataset_id=dataset_id, **kwargs)


# =============================================================================
# Core Sample Models
# =============================================================================


class Sample(BaseModel):
    """
    Canonical tokenized sample for training.

    This is the standard format consumed by the batching pipeline.
    All dataset loaders should produce samples conforming to this schema.

    Fields:
        input_ids: Token IDs for the input sequence
        loss_mask: Binary mask (1 = compute loss, 0 = ignore)
        segment_ids: Segment IDs for packed sequences (optional)
        meta: Sample metadata for tracking and curriculum
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Core fields
    input_ids: tuple[int, ...] = Field(
        description="Token IDs for the input sequence",
    )
    loss_mask: tuple[int, ...] = Field(
        description="Binary mask: 1 = compute loss, 0 = ignore (prompt masking)",
    )

    # Packing support (Phase 3)
    segment_ids: tuple[int, ...] | None = Field(
        default=None,
        description="Segment IDs for packed sequences (0, 0, 0, 1, 1, 2, ...)",
    )

    # Metadata
    meta: SampleMeta = Field(description="Sample metadata")

    # Sample type
    sample_type: SampleType = Field(
        default=SampleType.SFT,
        description="Type of training sample",
    )

    @field_validator("input_ids", "loss_mask", "segment_ids", mode="before")
    @classmethod
    def convert_to_tuple(cls, v):
        """Convert lists to tuples for immutability."""
        if v is None:
            return v
        if isinstance(v, (list, tuple)):
            return tuple(v)
        raise ValueError(f"Expected list or tuple, got {type(v)}")

    @model_validator(mode="after")
    def validate_lengths(self) -> Self:
        """Validate that all sequence fields have matching lengths."""
        input_len = len(self.input_ids)
        mask_len = len(self.loss_mask)

        if input_len != mask_len:
            raise SampleValidationError(
                f"input_ids length ({input_len}) != loss_mask length ({mask_len})"
            )

        if self.segment_ids is not None:
            seg_len = len(self.segment_ids)
            if seg_len != input_len:
                raise SampleValidationError(
                    f"segment_ids length ({seg_len}) != input_ids length ({input_len})"
                )

        return self

    @model_validator(mode="after")
    def validate_loss_mask_values(self) -> Self:
        """Validate loss_mask contains only 0 or 1."""
        invalid = [v for v in self.loss_mask if v not in (0, 1)]
        if invalid:
            raise SampleValidationError(
                f"loss_mask must contain only 0 or 1, found: {set(invalid)}"
            )
        return self

    @model_validator(mode="after")
    def validate_segment_ids_monotonic(self) -> Self:
        """Validate segment_ids are monotonically non-decreasing."""
        if self.segment_ids is None:
            return self

        for i in range(1, len(self.segment_ids)):
            if self.segment_ids[i] < self.segment_ids[i - 1]:
                raise SampleValidationError(
                    f"segment_ids must be monotonically non-decreasing, "
                    f"found {self.segment_ids[i]} < {self.segment_ids[i - 1]} at index {i}"
                )
        return self

    @property
    def length(self) -> int:
        """Sequence length."""
        return len(self.input_ids)

    @property
    def num_loss_tokens(self) -> int:
        """Number of tokens contributing to loss."""
        return sum(self.loss_mask)

    @property
    def num_segments(self) -> int:
        """Number of segments (for packed sequences)."""
        if self.segment_ids is None:
            return 1
        return max(self.segment_ids) + 1 if self.segment_ids else 0

    def to_lists(self) -> dict:
        """Convert to dict with lists (for serialization)."""
        return {
            "input_ids": list(self.input_ids),
            "loss_mask": list(self.loss_mask),
            "segment_ids": list(self.segment_ids) if self.segment_ids else None,
            "meta": self.meta.model_dump(),
            "sample_type": self.sample_type.value,
        }

    @classmethod
    def from_lists(cls, data: dict) -> Sample:
        """Create from dict with lists (for deserialization)."""
        meta_data = data.get("meta", {})
        if isinstance(meta_data, dict):
            meta = SampleMeta(**meta_data)
        else:
            meta = meta_data

        return cls(
            input_ids=data["input_ids"],
            loss_mask=data["loss_mask"],
            segment_ids=data.get("segment_ids"),
            meta=meta,
            sample_type=SampleType(data.get("sample_type", SampleType.SFT.value)),
        )


class PreferenceSample(BaseModel):
    """
    Preference pair for DPO training.

    Contains chosen and rejected samples with shared metadata.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Chosen response
    chosen_input_ids: tuple[int, ...] = Field(
        description="Token IDs for chosen (preferred) response",
    )
    chosen_loss_mask: tuple[int, ...] = Field(
        description="Loss mask for chosen response",
    )

    # Rejected response
    rejected_input_ids: tuple[int, ...] = Field(
        description="Token IDs for rejected response",
    )
    rejected_loss_mask: tuple[int, ...] = Field(
        description="Loss mask for rejected response",
    )

    # Shared prompt length (for log prob computation)
    prompt_length: int = Field(
        ge=0,
        description="Length of shared prompt tokens",
    )

    # Metadata
    meta: SampleMeta = Field(description="Sample metadata")

    @field_validator(
        "chosen_input_ids",
        "chosen_loss_mask",
        "rejected_input_ids",
        "rejected_loss_mask",
        mode="before",
    )
    @classmethod
    def convert_to_tuple(cls, v):
        """Convert lists to tuples for immutability."""
        if isinstance(v, (list, tuple)):
            return tuple(v)
        raise ValueError(f"Expected list or tuple, got {type(v)}")

    @model_validator(mode="after")
    def validate_lengths(self) -> Self:
        """Validate that chosen/rejected sequences are internally consistent."""
        if len(self.chosen_input_ids) != len(self.chosen_loss_mask):
            raise SampleValidationError(
                f"chosen_input_ids length ({len(self.chosen_input_ids)}) != "
                f"chosen_loss_mask length ({len(self.chosen_loss_mask)})"
            )
        if len(self.rejected_input_ids) != len(self.rejected_loss_mask):
            raise SampleValidationError(
                f"rejected_input_ids length ({len(self.rejected_input_ids)}) != "
                f"rejected_loss_mask length ({len(self.rejected_loss_mask)})"
            )
        return self

    @model_validator(mode="after")
    def validate_prompt_length(self) -> Self:
        """Validate prompt_length doesn't exceed sequence lengths."""
        if self.prompt_length > len(self.chosen_input_ids):
            raise SampleValidationError(
                f"prompt_length ({self.prompt_length}) > "
                f"chosen_input_ids length ({len(self.chosen_input_ids)})"
            )
        if self.prompt_length > len(self.rejected_input_ids):
            raise SampleValidationError(
                f"prompt_length ({self.prompt_length}) > "
                f"rejected_input_ids length ({len(self.rejected_input_ids)})"
            )
        return self

    @property
    def chosen_length(self) -> int:
        """Length of chosen sequence."""
        return len(self.chosen_input_ids)

    @property
    def rejected_length(self) -> int:
        """Length of rejected sequence."""
        return len(self.rejected_input_ids)

    @property
    def max_length(self) -> int:
        """Maximum of chosen and rejected lengths."""
        return max(self.chosen_length, self.rejected_length)

    def to_samples(self) -> tuple[Sample, Sample]:
        """Convert to two separate Sample objects (chosen, rejected)."""
        chosen = Sample(
            input_ids=self.chosen_input_ids,
            loss_mask=self.chosen_loss_mask,
            meta=self.meta,
            sample_type=SampleType.DPO,
        )
        rejected = Sample(
            input_ids=self.rejected_input_ids,
            loss_mask=self.rejected_loss_mask,
            meta=SampleMeta(
                sample_id=f"{self.meta.sample_id}_rejected",
                dataset_id=self.meta.dataset_id,
                source=self.meta.source,
                difficulty=self.meta.difficulty,
                difficulty_score=self.meta.difficulty_score,
            ),
            sample_type=SampleType.DPO,
        )
        return chosen, rejected


# =============================================================================
# Dataset Fingerprinting
# =============================================================================


class DatasetFingerprint(BaseModel):
    """
    Content-addressable hash of a dataset for cache invalidation.

    Combines dataset content hash with tokenizer hash to ensure
    cached artifacts (length caches, batch plans) are invalidated
    when either changes.
    """

    model_config = ConfigDict(frozen=True)

    # Primary fingerprint
    fingerprint: str = Field(description="Short fingerprint (16 chars)")
    full_hash: str = Field(description="Full SHA-256 hash")

    # Component hashes
    content_hash: str = Field(description="Hash of dataset content")
    tokenizer_hash: str = Field(description="Hash of tokenizer config")

    # Metadata
    num_samples: int = Field(ge=0, description="Number of samples in dataset")
    source_path: str | None = Field(default=None, description="Source file path")
    algorithm: str = Field(default="sha256", description="Hash algorithm")
    version: int = Field(default=1, description="Fingerprint format version")

    def matches(self, other: DatasetFingerprint) -> bool:
        """Check if fingerprints match."""
        return self.full_hash == other.full_hash

    def matches_content(self, other: DatasetFingerprint) -> bool:
        """Check if content matches (ignoring tokenizer)."""
        return self.content_hash == other.content_hash


def compute_dataset_fingerprint(
    dataset_path: str | Path,
    tokenizer_hash: str,
    sample_limit: int | None = None,
) -> DatasetFingerprint:
    """
    Compute a fingerprint for a dataset file.

    Args:
        dataset_path: Path to dataset file (JSONL)
        tokenizer_hash: Tokenizer fingerprint hash
        sample_limit: Max samples to hash (None = all)

    Returns:
        DatasetFingerprint for cache invalidation
    """
    path = Path(dataset_path)

    # Hash file content
    hasher = hashlib.sha256()
    num_samples = 0

    with open(path) as f:
        for i, line in enumerate(f):
            if sample_limit is not None and i >= sample_limit:
                break
            if line.strip():
                hasher.update(line.encode("utf-8"))
                num_samples += 1

    content_hash = hasher.hexdigest()

    # Combine with tokenizer hash
    combined = f"{content_hash}:{tokenizer_hash}"
    combined_hasher = hashlib.sha256()
    combined_hasher.update(combined.encode("utf-8"))
    full_hash = combined_hasher.hexdigest()

    return DatasetFingerprint(
        fingerprint=full_hash[:16],
        full_hash=full_hash,
        content_hash=content_hash[:16],
        tokenizer_hash=tokenizer_hash[:16],
        num_samples=num_samples,
        source_path=str(path.absolute()),
    )


# =============================================================================
# I/O Utilities
# =============================================================================


def save_samples(samples: list[Sample], path: str | Path) -> None:
    """Save samples to JSONL file."""
    path = Path(path)
    with open(path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample.to_lists()) + "\n")


def load_samples(path: str | Path) -> list[Sample]:
    """Load samples from JSONL file."""
    path = Path(path)
    samples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                samples.append(Sample.from_lists(data))
    return samples


async def save_samples_async(samples: list[Sample], path: str | Path) -> None:
    """Async: Save samples to JSONL file."""
    import aiofiles

    path = Path(path)
    async with aiofiles.open(path, "w") as f:
        for sample in samples:
            await f.write(json.dumps(sample.to_lists()) + "\n")


async def load_samples_async(path: str | Path) -> list[Sample]:
    """Async: Load samples from JSONL file."""
    import aiofiles

    path = Path(path)
    samples = []
    async with aiofiles.open(path) as f:
        async for line in f:
            if line.strip():
                data = json.loads(line)
                samples.append(Sample.from_lists(data))
    return samples
