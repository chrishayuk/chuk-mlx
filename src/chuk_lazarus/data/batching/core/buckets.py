"""
Bucket configuration for length-based batching.

Buckets group sequences by length to reduce padding waste.
Each bucket has a max length, and sequences are padded to that length.
"""

from __future__ import annotations

from typing import NewType

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Type-safe bucket identifier
BucketId = NewType("BucketId", int)


class BucketSpec(BaseModel):
    """
    Configuration for length-based bucketing.

    Sequences are assigned to buckets based on their length.
    Each bucket has a maximum length that sequences are padded to.

    Example:
        edges = [128, 256, 512, 1024]
        - Bucket 0: lengths 1-128, pad to 128
        - Bucket 1: lengths 129-256, pad to 256
        - Bucket 2: lengths 257-512, pad to 512
        - Bucket 3: lengths 513-1024, pad to 1024
        - Overflow: lengths > 1024, pad to overflow_max
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    edges: tuple[int, ...] = Field(
        description="Bucket edge lengths (must be sorted ascending)",
    )
    overflow_max: int = Field(
        gt=0,
        description="Maximum length for overflow bucket (sequences longer than last edge)",
    )
    min_length: int = Field(
        default=1,
        ge=1,
        description="Minimum sequence length (shorter sequences are skipped)",
    )

    @field_validator("edges", mode="before")
    @classmethod
    def convert_to_tuple(cls, v):
        """Convert list to tuple."""
        if isinstance(v, (list, tuple)):
            return tuple(v)
        raise ValueError(f"Expected list or tuple, got {type(v)}")

    @model_validator(mode="after")
    def validate_edges_sorted(self):
        """Validate edges are sorted ascending."""
        for i in range(1, len(self.edges)):
            if self.edges[i] <= self.edges[i - 1]:
                raise ValueError(
                    f"edges must be strictly ascending, "
                    f"found {self.edges[i]} <= {self.edges[i - 1]} at index {i}"
                )
        return self

    @model_validator(mode="after")
    def validate_overflow_gt_last_edge(self):
        """Validate overflow_max > last edge."""
        if self.edges and self.overflow_max <= self.edges[-1]:
            raise ValueError(
                f"overflow_max ({self.overflow_max}) must be > last edge ({self.edges[-1]})"
            )
        return self

    @property
    def num_buckets(self) -> int:
        """Number of buckets (including overflow)."""
        return len(self.edges) + 1

    @property
    def overflow_bucket_id(self) -> BucketId:
        """ID of the overflow bucket."""
        return BucketId(len(self.edges))

    def get_bucket_id(self, length: int) -> BucketId:
        """
        Get bucket ID for a sequence length.

        Args:
            length: Sequence length

        Returns:
            Bucket ID (0-indexed)
        """
        for i, edge in enumerate(self.edges):
            if length <= edge:
                return BucketId(i)
        return self.overflow_bucket_id

    def get_bucket_max_length(self, bucket_id: BucketId) -> int:
        """
        Get maximum length for a bucket.

        Args:
            bucket_id: Bucket ID

        Returns:
            Maximum sequence length for this bucket
        """
        if bucket_id < len(self.edges):
            return self.edges[bucket_id]
        return self.overflow_max

    def get_bucket_range(self, bucket_id: BucketId) -> tuple[int, int]:
        """
        Get (min_length, max_length) range for a bucket.

        Args:
            bucket_id: Bucket ID

        Returns:
            Tuple of (min_length, max_length) inclusive
        """
        if bucket_id == 0:
            min_len = self.min_length
        else:
            min_len = self.edges[bucket_id - 1] + 1

        max_len = self.get_bucket_max_length(bucket_id)
        return (min_len, max_len)

    def is_overflow(self, bucket_id: BucketId) -> bool:
        """Check if bucket is the overflow bucket."""
        return bucket_id == self.overflow_bucket_id

    def should_skip(self, length: int) -> bool:
        """Check if a sequence should be skipped (too short or too long)."""
        return length < self.min_length or length > self.overflow_max

    @classmethod
    def default(cls) -> BucketSpec:
        """Create default bucket spec (128, 256, 512, 1024, overflow=2048)."""
        return cls(
            edges=(128, 256, 512, 1024),
            overflow_max=2048,
        )

    @classmethod
    def from_max_length(cls, max_length: int, num_buckets: int = 4) -> BucketSpec:
        """
        Create bucket spec from max length.

        Divides range evenly into buckets.

        Args:
            max_length: Maximum sequence length
            num_buckets: Number of buckets (including overflow)

        Returns:
            BucketSpec with evenly spaced edges
        """
        if num_buckets < 2:
            raise ValueError("num_buckets must be >= 2")

        step = max_length // num_buckets
        edges = tuple(step * (i + 1) for i in range(num_buckets - 1))
        return cls(edges=edges, overflow_max=max_length)


class BucketStats(BaseModel):
    """
    Statistics for a single bucket.

    Tracks sample count, token count, and padding waste.
    """

    model_config = ConfigDict(frozen=False)

    bucket_id: BucketId = Field(description="Bucket identifier")
    bucket_max_length: int = Field(description="Max length for this bucket")
    sample_count: int = Field(default=0, ge=0, description="Number of samples")
    total_tokens: int = Field(default=0, ge=0, description="Total tokens before padding")
    padded_tokens: int = Field(default=0, ge=0, description="Total tokens after padding")

    @property
    def padding_tokens(self) -> int:
        """Number of padding tokens added."""
        return self.padded_tokens - self.total_tokens

    @property
    def padding_ratio(self) -> float:
        """Fraction of tokens that are padding (0.0 = no waste)."""
        if self.padded_tokens == 0:
            return 0.0
        return self.padding_tokens / self.padded_tokens

    @property
    def efficiency(self) -> float:
        """Fraction of tokens that are real (1.0 = perfect)."""
        return 1.0 - self.padding_ratio

    @property
    def avg_length(self) -> float:
        """Average sequence length."""
        if self.sample_count == 0:
            return 0.0
        return self.total_tokens / self.sample_count

    def add_sample(self, length: int) -> None:
        """Record a sample added to this bucket."""
        self.sample_count += 1
        self.total_tokens += length
        self.padded_tokens += self.bucket_max_length
