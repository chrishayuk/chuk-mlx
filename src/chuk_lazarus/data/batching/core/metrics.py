"""
Batching metrics for monitoring efficiency.

Tracks padding waste, throughput, and bucket utilization.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .buckets import BucketId, BucketStats


class BatchShapeHistogram(BaseModel):
    """
    Histogram of batch shapes by bucket.

    Tracks how batches are distributed across buckets.
    """

    model_config = ConfigDict(frozen=False)

    bucket_counts: dict[int, int] = Field(
        default_factory=dict,
        description="Number of batches per bucket ID",
    )
    bucket_samples: dict[int, int] = Field(
        default_factory=dict,
        description="Number of samples per bucket ID",
    )

    def record_batch(self, bucket_id: BucketId, batch_size: int) -> None:
        """Record a batch for histogram."""
        bid = int(bucket_id)
        self.bucket_counts[bid] = self.bucket_counts.get(bid, 0) + 1
        self.bucket_samples[bid] = self.bucket_samples.get(bid, 0) + batch_size

    @property
    def total_batches(self) -> int:
        """Total batches across all buckets."""
        return sum(self.bucket_counts.values())

    @property
    def total_samples(self) -> int:
        """Total samples across all buckets."""
        return sum(self.bucket_samples.values())

    def bucket_fraction(self, bucket_id: BucketId) -> float:
        """Fraction of samples in a bucket."""
        total = self.total_samples
        if total == 0:
            return 0.0
        return self.bucket_samples.get(int(bucket_id), 0) / total


class BatchMetrics(BaseModel):
    """
    Comprehensive batching metrics.

    Tracks efficiency, throughput, and resource utilization.
    """

    model_config = ConfigDict(frozen=False)

    # Token counts
    total_tokens: int = Field(
        default=0,
        ge=0,
        description="Total tokens before padding",
    )
    padded_tokens: int = Field(
        default=0,
        ge=0,
        description="Total tokens after padding",
    )
    loss_tokens: int = Field(
        default=0,
        ge=0,
        description="Tokens contributing to loss (sum of loss_mask)",
    )

    # Sample counts
    total_samples: int = Field(
        default=0,
        ge=0,
        description="Total samples processed",
    )
    skipped_samples: int = Field(
        default=0,
        ge=0,
        description="Samples skipped (too short/long)",
    )

    # Batch counts
    total_batches: int = Field(
        default=0,
        ge=0,
        description="Total batches formed",
    )

    # Timing (optional, filled during training)
    total_time_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Total processing time",
    )

    # Per-bucket stats
    bucket_stats: dict[int, BucketStats] = Field(
        default_factory=dict,
        description="Stats per bucket ID",
    )

    # Batch shape histogram
    shape_histogram: BatchShapeHistogram = Field(
        default_factory=BatchShapeHistogram,
        description="Distribution of batches across buckets",
    )

    # =========================================================================
    # Derived Metrics
    # =========================================================================

    @property
    def padding_tokens(self) -> int:
        """Number of padding tokens."""
        return self.padded_tokens - self.total_tokens

    @property
    def padding_waste(self) -> float:
        """Fraction of tokens that are padding (0.0 = no waste)."""
        if self.padded_tokens == 0:
            return 0.0
        return self.padding_tokens / self.padded_tokens

    @property
    def efficiency(self) -> float:
        """Fraction of tokens that are real (1.0 = perfect)."""
        return 1.0 - self.padding_waste

    @property
    def loss_efficiency(self) -> float:
        """Fraction of padded tokens that contribute to loss."""
        if self.padded_tokens == 0:
            return 0.0
        return self.loss_tokens / self.padded_tokens

    @property
    def tokens_per_second(self) -> float:
        """Throughput in tokens per second."""
        if self.total_time_seconds == 0:
            return 0.0
        return self.total_tokens / self.total_time_seconds

    @property
    def effective_tokens_per_second(self) -> float:
        """Effective throughput (loss tokens per second)."""
        if self.total_time_seconds == 0:
            return 0.0
        return self.loss_tokens / self.total_time_seconds

    @property
    def samples_per_second(self) -> float:
        """Sample throughput."""
        if self.total_time_seconds == 0:
            return 0.0
        return self.total_samples / self.total_time_seconds

    @property
    def avg_batch_size(self) -> float:
        """Average samples per batch."""
        if self.total_batches == 0:
            return 0.0
        return self.total_samples / self.total_batches

    @property
    def skip_rate(self) -> float:
        """Fraction of samples skipped."""
        total = self.total_samples + self.skipped_samples
        if total == 0:
            return 0.0
        return self.skipped_samples / total

    # =========================================================================
    # Recording Methods
    # =========================================================================

    def record_sample(
        self,
        bucket_id: BucketId,
        length: int,
        loss_tokens: int,
        bucket_max_length: int,
    ) -> None:
        """Record a sample being added to a batch."""
        self.total_samples += 1
        self.total_tokens += length
        self.padded_tokens += bucket_max_length
        self.loss_tokens += loss_tokens

        # Update bucket stats
        bid = int(bucket_id)
        if bid not in self.bucket_stats:
            self.bucket_stats[bid] = BucketStats(
                bucket_id=bucket_id,
                bucket_max_length=bucket_max_length,
            )
        self.bucket_stats[bid].add_sample(length)

    def record_batch(self, bucket_id: BucketId, batch_size: int) -> None:
        """Record a batch being formed."""
        self.total_batches += 1
        self.shape_histogram.record_batch(bucket_id, batch_size)

    def record_skip(self) -> None:
        """Record a sample being skipped."""
        self.skipped_samples += 1

    def record_time(self, seconds: float) -> None:
        """Record processing time."""
        self.total_time_seconds += seconds

    # =========================================================================
    # Reporting
    # =========================================================================

    def summary(self) -> dict:
        """Get summary metrics as dict."""
        return {
            "total_samples": self.total_samples,
            "total_batches": self.total_batches,
            "total_tokens": self.total_tokens,
            "loss_tokens": self.loss_tokens,
            "padding_waste": f"{self.padding_waste:.2%}",
            "efficiency": f"{self.efficiency:.2%}",
            "loss_efficiency": f"{self.loss_efficiency:.2%}",
            "avg_batch_size": f"{self.avg_batch_size:.1f}",
            "skip_rate": f"{self.skip_rate:.2%}",
            "tokens_per_second": f"{self.tokens_per_second:.0f}",
            "effective_tokens_per_second": f"{self.effective_tokens_per_second:.0f}",
        }

    def bucket_summary(self) -> list[dict]:
        """Get per-bucket summary."""
        summaries = []
        for bid, stats in sorted(self.bucket_stats.items()):
            summaries.append(
                {
                    "bucket_id": bid,
                    "max_length": stats.bucket_max_length,
                    "samples": stats.sample_count,
                    "efficiency": f"{stats.efficiency:.2%}",
                    "avg_length": f"{stats.avg_length:.1f}",
                }
            )
        return summaries
