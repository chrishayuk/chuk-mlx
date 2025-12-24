"""
Async-native token-budget batch sampler.

Forms batches by token budget rather than sample count,
maximizing GPU utilization while respecting memory constraints.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import AsyncIterator

from pydantic import BaseModel, ConfigDict, Field

from .buckets import BucketId, BucketSpec
from .metrics import BatchMetrics


class BatchSpec(BaseModel):
    """
    Specification for a single batch.

    Contains sample IDs and bucket information for forming a batch.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_ids: tuple[str, ...] = Field(
        description="Sample IDs in this batch",
    )
    bucket_id: BucketId = Field(
        description="Bucket ID for this batch",
    )
    max_length: int = Field(
        description="Maximum sequence length (pad target)",
    )
    token_count: int = Field(
        description="Total tokens in batch (before padding)",
    )

    @property
    def batch_size(self) -> int:
        """Number of samples in batch."""
        return len(self.sample_ids)

    @property
    def padded_token_count(self) -> int:
        """Total tokens after padding."""
        return self.batch_size * self.max_length


class TokenBudgetBatchSampler:
    """
    Async-native batch sampler using token budget.

    Groups samples by bucket, then forms batches that fit within
    a token budget. Deterministic with seed control.

    Usage:
        sampler = TokenBudgetBatchSampler(
            lengths={"s1": 100, "s2": 200, ...},
            bucket_spec=BucketSpec.default(),
            token_budget=4096,
            seed=42,
        )

        async for batch_spec in sampler.iter_epoch(epoch=0):
            # batch_spec.sample_ids, batch_spec.max_length, etc.
            ...
    """

    def __init__(
        self,
        lengths: dict[str, int],
        bucket_spec: BucketSpec,
        token_budget: int,
        seed: int = 42,
        drop_last: bool = False,
    ):
        """
        Initialize sampler.

        Args:
            lengths: Map of sample_id -> sequence length
            bucket_spec: Bucket configuration
            token_budget: Maximum tokens per batch
            seed: Base random seed (actual seed = base + epoch)
            drop_last: If True, drop last incomplete batch per bucket
        """
        self._lengths = lengths
        self._bucket_spec = bucket_spec
        self._token_budget = token_budget
        self._base_seed = seed
        self._drop_last = drop_last

        # Pre-assign samples to buckets
        self._buckets: dict[BucketId, list[str]] = defaultdict(list)
        self._skipped: list[str] = []

        for sample_id, length in lengths.items():
            if bucket_spec.should_skip(length):
                self._skipped.append(sample_id)
            else:
                bucket_id = bucket_spec.get_bucket_id(length)
                self._buckets[bucket_id].append(sample_id)

    @property
    def bucket_spec(self) -> BucketSpec:
        """Bucket configuration."""
        return self._bucket_spec

    @property
    def token_budget(self) -> int:
        """Token budget per batch."""
        return self._token_budget

    @property
    def num_samples(self) -> int:
        """Total samples (excluding skipped)."""
        return sum(len(samples) for samples in self._buckets.values())

    @property
    def num_skipped(self) -> int:
        """Number of skipped samples."""
        return len(self._skipped)

    def bucket_sizes(self) -> dict[BucketId, int]:
        """Get sample count per bucket."""
        return {bid: len(samples) for bid, samples in self._buckets.items()}

    # =========================================================================
    # Epoch Iteration
    # =========================================================================

    async def iter_epoch(
        self,
        epoch: int = 0,
        interleave_buckets: bool = True,
    ) -> AsyncIterator[BatchSpec]:
        """
        Async iterate over batches for one epoch.

        Args:
            epoch: Epoch number (affects shuffle seed)
            interleave_buckets: If True, interleave batches from different buckets

        Yields:
            BatchSpec for each batch
        """
        # Deterministic seed for this epoch
        rng = random.Random(self._base_seed + epoch)

        # Shuffle samples within each bucket
        shuffled_buckets: dict[BucketId, list[str]] = {}
        for bucket_id, sample_ids in self._buckets.items():
            shuffled = list(sample_ids)
            rng.shuffle(shuffled)
            shuffled_buckets[bucket_id] = shuffled

        # Form batches per bucket
        bucket_batches: dict[BucketId, list[BatchSpec]] = {}
        for bucket_id, sample_ids in shuffled_buckets.items():
            batches = list(self._form_batches(bucket_id, sample_ids))
            if batches:
                bucket_batches[bucket_id] = batches

        if interleave_buckets:
            # Interleave batches from different buckets for balanced sharding
            async for batch in self._interleave_batches(bucket_batches, rng):
                yield batch
        else:
            # Yield all batches from each bucket in sequence
            for bucket_id in sorted(bucket_batches.keys()):
                for batch in bucket_batches[bucket_id]:
                    yield batch

    def _form_batches(
        self,
        bucket_id: BucketId,
        sample_ids: list[str],
    ) -> list[BatchSpec]:
        """Form batches from samples in a bucket."""
        max_length = self._bucket_spec.get_bucket_max_length(bucket_id)
        max_samples_per_batch = self._token_budget // max_length

        if max_samples_per_batch == 0:
            # Token budget too small for even one sample
            max_samples_per_batch = 1

        batches = []
        current_batch: list[str] = []
        current_tokens = 0

        for sample_id in sample_ids:
            # Check if adding this sample would exceed budget
            would_exceed = (
                current_tokens + max_length > self._token_budget
                or len(current_batch) >= max_samples_per_batch
            )

            if would_exceed and current_batch:
                # Emit current batch
                batches.append(self._make_batch_spec(current_batch, bucket_id, max_length))
                current_batch = []
                current_tokens = 0

            current_batch.append(sample_id)
            current_tokens += max_length

        # Handle remaining samples
        if current_batch:
            if not self._drop_last or len(current_batch) == max_samples_per_batch:
                batches.append(self._make_batch_spec(current_batch, bucket_id, max_length))

        return batches

    def _make_batch_spec(
        self,
        sample_ids: list[str],
        bucket_id: BucketId,
        max_length: int,
    ) -> BatchSpec:
        """Create a BatchSpec from sample IDs."""
        token_count = sum(self._lengths[sid] for sid in sample_ids)
        return BatchSpec(
            sample_ids=tuple(sample_ids),
            bucket_id=bucket_id,
            max_length=max_length,
            token_count=token_count,
        )

    async def _interleave_batches(
        self,
        bucket_batches: dict[BucketId, list[BatchSpec]],
        rng: random.Random,
    ) -> AsyncIterator[BatchSpec]:
        """
        Interleave batches from different buckets.

        Uses round-robin with shuffled bucket order per round.
        """
        # Track position in each bucket
        positions: dict[BucketId, int] = dict.fromkeys(bucket_batches, 0)
        active_buckets = list(bucket_batches.keys())

        while active_buckets:
            # Shuffle bucket order for this round
            rng.shuffle(active_buckets)

            exhausted = []
            for bucket_id in active_buckets:
                pos = positions[bucket_id]
                batches = bucket_batches[bucket_id]

                if pos < len(batches):
                    yield batches[pos]
                    positions[bucket_id] = pos + 1
                else:
                    exhausted.append(bucket_id)

            # Remove exhausted buckets
            for bid in exhausted:
                active_buckets.remove(bid)

    # =========================================================================
    # Metrics
    # =========================================================================

    def compute_metrics(self, loss_tokens_per_sample: dict[str, int] | None = None) -> BatchMetrics:
        """
        Compute metrics for the current configuration.

        Args:
            loss_tokens_per_sample: Optional map of sample_id -> loss token count

        Returns:
            BatchMetrics with efficiency statistics
        """
        metrics = BatchMetrics()

        # Record skipped samples
        for _ in self._skipped:
            metrics.record_skip()

        # Simulate one epoch to compute metrics
        for bucket_id, sample_ids in self._buckets.items():
            max_length = self._bucket_spec.get_bucket_max_length(bucket_id)

            for sample_id in sample_ids:
                length = self._lengths[sample_id]
                loss_tokens = (
                    loss_tokens_per_sample.get(sample_id, length)
                    if loss_tokens_per_sample
                    else length
                )
                metrics.record_sample(bucket_id, length, loss_tokens, max_length)

            # Count batches
            batches = self._form_batches(bucket_id, sample_ids)
            for batch in batches:
                metrics.record_batch(bucket_id, batch.batch_size)

        return metrics

    # =========================================================================
    # Utilities
    # =========================================================================

    def estimate_batches_per_epoch(self) -> int:
        """Estimate total batches per epoch."""
        total = 0
        for bucket_id, sample_ids in self._buckets.items():
            batches = self._form_batches(bucket_id, list(sample_ids))
            total += len(batches)
        return total

    def get_bucket_distribution(self) -> dict[BucketId, float]:
        """Get fraction of samples in each bucket."""
        total = self.num_samples
        if total == 0:
            return {}
        return {bid: len(samples) / total for bid, samples in self._buckets.items()}
