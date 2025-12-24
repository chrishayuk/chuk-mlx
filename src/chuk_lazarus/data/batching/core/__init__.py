"""
Core batching primitives.

This submodule provides:
- BucketSpec: Length bucket configuration
- TokenBudgetBatchSampler: Form batches by token budget
- BatchMetrics: Padding waste, throughput tracking
"""

from .buckets import (
    BucketId,
    BucketSpec,
    BucketStats,
)
from .metrics import (
    BatchMetrics,
    BatchShapeHistogram,
)
from .sampler import (
    BatchSpec,
    TokenBudgetBatchSampler,
)

__all__ = [
    # Buckets
    "BucketSpec",
    "BucketId",
    "BucketStats",
    # Sampler
    "TokenBudgetBatchSampler",
    "BatchSpec",
    # Metrics
    "BatchMetrics",
    "BatchShapeHistogram",
]
