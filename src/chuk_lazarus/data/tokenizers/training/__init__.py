"""
Training-time tokenizer utilities.

Modules:
- packer: Smart sequence packing for efficient batching
- throughput: Token throughput profiling
"""

from .packer import (
    PackedBatch,
    PackedSequence,
    PackingConfig,
    PackingStats,
    calculate_packing_efficiency,
    create_packed_batch,
    pack_sequences,
)
from .throughput import (
    BatchMetrics,
    ThroughputMetrics,
    ThroughputProfiler,
    estimate_training_tokens,
    profile_tokenization,
)

__all__ = [
    # Packer
    "PackedBatch",
    "PackedSequence",
    "PackingConfig",
    "PackingStats",
    "pack_sequences",
    "create_packed_batch",
    "calculate_packing_efficiency",
    # Throughput
    "BatchMetrics",
    "ThroughputMetrics",
    "ThroughputProfiler",
    "profile_tokenization",
    "estimate_training_tokens",
]
