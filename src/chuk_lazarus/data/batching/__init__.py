"""
Async-native batching infrastructure.

This module provides:
- BucketSpec: Length bucket configuration
- LengthCache: Async cache for pre-computed sequence lengths
- TokenBudgetBatchSampler: Form batches by token budget, not sample count
- BatchMetrics: Padding waste, throughput, bucket utilization
- PadPolicy, BatchingConfig: Predictability mode for reproducibility
- BatchFingerprint: Verify batch contents for replay
- Packing: Sequence concatenation with segment-aware attention
- BatchPlan: Precomputed schedules for distributed training
- Analysis: Efficiency reports, bucket optimization, histograms

Submodules:
- core: Bucket configuration, sampler, metrics
- planning: BatchPlan, packing, predictability
- generation: LengthCache, BatchWriter, BatchReader
- analyze: Efficiency analysis, histograms, suggestions

Design principles:
- Async-native: All I/O is async (aiofiles, async iterators)
- Pydantic-native: All structures use BaseModel for validation
- No magic strings: Enums and constants for type safety
- Deterministic: Reproducible batching with seeds
- Verifiable: Fingerprints for CI/CD validation
- Distributed-ready: Plans shard cleanly across ranks
"""

# Import submodules for namespace access
from . import analyze, core, generation, planning, streaming

# Analysis (Phase 5.6 - unified pipeline)
from .analyze import (
    BatchingEfficiencyReport,
    BucketAnalysis,
    BucketEfficiency,
    BucketSuggestion,
    HistogramBin,
    LengthHistogram,
    OptimizationGoal,
    analyze_bucket_efficiency,
    compute_length_histogram,
    create_efficiency_report,
    suggest_bucket_edges,
    visualize_buckets,
)

# Core: Buckets, Sampler, Metrics
from .core import (
    BatchMetrics,
    BatchShapeHistogram,
    BatchSpec,
    BucketId,
    BucketSpec,
    BucketStats,
    TokenBudgetBatchSampler,
)

# Generation: LengthCache, I/O
from .generation import (
    BatchReader,
    BatchWriter,
    CollatedBatch,
    LengthCache,
    LengthEntry,
    default_collate,
    pad_sequences,
)

# Planning: BatchPlan, Packing, Predictability
from .planning import (
    BatchFingerprint,
    BatchingConfig,
    BatchingMode,
    BatchPlan,
    BatchPlanBuilder,
    BatchPlanMeta,
    EpochPlan,
    MicrobatchSpec,
    PackedSequence,
    PackingConfig,
    PackingMetrics,
    PackingMode,
    PadPolicy,
    SequenceToPack,
    compute_batch_fingerprint,
    compute_batch_fingerprint_async,
    compute_packing_metrics,
    create_segment_attention_mask,
    load_batch_plan,
    load_batch_plan_async,
    load_fingerprint,
    pack_sequences,
    save_batch_plan,
    save_batch_plan_async,
    save_fingerprint,
    verify_batch_fingerprint,
)

__all__ = [
    # Submodules
    "core",
    "planning",
    "generation",
    "analyze",
    "streaming",
    # Analysis (Phase 5.6 - unified pipeline)
    "LengthHistogram",
    "HistogramBin",
    "BucketEfficiency",
    "BucketAnalysis",
    "BucketSuggestion",
    "OptimizationGoal",
    "BatchingEfficiencyReport",
    "compute_length_histogram",
    "analyze_bucket_efficiency",
    "suggest_bucket_edges",
    "create_efficiency_report",
    "visualize_buckets",
    # Bucket configuration
    "BucketSpec",
    "BucketId",
    "BucketStats",
    # Length cache
    "LengthCache",
    "LengthEntry",
    # Sampler
    "TokenBudgetBatchSampler",
    "BatchSpec",
    # Metrics
    "BatchMetrics",
    "BatchShapeHistogram",
    # Predictability mode (Phase 2)
    "PadPolicy",
    "BatchingMode",
    "BatchingConfig",
    "BatchFingerprint",
    "compute_batch_fingerprint",
    "compute_batch_fingerprint_async",
    "verify_batch_fingerprint",
    "save_fingerprint",
    "load_fingerprint",
    # Packing (Phase 3)
    "PackingMode",
    "PackingConfig",
    "PackedSequence",
    "SequenceToPack",
    "pack_sequences",
    "create_segment_attention_mask",
    "PackingMetrics",
    "compute_packing_metrics",
    # BatchPlan (Phase 4)
    "BatchPlan",
    "BatchPlanMeta",
    "BatchPlanBuilder",
    "EpochPlan",
    "MicrobatchSpec",
    "save_batch_plan",
    "load_batch_plan",
    "save_batch_plan_async",
    "load_batch_plan_async",
    # Batch I/O (Phase 5.6 - unified pipeline)
    "BatchWriter",
    "BatchReader",
    "CollatedBatch",
    "default_collate",
    "pad_sequences",
]
