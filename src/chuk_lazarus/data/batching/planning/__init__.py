"""
Batch planning and reproducibility.

This submodule provides:
- BatchPlan: Precomputed batch schedules
- BatchPlanBuilder: Build plans from lengths
- BatchingConfig: Complete batching configuration
- Packing: Sequence concatenation with segment masks
- Fingerprinting: Verify batch ordering
"""

from .batch_plan import (
    BatchPlan,
    BatchPlanBuilder,
    BatchPlanMeta,
    EpochPlan,
    MicrobatchSpec,
    load_batch_plan,
    load_batch_plan_async,
    save_batch_plan,
    save_batch_plan_async,
)
from .packing import (
    PackedSequence,
    PackingConfig,
    PackingMetrics,
    PackingMode,
    SequenceToPack,
    compute_packing_metrics,
    create_segment_attention_mask,
    pack_sequences,
)
from .predictability import (
    BatchFingerprint,
    BatchingConfig,
    BatchingMode,
    PadPolicy,
    compute_batch_fingerprint,
    compute_batch_fingerprint_async,
    load_fingerprint,
    save_fingerprint,
    verify_batch_fingerprint,
)

__all__ = [
    # BatchPlan
    "BatchPlan",
    "BatchPlanMeta",
    "BatchPlanBuilder",
    "EpochPlan",
    "MicrobatchSpec",
    "save_batch_plan",
    "load_batch_plan",
    "save_batch_plan_async",
    "load_batch_plan_async",
    # Predictability
    "PadPolicy",
    "BatchingMode",
    "BatchingConfig",
    "BatchFingerprint",
    "compute_batch_fingerprint",
    "compute_batch_fingerprint_async",
    "verify_batch_fingerprint",
    "save_fingerprint",
    "load_fingerprint",
    # Packing
    "PackingMode",
    "PackingConfig",
    "PackedSequence",
    "SequenceToPack",
    "pack_sequences",
    "create_segment_attention_mask",
    "PackingMetrics",
    "compute_packing_metrics",
]
