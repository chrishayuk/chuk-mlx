"""
Canonical sample schema for training.

This module defines the standard sample format used throughout the batching
and training pipeline. All samples conform to a validated Pydantic schema
for reproducibility and type safety.

Key types:
- Sample: Base tokenized sample with input_ids, loss_mask, metadata
- PreferenceSample: Paired chosen/rejected samples for DPO
- SampleMeta: Metadata for tracking, curriculum, and reproducibility

Enums:
- SampleType: SFT, DPO, PRETRAIN, RL
- DatasetSource: LOCAL, HUGGINGFACE, GYM, SYNTHETIC

Functions:
- dataset_fingerprint(): Hash dataset content for cache invalidation
"""

from .schema import (
    DatasetFingerprint,
    DatasetSource,
    PreferenceSample,
    Sample,
    SampleMeta,
    SampleType,
    SampleValidationError,
    compute_dataset_fingerprint,
)

__all__ = [
    # Core models
    "Sample",
    "SampleMeta",
    "PreferenceSample",
    # Enums
    "SampleType",
    "DatasetSource",
    # Fingerprinting
    "DatasetFingerprint",
    "compute_dataset_fingerprint",
    # Errors
    "SampleValidationError",
]
