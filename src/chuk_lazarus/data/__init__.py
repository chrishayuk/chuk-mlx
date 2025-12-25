"""
Data handling for training.

This module provides:
- Canonical sample schema (Pydantic-native, no magic strings)
- Async-native batching infrastructure (buckets, token-budget sampling)
- Base dataset class for inheritance
- Dataset classes for different training paradigms (SFT, DPO, RL)
- Batch dataset for pre-tokenized NPZ data
- Data generators for synthetic data
- Batch I/O for writing/reading NPZ batch files
- Tokenizer utilities
"""

# Base class and protocols
from .base_dataset import BaseDataset

# Batch datasets (for pre-tokenized NPZ data)
from .batch_dataset_base import BatchDatasetBase

# Async-native batching (unified pipeline)
from .batching import (
    BatchFingerprint,
    BatchingConfig,
    BatchingMode,
    BatchMetrics,
    BatchPlan,
    BatchPlanBuilder,
    BatchPlanMeta,
    BatchReader,
    BatchShapeHistogram,
    BatchSpec,
    BatchWriter,
    BucketId,
    BucketSpec,
    BucketStats,
    CollatedBatch,
    EpochPlan,
    LengthCache,
    LengthEntry,
    MicrobatchSpec,
    PackedSequence,
    PackingConfig,
    PackingMetrics,
    PackingMode,
    PadPolicy,
    SequenceToPack,
    TokenBudgetBatchSampler,
    compute_batch_fingerprint,
    compute_packing_metrics,
    create_segment_attention_mask,
    default_collate,
    load_batch_plan,
    pack_sequences,
    pad_sequences,
    save_batch_plan,
    verify_batch_fingerprint,
)

# Classification dataset
from .classification_dataset import (
    ClassificationDataset,
    ClassificationSample,
    load_classification_data,
)

# Data generators
from .generators import (
    MathProblem,
    MathProblemGenerator,
    ProblemType,
    ToolCallTrace,
    TrainingSample,
    generate_lazarus_dataset,
)
from .preference_dataset import PreferenceDataset, PreferencePair, load_preference_data
from .protocols import (
    BatchableDataset,
    ClassificationDatasetProtocol,
    Dataset,
    PreferenceDatasetProtocol,
    SFTDatasetProtocol,
    TokenizerProtocol,
)
from .rollout_buffer import Episode, RolloutBuffer, Transition

# Canonical sample schema (Phase 0 - Pydantic-native)
from .samples import (
    DatasetFingerprint,
    DatasetSource,
    PreferenceSample,
    Sample,
    SampleMeta,
    SampleType,
    SampleValidationError,
    compute_dataset_fingerprint,
)

# RL/SFT datasets
from .sft_dataset import SFTDataset, SFTSample

# Tokenizer utilities
from .tokenizers import (
    BoWCharacterTokenizer,
    BoWTokenizerConfig,
    CharacterTokenizer,
    CharacterTokenizerConfig,
    CustomTokenizer,
    load_vocabulary,
    save_vocabulary,
)
from .train_batch_dataset import TrainBatchDataset

__all__ = [
    # Base and Protocols
    "BaseDataset",
    "Dataset",
    "BatchableDataset",
    "SFTDatasetProtocol",
    "PreferenceDatasetProtocol",
    "ClassificationDatasetProtocol",
    "TokenizerProtocol",
    # Batch datasets
    "BatchDatasetBase",
    "TrainBatchDataset",
    # Classification
    "ClassificationDataset",
    "ClassificationSample",
    "load_classification_data",
    # Async-native batching (unified pipeline)
    "BucketSpec",
    "BucketId",
    "BucketStats",
    "LengthCache",
    "LengthEntry",
    "TokenBudgetBatchSampler",
    "BatchSpec",
    "BatchMetrics",
    "BatchShapeHistogram",
    # Predictability mode
    "PadPolicy",
    "BatchingMode",
    "BatchingConfig",
    "BatchFingerprint",
    "compute_batch_fingerprint",
    "verify_batch_fingerprint",
    # Packing
    "PackingMode",
    "PackingConfig",
    "PackedSequence",
    "SequenceToPack",
    "pack_sequences",
    "create_segment_attention_mask",
    "PackingMetrics",
    "compute_packing_metrics",
    # BatchPlan
    "BatchPlan",
    "BatchPlanMeta",
    "BatchPlanBuilder",
    "EpochPlan",
    "MicrobatchSpec",
    "save_batch_plan",
    "load_batch_plan",
    # Batch I/O (unified pipeline)
    "BatchWriter",
    "BatchReader",
    "CollatedBatch",
    "default_collate",
    "pad_sequences",
    # Canonical sample schema
    "Sample",
    "SampleMeta",
    "SampleType",
    "DatasetSource",
    "PreferenceSample",
    "SampleValidationError",
    "DatasetFingerprint",
    "compute_dataset_fingerprint",
    # Generators
    "MathProblem",
    "MathProblemGenerator",
    "ProblemType",
    "ToolCallTrace",
    "TrainingSample",
    "generate_lazarus_dataset",
    # Preference
    "PreferenceDataset",
    "PreferencePair",
    "load_preference_data",
    # Rollout
    "Episode",
    "RolloutBuffer",
    "Transition",
    # SFT
    "SFTDataset",
    "SFTSample",
    # Tokenizers
    "BoWCharacterTokenizer",
    "BoWTokenizerConfig",
    "CharacterTokenizer",
    "CharacterTokenizerConfig",
    "CustomTokenizer",
    "load_vocabulary",
    "save_vocabulary",
]
