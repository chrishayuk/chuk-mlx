"""
Data handling for training.

This module provides:
- Base dataset class for inheritance
- Dataset classes for different training paradigms (SFT, DPO, RL)
- Batch dataset for pre-tokenized NPZ data
- Data generators for synthetic data
- Preprocessing utilities (batching, padding, bucketing)
- Tokenizer utilities
"""

# Base class
from .base_dataset import BaseDataset

# Batch datasets (for pre-tokenized NPZ data)
from .batch_dataset_base import BatchDatasetBase

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

# Preprocessing utilities (batching, padding, tokenization)
from .preprocessing import (
    BatchBase,
    FineTuneBatch,
    LLaMAFineTuneBatch,
    PretrainBatchGenerator,
    add_to_buckets,
    get_batch_from_buckets,
    pad_sequences,
)
from .rollout_buffer import Episode, RolloutBuffer, Transition

# RL/SFT datasets
from .sft_dataset import SFTDataset, SFTSample

# Tokenizer utilities
from .tokenizers import (
    CustomTokenizer,
    load_vocabulary,
    save_vocabulary,
)
from .train_batch_dataset import TrainBatchDataset

__all__ = [
    # Base
    "BaseDataset",
    # Batch datasets
    "BatchDatasetBase",
    "TrainBatchDataset",
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
    # Preprocessing
    "BatchBase",
    "FineTuneBatch",
    "LLaMAFineTuneBatch",
    "PretrainBatchGenerator",
    "add_to_buckets",
    "get_batch_from_buckets",
    "pad_sequences",
    # Rollout
    "Episode",
    "RolloutBuffer",
    "Transition",
    # SFT
    "SFTDataset",
    "SFTSample",
    # Tokenizers
    "CustomTokenizer",
    "load_vocabulary",
    "save_vocabulary",
]
