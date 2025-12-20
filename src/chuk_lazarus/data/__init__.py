"""
Data handling for training.

This module provides:
- Dataset classes for different training paradigms (SFT, DPO, RL)
- Batch dataset for pre-tokenized NPZ data
- Data generators for synthetic data
- Preprocessing utilities (batching, padding, bucketing)
- Tokenizer utilities
"""

# RL/SFT datasets
from .sft_dataset import SFTDataset, SFTSample
from .preference_dataset import PreferenceDataset, PreferencePair, load_preference_data
from .rollout_buffer import RolloutBuffer, Transition, Episode

# Batch datasets (for pre-tokenized NPZ data)
from .batch_dataset_base import BatchDatasetBase
from .train_batch_dataset import TrainBatchDataset

# Data generators
from .generators import (
    MathProblemGenerator,
    MathProblem,
    ProblemType,
    ToolCallTrace,
    TrainingSample,
    generate_lazarus_dataset,
)

# Preprocessing utilities (batching, padding, tokenization)
from .preprocessing import (
    BatchBase,
    FineTuneBatch,
    PretrainBatchGenerator,
    LLaMAFineTuneBatch,
    pad_sequences,
    add_to_buckets,
    get_batch_from_buckets,
)

# Tokenizer utilities
from .tokenizers import (
    CustomTokenizer,
    load_vocabulary,
    save_vocabulary,
)
