"""Training CLI commands.

This module provides commands for model training and data generation.

Commands:
    train_sft: Run supervised fine-tuning
    train_dpo: Run direct preference optimization
    generate_data: Generate synthetic training data
"""

from ._types import (
    DataGenConfig,
    DataGenResult,
    DataGenType,
    DPOConfig,
    SFTConfig,
    TrainMode,
    TrainResult,
)
from .datagen import generate_data, generate_data_cmd
from .dpo import train_dpo, train_dpo_cmd
from .sft import train_sft, train_sft_cmd

__all__ = [
    # Types
    "DataGenConfig",
    "DataGenResult",
    "DataGenType",
    "DPOConfig",
    "SFTConfig",
    "TrainMode",
    "TrainResult",
    # SFT Commands
    "train_sft",
    "train_sft_cmd",
    # DPO Commands
    "train_dpo",
    "train_dpo_cmd",
    # Data Generation Commands
    "generate_data",
    "generate_data_cmd",
]
