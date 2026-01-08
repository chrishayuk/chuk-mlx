"""Training CLI commands.

This module provides commands for model training and data generation.

Commands:
    train_sft_cmd: Run supervised fine-tuning
    train_dpo_cmd: Run direct preference optimization
    train_grpo_cmd: Run GRPO training
    generate_data_cmd: Generate synthetic training data
"""

from ._types import (
    DataGenConfig,
    DataGenResult,
    DataGenType,
    DPOConfig,
    GRPOConfig,
    SFTConfig,
    TrainMode,
    TrainResult,
)
from .datagen import generate_data, generate_data_cmd
from .dpo import train_dpo_cmd
from .grpo import train_grpo_cmd
from .sft import train_sft_cmd

# Backwards compatibility aliases
train_dpo = train_dpo_cmd
train_grpo = train_grpo_cmd
train_sft = train_sft_cmd

__all__ = [
    # Types
    "DataGenConfig",
    "DataGenResult",
    "DataGenType",
    "DPOConfig",
    "GRPOConfig",
    "SFTConfig",
    "TrainMode",
    "TrainResult",
    # SFT Commands
    "train_sft",
    "train_sft_cmd",
    # DPO Commands
    "train_dpo",
    "train_dpo_cmd",
    # GRPO Commands
    "train_grpo",
    "train_grpo_cmd",
    # Data Generation Commands
    "generate_data",
    "generate_data_cmd",
]
