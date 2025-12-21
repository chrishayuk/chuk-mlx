"""
Training infrastructure for chuk-lazarus.

This module provides:
- Loss functions (DPO, GRPO, PPO, SFT)
- Trainer classes for different training paradigms
- Utility functions for RL training
- Learning rate schedulers
"""

# Loss functions
from .losses import (
    DPOConfig,
    dpo_loss,
    create_dpo_loss_fn,
    GRPOConfig,
    GRPOBatch,
    grpo_loss,
    compute_grpo_advantages,
    PPOConfig,
    ppo_loss,
    compute_ppo_loss_for_batch,
    sft_loss,
    SFTConfig,
)

# Base Trainer
from .base_trainer import BaseTrainer, BaseTrainerConfig

# Trainers
from .trainers import (
    SFTTrainer,
    SFTConfig,
    DPOTrainer,
    DPOTrainerConfig,
    GRPOTrainer,
    GRPOTrainerConfig,
    PPOTrainer,
    PPOTrainerConfig,
)

# Learning rate schedulers
from .schedulers import schedule_learning_rate

# Utilities
from .utils import (
    extract_log_probs,
    compute_log_probs_from_logits,
    compute_sequence_log_prob,
    compute_kl_divergence,
    compute_approx_kl,
    compute_gae,
    compute_returns,
    normalize_advantages,
)
