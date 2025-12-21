"""
Direct Preference Optimization (DPO) Loss

DPO is a simpler alternative to RLHF/PPO that directly optimizes the policy
using preference pairs without requiring a separate reward model.

Key insight: The optimal policy under a KL-constrained reward maximization
objective can be expressed in closed form, which lets us derive a simple
classification-style loss.

Paper: https://arxiv.org/abs/2305.18290
"""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from ..utils.log_probs import compute_sequence_log_prob, extract_log_probs


@dataclass
class DPOConfig:
    """Configuration for DPO training."""

    beta: float = 0.1  # KL penalty coefficient (higher = stay closer to reference)
    label_smoothing: float = 0.0  # Optional label smoothing
    reference_free: bool = False  # If True, skip reference model (simpler but less stable)


def dpo_loss(
    policy_model: nn.Module,
    reference_model: nn.Module,
    chosen_input_ids: mx.array,
    rejected_input_ids: mx.array,
    chosen_attention_mask: mx.array = None,
    rejected_attention_mask: mx.array = None,
    config: DPOConfig = None,
) -> tuple[mx.array, dict]:
    """
    Compute DPO loss for a batch of preference pairs.

    The DPO loss is:
        L = -log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))

    where log_ratio = log(pi(y|x)) - log(pi_ref(y|x))

    Args:
        policy_model: The model being trained
        reference_model: Frozen reference model (usually the SFT model)
        chosen_input_ids: (batch, seq_len) - preferred completions
        rejected_input_ids: (batch, seq_len) - rejected completions
        chosen_attention_mask: Optional mask for chosen sequences
        rejected_attention_mask: Optional mask for rejected sequences
        config: DPO configuration

    Returns:
        loss: Scalar loss value
        metrics: Dict with debugging info (log_probs, rewards, etc.)
    """
    if config is None:
        config = DPOConfig()

    # Get log probs from policy model
    policy_chosen_log_probs, _ = extract_log_probs(
        policy_model, chosen_input_ids, chosen_attention_mask
    )
    policy_rejected_log_probs, _ = extract_log_probs(
        policy_model, rejected_input_ids, rejected_attention_mask
    )

    # Sum to get sequence-level log probs
    # Shift masks to match log_probs shape (seq_len - 1)
    chosen_mask_shifted = (
        chosen_attention_mask[:, 1:] if chosen_attention_mask is not None else None
    )
    rejected_mask_shifted = (
        rejected_attention_mask[:, 1:] if rejected_attention_mask is not None else None
    )

    policy_chosen_seq = compute_sequence_log_prob(policy_chosen_log_probs, chosen_mask_shifted)
    policy_rejected_seq = compute_sequence_log_prob(
        policy_rejected_log_probs, rejected_mask_shifted
    )

    if config.reference_free:
        # Simpler version: no reference model
        ref_chosen_seq = mx.zeros_like(policy_chosen_seq)
        ref_rejected_seq = mx.zeros_like(policy_rejected_seq)
    else:
        # Get log probs from reference model (no gradients needed)
        ref_chosen_log_probs, _ = extract_log_probs(
            reference_model, chosen_input_ids, chosen_attention_mask
        )
        ref_rejected_log_probs, _ = extract_log_probs(
            reference_model, rejected_input_ids, rejected_attention_mask
        )
        ref_chosen_seq = compute_sequence_log_prob(ref_chosen_log_probs, chosen_mask_shifted)
        ref_rejected_seq = compute_sequence_log_prob(ref_rejected_log_probs, rejected_mask_shifted)

    # Compute log ratios (implicit rewards in DPO)
    chosen_log_ratio = policy_chosen_seq - ref_chosen_seq
    rejected_log_ratio = policy_rejected_seq - ref_rejected_seq

    # DPO loss: negative log sigmoid of scaled preference
    logits = config.beta * (chosen_log_ratio - rejected_log_ratio)

    if config.label_smoothing > 0:
        # Soft labels: slightly prefer chosen but not absolutely
        loss = -config.label_smoothing * mx.log(mx.sigmoid(-logits) + 1e-10) - (
            1 - config.label_smoothing
        ) * mx.log(mx.sigmoid(logits) + 1e-10)
    else:
        # Standard DPO loss
        loss = -mx.log(mx.sigmoid(logits) + 1e-10)

    loss = mx.mean(loss)

    # Compute metrics for monitoring
    metrics = {
        "loss": loss,
        "chosen_reward": mx.mean(config.beta * chosen_log_ratio),
        "rejected_reward": mx.mean(config.beta * rejected_log_ratio),
        "reward_margin": mx.mean(config.beta * (chosen_log_ratio - rejected_log_ratio)),
        "accuracy": mx.mean((chosen_log_ratio > rejected_log_ratio).astype(mx.float32)),
    }

    return loss, metrics


def create_dpo_loss_fn(
    policy_model: nn.Module, reference_model: nn.Module, config: DPOConfig = None
):
    """
    Create a loss function suitable for use with MLX's value_and_grad.

    Returns a function that takes a batch dict and returns (loss, metrics).
    """
    if config is None:
        config = DPOConfig()

    def loss_fn(batch: dict) -> tuple[mx.array, dict]:
        return dpo_loss(
            policy_model=policy_model,
            reference_model=reference_model,
            chosen_input_ids=batch["chosen_input_ids"],
            rejected_input_ids=batch["rejected_input_ids"],
            chosen_attention_mask=batch.get("chosen_attention_mask"),
            rejected_attention_mask=batch.get("rejected_attention_mask"),
            config=config,
        )

    return loss_fn
