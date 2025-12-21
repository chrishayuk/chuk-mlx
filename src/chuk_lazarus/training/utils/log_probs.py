"""
Log probability extraction for RL training.

This is the foundation for all RL methods - we need to compute
the probability of actions under the policy.
"""

import mlx.core as mx
import mlx.nn as nn


def compute_log_probs_from_logits(logits: mx.array, actions: mx.array) -> mx.array:
    """
    Compute log probabilities of actions given logits.

    Args:
        logits: Shape (batch, seq_len, vocab_size) - raw model outputs
        actions: Shape (batch, seq_len) - token ids that were selected

    Returns:
        log_probs: Shape (batch, seq_len) - log probability of each action
    """
    # Log softmax for numerical stability
    log_probs_all = mx.log(mx.softmax(logits, axis=-1) + 1e-10)

    # Gather the log probs for the actual actions taken
    # actions shape: (batch, seq_len) -> need to index into vocab dimension
    batch_size, seq_len, vocab_size = logits.shape

    # Flatten for gathering
    logits_flat = log_probs_all.reshape(-1, vocab_size)  # (batch*seq, vocab)
    actions_flat = actions.reshape(-1)  # (batch*seq,)

    # Gather log probs for selected actions
    # MLX doesn't have gather, so we use indexing
    indices = mx.arange(batch_size * seq_len)
    log_probs = logits_flat[indices, actions_flat]

    # Reshape back
    log_probs = log_probs.reshape(batch_size, seq_len)

    return log_probs


def extract_log_probs(
    model: nn.Module, input_ids: mx.array, attention_mask: mx.array = None
) -> tuple[mx.array, mx.array]:
    """
    Extract log probabilities from a model for given inputs.

    This runs a forward pass and computes log probs for the next-token
    predictions (shifted by 1 from inputs).

    Args:
        model: The language model
        input_ids: Shape (batch, seq_len) - input token ids
        attention_mask: Shape (batch, seq_len) - optional attention mask

    Returns:
        log_probs: Shape (batch, seq_len-1) - log prob of each predicted token
        logits: Shape (batch, seq_len-1, vocab_size) - raw logits
    """
    # Forward pass (no cache for training)
    logits, _ = model(input_ids, cache=None)

    # Shift: logits[t] predicts token[t+1]
    # So we compare logits[:-1] with input_ids[1:]
    shifted_logits = logits[:, :-1, :]  # (batch, seq_len-1, vocab)
    targets = input_ids[:, 1:]  # (batch, seq_len-1)

    log_probs = compute_log_probs_from_logits(shifted_logits, targets)

    # Apply attention mask if provided (mask out padding)
    if attention_mask is not None:
        # Shift mask to match
        shifted_mask = attention_mask[:, 1:]
        log_probs = log_probs * shifted_mask

    return log_probs, shifted_logits


def compute_sequence_log_prob(log_probs: mx.array, attention_mask: mx.array = None) -> mx.array:
    """
    Sum log probs across sequence to get total sequence log probability.

    Args:
        log_probs: Shape (batch, seq_len) - per-token log probs
        attention_mask: Shape (batch, seq_len) - mask for valid tokens

    Returns:
        sequence_log_probs: Shape (batch,) - total log prob per sequence
    """
    if attention_mask is not None:
        # Only sum over valid tokens
        return mx.sum(log_probs * attention_mask, axis=-1)
    else:
        return mx.sum(log_probs, axis=-1)
