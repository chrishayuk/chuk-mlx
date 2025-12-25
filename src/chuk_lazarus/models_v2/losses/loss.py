"""
Loss functions for language model training.
"""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


def compute_lm_loss(
    model: Any,
    input_ids: mx.array,
    labels: mx.array,
    attention_mask: mx.array | None = None,
) -> tuple[mx.array, int]:
    """
    Compute language modeling loss with optional attention mask.

    Args:
        model: Language model (CausalLM or similar)
        input_ids: Input token IDs, shape (batch, seq_len)
        labels: Target token IDs, shape (batch, seq_len)
        attention_mask: Optional mask, shape (batch, seq_len). 1 for valid tokens.

    Returns:
        Tuple of (loss, num_tokens)

    Example:
        >>> loss, ntoks = compute_lm_loss(model, input_ids, labels, attention_mask)
        >>> loss.backward()
    """
    try:
        # Forward pass
        output = model(input_ids, labels=labels)

        if output.loss is not None:
            # Model computed loss internally
            loss = output.loss
            if attention_mask is not None:
                # Apply mask if provided
                ntoks = mx.maximum(attention_mask.sum(), 1)
            else:
                ntoks = labels.size
            return loss, int(ntoks)

        # Compute loss from logits
        logits = output.logits

        # Cross-entropy loss
        ce = nn.losses.cross_entropy(logits, labels)

        if attention_mask is not None:
            # Apply the attention mask
            ce = ce * attention_mask
            ntoks = mx.maximum(attention_mask.sum(), 1)
        else:
            ntoks = mx.array(labels.size)

        # Average loss
        loss = ce.sum() / ntoks

        return loss, int(ntoks)

    except Exception as e:
        logger.error(f"Error during loss computation: {e}")
        return mx.array(0.0), 1


def compute_cross_entropy_loss(
    logits: mx.array,
    labels: mx.array,
    attention_mask: mx.array | None = None,
    label_smoothing: float = 0.0,
) -> tuple[mx.array, int]:
    """
    Compute cross-entropy loss from logits.

    Args:
        logits: Model logits, shape (batch, seq_len, vocab_size)
        labels: Target token IDs, shape (batch, seq_len)
        attention_mask: Optional mask, shape (batch, seq_len)
        label_smoothing: Label smoothing factor (0.0 = no smoothing)

    Returns:
        Tuple of (loss, num_tokens)
    """
    # Cross-entropy loss
    ce = nn.losses.cross_entropy(logits, labels, label_smoothing=label_smoothing)

    if attention_mask is not None:
        ce = ce * attention_mask
        ntoks = mx.maximum(attention_mask.sum(), 1)
    else:
        ntoks = mx.array(labels.size)

    loss = ce.sum() / ntoks

    return loss, int(ntoks)
