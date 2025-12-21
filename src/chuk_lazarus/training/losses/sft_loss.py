"""
Supervised Fine-Tuning (SFT) Loss

Standard cross-entropy loss for language model training.
"""

from dataclasses import dataclass

import mlx.core as mx


@dataclass
class SFTConfig:
    """Configuration for SFT training."""

    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    max_seq_length: int = 512
    mask_prompt: bool = True
    log_interval: int = 10
    eval_interval: int = 100
    checkpoint_interval: int = 500
    checkpoint_dir: str = "./checkpoints/sft"
    max_steps: int | None = None
    min_loss: float | None = None


def sft_loss(
    logits: mx.array, labels: mx.array, loss_mask: mx.array
) -> tuple[mx.array, dict[str, mx.array]]:
    """
    Compute SFT cross-entropy loss.

    Args:
        logits: Model output, shape (batch, seq_len, vocab_size)
        labels: Target token ids, shape (batch, seq_len)
        loss_mask: Mask for valid tokens, shape (batch, seq_len)

    Returns:
        loss: Scalar loss
        metrics: Dict with token count, perplexity, etc.
    """
    batch_size, seq_len, vocab_size = logits.shape

    # Reshape for cross entropy
    logits_flat = logits.reshape(-1, vocab_size)
    labels_flat = labels.reshape(-1)
    mask_flat = loss_mask.reshape(-1)

    # Cross entropy per token
    log_probs = mx.log(mx.softmax(logits_flat, axis=-1) + 1e-10)

    # Gather log probs for labels
    indices = mx.arange(logits_flat.shape[0])
    token_log_probs = log_probs[indices, labels_flat]

    # Apply mask and compute mean loss
    masked_log_probs = token_log_probs * mask_flat
    num_tokens = mx.sum(mask_flat) + 1e-10
    loss = -mx.sum(masked_log_probs) / num_tokens

    # Metrics
    perplexity = mx.exp(loss)

    metrics = {
        "loss": loss,
        "perplexity": perplexity,
        "num_tokens": num_tokens,
    }

    return loss, metrics
