"""
Dual-Reward Loss for Classifier Emergence

Combines:
1. Classification loss at intermediate layer (vocab-aligned classifier)
2. Answer loss at final layer (correct outputs)

This loss function trains V/O projections to create vocabulary-mappable
classifiers while maintaining answer quality.
"""

from dataclasses import dataclass, field

import mlx.core as mx


@dataclass
class DualRewardLossConfig:
    """Configuration for dual-reward loss."""

    # Intermediate classification loss
    classifier_layer: int = -1  # -1 means 55% depth
    classifier_weight: float = 0.4

    # Target tokens for classification (operation -> token_id)
    classifier_targets: dict[str, int] = field(default_factory=dict)

    # Whether to use softmax or direct logit for classification
    use_softmax: bool = True


def dual_reward_loss(
    final_logits: mx.array,
    classifier_logits: mx.array,
    labels: mx.array,
    classifier_labels: mx.array,
    loss_mask: mx.array,
    config: DualRewardLossConfig,
) -> tuple[mx.array, dict[str, mx.array]]:
    """
    Compute dual-reward loss combining classification and answer losses.

    Args:
        final_logits: Logits from final layer, shape (batch, seq_len, vocab_size)
        classifier_logits: Logits from classifier layer, shape (batch, seq_len, vocab_size)
        labels: Target answer token ids, shape (batch, seq_len)
        classifier_labels: Target classification token ids, shape (batch,)
        loss_mask: Mask for answer tokens, shape (batch, seq_len)
        config: Loss configuration

    Returns:
        total_loss: Combined loss
        metrics: Dict with individual losses and metrics
    """
    batch_size = final_logits.shape[0]
    vocab_size = final_logits.shape[-1]

    # === Answer Loss (final layer) ===
    # Standard cross-entropy on response tokens
    logits_flat = final_logits.reshape(-1, vocab_size)
    labels_flat = labels.reshape(-1)
    mask_flat = loss_mask.reshape(-1)

    log_probs = mx.log(mx.softmax(logits_flat, axis=-1) + 1e-10)
    indices = mx.arange(logits_flat.shape[0])
    token_log_probs = log_probs[indices, labels_flat]

    masked_log_probs = token_log_probs * mask_flat
    num_tokens = mx.sum(mask_flat) + 1e-10
    answer_loss = -mx.sum(masked_log_probs) / num_tokens

    # === Classification Loss (intermediate layer) ===
    # Cross-entropy on last token position for classification target
    # Use last token of each sequence
    cls_logits = classifier_logits[:, -1, :]  # (batch, vocab_size)

    if config.use_softmax:
        cls_probs = mx.softmax(cls_logits, axis=-1)
        cls_log_probs = mx.log(cls_probs + 1e-10)
    else:
        cls_log_probs = mx.log_softmax(cls_logits, axis=-1)

    # Gather log probs for classifier targets
    batch_indices = mx.arange(batch_size)
    cls_token_log_probs = cls_log_probs[batch_indices, classifier_labels]
    classifier_loss = -mx.mean(cls_token_log_probs)

    # === Combined Loss ===
    cls_weight = config.classifier_weight
    ans_weight = 1.0 - cls_weight

    total_loss = cls_weight * classifier_loss + ans_weight * answer_loss

    # Metrics
    answer_perplexity = mx.exp(answer_loss)

    # Classification accuracy (for logging)
    cls_predictions = mx.argmax(cls_logits, axis=-1)
    cls_correct = mx.sum(cls_predictions == classifier_labels)
    cls_accuracy = cls_correct / batch_size

    metrics = {
        "loss": total_loss,
        "answer_loss": answer_loss,
        "classifier_loss": classifier_loss,
        "answer_perplexity": answer_perplexity,
        "classifier_accuracy": cls_accuracy,
        "num_tokens": num_tokens,
    }

    return total_loss, metrics


def classification_only_loss(
    classifier_logits: mx.array,
    classifier_labels: mx.array,
) -> tuple[mx.array, dict[str, mx.array]]:
    """
    Compute classification-only loss (for probing/evaluation).

    Args:
        classifier_logits: Logits from classifier layer, shape (batch, vocab_size)
        classifier_labels: Target classification token ids, shape (batch,)

    Returns:
        loss: Classification loss
        metrics: Dict with accuracy, etc.
    """
    batch_size = classifier_logits.shape[0]

    cls_probs = mx.softmax(classifier_logits, axis=-1)
    cls_log_probs = mx.log(cls_probs + 1e-10)

    batch_indices = mx.arange(batch_size)
    cls_token_log_probs = cls_log_probs[batch_indices, classifier_labels]
    loss = -mx.mean(cls_token_log_probs)

    # Accuracy
    predictions = mx.argmax(classifier_logits, axis=-1)
    correct = mx.sum(predictions == classifier_labels)
    accuracy = correct / batch_size

    metrics = {
        "loss": loss,
        "accuracy": accuracy,
    }

    return loss, metrics
