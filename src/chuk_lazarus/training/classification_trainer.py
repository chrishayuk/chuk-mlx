"""
Classification Trainer for text classification tasks.

Provides a ready-to-use trainer for classification with:
- Automatic batching from ClassificationDataset
- Cross-entropy loss with accuracy tracking
- Evaluation utilities
"""

from __future__ import annotations

import logging
import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import mlx.core as mx
import mlx.nn as nn

from chuk_lazarus.training.base_trainer import BaseTrainer, BaseTrainerConfig

if TYPE_CHECKING:
    from chuk_lazarus.data import ClassificationDataset

logger = logging.getLogger(__name__)


class BoWTokenizerProtocol(Protocol):
    """Protocol for bag-of-words tokenizers."""

    def encode(self, text: str) -> list[float]: ...

    @property
    def vocab_size(self) -> int: ...


@dataclass
class ClassificationTrainerConfig(BaseTrainerConfig):
    """Configuration for classification trainer."""

    batch_size: int = 32
    learning_rate: float = 0.01
    log_interval: int = 10
    checkpoint_interval: int = 100
    label_names: list[str] = field(default_factory=list)


class ClassificationTrainer(BaseTrainer):
    """
    Trainer for text classification tasks.

    Works with ClassificationDataset and any tokenizer implementing
    the BoWTokenizerProtocol (encode() -> list[float]).

    Example:
        >>> from chuk_lazarus.data import ClassificationDataset
        >>> from chuk_lazarus.data.tokenizers import BoWCharacterTokenizer
        >>> from chuk_lazarus.training import ClassificationTrainer, ClassificationTrainerConfig
        >>>
        >>> dataset = ClassificationDataset.from_jsonl("train.jsonl")
        >>> tokenizer = BoWCharacterTokenizer.from_corpus(dataset.texts)
        >>> model = MyClassifier(vocab_size=tokenizer.vocab_size, num_classes=2)
        >>>
        >>> config = ClassificationTrainerConfig(batch_size=32, max_steps=100)
        >>> trainer = ClassificationTrainer(model, tokenizer, config)
        >>> trainer.train(dataset, num_epochs=10)
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: BoWTokenizerProtocol,
        config: ClassificationTrainerConfig,
    ):
        super().__init__(model, tokenizer, config)
        self.config: ClassificationTrainerConfig = config

    def compute_loss(self, batch: dict[str, Any]) -> tuple[mx.array, dict[str, Any]]:
        """Compute cross-entropy loss and accuracy."""
        x = batch["input"]
        y = batch["label"]

        logits = self.model(x)
        loss = mx.mean(nn.losses.cross_entropy(logits, y))

        predictions = mx.argmax(logits, axis=-1)
        accuracy = mx.mean(predictions == y)

        return loss, {"loss": loss, "accuracy": accuracy}

    def get_train_batches(self, dataset: ClassificationDataset) -> Iterator[dict[str, Any]]:
        """Create batches from ClassificationDataset."""
        batch_size = self.config.batch_size

        indices = list(range(len(dataset)))
        random.shuffle(indices)

        for i in range(0, len(dataset), batch_size):
            batch_indices = indices[i : i + batch_size]
            batch_samples = [dataset[j] for j in batch_indices]

            inputs = mx.stack([mx.array(self.tokenizer.encode(s.text)) for s in batch_samples])
            labels = mx.array([s.label for s in batch_samples])

            yield {"input": inputs, "label": labels}

    def _create_epoch_metrics(self) -> dict[str, list[float]]:
        """Track loss and accuracy."""
        return {"loss": [], "accuracy": []}

    def _log_metrics(self, metrics: dict[str, float]):
        """Log with accuracy."""
        loss = metrics.get("loss", 0)
        acc = metrics.get("accuracy", 0)
        logger.info(f"Step {self.global_step:3d} | Loss: {loss:.4f} | Acc: {acc:.2%}")


def evaluate_classifier(
    model: nn.Module,
    tokenizer: BoWTokenizerProtocol,
    dataset: ClassificationDataset,
    label_names: list[str] | None = None,
) -> dict[str, Any]:
    """
    Evaluate a classification model.

    Args:
        model: The classifier model.
        tokenizer: Tokenizer with encode() method.
        dataset: ClassificationDataset to evaluate on.
        label_names: Optional names for labels (e.g., ["negative", "positive"]).

    Returns:
        Dictionary with:
        - accuracy: Overall accuracy
        - predictions: List of (text, true_label, pred_label, confidence) tuples
    """
    inputs = mx.stack([mx.array(tokenizer.encode(s.text)) for s in dataset])
    labels = mx.array([s.label for s in dataset])

    logits = model(inputs)
    probs = mx.softmax(logits, axis=-1)
    predictions = mx.argmax(logits, axis=-1)

    accuracy = float(mx.mean(predictions == labels))

    results = []
    for i, sample in enumerate(dataset):
        pred = int(predictions[i])
        conf = float(probs[i, pred])
        pred_name = label_names[pred] if label_names else str(pred)
        results.append(
            {
                "text": sample.text,
                "true_label": sample.label,
                "pred_label": pred,
                "pred_name": pred_name,
                "confidence": conf,
                "correct": sample.label == pred,
            }
        )

    return {
        "accuracy": accuracy,
        "predictions": results,
    }
