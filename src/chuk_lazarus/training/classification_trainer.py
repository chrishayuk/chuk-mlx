"""
Classification Trainer.

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


class EncoderProtocol(Protocol):
    """
    Protocol for feature encoders.

    Converts sample text/data into a fixed-size feature vector.
    Examples:
    - BoWCharacterTokenizer: text -> character frequency vector
    - FeatureEncoder: "72,85" -> [0.72, 0.85] (numeric features)
    """

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
    Trainer for classification tasks.

    Example (text classification):
        >>> dataset = ClassificationDataset.from_jsonl("train.jsonl")
        >>> encoder = BoWCharacterTokenizer.from_corpus(dataset.texts)
        >>> model = LinearClassifier(input_size=encoder.vocab_size, num_labels=2)
        >>> trainer = ClassificationTrainer(model, encoder, config)
        >>> trainer.train(dataset, num_epochs=10)

    Example (numeric features - no encoder needed):
        >>> dataset = ClassificationDataset.from_features(X, y)
        >>> model = LinearClassifier(input_size=2, num_labels=2)
        >>> trainer = ClassificationTrainer(model, None, config)
        >>> trainer.train(dataset, num_epochs=10)
    """

    def __init__(
        self,
        model: nn.Module,
        encoder: EncoderProtocol | None,
        config: ClassificationTrainerConfig,
    ):
        super().__init__(model, encoder, config)
        self.config: ClassificationTrainerConfig = config
        self.encoder = encoder

    def compute_loss(self, batch: dict[str, Any]) -> tuple[mx.array, dict[str, Any]]:
        """Compute cross-entropy loss and accuracy."""
        x = batch["input"]
        y = batch["label"]

        logits = self.model(x)
        loss = mx.mean(nn.losses.cross_entropy(logits, y))

        predictions = mx.argmax(logits, axis=-1)
        accuracy = mx.mean(predictions == y)

        return loss, {"loss": loss, "accuracy": accuracy}

    def _get_features(self, sample) -> list[float]:
        """Extract features from a sample."""
        if sample.features is not None:
            return sample.features
        if self.encoder is not None and sample.text is not None:
            return self.encoder.encode(sample.text)
        raise ValueError("Sample must have features or text (with encoder)")

    def get_train_batches(self, dataset: ClassificationDataset) -> Iterator[dict[str, Any]]:
        """Create batches from ClassificationDataset."""
        batch_size = self.config.batch_size

        indices = list(range(len(dataset)))
        random.shuffle(indices)

        for i in range(0, len(dataset), batch_size):
            batch_indices = indices[i : i + batch_size]
            batch_samples = [dataset[j] for j in batch_indices]

            inputs = mx.stack([mx.array(self._get_features(s)) for s in batch_samples])
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


def evaluate(
    model: nn.Module,
    dataset: ClassificationDataset,
    encoder: EncoderProtocol | None = None,
) -> dict[str, float]:
    """
    Evaluate a classification model.

    Args:
        model: The classifier model.
        dataset: ClassificationDataset to evaluate on.
        encoder: Optional encoder (only needed for text classification).

    Returns:
        Dictionary with accuracy.
    """

    def get_features(sample) -> list[float]:
        if sample.features is not None:
            return sample.features
        if encoder is not None and sample.text is not None:
            return encoder.encode(sample.text)
        raise ValueError("Sample must have features or text (with encoder)")

    inputs = mx.stack([mx.array(get_features(s)) for s in dataset])
    labels = mx.array([s.label for s in dataset])

    logits = model(inputs)
    predictions = mx.argmax(logits, axis=-1)
    accuracy = float(mx.mean(predictions == labels))

    return {"accuracy": accuracy}


def predict(model: nn.Module, features: list[float]) -> tuple[int, float]:
    """
    Make a prediction for a single sample.

    Args:
        model: Trained classifier.
        features: Feature vector (already normalized).

    Returns:
        Tuple of (predicted_class, confidence).
    """
    x = mx.array([features])
    logits = model(x)
    probs = mx.softmax(logits, axis=-1)
    pred = int(mx.argmax(logits, axis=-1)[0])
    return pred, float(probs[0, pred])


def train_classifier(
    model: nn.Module,
    dataset,
    epochs: int = 10,
    lr: float = 0.01,
    batch_size: int = 32,
) -> None:
    """
    Simple training helper for classification.

    Args:
        model: Classifier model.
        dataset: Training dataset with features.
        epochs: Number of epochs.
        lr: Learning rate.
        batch_size: Batch size.
    """
    config = ClassificationTrainerConfig(
        batch_size=batch_size,
        learning_rate=lr,
        log_interval=max(1, len(dataset) // batch_size),  # Log once per epoch
        checkpoint_interval=999999,  # Don't save checkpoints
    )
    trainer = ClassificationTrainer(model, None, config)
    trainer.train(dataset=dataset, num_epochs=epochs)
