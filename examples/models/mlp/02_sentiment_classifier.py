"""
Sentiment Classification with Bag-of-Words MLP

Demonstrates training a simple MLP for sentiment classification using:
- ClassificationDataset and BoWCharacterTokenizer from the data module
- ClassificationTrainer from the training module
- MLP components from models_v2

Run with:
    uv run python examples/models/mlp/02_sentiment_classifier.py
"""

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from chuk_lazarus.data import ClassificationDataset
from chuk_lazarus.data.tokenizers import BoWCharacterTokenizer
from chuk_lazarus.models_v2.components.ffn import MLP
from chuk_lazarus.models_v2.components.normalization import RMSNorm
from chuk_lazarus.models_v2.core.config import FFNConfig
from chuk_lazarus.models_v2.core.enums import ActivationType
from chuk_lazarus.training import (
    ClassificationTrainer,
    ClassificationTrainerConfig,
    evaluate_classifier,
)


class SentimentMLP(nn.Module):
    """Simple MLP for sentiment classification."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 64,
        intermediate_size: int = 128,
        num_classes: int = 2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(vocab_size, hidden_size)
        self.mlp = MLP(FFNConfig(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation=ActivationType.GELU,
        ))
        self.norm = RMSNorm(dims=hidden_size, eps=1e-5)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.input_proj(x)
        x = self.mlp(x)
        x = self.norm(x)
        return self.classifier(x)


def main():
    # Load data
    data_dir = Path(__file__).parent / "data"
    train_data = ClassificationDataset.from_jsonl(data_dir / "train.jsonl")
    test_data = ClassificationDataset.from_jsonl(data_dir / "test.jsonl")
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")

    # Build tokenizer from corpus
    tokenizer = BoWCharacterTokenizer.from_corpus(train_data.texts)
    print(f"Vocab size: {tokenizer.vocab_size}")

    # Create model
    model = SentimentMLP(vocab_size=tokenizer.vocab_size)

    # Train
    config = ClassificationTrainerConfig(
        learning_rate=0.05,
        batch_size=30,
        log_interval=10,
        max_steps=100,
    )
    trainer = ClassificationTrainer(model, tokenizer, config)
    trainer.train(train_data, num_epochs=100)

    # Evaluate
    results = evaluate_classifier(
        model, tokenizer, test_data,
        label_names=["negative", "positive"],
    )
    print(f"\nTest accuracy: {results['accuracy']:.2%}")
    for p in results["predictions"]:
        mark = "V" if p["correct"] else "X"
        print(f"  {mark} '{p['text']}' -> {p['pred_name']} ({p['confidence']:.0%})")


if __name__ == "__main__":
    main()
