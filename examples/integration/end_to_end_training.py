#!/usr/bin/env python3
"""
End-to-End Integration Example

This example demonstrates how all three modules work together:
- models_v2: Model loading, architecture, and configuration
- training: Trainers (SFT, DPO) and loss functions
- data: Datasets and tokenizers

Architecture Overview:
    ┌─────────────────┐
    │   models_v2     │  ← Pure model definitions, no training deps
    │  (CausalLM,     │
    │   Backbones,    │
    │   LoRA)         │
    └────────┬────────┘
             │ model + tokenizer
             ▼
    ┌─────────────────┐
    │    training     │  ← Trainers orchestrate model + data
    │  (SFTTrainer,   │
    │   DPOTrainer,   │
    │   losses)       │
    └────────┬────────┘
             │ uses datasets
             ▼
    ┌─────────────────┐
    │      data       │  ← Datasets, batching, tokenizers
    │  (SFTDataset,   │
    │   batching,     │
    │   tokenizers)   │
    └─────────────────┘

Usage:
    python examples/integration/end_to_end_training.py

This creates synthetic data, trains a tiny model, and runs inference.
"""

import json
import tempfile
from pathlib import Path

# === 1. Imports from all three modules ===
# models_v2: Model loading and architecture
# data: Datasets and tokenizers
from chuk_lazarus.data import (
    BoWCharacterTokenizer,
    ClassificationDataset,  # Protocol for type checking
)
from chuk_lazarus.models_v2 import (
    LoRAConfig,
)

# training: Trainers and loss functions
from chuk_lazarus.training import (
    ClassificationTrainer,
    ClassificationTrainerConfig,
)


def create_synthetic_sft_data(output_dir: Path, num_samples: int = 50) -> Path:
    """Create synthetic SFT training data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "sft_train.jsonl"

    samples = []
    for i in range(num_samples):
        a, b = i % 10 + 1, (i * 3) % 10 + 1
        samples.append(
            {
                "prompt": f"What is {a} + {b}?",
                "response": f"The sum of {a} and {b} is {a + b}.",
            }
        )

    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Created {num_samples} SFT samples at {output_path}")
    return output_path


def create_synthetic_classification_data(output_dir: Path, num_samples: int = 100) -> Path:
    """Create synthetic classification training data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "classification_train.jsonl"

    positive_words = ["great", "amazing", "wonderful", "excellent", "fantastic"]
    negative_words = ["bad", "terrible", "awful", "horrible", "poor"]

    samples = []
    for i in range(num_samples):
        if i % 2 == 0:
            word = positive_words[i % len(positive_words)]
            samples.append({"text": f"This is {word}!", "label": 1})
        else:
            word = negative_words[i % len(negative_words)]
            samples.append({"text": f"This is {word}!", "label": 0})

    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Created {num_samples} classification samples at {output_path}")
    return output_path


def demo_classification_pipeline(data_dir: Path):
    """
    Demonstrate the classification training pipeline.

    This shows data → model → training → evaluation flow.
    """
    print("\n" + "=" * 60)
    print("Classification Training Pipeline Demo")
    print("=" * 60)

    # 1. Create synthetic data
    data_path = create_synthetic_classification_data(data_dir)

    # 2. Load dataset
    dataset = ClassificationDataset.from_jsonl(data_path)
    train_dataset, eval_dataset = dataset.split(train_ratio=0.8, seed=42)
    print(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")

    # 3. Create tokenizer from corpus
    tokenizer = BoWCharacterTokenizer.from_corpus(dataset.texts)
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # 4. Create a simple classifier model
    import mlx.nn as nn

    class SimpleClassifier(nn.Module):
        """Simple bag-of-words classifier."""

        def __init__(self, vocab_size: int, num_classes: int):
            super().__init__()
            self.fc = nn.Linear(vocab_size, num_classes)

        def __call__(self, x):
            return self.fc(x)

    model = SimpleClassifier(vocab_size=tokenizer.vocab_size, num_classes=2)
    print(f"Model created with {sum(p.size for p in model.parameters().values())} parameters")

    # 5. Configure and create trainer
    config = ClassificationTrainerConfig(
        batch_size=8,
        learning_rate=0.01,
        max_steps=50,
        log_interval=10,
    )
    trainer = ClassificationTrainer(model=model, tokenizer=tokenizer, config=config)

    # 6. Train
    print("\nTraining...")
    trainer.train(train_dataset)

    # 7. Evaluate
    from chuk_lazarus.training import evaluate_classifier

    results = evaluate_classifier(
        model, tokenizer, eval_dataset, label_names=["negative", "positive"]
    )
    print(f"\nEvaluation accuracy: {results['accuracy']:.2%}")


def demo_model_architecture():
    """
    Demonstrate the models_v2 architecture.

    Shows how models are structured without training.
    """
    print("\n" + "=" * 60)
    print("Model Architecture Demo (models_v2)")
    print("=" * 60)

    # 1. Show model config structure
    print("\n1. Model Configuration:")
    print("   LlamaConfig fields: hidden_size, num_layers, vocab_size, etc.")

    # 2. Show model registry
    from chuk_lazarus.models_v2 import list_models

    registered = list_models()
    print(f"\n2. Registered model types: {registered['types']}")

    # 3. Show LoRA config
    print("\n3. LoRA Configuration:")
    lora_config = LoRAConfig(rank=8, alpha=16, target_modules=["q_proj", "v_proj"])
    print(f"   Rank: {lora_config.rank}, Alpha: {lora_config.alpha}")
    print(f"   Target modules: {lora_config.target_modules}")

    # 4. Show loss function (the only training-related thing in models_v2)
    print("\n4. Basic LM Loss (models_v2.losses.loss):")
    print("   compute_lm_loss(model, input_ids, labels, attention_mask)")
    print("   → Returns (loss, num_tokens)")


def demo_data_module():
    """
    Demonstrate the data module capabilities.

    Shows datasets, tokenizers, and batching.
    """
    print("\n" + "=" * 60)
    print("Data Module Demo")
    print("=" * 60)

    # 1. Dataset protocols
    print("\n1. Dataset Protocols:")
    print("   - Dataset[T]: Basic __len__, __getitem__, __iter__")
    print("   - BatchableDataset: Adds iter_batches()")
    print("   - SFTDatasetProtocol: SFT-specific interface")
    print("   - PreferenceDatasetProtocol: DPO-specific interface")

    # 2. Available tokenizers
    print("\n2. Tokenizers:")
    print("   - BoWCharacterTokenizer: Bag-of-words character tokenizer")
    print("   - CharacterTokenizer: Simple character tokenizer")
    print("   - CustomTokenizer: Configurable tokenizer")

    # 3. Batching infrastructure
    print("\n3. Batching Infrastructure:")
    print("   - TokenBudgetBatchSampler: Token-budget based batching")
    print("   - BatchWriter/BatchReader: Async batch I/O")
    print("   - pack_sequences: Sequence packing for efficiency")


def show_import_patterns():
    """Show the recommended import patterns."""
    print("\n" + "=" * 60)
    print("Recommended Import Patterns")
    print("=" * 60)

    print("""
# Top-level convenience imports
from chuk_lazarus import load_model, create_model

# Model architecture (pure, no training deps)
from chuk_lazarus.models_v2 import (
    CausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    LoRAConfig,
    apply_lora,
    compute_lm_loss,  # Basic loss only
)

# Training (orchestrates model + data)
from chuk_lazarus.training import (
    SFTTrainer,
    SFTConfig,
    DPOTrainer,
    DPOTrainerConfig,
    ClassificationTrainer,
    sft_loss,
    dpo_loss,
)

# Data (datasets, tokenizers, batching)
from chuk_lazarus.data import (
    SFTDataset,
    PreferenceDataset,
    ClassificationDataset,
    BoWCharacterTokenizer,
    BatchWriter,
    BatchReader,
)
""")


def main():
    """Run all demos."""
    print("=" * 60)
    print("CHUK-LAZARUS: End-to-End Integration Example")
    print("=" * 60)
    print("\nThis demonstrates how models_v2, training, and data work together.")

    # Create temp directory for demo data
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)

        # Run demos
        show_import_patterns()
        demo_model_architecture()
        demo_data_module()
        demo_classification_pipeline(data_dir)

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. models_v2 is PURE - no training or dataset dependencies")
    print("2. training uses models_v2 and data, orchestrating the training loop")
    print("3. data provides datasets, tokenizers, and batching infrastructure")
    print("4. Use protocols (SFTDatasetProtocol, etc.) for type-safe interfaces")


if __name__ == "__main__":
    main()
