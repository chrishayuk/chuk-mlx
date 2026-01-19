"""
Vocabulary Transfer Experiment

Hypothesis: The model fails because Collatz vocabulary is unseen, not because
of structural differences.

Test: Add Collatz-style NL descriptions to training (but using sum_even as
the underlying program). If the model then generalizes to actual Collatz,
it proves the issue is vocabulary, not structure.

This is a diagnostic experiment to understand the failure mode.
"""

import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from templates import TemplateID, NUM_TEMPLATES, template_name


# =============================================================================
# SYNTHETIC COLLATZ-VOCABULARY EXAMPLES
# =============================================================================

def generate_collatz_vocabulary_examples(n: int = 200) -> list[dict]:
    """
    Generate training examples that use Collatz vocabulary but are
    labeled as LOOP_CONDITIONAL_ACCUMULATE.

    These are "fake Collatz" - they use Collatz words but we train
    the model to recognize them as the same template as sum_even.
    """
    templates = [
        "Collatz-like computation for {0}",
        "Iterative steps starting at {0}",
        "Sequence length from {0}",
        "Count iterations for {0}",
        "Steps to converge from {0}",
        "Iterations until termination for {0}",
        "Process {0} until done",
        "How many steps for {0}?",
        "Length of sequence starting at {0}",
        "Convergence steps for {0}",
    ]

    examples = []
    for _ in range(n):
        num = random.randint(10, 1000)
        template = random.choice(templates)
        nl = template.format(num)

        # Label as LOOP_CONDITIONAL_ACCUMULATE (same as sum_even and actual collatz)
        examples.append({
            "nl_input": nl,
            "program_name": "vocabulary_transfer",
            "template_id": int(TemplateID.LOOP_CONDITIONAL_ACCUMULATE),
            "template_name": "LOOP_CONDITIONAL_ACCUMULATE",
            "operands": [num],
            "expected_result": 0,  # Not used for classification
        })

    return examples


# =============================================================================
# MODEL
# =============================================================================

class TemplateClassifier(nn.Module):
    def __init__(self, hidden_dim: int, num_templates: int = NUM_TEMPLATES, hidden_size: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_templates)
        self.dropout = nn.Dropout(0.1)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.gelu(self.fc1(x))
        x = self.dropout(x)
        x = nn.gelu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


# =============================================================================
# UTILITIES
# =============================================================================

@dataclass
class Config:
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 20
    hidden_layer: int = 13


def load_dataset(path: Path) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data["examples"]


def get_hidden_state(model, tokenizer, text: str, layer: int) -> mx.array:
    tokens = tokenizer.encode(text)
    input_ids = mx.array([tokens])
    hidden = model.model.embed_tokens(input_ids)

    for i, block in enumerate(model.model.layers[:layer]):
        hidden = block(hidden, mask=None, cache=None)
        if hasattr(hidden, 'hidden_states'):
            hidden = hidden.hidden_states
        elif isinstance(hidden, tuple):
            hidden = hidden[0]

    return hidden[0, -1, :]


def train_classifier(model, tokenizer, train_examples, config):
    import random

    hidden_dim = model.model.layers[0].self_attn.q_proj.weight.shape[0]
    classifier = TemplateClassifier(hidden_dim=hidden_dim)
    optimizer = optim.Adam(learning_rate=config.learning_rate)

    def loss_fn(classifier, hidden_states, labels):
        logits = classifier(hidden_states)
        return nn.losses.cross_entropy(logits, labels).mean()

    loss_and_grad = nn.value_and_grad(classifier, loss_fn)

    for epoch in range(config.epochs):
        random.shuffle(train_examples)

        for i in range(0, len(train_examples), config.batch_size):
            batch = train_examples[i:i + config.batch_size]

            hidden_states = []
            labels = []

            for ex in batch:
                hidden = get_hidden_state(model, tokenizer, ex["nl_input"], config.hidden_layer)
                hidden_states.append(hidden)
                labels.append(ex["template_id"])

            hidden_states = mx.stack(hidden_states)
            labels = mx.array(labels)

            loss, grads = loss_and_grad(classifier, hidden_states, labels)
            optimizer.update(classifier, grads)
            mx.eval(classifier.parameters(), optimizer.state)

    return classifier


def evaluate(classifier, model, tokenizer, examples, config):
    correct = 0
    total = 0
    by_pred = {}

    for ex in examples:
        total += 1
        hidden = get_hidden_state(model, tokenizer, ex["nl_input"], config.hidden_layer)
        logits = classifier(hidden[None, :])
        pred = int(mx.argmax(logits[0]).item())
        true = ex["template_id"]

        if pred == true:
            correct += 1

        pred_name = template_name(TemplateID(pred))
        by_pred[pred_name] = by_pred.get(pred_name, 0) + 1

    return {
        "accuracy": correct / total if total > 0 else 0,
        "correct": correct,
        "total": total,
        "predictions": by_pred,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("Vocabulary Transfer Experiment")
    print("=" * 60)

    # Load model
    print("\n1. Loading base model...")
    from chuk_lazarus.models_v2.loader import load_model

    loaded = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = loaded.model
    tokenizer = loaded.tokenizer
    model.freeze()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    mx.eval(model.parameters())
    print("   Model loaded.")

    # Load datasets
    print("\n2. Loading datasets...")
    results_dir = Path(__file__).parent / "results"
    train_examples = load_dataset(results_dir / "train_dataset.json")
    test_examples = load_dataset(results_dir / "test_dataset.json")

    # Generate vocabulary transfer examples
    print("\n3. Generating vocabulary transfer examples...")
    vocab_transfer = generate_collatz_vocabulary_examples(200)
    print(f"   Generated {len(vocab_transfer)} vocabulary transfer examples")
    print("   Sample:")
    for ex in vocab_transfer[:3]:
        print(f"     '{ex['nl_input']}' → {ex['template_name']}")

    # Experiment 1: Baseline (no vocabulary transfer)
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Baseline (no vocabulary transfer)")
    print("=" * 60)

    config = Config()
    classifier1 = train_classifier(model, tokenizer, train_examples, config)

    train_eval1 = evaluate(classifier1, model, tokenizer, train_examples[:200], config)
    test_eval1 = evaluate(classifier1, model, tokenizer, test_examples, config)

    print(f"\n   Train accuracy: {train_eval1['accuracy']:.1%}")
    print(f"   Test accuracy (Collatz): {test_eval1['accuracy']:.1%}")
    print(f"   Test predictions: {test_eval1['predictions']}")

    # Experiment 2: With vocabulary transfer
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: With vocabulary transfer")
    print("=" * 60)

    # Add vocabulary transfer examples to training
    train_with_vocab = train_examples + vocab_transfer
    random.shuffle(train_with_vocab)

    print(f"   Training with {len(train_with_vocab)} examples")
    print(f"   (Original {len(train_examples)} + {len(vocab_transfer)} vocab transfer)")

    classifier2 = train_classifier(model, tokenizer, train_with_vocab, config)

    train_eval2 = evaluate(classifier2, model, tokenizer, train_examples[:200], config)
    vocab_eval2 = evaluate(classifier2, model, tokenizer, vocab_transfer[:50], config)
    test_eval2 = evaluate(classifier2, model, tokenizer, test_examples, config)

    print(f"\n   Train accuracy: {train_eval2['accuracy']:.1%}")
    print(f"   Vocab transfer accuracy: {vocab_eval2['accuracy']:.1%}")
    print(f"   Test accuracy (Collatz): {test_eval2['accuracy']:.1%}")
    print(f"   Test predictions: {test_eval2['predictions']}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"""
| Experiment              | Train Acc | Vocab Transfer | Collatz Test |
|-------------------------|-----------|----------------|--------------|
| Baseline (no transfer)  | {train_eval1['accuracy']:>9.1%} | N/A            | {test_eval1['accuracy']:>12.1%} |
| With vocab transfer     | {train_eval2['accuracy']:>9.1%} | {vocab_eval2['accuracy']:>14.1%} | {test_eval2['accuracy']:>12.1%} |

Interpretation:
  - If vocab transfer improves Collatz test accuracy:
    The issue is vocabulary/semantics, not structure.

  - If vocab transfer doesn't help:
    The hidden states don't encode useful features for
    cross-vocabulary generalization.
""")
    print("=" * 60)


if __name__ == "__main__":
    main()
