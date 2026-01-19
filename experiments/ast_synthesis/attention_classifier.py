"""
Attention-Based Template Classifier

Tests whether attention features can enable vocabulary-agnostic classification.

Two approaches:
1. Attention features classifier: Use computed attention features
2. Combined classifier: Use both hidden states and attention features

Key test: Can we achieve >0% on Collatz WITHOUT vocabulary transfer?
"""

import json
import random
import math
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from templates import TemplateID, NUM_TEMPLATES, template_name


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_attention_features(model, tokenizer, text: str, layer: int = 13) -> mx.array:
    """Extract attention pattern features from a single layer."""
    tokens = tokenizer.encode(text)
    input_ids = mx.array([tokens])
    seq_len = len(tokens)

    # Forward through embedding
    hidden = model.model.embed_tokens(input_ids)

    # Create mask
    mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
    mask = mask.astype(hidden.dtype)

    # Forward to target layer
    for i, block in enumerate(model.model.layers[:layer]):
        if i == layer - 1:
            # Extract attention at this layer
            attn = block.self_attn

            q = attn.q_proj(hidden)
            k = attn.k_proj(hidden)

            num_heads = attn.num_heads
            num_kv_heads = attn.num_kv_heads
            head_dim = q.shape[-1] // num_heads

            q = q.reshape(1, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
            kv_head_dim = k.shape[-1] // num_kv_heads
            k = k.reshape(1, seq_len, num_kv_heads, kv_head_dim).transpose(0, 2, 1, 3)

            repeat_factor = num_heads // num_kv_heads
            k = mx.repeat(k, repeat_factor, axis=1)

            scale = math.sqrt(head_dim)
            scores = (q @ k.transpose(0, 1, 3, 2)) / scale
            scores = scores + mask
            attn_weights = mx.softmax(scores, axis=-1)[0]  # (num_heads, seq, seq)

            # Compute features
            features = []
            for h in range(num_heads):
                head_attn = attn_weights[h]
                eps = 1e-10

                # Entropy per position
                entropy = -mx.sum(head_attn * mx.log(head_attn + eps), axis=-1)
                features.append(mx.mean(entropy).item())

                # Attention to first token
                features.append(mx.mean(head_attn[:, 0]).item())

                # Attention to last token
                features.append(mx.mean(head_attn[:, -1]).item())

                # Self-attention (diagonal)
                diag = [head_attn[i, i].item() for i in range(min(seq_len, 10))]
                features.append(sum(diag) / len(diag) if diag else 0)

            return mx.array(features)

        hidden = block(hidden, mask=mask, cache=None)
        if hasattr(hidden, 'hidden_states'):
            hidden = hidden.hidden_states
        elif isinstance(hidden, tuple):
            hidden = hidden[0]

    return mx.zeros(128)  # Fallback


def get_hidden_state(model, tokenizer, text: str, layer: int) -> mx.array:
    """Get hidden state from model."""
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


def get_combined_features(model, tokenizer, text: str, layer: int = 13) -> mx.array:
    """Get both hidden state and attention features."""
    hidden = get_hidden_state(model, tokenizer, text, layer)
    attn_features = extract_attention_features(model, tokenizer, text, layer)

    # Concatenate
    return mx.concatenate([hidden, attn_features])


# =============================================================================
# MODELS
# =============================================================================

class AttentionClassifier(nn.Module):
    """Classifier using only attention features."""
    def __init__(self, input_dim: int = 128, num_templates: int = NUM_TEMPLATES):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_templates)
        self.dropout = nn.Dropout(0.1)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.gelu(self.fc1(x))
        x = self.dropout(x)
        x = nn.gelu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


class CombinedClassifier(nn.Module):
    """Classifier using both hidden state and attention features."""
    def __init__(self, hidden_dim: int, attn_dim: int = 128, num_templates: int = NUM_TEMPLATES):
        super().__init__()
        combined_dim = hidden_dim + attn_dim
        self.fc1 = nn.Linear(combined_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_templates)
        self.dropout = nn.Dropout(0.1)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.gelu(self.fc1(x))
        x = self.dropout(x)
        x = nn.gelu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


class HiddenStateClassifier(nn.Module):
    """Baseline: Hidden state only classifier."""
    def __init__(self, hidden_dim: int, num_templates: int = NUM_TEMPLATES):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, num_templates)
        self.dropout = nn.Dropout(0.1)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.gelu(self.fc1(x))
        x = self.dropout(x)
        x = nn.gelu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


# =============================================================================
# TRAINING
# =============================================================================

@dataclass
class Config:
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 20
    layer: int = 13


def load_dataset(path: Path) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data["examples"]


def train_classifier(classifier, feature_fn, train_examples, config, optimizer=None):
    """Train a classifier with given feature extraction function."""
    if optimizer is None:
        optimizer = optim.Adam(learning_rate=config.learning_rate)

    def loss_fn(classifier, features, labels):
        logits = classifier(features)
        return nn.losses.cross_entropy(logits, labels).mean()

    loss_and_grad = nn.value_and_grad(classifier, loss_fn)

    for epoch in range(config.epochs):
        random.shuffle(train_examples)

        for i in range(0, len(train_examples), config.batch_size):
            batch = train_examples[i:i + config.batch_size]

            features_list = []
            labels = []

            for ex in batch:
                feat = feature_fn(ex["nl_input"])
                features_list.append(feat)
                labels.append(ex["template_id"])

            features = mx.stack(features_list)
            labels = mx.array(labels)

            loss, grads = loss_and_grad(classifier, features, labels)
            optimizer.update(classifier, grads)
            mx.eval(classifier.parameters(), optimizer.state)

    return classifier


def evaluate_classifier(classifier, feature_fn, examples):
    """Evaluate classifier accuracy."""
    correct = 0
    total = 0
    by_pred = {}

    for ex in examples:
        total += 1
        features = feature_fn(ex["nl_input"])
        logits = classifier(features[None, :])
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
    print("=" * 70)
    print("Attention-Based Template Classification Experiment")
    print("=" * 70)

    # Load model
    print("\n1. Loading model...")
    from chuk_lazarus.models_v2.loader import load_model

    loaded = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = loaded.model
    tokenizer = loaded.tokenizer
    model.freeze()
    mx.eval(model.parameters())

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hidden_dim = model.model.layers[0].self_attn.q_proj.weight.shape[0]
    print(f"   Model loaded. Hidden dim: {hidden_dim}")

    # Load datasets
    print("\n2. Loading datasets...")
    results_dir = Path(__file__).parent / "results"
    train_examples = load_dataset(results_dir / "train_dataset.json")
    test_examples = load_dataset(results_dir / "test_dataset.json")

    print(f"   Train: {len(train_examples)} (NO vocab transfer)")
    print(f"   Test: {len(test_examples)} (Collatz - held out)")

    config = Config()

    # Create feature extraction functions
    def hidden_feature_fn(text):
        return get_hidden_state(model, tokenizer, text, config.layer)

    def attn_feature_fn(text):
        return extract_attention_features(model, tokenizer, text, config.layer)

    def combined_feature_fn(text):
        return get_combined_features(model, tokenizer, text, config.layer)

    # Get attention feature dimension
    sample_attn = attn_feature_fn(train_examples[0]["nl_input"])
    attn_dim = sample_attn.shape[0]
    print(f"   Attention feature dim: {attn_dim}")

    # ==========================================================================
    # EXPERIMENT 1: Hidden State Only (Baseline)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Hidden State Only (Baseline, no vocab transfer)")
    print("=" * 70)

    classifier1 = HiddenStateClassifier(hidden_dim)
    print("   Training...")
    start = time.perf_counter()
    train_classifier(classifier1, hidden_feature_fn, train_examples, config)
    elapsed = time.perf_counter() - start
    print(f"   Done in {elapsed:.1f}s")

    train_eval1 = evaluate_classifier(classifier1, hidden_feature_fn, train_examples[:200])
    test_eval1 = evaluate_classifier(classifier1, hidden_feature_fn, test_examples)

    print(f"\n   Train accuracy: {train_eval1['accuracy']:.1%}")
    print(f"   Test accuracy (Collatz): {test_eval1['accuracy']:.1%}")
    print(f"   Test predictions: {test_eval1['predictions']}")

    # ==========================================================================
    # EXPERIMENT 2: Attention Features Only
    # ==========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Attention Features Only (no vocab transfer)")
    print("=" * 70)

    classifier2 = AttentionClassifier(input_dim=attn_dim)
    print("   Training...")
    start = time.perf_counter()
    train_classifier(classifier2, attn_feature_fn, train_examples, config)
    elapsed = time.perf_counter() - start
    print(f"   Done in {elapsed:.1f}s")

    train_eval2 = evaluate_classifier(classifier2, attn_feature_fn, train_examples[:200])
    test_eval2 = evaluate_classifier(classifier2, attn_feature_fn, test_examples)

    print(f"\n   Train accuracy: {train_eval2['accuracy']:.1%}")
    print(f"   Test accuracy (Collatz): {test_eval2['accuracy']:.1%}")
    print(f"   Test predictions: {test_eval2['predictions']}")

    # ==========================================================================
    # EXPERIMENT 3: Combined (Hidden + Attention)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Combined Features (no vocab transfer)")
    print("=" * 70)

    classifier3 = CombinedClassifier(hidden_dim, attn_dim)
    print("   Training...")
    start = time.perf_counter()
    train_classifier(classifier3, combined_feature_fn, train_examples, config)
    elapsed = time.perf_counter() - start
    print(f"   Done in {elapsed:.1f}s")

    train_eval3 = evaluate_classifier(classifier3, combined_feature_fn, train_examples[:200])
    test_eval3 = evaluate_classifier(classifier3, combined_feature_fn, test_examples)

    print(f"\n   Train accuracy: {train_eval3['accuracy']:.1%}")
    print(f"   Test accuracy (Collatz): {test_eval3['accuracy']:.1%}")
    print(f"   Test predictions: {test_eval3['predictions']}")

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY (No Vocabulary Transfer)")
    print("=" * 70)
    print(f"""
| Features        | Train Acc | Test Acc (Collatz) |
|-----------------|-----------|-------------------|
| Hidden State    | {train_eval1['accuracy']:>9.1%} | {test_eval1['accuracy']:>17.1%} |
| Attention Only  | {train_eval2['accuracy']:>9.1%} | {test_eval2['accuracy']:>17.1%} |
| Combined        | {train_eval3['accuracy']:>9.1%} | {test_eval3['accuracy']:>17.1%} |

Comparison to WITH vocab transfer:
  Hidden State + vocab transfer: 98% test accuracy

Interpretation:
  If attention improves zero-shot test accuracy:
    → Attention features ARE more vocabulary-agnostic
  If not:
    → Attention encodes similar semantic features as hidden states
""")
    print("=" * 70)

    # Save results
    results = {
        "hidden_only": {"train": train_eval1["accuracy"], "test": test_eval1["accuracy"]},
        "attention_only": {"train": train_eval2["accuracy"], "test": test_eval2["accuracy"]},
        "combined": {"train": train_eval3["accuracy"], "test": test_eval3["accuracy"]},
    }

    results_path = results_dir / "attention_classifier_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {results_path}")


if __name__ == "__main__":
    main()
