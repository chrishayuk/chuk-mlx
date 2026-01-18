"""
Fast Attention-Based Classification Experiment

Simplified version that:
1. Uses fewer examples
2. Simpler attention features (just last layer)
3. Caches features to avoid recomputation
"""

import json
import random
import math
import time
from pathlib import Path
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from templates import TemplateID, NUM_TEMPLATES, template_name


# =============================================================================
# SIMPLE FEATURE EXTRACTION
# =============================================================================

def get_hidden_state(model, tokenizer, text: str, layer: int = 13) -> mx.array:
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


def get_simple_attention_features(model, tokenizer, text: str, layer: int = 13) -> mx.array:
    """
    Simpler attention feature extraction.
    Just compute attention entropy for the last token position.
    """
    tokens = tokenizer.encode(text)
    input_ids = mx.array([tokens])
    seq_len = len(tokens)

    hidden = model.model.embed_tokens(input_ids)

    # Forward to layer before target
    for i, block in enumerate(model.model.layers[:layer-1]):
        hidden = block(hidden, mask=None, cache=None)
        if hasattr(hidden, 'hidden_states'):
            hidden = hidden.hidden_states
        elif isinstance(hidden, tuple):
            hidden = hidden[0]

    # Extract attention at target layer
    attn = model.model.layers[layer-1].self_attn

    q = attn.q_proj(hidden)
    k = attn.k_proj(hidden)

    num_heads = attn.num_heads
    num_kv_heads = attn.num_kv_heads
    head_dim = q.shape[-1] // num_heads
    kv_head_dim = k.shape[-1] // num_kv_heads

    # Just look at last position's attention
    q_last = q[0, -1, :].reshape(num_heads, head_dim)  # (num_heads, head_dim)
    k_all = k[0, :, :].reshape(seq_len, num_kv_heads, kv_head_dim)  # (seq, num_kv_heads, kv_head_dim)

    # Expand KV heads
    repeat_factor = num_heads // num_kv_heads
    k_expanded = mx.repeat(k_all, repeat_factor, axis=1)  # (seq, num_heads, head_dim)

    # Compute attention scores for last position
    scale = math.sqrt(head_dim)
    scores = []
    for h in range(num_heads):
        score = mx.sum(q_last[h] * k_expanded[:, h, :], axis=-1) / scale
        scores.append(score)

    scores = mx.stack(scores)  # (num_heads, seq_len)

    # Apply softmax
    attn_weights = mx.softmax(scores, axis=-1)  # (num_heads, seq_len)

    # Extract simple features per head
    features = []
    eps = 1e-10
    for h in range(num_heads):
        w = attn_weights[h]
        # Entropy
        entropy = -mx.sum(w * mx.log(w + eps)).item()
        features.append(entropy)
        # Attention to first
        features.append(w[0].item())
        # Attention to last
        features.append(w[-1].item())
        # Max attention
        features.append(mx.max(w).item())

    return mx.array(features)


# =============================================================================
# MODEL
# =============================================================================

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim: int, num_templates: int = NUM_TEMPLATES):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, num_templates)
        self.dropout = nn.Dropout(0.1)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.gelu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# =============================================================================
# TRAINING
# =============================================================================

@dataclass
class Config:
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 15
    layer: int = 13
    train_samples: int = 500  # Limit for speed


def load_dataset(path: Path) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data["examples"]


def train_and_evaluate(classifier, train_features, train_labels, test_features, test_labels, config):
    """Train classifier and evaluate."""
    optimizer = optim.Adam(learning_rate=config.learning_rate)

    def loss_fn(classifier, features, labels):
        logits = classifier(features)
        return nn.losses.cross_entropy(logits, labels).mean()

    loss_and_grad = nn.value_and_grad(classifier, loss_fn)

    num_train = train_features.shape[0]
    indices = list(range(num_train))

    for epoch in range(config.epochs):
        random.shuffle(indices)

        for i in range(0, num_train, config.batch_size):
            batch_idx = indices[i:i + config.batch_size]
            batch_features = train_features[batch_idx]
            batch_labels = train_labels[batch_idx]

            loss, grads = loss_and_grad(classifier, batch_features, batch_labels)
            optimizer.update(classifier, grads)
            mx.eval(classifier.parameters(), optimizer.state)

    # Evaluate
    train_logits = classifier(train_features)
    train_preds = mx.argmax(train_logits, axis=-1)
    train_acc = mx.mean(train_preds == train_labels).item()

    test_logits = classifier(test_features)
    test_preds = mx.argmax(test_logits, axis=-1)
    test_acc = mx.mean(test_preds == test_labels).item()

    # Get predictions distribution
    preds_list = test_preds.tolist()
    preds_dist = {}
    for p in preds_list:
        name = template_name(TemplateID(int(p)))
        preds_dist[name] = preds_dist.get(name, 0) + 1

    return train_acc, test_acc, preds_dist


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("Fast Attention-Based Classification Experiment")
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
    num_heads = model.model.layers[0].self_attn.num_heads
    print(f"   Model loaded. Hidden dim: {hidden_dim}, Heads: {num_heads}")

    # Load datasets
    print("\n2. Loading datasets...")
    results_dir = Path(__file__).parent / "results"
    train_examples = load_dataset(results_dir / "train_dataset.json")
    test_examples = load_dataset(results_dir / "test_dataset.json")

    config = Config()

    # Limit train examples for speed
    train_examples = random.sample(train_examples, min(config.train_samples, len(train_examples)))

    print(f"   Train: {len(train_examples)} (NO vocab transfer)")
    print(f"   Test: {len(test_examples)} (Collatz - held out)")

    # Extract features
    print("\n3. Extracting features...")

    print("   Extracting hidden states...")
    start = time.perf_counter()
    train_hidden = mx.stack([get_hidden_state(model, tokenizer, ex["nl_input"], config.layer) for ex in train_examples])
    test_hidden = mx.stack([get_hidden_state(model, tokenizer, ex["nl_input"], config.layer) for ex in test_examples])
    print(f"   Done in {time.perf_counter() - start:.1f}s")

    print("   Extracting attention features...")
    start = time.perf_counter()
    train_attn = mx.stack([get_simple_attention_features(model, tokenizer, ex["nl_input"], config.layer) for ex in train_examples])
    test_attn = mx.stack([get_simple_attention_features(model, tokenizer, ex["nl_input"], config.layer) for ex in test_examples])
    attn_dim = train_attn.shape[1]
    print(f"   Done in {time.perf_counter() - start:.1f}s, dim={attn_dim}")

    # Combined features
    train_combined = mx.concatenate([train_hidden, train_attn], axis=1)
    test_combined = mx.concatenate([test_hidden, test_attn], axis=1)

    # Labels
    train_labels = mx.array([ex["template_id"] for ex in train_examples])
    test_labels = mx.array([ex["template_id"] for ex in test_examples])

    # ==========================================================================
    # EXPERIMENTS
    # ==========================================================================

    print("\n4. Training classifiers...")

    # Experiment 1: Hidden State Only
    print("\n   [1/3] Hidden State Only...")
    classifier1 = SimpleClassifier(hidden_dim)
    train_acc1, test_acc1, preds1 = train_and_evaluate(
        classifier1, train_hidden, train_labels, test_hidden, test_labels, config
    )
    print(f"       Train: {train_acc1:.1%}, Test: {test_acc1:.1%}")

    # Experiment 2: Attention Only
    print("\n   [2/3] Attention Features Only...")
    classifier2 = SimpleClassifier(attn_dim)
    train_acc2, test_acc2, preds2 = train_and_evaluate(
        classifier2, train_attn, train_labels, test_attn, test_labels, config
    )
    print(f"       Train: {train_acc2:.1%}, Test: {test_acc2:.1%}")

    # Experiment 3: Combined
    print("\n   [3/3] Combined Features...")
    classifier3 = SimpleClassifier(hidden_dim + attn_dim)
    train_acc3, test_acc3, preds3 = train_and_evaluate(
        classifier3, train_combined, train_labels, test_combined, test_labels, config
    )
    print(f"       Train: {train_acc3:.1%}, Test: {test_acc3:.1%}")

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "=" * 70)
    print("RESULTS (No Vocabulary Transfer)")
    print("=" * 70)
    print(f"""
| Features        | Train Acc | Test Acc (Collatz) |
|-----------------|-----------|-------------------|
| Hidden State    | {train_acc1:>9.1%} | {test_acc1:>17.1%} |
| Attention Only  | {train_acc2:>9.1%} | {test_acc2:>17.1%} |
| Combined        | {train_acc3:>9.1%} | {test_acc3:>17.1%} |

Test Predictions:
  Hidden State:   {preds1}
  Attention Only: {preds2}
  Combined:       {preds3}

With vocab transfer (baseline): 98% test accuracy

Key Finding:
  If attention features don't improve zero-shot accuracy:
  → Vocabulary-agnostic generalization needs different approach
""")
    print("=" * 70)

    # Save results
    results = {
        "hidden_only": {"train": train_acc1, "test": test_acc1, "predictions": preds1},
        "attention_only": {"train": train_acc2, "test": test_acc2, "predictions": preds2},
        "combined": {"train": train_acc3, "test": test_acc3, "predictions": preds3},
    }

    results_path = results_dir / "attention_classifier_fast_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {results_path}")


if __name__ == "__main__":
    main()
