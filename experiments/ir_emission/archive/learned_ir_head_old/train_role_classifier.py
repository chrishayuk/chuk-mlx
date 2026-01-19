"""
Train role classifier on synthetic feature-coverage data.
Test on GSM8K to measure transfer.
"""

import re
import json
import random
from collections import Counter
from pathlib import Path

import functools
print = functools.partial(print, flush=True)

import numpy as np


# =============================================================================
# FEATURE EXTRACTION (same as before)
# =============================================================================

def extract_numbers_with_features(text: str) -> list[dict]:
    """Extract numbers and compute features."""
    numbers = []
    words = text.lower().split()
    text_lower = text.lower()

    num_positions = []
    for i, word in enumerate(words):
        clean = word.strip('.,!?:;()[]"\'')
        num_match = re.match(r'^\$?(\d+(?:\.\d+)?)\%?$', clean)
        if num_match:
            num_positions.append((i, num_match.group(1), word))

    total_nums = len(num_positions)

    for idx, (pos, value, original) in enumerate(num_positions):
        start = max(0, pos - 5)
        end = min(len(words), pos + 6)
        left_ctx = ' '.join(words[start:pos])
        right_ctx = ' '.join(words[pos+1:end])
        full_ctx = f"{left_ctx} {right_ctx}"

        features = {
            'value': value,
            'is_first': idx == 0,
            'is_last': idx == total_nums - 1,
            'relative_position': idx / max(1, total_nums - 1) if total_nums > 1 else 0.5,

            'has_each': 'each' in full_ctx,
            'has_per': 'per' in full_ctx,
            'has_every': 'every' in full_ctx,
            'has_times': 'times' in full_ctx or 'twice' in full_ctx,

            'has_among': 'among' in full_ctx,
            'has_between': 'between' in full_ctx,
            'has_split': 'split' in full_ctx or 'divide' in full_ctx,
            'has_equal': 'equal' in full_ctx,

            'has_more': 'more' in full_ctx,
            'has_additional': 'additional' in full_ctx,

            'has_left': 'left' in text_lower or 'remain' in text_lower,

            'is_dollar': original.startswith('$'),

            'near_eat': any(w in left_ctx for w in ['eat', 'eats', 'ate']),
            'near_spend': any(w in left_ctx for w in ['spend', 'spends', 'spent']),
            'near_give': any(w in left_ctx for w in ['give', 'gives', 'gave']),
            'near_use': any(w in left_ctx for w in ['use', 'uses', 'used']),
            'near_lose': any(w in left_ctx for w in ['lose', 'loses', 'lost']),
            'near_take': any(w in left_ctx for w in ['take', 'takes', 'took']),
            'near_find': any(w in left_ctx for w in ['find', 'finds', 'found']),
            'near_get': any(w in left_ctx for w in ['get', 'gets', 'got']),
            'near_receive': any(w in left_ctx for w in ['receive', 'receives', 'received']),
        }

        numbers.append(features)

    return numbers


def features_to_vector(features: dict) -> np.ndarray:
    """Convert feature dict to numeric vector."""
    bool_features = [
        'is_first', 'is_last',
        'has_each', 'has_per', 'has_every', 'has_times',
        'has_among', 'has_between', 'has_split', 'has_equal',
        'has_more', 'has_additional', 'has_left',
        'is_dollar',
        'near_eat', 'near_spend', 'near_give', 'near_use',
        'near_lose', 'near_take', 'near_find', 'near_get', 'near_receive',
    ]

    vec = [float(features.get(f, False)) for f in bool_features]
    vec.append(features.get('relative_position', 0.5))

    return np.array(vec)


FEATURE_NAMES = [
    'is_first', 'is_last',
    'has_each', 'has_per', 'has_every', 'has_times',
    'has_among', 'has_between', 'has_split', 'has_equal',
    'has_more', 'has_additional', 'has_left',
    'is_dollar',
    'near_eat', 'near_spend', 'near_give', 'near_use',
    'near_lose', 'near_take', 'near_find', 'near_get', 'near_receive',
    'relative_position'
]


# =============================================================================
# CLASSIFIER
# =============================================================================

class RoleClassifier:
    """Logistic regression for number role classification."""

    def __init__(self):
        self.roles = ['MUL_LEFT', 'MUL_RIGHT', 'ADD_LEFT', 'ADD_RIGHT',
                      'SUB_LEFT', 'SUB_RIGHT', 'DIV_LEFT', 'DIV_RIGHT']
        self.role_to_idx = {r: i for i, r in enumerate(self.roles)}
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 200, lr: float = 0.5):
        """Train with gradient descent."""
        n_samples, n_features = X.shape
        n_classes = len(self.roles)

        self.weights = np.random.randn(n_features, n_classes) * 0.01
        self.bias = np.zeros(n_classes)

        for epoch in range(epochs):
            logits = X @ self.weights + self.bias
            exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

            targets = np.zeros((n_samples, n_classes))
            targets[np.arange(n_samples), y] = 1

            grad_logits = (probs - targets) / n_samples
            grad_weights = X.T @ grad_logits
            grad_bias = grad_logits.sum(axis=0)

            self.weights -= lr * grad_weights
            self.bias -= lr * grad_bias

            if epoch % 50 == 0:
                loss = -np.log(probs[np.arange(n_samples), y] + 1e-8).mean()
                acc = (probs.argmax(axis=1) == y).mean()
                print(f"  Epoch {epoch}: loss={loss:.4f}, acc={acc:.2%}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        logits = X @ self.weights + self.bias
        return logits.argmax(axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = X @ self.weights + self.bias
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_synthetic_data():
    """Load feature-coverage synthetic data."""
    path = Path(__file__).parent / "feature_coverage_data" / "train.json"
    with open(path) as f:
        return json.load(f)


def load_gsm8k_with_roles():
    """Load GSM8K with extracted roles."""
    path = Path(__file__).parent / "number_role_data.json"
    with open(path) as f:
        return json.load(f)


def prepare_synthetic_examples(data: list[dict]) -> tuple:
    """Prepare training data from synthetic examples."""
    classifier = RoleClassifier()
    X_list = []
    y_list = []

    for ex in data:
        problem = ex["problem"]
        numbers = extract_numbers_with_features(problem)

        # Match extracted numbers to labeled numbers
        for labeled in ex["numbers"]:
            value = labeled["value"]
            role = labeled["role"]

            # Find matching extracted number
            for num in numbers:
                if num["value"] == value:
                    X_list.append(features_to_vector(num))
                    y_list.append(classifier.role_to_idx[role])
                    break

    return np.array(X_list), np.array(y_list), classifier


# =============================================================================
# MAIN
# =============================================================================

def main():
    random.seed(42)
    np.random.seed(42)

    print("=" * 70)
    print("  ROLE CLASSIFIER TRAINING")
    print("  Train on synthetic feature-coverage data")
    print("  Test on GSM8K")
    print("=" * 70)

    # Load synthetic data
    print("\nLoading synthetic data...")
    synth_data = load_synthetic_data()
    print(f"Synthetic examples: {len(synth_data)}")

    # Prepare training data
    print("\nPreparing training data...")
    X, y, classifier = prepare_synthetic_examples(synth_data)
    print(f"Training samples: {len(X)}")

    # Split
    n = len(X)
    indices = np.random.permutation(n)
    train_idx = indices[:int(0.8*n)]
    test_idx = indices[int(0.8*n):]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")

    # Train
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    classifier.fit(X_train, y_train, epochs=200, lr=1.0)

    # Evaluate on synthetic test
    print("\n" + "=" * 70)
    print("SYNTHETIC TEST ACCURACY")
    print("=" * 70)

    synth_preds = classifier.predict(X_test)
    synth_acc = (synth_preds == y_test).mean()
    print(f"\nOverall: {synth_acc:.2%}")

    print("\nPer-class:")
    for i, role in enumerate(classifier.roles):
        mask = y_test == i
        if mask.sum() > 0:
            acc = (synth_preds[mask] == y_test[mask]).mean()
            print(f"  {role}: {acc:.2%} (n={mask.sum()})")

    # Feature importance
    print("\n" + "=" * 70)
    print("LEARNED FEATURE WEIGHTS")
    print("=" * 70)

    for i, role in enumerate(classifier.roles):
        weights = classifier.weights[:, i]
        top_idx = np.argsort(weights)[-3:][::-1]
        bot_idx = np.argsort(weights)[:2]

        print(f"\n  {role}:")
        print(f"    Positive: ", end="")
        for idx in top_idx:
            print(f"{FEATURE_NAMES[idx]}({weights[idx]:.2f}) ", end="")
        print(f"\n    Negative: ", end="")
        for idx in bot_idx:
            print(f"{FEATURE_NAMES[idx]}({weights[idx]:.2f}) ", end="")
        print()

    # Test on GSM8K
    print("\n" + "=" * 70)
    print("GSM8K TRANSFER TEST")
    print("=" * 70)

    gsm8k_roles = load_gsm8k_with_roles()
    print(f"\nGSM8K role examples: {len(gsm8k_roles)}")

    # Prepare GSM8K test data
    gsm8k_X = []
    gsm8k_y = []

    for item in gsm8k_roles:
        # Extract features from the context
        # Use a dummy problem to get features
        dummy = f"prefix {item['value']} suffix {item['context']}"
        numbers = extract_numbers_with_features(dummy)

        if numbers:
            gsm8k_X.append(features_to_vector(numbers[0]))
            gsm8k_y.append(classifier.role_to_idx.get(item['role'], 0))

    gsm8k_X = np.array(gsm8k_X)
    gsm8k_y = np.array(gsm8k_y)

    # Evaluate
    gsm8k_preds = classifier.predict(gsm8k_X)
    gsm8k_acc = (gsm8k_preds == gsm8k_y).mean()

    print(f"\nGSM8K Overall Accuracy: {gsm8k_acc:.2%}")

    print("\nPer-class:")
    for i, role in enumerate(classifier.roles):
        mask = gsm8k_y == i
        if mask.sum() > 0:
            acc = (gsm8k_preds[mask] == gsm8k_y[mask]).mean()
            print(f"  {role}: {acc:.2%} (n={mask.sum()})")

    # Confusion analysis
    print("\n" + "=" * 70)
    print("CONFUSION ANALYSIS")
    print("=" * 70)

    for true_role in ['MUL_RIGHT', 'SUB_RIGHT', 'DIV_RIGHT', 'ADD_RIGHT']:
        true_idx = classifier.role_to_idx[true_role]
        mask = gsm8k_y == true_idx

        if mask.sum() > 0:
            pred_counts = Counter(gsm8k_preds[mask])
            print(f"\n  True {true_role} predicted as:")
            for pred_idx, count in pred_counts.most_common(3):
                pct = count / mask.sum()
                print(f"    {classifier.roles[pred_idx]}: {pct:.1%}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  Synthetic test: {synth_acc:.1%}")
    print(f"  GSM8K transfer: {gsm8k_acc:.1%}")
    print(f"  Gap: {synth_acc - gsm8k_acc:.1%}")


if __name__ == "__main__":
    main()
