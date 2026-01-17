"""
Number Role Classifier

Decomposed approach:
1. Extract numbers with context
2. Classify number → role (tiny classifier)
3. Map question type → template
4. Execute deterministically
"""

import re
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import functools
print = functools.partial(print, flush=True)

import numpy as np


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_numbers_with_features(text: str) -> list[dict]:
    """Extract numbers and compute features for role classification."""
    numbers = []
    words = text.lower().split()
    text_lower = text.lower()

    # Find all numbers
    num_positions = []
    for i, word in enumerate(words):
        clean = word.strip('.,!?:;()[]"\'')
        num_match = re.match(r'^\$?(\d+(?:\.\d+)?)\%?$', clean)
        if num_match:
            num_positions.append((i, num_match.group(1), word))

    total_nums = len(num_positions)

    for idx, (pos, value, original) in enumerate(num_positions):
        # Context window (±5 words)
        start = max(0, pos - 5)
        end = min(len(words), pos + 6)
        left_ctx = ' '.join(words[start:pos])
        right_ctx = ' '.join(words[pos+1:end])
        full_ctx = f"{left_ctx} {right_ctx}"

        # Feature extraction
        features = {
            'value': value,
            'original': original,
            'position': pos,

            # Positional features
            'is_first': idx == 0,
            'is_last': idx == total_nums - 1,
            'relative_position': idx / max(1, total_nums - 1) if total_nums > 1 else 0.5,

            # Context pattern features
            'has_each': 'each' in full_ctx,
            'has_per': 'per' in full_ctx,
            'has_every': 'every' in full_ctx,
            'has_times': 'times' in full_ctx or 'twice' in full_ctx,

            'has_half': 'half' in full_ctx,
            'has_third': 'third' in full_ctx,
            'has_quarter': 'quarter' in full_ctx,

            'has_left': 'left' in text_lower or 'remaining' in text_lower,
            'has_total': 'total' in text_lower or 'all' in full_ctx,

            'is_dollar': original.startswith('$'),
            'is_percent': original.endswith('%'),

            # Verb proximity (within 3 words)
            'near_eat': any(w in left_ctx for w in ['eat', 'eats', 'ate']),
            'near_spend': any(w in left_ctx for w in ['spend', 'spends', 'spent']),
            'near_give': any(w in left_ctx for w in ['give', 'gives', 'gave']),
            'near_sell': any(w in left_ctx for w in ['sell', 'sells', 'sold']),
            'near_buy': any(w in left_ctx for w in ['buy', 'buys', 'bought']),
            'near_earn': any(w in left_ctx for w in ['earn', 'earns', 'earned']),
            'near_has': any(w in left_ctx for w in ['has', 'have', 'had']),

            # Raw context for debugging
            'left_context': left_ctx,
            'right_context': right_ctx,
        }

        numbers.append(features)

    return numbers


def features_to_vector(features: dict) -> np.ndarray:
    """Convert feature dict to numeric vector."""
    bool_features = [
        'is_first', 'is_last',
        'has_each', 'has_per', 'has_every', 'has_times',
        'has_half', 'has_third', 'has_quarter',
        'has_left', 'has_total',
        'is_dollar', 'is_percent',
        'near_eat', 'near_spend', 'near_give', 'near_sell',
        'near_buy', 'near_earn', 'near_has',
    ]

    vec = [float(features.get(f, False)) for f in bool_features]
    vec.append(features.get('relative_position', 0.5))

    return np.array(vec)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_number_role_data():
    """Load the extracted number→role mappings."""
    path = Path(__file__).parent / "number_role_data.json"
    with open(path) as f:
        data = json.load(f)
    return data


def prepare_training_data():
    """Prepare training data for role classifier."""
    from datasets import load_dataset

    print("Loading GSM8K...")
    ds = load_dataset("gsm8k", "main", split="train")

    # Load pre-extracted roles
    role_data = load_number_role_data()

    # Create lookup: (value, context_snippet) → role
    role_lookup = {}
    for item in role_data:
        key = (item['value'], item['context'][:50])
        role_lookup[key] = item['role']

    # Process each problem
    examples = []

    for item in ds:
        question = item["question"]
        numbers = extract_numbers_with_features(question)

        # Try to match numbers to roles
        for num in numbers:
            # Find matching role
            for role_item in role_data:
                if role_item['value'] == num['value']:
                    # Check context overlap
                    if num['left_context'][:20] in role_item['context'] or \
                       num['right_context'][:20] in role_item['context']:
                        examples.append({
                            'features': num,
                            'role': role_item['role'],
                            'question': question,
                        })
                        break

    print(f"Prepared {len(examples)} training examples")
    return examples


# =============================================================================
# CLASSIFIER
# =============================================================================

class SimpleRoleClassifier:
    """Simple logistic regression for number role classification."""

    def __init__(self):
        self.roles = ['MUL_LEFT', 'MUL_RIGHT', 'ADD_LEFT', 'ADD_RIGHT',
                      'SUB_LEFT', 'SUB_RIGHT', 'DIV_LEFT', 'DIV_RIGHT']
        self.role_to_idx = {r: i for i, r in enumerate(self.roles)}
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, lr: float = 0.1):
        """Train with simple gradient descent."""
        n_samples, n_features = X.shape
        n_classes = len(self.roles)

        # Initialize weights
        self.weights = np.random.randn(n_features, n_classes) * 0.01
        self.bias = np.zeros(n_classes)

        for epoch in range(epochs):
            # Forward pass (softmax)
            logits = X @ self.weights + self.bias
            exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

            # One-hot targets
            targets = np.zeros((n_samples, n_classes))
            targets[np.arange(n_samples), y] = 1

            # Gradient
            grad_logits = (probs - targets) / n_samples
            grad_weights = X.T @ grad_logits
            grad_bias = grad_logits.sum(axis=0)

            # Update
            self.weights -= lr * grad_weights
            self.bias -= lr * grad_bias

            # Loss
            if epoch % 20 == 0:
                loss = -np.log(probs[np.arange(n_samples), y] + 1e-8).mean()
                acc = (probs.argmax(axis=1) == y).mean()
                print(f"  Epoch {epoch}: loss={loss:.4f}, acc={acc:.2%}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict roles."""
        logits = X @ self.weights + self.bias
        return logits.argmax(axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        logits = X @ self.weights + self.bias
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)


# =============================================================================
# QUESTION TYPE CLASSIFIER
# =============================================================================

def classify_question_type(question: str) -> str:
    """Classify question into template type."""
    q_lower = question.lower()

    # Check question ending
    if 'how many' in q_lower and 'left' in q_lower:
        return 'REMAINING'  # X - Y
    if 'how many' in q_lower and 'total' in q_lower:
        return 'TOTAL'  # X + Y or X * Y
    if 'how many' in q_lower and 'each' in q_lower:
        return 'EACH_GETS'  # X / Y
    if 'how much' in q_lower and ('earn' in q_lower or 'make' in q_lower or 'profit' in q_lower):
        return 'EARNINGS'  # (X - Y) * Z or X * Y
    if 'how much' in q_lower and 'left' in q_lower:
        return 'MONEY_LEFT'  # X - Y
    if 'how much' in q_lower and 'spend' in q_lower:
        return 'SPENDING'  # X * Y or sum
    if 'how much' in q_lower:
        return 'AMOUNT'  # generic
    if 'how many' in q_lower:
        return 'COUNT'  # generic

    return 'UNKNOWN'


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  NUMBER ROLE CLASSIFIER")
    print("  Decomposed approach: Extract → Classify → Template → Execute")
    print("=" * 70)

    # Prepare data
    print("\nPreparing training data...")
    examples = prepare_training_data()

    if len(examples) < 100:
        print("Not enough matched examples. Running with synthetic features.")
        # Fallback: use the raw role data
        role_data = load_number_role_data()

        examples = []
        for item in role_data:
            # Extract features from context
            features = extract_numbers_with_features(f"prefix {item['value']} suffix " + item['context'])
            if features:
                examples.append({
                    'features': features[0],
                    'role': item['role'],
                })

    print(f"Training examples: {len(examples)}")

    # Convert to arrays
    classifier = SimpleRoleClassifier()

    X = np.array([features_to_vector(ex['features']) for ex in examples])
    y = np.array([classifier.role_to_idx.get(ex['role'], 0) for ex in examples])

    # Split train/test
    n = len(examples)
    indices = list(range(n))
    random.shuffle(indices)
    train_idx = indices[:int(0.8*n)]
    test_idx = indices[int(0.8*n):]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"\nTrain: {len(train_idx)}, Test: {len(test_idx)}")

    # Train
    print("\nTraining classifier...")
    classifier.fit(X_train, y_train, epochs=100, lr=0.5)

    # Evaluate
    train_preds = classifier.predict(X_train)
    test_preds = classifier.predict(X_test)

    train_acc = (train_preds == y_train).mean()
    test_acc = (test_preds == y_test).mean()

    print(f"\nTrain accuracy: {train_acc:.2%}")
    print(f"Test accuracy:  {test_acc:.2%}")

    # Per-class accuracy
    print("\nPer-class accuracy:")
    for i, role in enumerate(classifier.roles):
        mask = y_test == i
        if mask.sum() > 0:
            acc = (test_preds[mask] == y_test[mask]).mean()
            print(f"  {role}: {acc:.2%} (n={mask.sum()})")

    # Feature importance (weight magnitudes)
    print("\nTop features per role:")
    feature_names = [
        'is_first', 'is_last',
        'has_each', 'has_per', 'has_every', 'has_times',
        'has_half', 'has_third', 'has_quarter',
        'has_left', 'has_total',
        'is_dollar', 'is_percent',
        'near_eat', 'near_spend', 'near_give', 'near_sell',
        'near_buy', 'near_earn', 'near_has',
        'relative_position'
    ]

    for i, role in enumerate(classifier.roles[:4]):  # Top 4 roles
        weights = classifier.weights[:, i]
        top_idx = np.argsort(np.abs(weights))[-5:][::-1]
        print(f"\n  {role}:")
        for idx in top_idx:
            print(f"    {feature_names[idx]}: {weights[idx]:.3f}")

    # Test on example
    print("\n" + "=" * 70)
    print("EXAMPLE INFERENCE")
    print("=" * 70)

    test_question = "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and uses 4 for baking. She sells the rest for $2 each."

    print(f"\nQuestion: {test_question}")

    numbers = extract_numbers_with_features(test_question)
    print(f"\nExtracted {len(numbers)} numbers:")

    for num in numbers:
        vec = features_to_vector(num).reshape(1, -1)
        probs = classifier.predict_proba(vec)[0]
        pred_role = classifier.roles[probs.argmax()]
        confidence = probs.max()

        print(f"\n  {num['value']}:")
        print(f"    Context: ...{num['left_context'][-20:]} [{num['value']}] {num['right_context'][:20]}...")
        print(f"    Features: each={num['has_each']}, per={num['has_per']}, first={num['is_first']}")
        print(f"    Predicted: {pred_role} ({confidence:.2%})")

    # Question type
    q_type = classify_question_type(test_question)
    print(f"\n  Question type: {q_type}")


if __name__ == "__main__":
    main()
