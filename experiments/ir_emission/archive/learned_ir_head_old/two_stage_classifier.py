"""
Two-Stage Classifier

Stage 1: Question → Template (question-level features)
Stage 2: Number → Slot (number-level features only)

The key insight: "has_left" is about the QUESTION, not individual numbers.
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
# STAGE 1: QUESTION → TEMPLATE
# =============================================================================

def extract_question_features(text: str) -> dict:
    """Extract question-level features."""
    text_lower = text.lower()

    return {
        # Question type indicators
        'has_left': 'left' in text_lower or 'remain' in text_lower,
        'has_total': 'total' in text_lower or 'altogether' in text_lower,
        'has_each_get': 'each get' in text_lower or 'per person' in text_lower or 'each group' in text_lower,
        'has_how_many': 'how many' in text_lower,
        'has_how_much': 'how much' in text_lower,

        # Operation indicators (question level)
        'has_split': 'split' in text_lower or 'divide' in text_lower or 'share' in text_lower,
        'has_among': 'among' in text_lower or 'between' in text_lower,

        # Rate indicators (suggests MUL)
        'has_each': 'each' in text_lower,
        'has_per': ' per ' in text_lower,
        'has_every': 'every' in text_lower,
    }


def question_features_to_vector(features: dict) -> np.ndarray:
    """Convert question features to vector."""
    keys = ['has_left', 'has_total', 'has_each_get', 'has_how_many', 'has_how_much',
            'has_split', 'has_among', 'has_each', 'has_per', 'has_every']
    return np.array([float(features.get(k, False)) for k in keys])


QUESTION_FEATURE_NAMES = ['has_left', 'has_total', 'has_each_get', 'has_how_many', 'has_how_much',
                          'has_split', 'has_among', 'has_each', 'has_per', 'has_every']


# Templates: what operation sequence to use
TEMPLATES = ['MUL', 'SUB', 'ADD', 'DIV']


# =============================================================================
# STAGE 2: NUMBER → SLOT
# =============================================================================

def extract_number_features(text: str, num_value: str, num_idx: int, total_nums: int) -> dict:
    """Extract number-level features (NO question-level features)."""
    words = text.lower().split()

    # Find position of number
    num_pos = -1
    for i, word in enumerate(words):
        clean = word.strip('.,!?:;()[]"\'$')
        if clean == num_value:
            num_pos = i
            break

    if num_pos == -1:
        num_pos = 0  # fallback

    # Context window (±3 words)
    start = max(0, num_pos - 3)
    end = min(len(words), num_pos + 4)
    left_ctx = ' '.join(words[start:num_pos])
    right_ctx = ' '.join(words[num_pos+1:end])

    return {
        # Positional features
        'is_first': num_idx == 0,
        'is_last': num_idx == total_nums - 1,
        'relative_pos': num_idx / max(1, total_nums - 1) if total_nums > 1 else 0.5,

        # Immediate context (rate indicators)
        'ctx_each': 'each' in right_ctx,  # "5 each" → rate
        'ctx_per': 'per' in right_ctx,    # "5 per hour" → rate
        'ctx_dollar': '$' in words[num_pos] if num_pos < len(words) else False,

        # Verb proximity (consumption → subtract operand)
        'near_eat': any(w in left_ctx for w in ['eat', 'eats', 'ate']),
        'near_spend': any(w in left_ctx for w in ['spend', 'spends', 'spent']),
        'near_give': any(w in left_ctx for w in ['give', 'gives', 'gave']),
        'near_use': any(w in left_ctx for w in ['use', 'uses', 'used']),
        'near_take': any(w in left_ctx for w in ['take', 'takes', 'took']),
        'near_lose': any(w in left_ctx for w in ['lose', 'loses', 'lost']),

        # Addition proximity
        'near_find': any(w in left_ctx for w in ['find', 'finds', 'found']),
        'near_get': any(w in left_ctx for w in ['get', 'gets', 'got', 'more']),
        'near_receive': any(w in left_ctx for w in ['receive', 'receives']),

        # Division proximity
        'near_among': 'among' in right_ctx or 'between' in right_ctx,
    }


def number_features_to_vector(features: dict) -> np.ndarray:
    """Convert number features to vector."""
    keys = ['is_first', 'is_last', 'relative_pos',
            'ctx_each', 'ctx_per', 'ctx_dollar',
            'near_eat', 'near_spend', 'near_give', 'near_use', 'near_take', 'near_lose',
            'near_find', 'near_get', 'near_receive', 'near_among']
    return np.array([float(features.get(k, False)) for k in keys])


NUMBER_FEATURE_NAMES = ['is_first', 'is_last', 'relative_pos',
                        'ctx_each', 'ctx_per', 'ctx_dollar',
                        'near_eat', 'near_spend', 'near_give', 'near_use', 'near_take', 'near_lose',
                        'near_find', 'near_get', 'near_receive', 'near_among']


# Slots: LEFT or RIGHT operand
SLOTS = ['LEFT', 'RIGHT']


# =============================================================================
# CLASSIFIERS
# =============================================================================

class LogisticClassifier:
    """Simple logistic regression."""

    def __init__(self, classes: list):
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 200, lr: float = 1.0):
        n_samples, n_features = X.shape
        n_classes = len(self.classes)

        self.weights = np.random.randn(n_features, n_classes) * 0.01
        self.bias = np.zeros(n_classes)

        for epoch in range(epochs):
            logits = X @ self.weights + self.bias
            exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

            targets = np.zeros((n_samples, n_classes))
            targets[np.arange(n_samples), y] = 1

            grad_logits = (probs - targets) / n_samples
            self.weights -= lr * (X.T @ grad_logits)
            self.bias -= lr * grad_logits.sum(axis=0)

            if epoch % 50 == 0:
                loss = -np.log(probs[np.arange(n_samples), y] + 1e-8).mean()
                acc = (probs.argmax(axis=1) == y).mean()
                print(f"    Epoch {epoch}: loss={loss:.4f}, acc={acc:.2%}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        logits = X @ self.weights + self.bias
        return logits.argmax(axis=1)


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_synthetic_data(n: int) -> list[dict]:
    """Generate synthetic data with clear question-level and number-level features."""
    NAMES = ["Tom", "Sara", "Mike", "Emma", "John", "Lisa"]
    ITEMS = ["apples", "books", "cookies", "eggs", "stickers", "marbles"]

    examples = []

    for _ in range(n):
        template = random.choice(TEMPLATES)
        person = random.choice(NAMES)
        items = random.choice(ITEMS)

        if template == 'MUL':
            n1 = random.randint(2, 20)
            n2 = random.randint(2, 15)
            problems = [
                f"{person} has {n1} boxes with {n2} {items} each. How many in total?",
                f"{person} works {n1} hours at ${n2} per hour. How much does {person} earn?",
                f"There are {n1} rows with {n2} seats each. What is the total?",
            ]
            answer = n1 * n2

        elif template == 'SUB':
            n1 = random.randint(20, 100)
            n2 = random.randint(1, n1 - 1)
            problems = [
                f"{person} has {n1} {items}. {person} eats {n2}. How many are left?",
                f"There are {n1} {items}. {person} gives {n2} away. How many remain?",
                f"{person} had ${n1} and spent ${n2}. How much is left?",
            ]
            answer = n1 - n2

        elif template == 'ADD':
            n1 = random.randint(5, 50)
            n2 = random.randint(5, 30)
            problems = [
                f"{person} has {n1} {items}. {person} finds {n2} more. How many in total?",
                f"There are {n1} {items}. {person} gets {n2} more. What is the total?",
                f"{person} had {n1} and received {n2}. How many altogether?",
            ]
            answer = n1 + n2

        else:  # DIV
            n2 = random.randint(2, 10)
            n1 = n2 * random.randint(2, 15)
            problems = [
                f"{n1} {items} are split among {n2} kids. How many does each get?",
                f"{person} divides {n1} {items} between {n2} friends. How many per person?",
                f"There are {n1} {items} shared among {n2} groups. How many in each group?",
            ]
            answer = n1 // n2

        problem = random.choice(problems)

        examples.append({
            'problem': problem,
            'template': template,
            'n1': n1, 'n2': n2,
            'answer': answer,
            'n1_slot': 'LEFT',
            'n2_slot': 'RIGHT',
        })

    return examples


# =============================================================================
# MAIN
# =============================================================================

def main():
    random.seed(42)
    np.random.seed(42)

    print("=" * 70)
    print("  TWO-STAGE CLASSIFIER")
    print("  Stage 1: Question → Template")
    print("  Stage 2: Number → Slot")
    print("=" * 70)

    # Generate data
    print("\nGenerating synthetic data...")
    train_data = generate_synthetic_data(8000)
    test_data = generate_synthetic_data(2000)
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")

    # ==========================================================================
    # STAGE 1: Train Question → Template
    # ==========================================================================
    print("\n" + "=" * 70)
    print("STAGE 1: QUESTION → TEMPLATE")
    print("=" * 70)

    template_clf = LogisticClassifier(TEMPLATES)

    # Prepare data
    X1_train = np.array([question_features_to_vector(extract_question_features(ex['problem'])) for ex in train_data])
    y1_train = np.array([template_clf.class_to_idx[ex['template']] for ex in train_data])

    X1_test = np.array([question_features_to_vector(extract_question_features(ex['problem'])) for ex in test_data])
    y1_test = np.array([template_clf.class_to_idx[ex['template']] for ex in test_data])

    print("\nTraining...")
    template_clf.fit(X1_train, y1_train, epochs=150, lr=1.0)

    # Evaluate
    preds = template_clf.predict(X1_test)
    acc = (preds == y1_test).mean()
    print(f"\nTemplate accuracy: {acc:.2%}")

    print("\nPer-template:")
    for i, t in enumerate(TEMPLATES):
        mask = y1_test == i
        if mask.sum() > 0:
            t_acc = (preds[mask] == y1_test[mask]).mean()
            print(f"  {t}: {t_acc:.2%} (n={mask.sum()})")

    # Feature weights
    print("\nLearned weights:")
    for i, t in enumerate(TEMPLATES):
        top_idx = np.argsort(template_clf.weights[:, i])[-3:][::-1]
        print(f"  {t}: ", end="")
        for idx in top_idx:
            print(f"{QUESTION_FEATURE_NAMES[idx]}({template_clf.weights[idx, i]:.2f}) ", end="")
        print()

    # ==========================================================================
    # STAGE 2: Train Number → Slot
    # ==========================================================================
    print("\n" + "=" * 70)
    print("STAGE 2: NUMBER → SLOT")
    print("=" * 70)

    slot_clf = LogisticClassifier(SLOTS)

    # Prepare data (two numbers per problem)
    X2_train, y2_train = [], []
    for ex in train_data:
        # First number → LEFT
        f1 = extract_number_features(ex['problem'], str(ex['n1']), 0, 2)
        X2_train.append(number_features_to_vector(f1))
        y2_train.append(slot_clf.class_to_idx['LEFT'])

        # Second number → RIGHT
        f2 = extract_number_features(ex['problem'], str(ex['n2']), 1, 2)
        X2_train.append(number_features_to_vector(f2))
        y2_train.append(slot_clf.class_to_idx['RIGHT'])

    X2_train = np.array(X2_train)
    y2_train = np.array(y2_train)

    # Same for test
    X2_test, y2_test = [], []
    for ex in test_data:
        f1 = extract_number_features(ex['problem'], str(ex['n1']), 0, 2)
        X2_test.append(number_features_to_vector(f1))
        y2_test.append(slot_clf.class_to_idx['LEFT'])

        f2 = extract_number_features(ex['problem'], str(ex['n2']), 1, 2)
        X2_test.append(number_features_to_vector(f2))
        y2_test.append(slot_clf.class_to_idx['RIGHT'])

    X2_test = np.array(X2_test)
    y2_test = np.array(y2_test)

    print(f"\nTraining on {len(X2_train)} number examples...")
    slot_clf.fit(X2_train, y2_train, epochs=150, lr=1.0)

    # Evaluate
    preds2 = slot_clf.predict(X2_test)
    acc2 = (preds2 == y2_test).mean()
    print(f"\nSlot accuracy: {acc2:.2%}")

    print("\nPer-slot:")
    for i, s in enumerate(SLOTS):
        mask = y2_test == i
        if mask.sum() > 0:
            s_acc = (preds2[mask] == y2_test[mask]).mean()
            print(f"  {s}: {s_acc:.2%} (n={mask.sum()})")

    # Feature weights
    print("\nLearned weights:")
    for i, s in enumerate(SLOTS):
        weights = slot_clf.weights[:, i]
        top_idx = np.argsort(np.abs(weights))[-5:][::-1]
        print(f"  {s}: ", end="")
        for idx in top_idx:
            print(f"{NUMBER_FEATURE_NAMES[idx]}({weights[idx]:.2f}) ", end="")
        print()

    # ==========================================================================
    # END-TO-END EVALUATION
    # ==========================================================================
    print("\n" + "=" * 70)
    print("END-TO-END EVALUATION (Synthetic)")
    print("=" * 70)

    correct = 0
    for ex in test_data:
        # Stage 1: Predict template
        q_features = question_features_to_vector(extract_question_features(ex['problem']))
        pred_template = TEMPLATES[template_clf.predict(q_features.reshape(1, -1))[0]]

        # Stage 2: Predict slots
        f1 = number_features_to_vector(extract_number_features(ex['problem'], str(ex['n1']), 0, 2))
        f2 = number_features_to_vector(extract_number_features(ex['problem'], str(ex['n2']), 1, 2))

        slot1 = SLOTS[slot_clf.predict(f1.reshape(1, -1))[0]]
        slot2 = SLOTS[slot_clf.predict(f2.reshape(1, -1))[0]]

        # Assign numbers to slots
        if slot1 == 'LEFT':
            left_num, right_num = ex['n1'], ex['n2']
        else:
            left_num, right_num = ex['n2'], ex['n1']

        # Execute
        if pred_template == 'MUL':
            pred_answer = left_num * right_num
        elif pred_template == 'SUB':
            pred_answer = left_num - right_num
        elif pred_template == 'ADD':
            pred_answer = left_num + right_num
        else:  # DIV
            pred_answer = left_num // right_num if right_num != 0 else 0

        if pred_answer == ex['answer']:
            correct += 1

    e2e_acc = correct / len(test_data)
    print(f"\nEnd-to-end accuracy: {e2e_acc:.2%}")

    # ==========================================================================
    # GSM8K TRANSFER
    # ==========================================================================
    print("\n" + "=" * 70)
    print("GSM8K TRANSFER TEST")
    print("=" * 70)

    # Load GSM8K
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")

    gsm8k_examples = []
    for item in ds:
        # Extract answer
        match = re.search(r'####\s*(-?[\d,]+)', item["answer"])
        if match:
            gsm8k_examples.append({
                'problem': item["question"],
                'answer': int(match.group(1).replace(",", ""))
            })
        if len(gsm8k_examples) >= 200:
            break

    print(f"\nTesting on {len(gsm8k_examples)} GSM8K examples...")

    # Evaluate template prediction distribution
    template_preds = []
    for ex in gsm8k_examples:
        q_features = question_features_to_vector(extract_question_features(ex['problem']))
        pred = TEMPLATES[template_clf.predict(q_features.reshape(1, -1))[0]]
        template_preds.append(pred)

    print("\nTemplate predictions on GSM8K:")
    for t, count in Counter(template_preds).most_common():
        print(f"  {t}: {count} ({count/len(gsm8k_examples):.1%})")

    # Show examples
    print("\nSample predictions:")
    for ex, pred_t in list(zip(gsm8k_examples, template_preds))[:5]:
        print(f"\n  Q: {ex['problem'][:80]}...")
        print(f"  Answer: {ex['answer']}")
        print(f"  Predicted template: {pred_t}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  Stage 1 (Question→Template): {acc:.1%}")
    print(f"  Stage 2 (Number→Slot): {acc2:.1%}")
    print(f"  End-to-end synthetic: {e2e_acc:.1%}")


if __name__ == "__main__":
    main()
