"""
Test two-stage classifier on single-operation GSM8K problems.

Filter GSM8K to problems with exactly 1 computation step,
then run full pipeline and measure accuracy.
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
# CLASSIFIERS (same as before, but trained fresh)
# =============================================================================

class LogisticClassifier:
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

    def predict(self, X: np.ndarray) -> np.ndarray:
        logits = X @ self.weights + self.bias
        return logits.argmax(axis=1)


TEMPLATES = ['MUL', 'SUB', 'ADD', 'DIV']
SLOTS = ['LEFT', 'RIGHT']


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_question_features(text: str) -> np.ndarray:
    text_lower = text.lower()
    features = [
        'left' in text_lower or 'remain' in text_lower,  # has_left
        'total' in text_lower or 'altogether' in text_lower,  # has_total
        'each get' in text_lower or 'per person' in text_lower,  # has_each_get
        'how many' in text_lower,
        'how much' in text_lower,
        'split' in text_lower or 'divide' in text_lower or 'share' in text_lower,
        'among' in text_lower or 'between' in text_lower,
        'each' in text_lower,
        ' per ' in text_lower,
        'every' in text_lower,
    ]
    return np.array([float(f) for f in features])


def extract_numbers_from_text(text: str) -> list[tuple]:
    """Extract (value, position) for all numbers."""
    numbers = []
    words = text.split()
    for i, word in enumerate(words):
        clean = word.strip('.,!?:;()[]"\'$%')
        if re.match(r'^\d+(?:\.\d+)?$', clean):
            numbers.append((clean, i, word))
    return numbers


def extract_number_features(text: str, num_value: str, num_idx: int, total_nums: int) -> np.ndarray:
    words = text.lower().split()

    # Find position
    num_pos = -1
    for i, word in enumerate(words):
        clean = word.strip('.,!?:;()[]"\'$%')
        if clean == num_value:
            num_pos = i
            break
    if num_pos == -1:
        num_pos = 0

    start = max(0, num_pos - 3)
    end = min(len(words), num_pos + 4)
    left_ctx = ' '.join(words[start:num_pos])
    right_ctx = ' '.join(words[num_pos+1:end])

    features = [
        num_idx == 0,  # is_first
        num_idx == total_nums - 1,  # is_last
        num_idx / max(1, total_nums - 1) if total_nums > 1 else 0.5,  # relative_pos
        'each' in right_ctx,  # ctx_each
        'per' in right_ctx,  # ctx_per
        '$' in (words[num_pos] if num_pos < len(words) else ''),  # ctx_dollar
        any(w in left_ctx for w in ['eat', 'eats', 'ate']),
        any(w in left_ctx for w in ['spend', 'spends', 'spent']),
        any(w in left_ctx for w in ['give', 'gives', 'gave']),
        any(w in left_ctx for w in ['use', 'uses', 'used']),
        any(w in left_ctx for w in ['take', 'takes', 'took']),
        any(w in left_ctx for w in ['lose', 'loses', 'lost']),
        any(w in left_ctx for w in ['find', 'finds', 'found']),
        any(w in left_ctx for w in ['get', 'gets', 'got', 'more']),
        any(w in left_ctx for w in ['receive', 'receives']),
        'among' in right_ctx or 'between' in right_ctx,
    ]
    return np.array([float(f) for f in features])


# =============================================================================
# GSM8K LOADING
# =============================================================================

def load_single_op_gsm8k():
    """Load GSM8K problems with exactly 1 computation."""
    from datasets import load_dataset

    print("Loading GSM8K...")
    ds = load_dataset("gsm8k", "main", split="test")

    single_op = []
    multi_op = 0

    for item in ds:
        answer = item["answer"]
        question = item["question"]

        # Count computations
        computations = re.findall(r'<<([^>]+)>>', answer)
        if len(computations) != 1:
            multi_op += 1
            continue

        # Parse the single computation
        comp = computations[0]
        if '=' not in comp:
            continue

        expr, result = comp.rsplit('=', 1)
        expr = expr.strip()
        result = result.strip()

        # Determine operation
        if '+' in expr and '-' not in expr:
            op = 'ADD'
        elif '-' in expr and '+' not in expr:
            op = 'SUB'
        elif '*' in expr and '/' not in expr:
            op = 'MUL'
        elif '/' in expr and '*' not in expr:
            op = 'DIV'
        else:
            continue

        # Extract operands
        if op == 'ADD':
            parts = expr.split('+')
        elif op == 'SUB':
            parts = expr.split('-')
        elif op == 'MUL':
            parts = expr.split('*')
        else:
            parts = expr.split('/')

        if len(parts) != 2:
            continue

        try:
            left = float(parts[0].strip())
            right = float(parts[1].strip())
            answer_val = float(result)
        except:
            continue

        # Extract final answer
        final_match = re.search(r'####\s*(-?[\d,]+)', answer)
        if not final_match:
            continue
        final_answer = int(final_match.group(1).replace(",", ""))

        single_op.append({
            'question': question,
            'operation': op,
            'left': left,
            'right': right,
            'computed_result': answer_val,
            'final_answer': final_answer,
            'expression': expr,
        })

    print(f"Single-op: {len(single_op)}, Multi-op: {multi_op}")
    return single_op


# =============================================================================
# TRAINING DATA GENERATION
# =============================================================================

def generate_training_data(n: int) -> list[dict]:
    """Generate synthetic training data."""
    NAMES = ["Tom", "Sara", "Mike", "Emma", "John", "Lisa"]
    ITEMS = ["apples", "books", "cookies", "eggs", "stickers", "marbles"]

    examples = []
    for _ in range(n):
        template = random.choice(TEMPLATES)
        person = random.choice(NAMES)
        items = random.choice(ITEMS)

        if template == 'MUL':
            n1, n2 = random.randint(2, 20), random.randint(2, 15)
            problems = [
                f"{person} has {n1} boxes with {n2} {items} each. How many in total?",
                f"{person} works {n1} hours at ${n2} per hour. How much earned?",
                f"There are {n1} rows with {n2} seats each. Total?",
            ]
            answer = n1 * n2
        elif template == 'SUB':
            n1 = random.randint(20, 100)
            n2 = random.randint(1, n1 - 1)
            problems = [
                f"{person} has {n1} {items}. {person} eats {n2}. How many left?",
                f"There are {n1} {items}. {person} gives {n2} away. How many remain?",
                f"{person} had ${n1} and spent ${n2}. How much left?",
            ]
            answer = n1 - n2
        elif template == 'ADD':
            n1, n2 = random.randint(5, 50), random.randint(5, 30)
            problems = [
                f"{person} has {n1} {items}. {person} finds {n2} more. Total?",
                f"There are {n1} {items}. {n2} more arrive. How many altogether?",
            ]
            answer = n1 + n2
        else:  # DIV
            n2 = random.randint(2, 10)
            n1 = n2 * random.randint(2, 15)
            problems = [
                f"{n1} {items} split among {n2} kids. How many each?",
                f"{person} divides {n1} between {n2} friends. How many per person?",
            ]
            answer = n1 // n2

        examples.append({
            'problem': random.choice(problems),
            'template': template,
            'n1': n1, 'n2': n2,
            'answer': answer,
        })
    return examples


# =============================================================================
# MAIN
# =============================================================================

def main():
    random.seed(42)
    np.random.seed(42)

    print("=" * 70)
    print("  SINGLE-OP GSM8K TEST")
    print("  Two-stage classifier on single-operation problems")
    print("=" * 70)

    # Load GSM8K single-op
    gsm8k_data = load_single_op_gsm8k()

    # Generate training data
    print("\nGenerating training data...")
    train_data = generate_training_data(10000)

    # Train Stage 1: Question → Template
    print("\n" + "=" * 70)
    print("TRAINING STAGE 1: Question → Template")
    print("=" * 70)

    template_clf = LogisticClassifier(TEMPLATES)
    X1 = np.array([extract_question_features(ex['problem']) for ex in train_data])
    y1 = np.array([template_clf.class_to_idx[ex['template']] for ex in train_data])
    template_clf.fit(X1, y1, epochs=200, lr=1.0)

    # Train Stage 2: Number → Slot
    print("\nTRAINING STAGE 2: Number → Slot")

    slot_clf = LogisticClassifier(SLOTS)
    X2, y2 = [], []
    for ex in train_data:
        X2.append(extract_number_features(ex['problem'], str(ex['n1']), 0, 2))
        y2.append(0)  # LEFT
        X2.append(extract_number_features(ex['problem'], str(ex['n2']), 1, 2))
        y2.append(1)  # RIGHT
    slot_clf.fit(np.array(X2), np.array(y2), epochs=200, lr=1.0)

    # Test on GSM8K
    print("\n" + "=" * 70)
    print("TESTING ON SINGLE-OP GSM8K")
    print("=" * 70)

    correct = 0
    template_correct = 0
    results_by_op = {op: {'total': 0, 'correct': 0} for op in TEMPLATES}

    details = []

    for ex in gsm8k_data:
        question = ex['question']
        true_op = ex['operation']
        true_answer = ex['final_answer']
        true_left = ex['left']
        true_right = ex['right']

        # Extract numbers from question
        numbers = extract_numbers_from_text(question)

        if len(numbers) < 2:
            continue

        # Stage 1: Predict template
        q_features = extract_question_features(question)
        pred_op = TEMPLATES[template_clf.predict(q_features.reshape(1, -1))[0]]

        if pred_op == true_op:
            template_correct += 1

        # Stage 2: Predict slots for first two numbers
        num_vals = [n[0] for n in numbers[:2]]
        f1 = extract_number_features(question, num_vals[0], 0, len(numbers))
        f2 = extract_number_features(question, num_vals[1], 1, len(numbers))

        slot1 = SLOTS[slot_clf.predict(f1.reshape(1, -1))[0]]
        slot2 = SLOTS[slot_clf.predict(f2.reshape(1, -1))[0]]

        # Assign to left/right
        try:
            v1 = float(num_vals[0])
            v2 = float(num_vals[1])
        except:
            continue

        if slot1 == 'LEFT':
            left_num, right_num = v1, v2
        else:
            left_num, right_num = v2, v1

        # Execute
        if pred_op == 'MUL':
            pred_answer = left_num * right_num
        elif pred_op == 'SUB':
            pred_answer = left_num - right_num
        elif pred_op == 'ADD':
            pred_answer = left_num + right_num
        else:
            pred_answer = left_num / right_num if right_num != 0 else 0

        # Check
        is_correct = abs(pred_answer - true_answer) < 0.01 or int(pred_answer) == true_answer

        results_by_op[true_op]['total'] += 1
        if is_correct:
            correct += 1
            results_by_op[true_op]['correct'] += 1

        details.append({
            'question': question[:60],
            'true_op': true_op,
            'pred_op': pred_op,
            'true_answer': true_answer,
            'pred_answer': pred_answer,
            'correct': is_correct,
        })

    total = len([d for d in details])
    acc = correct / total if total > 0 else 0
    template_acc = template_correct / total if total > 0 else 0

    print(f"\nTotal single-op problems tested: {total}")
    print(f"Template accuracy: {template_acc:.1%}")
    print(f"End-to-end accuracy: {acc:.1%}")

    print("\nBy operation:")
    for op in TEMPLATES:
        r = results_by_op[op]
        if r['total'] > 0:
            op_acc = r['correct'] / r['total']
            print(f"  {op}: {r['correct']}/{r['total']} = {op_acc:.1%}")

    # Show examples
    print("\n" + "=" * 70)
    print("SAMPLE RESULTS")
    print("=" * 70)

    # Show some correct and incorrect
    print("\nCorrect predictions:")
    for d in [x for x in details if x['correct']][:5]:
        print(f"  Q: {d['question']}...")
        print(f"     {d['true_op']}: {d['true_answer']} (pred: {d['pred_op']}, {d['pred_answer']:.0f})")

    print("\nIncorrect predictions:")
    for d in [x for x in details if not x['correct']][:5]:
        print(f"  Q: {d['question']}...")
        print(f"     True: {d['true_op']}={d['true_answer']}, Pred: {d['pred_op']}={d['pred_answer']:.0f}")


if __name__ == "__main__":
    main()
