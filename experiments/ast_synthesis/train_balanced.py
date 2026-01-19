"""
Balanced Training for Template Classifier

The original training had severe class imbalance:
  - LOOP_CONDITIONAL_ACCUMULATE: 200 (14%)
  - LOOP_ACCUMULATE: 800 (57%)
  - IF_BRANCH: 400 (29%)

This caused the model to be biased towards LOOP_ACCUMULATE.

This version uses:
1. Balanced training data (equal examples per template)
2. Class-weighted loss
3. More epochs and better regularization
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

from templates import TemplateID, NUM_TEMPLATES, template_name, PROGRAM_TO_TEMPLATE
from slot_filler import fill_slots, detect_program_hint
from linearize import linearize, execute_ir


# =============================================================================
# MODEL (increased capacity)
# =============================================================================

class TemplateClassifier(nn.Module):
    """Enhanced template classifier with more capacity."""

    def __init__(self, hidden_dim: int, num_templates: int = NUM_TEMPLATES, hidden_size: int = 512):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 256)
        self.fc4 = nn.Linear(256, num_templates)
        self.dropout = nn.Dropout(0.2)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.gelu(self.fc1(x))
        x = self.dropout(x)
        x = nn.gelu(self.fc2(x))
        x = self.dropout(x)
        x = nn.gelu(self.fc3(x))
        x = self.dropout(x)
        return self.fc4(x)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 5e-4
    epochs: int = 30
    hidden_layer: int = 13
    examples_per_template: int = 400  # Balanced


# =============================================================================
# DATA LOADING AND BALANCING
# =============================================================================

def load_dataset(path: Path) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data["examples"]


def balance_dataset(examples: list[dict], examples_per_template: int) -> list[dict]:
    """Create a balanced dataset with equal examples per template."""
    # Group by template
    by_template = {}
    for ex in examples:
        tid = ex["template_id"]
        if tid not in by_template:
            by_template[tid] = []
        by_template[tid].append(ex)

    # Sample equally from each
    balanced = []
    for tid, exs in by_template.items():
        # Oversample if needed
        sampled = []
        while len(sampled) < examples_per_template:
            sampled.extend(random.sample(exs, min(len(exs), examples_per_template - len(sampled))))
        balanced.extend(sampled[:examples_per_template])

    random.shuffle(balanced)
    return balanced


# =============================================================================
# HIDDEN STATE EXTRACTION
# =============================================================================

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


def extract_operands(text: str) -> list[int]:
    numbers = re.findall(r'\b(\d+)\b', text)
    return [int(n) for n in numbers]


# =============================================================================
# TRAINING WITH CLASS WEIGHTS
# =============================================================================

def train_epoch(
    classifier: TemplateClassifier,
    optimizer: optim.Optimizer,
    model, tokenizer,
    examples: list[dict],
    config: TrainingConfig,
    class_weights: mx.array,
) -> float:
    """Train for one epoch with class-weighted loss."""
    random.shuffle(examples)

    def loss_fn(classifier, hidden_states, labels):
        logits = classifier(hidden_states)
        # Apply class weights
        ce = nn.losses.cross_entropy(logits, labels, reduction='none')
        weights = class_weights[labels]
        return (ce * weights).mean()

    loss_and_grad = nn.value_and_grad(classifier, loss_fn)

    total_loss = 0
    num_batches = 0

    for i in range(0, len(examples), config.batch_size):
        batch = examples[i:i + config.batch_size]

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

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_classification(
    classifier: TemplateClassifier,
    model, tokenizer,
    examples: list[dict],
    config: TrainingConfig,
) -> dict:
    correct = 0
    total = 0
    predictions_by_true = {}  # Track what we predict for each true class

    for ex in examples:
        total += 1

        hidden = get_hidden_state(model, tokenizer, ex["nl_input"], config.hidden_layer)
        logits = classifier(hidden[None, :])
        pred = int(mx.argmax(logits[0]).item())
        true = ex["template_id"]

        if pred == true:
            correct += 1

        key = (true, pred)
        predictions_by_true[key] = predictions_by_true.get(key, 0) + 1

    # Build confusion info
    confusion = {}
    for (true, pred), count in predictions_by_true.items():
        true_name = template_name(TemplateID(true))
        pred_name = template_name(TemplateID(pred))
        if true_name not in confusion:
            confusion[true_name] = {}
        confusion[true_name][pred_name] = count

    return {
        "accuracy": correct / total if total > 0 else 0,
        "correct": correct,
        "total": total,
        "confusion": confusion,
    }


def evaluate_execution(
    classifier: TemplateClassifier,
    model, tokenizer,
    examples: list[dict],
    config: TrainingConfig,
) -> dict:
    """Evaluate end-to-end execution accuracy."""
    template_correct = 0
    execution_correct = 0
    total = 0

    for ex in examples:
        total += 1
        nl = ex["nl_input"]
        expected = ex["expected_result"]
        true_template = TemplateID(ex["template_id"])

        # Classify
        hidden = get_hidden_state(model, tokenizer, nl, config.hidden_layer)
        logits = classifier(hidden[None, :])
        pred_template = TemplateID(int(mx.argmax(logits[0]).item()))

        if pred_template == true_template:
            template_correct += 1

        # Extract operands and execute
        operands = extract_operands(nl)
        program_hint = detect_program_hint(nl, pred_template)

        if program_hint is None:
            continue

        ast = fill_slots(pred_template, program_hint, operands)
        if ast is None:
            continue

        try:
            ir_opcodes = linearize(ast)
            success, result, error = execute_ir(ir_opcodes, operands)

            if success and result == expected:
                execution_correct += 1
        except:
            pass

    return {
        "template_accuracy": template_correct / total if total > 0 else 0,
        "execution_accuracy": execution_correct / total if total > 0 else 0,
        "total": total,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("AST-Based IR Synthesis - Balanced Training")
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

    # Load and balance datasets
    print("\n2. Loading and balancing datasets...")
    results_dir = Path(__file__).parent / "results"
    train_examples = load_dataset(results_dir / "train_dataset.json")
    test_examples = load_dataset(results_dir / "test_dataset.json")

    config = TrainingConfig()
    train_balanced = balance_dataset(train_examples, config.examples_per_template)

    print(f"   Original train: {len(train_examples)} examples")
    print(f"   Balanced train: {len(train_balanced)} examples")
    print(f"   Test: {len(test_examples)} examples")

    # Show balanced distribution
    print("\n   Balanced distribution:")
    counts = {}
    for ex in train_balanced:
        tid = ex["template_id"]
        counts[tid] = counts.get(tid, 0) + 1
    for tid, count in sorted(counts.items()):
        print(f"     {template_name(TemplateID(tid))}: {count}")

    # Compute class weights (inverse frequency)
    total = sum(counts.values())
    class_weights = mx.array([total / (NUM_TEMPLATES * counts.get(i, 1)) for i in range(NUM_TEMPLATES)])
    print(f"\n   Class weights: {[f'{w:.2f}' for w in class_weights.tolist()]}")

    # Create classifier
    print("\n3. Creating classifier...")
    hidden_dim = model.model.layers[0].self_attn.q_proj.weight.shape[0]
    classifier = TemplateClassifier(hidden_dim=hidden_dim)
    optimizer = optim.Adam(learning_rate=config.learning_rate)

    # Training loop
    print("\n4. Training...")
    print("-" * 60)
    print(f"{'Epoch':>6} | {'Loss':>8} | {'Train':>10} | {'Test':>10}")
    print("-" * 60)

    best_test_acc = 0

    for epoch in range(config.epochs):
        start = time.perf_counter()

        train_loss = train_epoch(
            classifier, optimizer, model, tokenizer,
            train_balanced, config, class_weights
        )

        elapsed = time.perf_counter() - start

        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == config.epochs - 1:
            train_eval = evaluate_classification(
                classifier, model, tokenizer,
                train_balanced[:200], config
            )
            test_eval = evaluate_classification(
                classifier, model, tokenizer,
                test_examples, config
            )

            print(f"{epoch+1:6d} | {train_loss:8.4f} | {train_eval['accuracy']:10.1%} | "
                  f"{test_eval['accuracy']:10.1%} | {elapsed:.1f}s")

            if test_eval['accuracy'] > best_test_acc:
                best_test_acc = test_eval['accuracy']
        else:
            print(f"{epoch+1:6d} | {train_loss:8.4f} | {'--':>10} | {'--':>10} | {elapsed:.1f}s")

    print("-" * 60)

    # Final evaluation
    print("\n5. Final Evaluation...")

    print("\n   Training set (balanced):")
    train_final = evaluate_classification(
        classifier, model, tokenizer,
        train_balanced, config
    )
    print(f"   Accuracy: {train_final['accuracy']:.1%}")
    print("   Confusion matrix:")
    for true_name, preds in train_final["confusion"].items():
        print(f"     {true_name}:")
        for pred_name, count in sorted(preds.items(), key=lambda x: -x[1]):
            print(f"       → {pred_name}: {count}")

    print("\n   Test set (Collatz - HELD OUT):")
    test_final = evaluate_classification(
        classifier, model, tokenizer,
        test_examples, config
    )
    print(f"   Accuracy: {test_final['accuracy']:.1%}")
    print("   Confusion matrix:")
    for true_name, preds in test_final["confusion"].items():
        print(f"     {true_name}:")
        for pred_name, count in sorted(preds.items(), key=lambda x: -x[1]):
            print(f"       → {pred_name}: {count}")

    # End-to-end execution
    print("\n6. End-to-End Execution Test...")
    exec_eval = evaluate_execution(
        classifier, model, tokenizer,
        test_examples, config
    )
    print(f"   Template Accuracy: {exec_eval['template_accuracy']:.1%}")
    print(f"   Execution Accuracy: {exec_eval['execution_accuracy']:.1%}")

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"""
Training (Balanced):
  Template Accuracy: {train_final['accuracy']:.1%}

Test (Collatz - UNSEEN):
  Template Accuracy: {test_final['accuracy']:.1%}
  Execution Accuracy: {exec_eval['execution_accuracy']:.1%}

Best Test Accuracy: {best_test_acc:.1%}

Verdict: {"SUCCESS" if test_final['accuracy'] > 0.5 else "PARTIAL" if test_final['accuracy'] > 0.1 else "NEEDS MORE WORK"}
""")
    print("=" * 60)

    # Save results
    results = {
        "config": {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "examples_per_template": config.examples_per_template,
        },
        "train_accuracy": train_final["accuracy"],
        "test_accuracy": test_final["accuracy"],
        "execution_accuracy": exec_eval["execution_accuracy"],
        "best_test_accuracy": best_test_acc,
    }

    results_path = results_dir / "balanced_training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to: {results_path}")


if __name__ == "__main__":
    main()
