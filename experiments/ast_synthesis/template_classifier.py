"""
Template Classifier for AST-Based IR Synthesis

Classifies NL input into template IDs using frozen LLM hidden states.

This is the core experiment:
  - Train: Learn mapping from hidden states to templates
  - Test: Can the model generalize to Collatz?

Key insight: We classify into TEMPLATES (structural equivalence classes),
not PROGRAMS. This should enable compositional generalization.
"""

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from templates import TemplateID, NUM_TEMPLATES, template_name


# =============================================================================
# MODEL
# =============================================================================

class TemplateClassifier(nn.Module):
    """
    Classifies NL input into one of the template IDs.

    Architecture matches the proven operation classifier from
    previous experiments (which achieved 100% accuracy).

    Input: Hidden state from LLM at last token position (Layer 13)
    Output: Logits over template classes
    """

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
# OPERAND EXTRACTION
# =============================================================================

def extract_operands(text: str) -> list[int]:
    """Extract numeric operands from NL text."""
    numbers = re.findall(r'\b(\d+)\b', text)
    return [int(n) for n in numbers]


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 20
    hidden_layer: int = 13  # Which layer to extract hidden state from (L13)
    eval_every: int = 5


# =============================================================================
# DATA LOADING
# =============================================================================

def load_dataset(path: Path) -> list[dict]:
    """Load dataset from JSON."""
    with open(path) as f:
        data = json.load(f)
    return data["examples"]


# =============================================================================
# HIDDEN STATE EXTRACTION
# =============================================================================

def get_hidden_state(model, tokenizer, text: str, layer: int) -> mx.array:
    """
    Get hidden state from model at specified layer.

    This is from the frozen TinyLlama - we're reading out what
    the model already knows about structure via attention.
    """
    tokens = tokenizer.encode(text)
    input_ids = mx.array([tokens])

    # Get hidden states by running partial forward
    hidden = model.model.embed_tokens(input_ids)

    for i, block in enumerate(model.model.layers[:layer]):
        hidden = block(hidden, mask=None, cache=None)
        # Handle BlockOutput wrapper if present
        if hasattr(hidden, 'hidden_states'):
            hidden = hidden.hidden_states
        elif isinstance(hidden, tuple):
            hidden = hidden[0]

    # Return last token position
    return hidden[0, -1, :]


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(
    classifier: TemplateClassifier,
    optimizer: optim.Optimizer,
    model,
    tokenizer,
    examples: list[dict],
    config: TrainingConfig,
) -> float:
    """Train for one epoch."""
    import random
    random.shuffle(examples)

    def loss_fn(classifier, hidden_states, labels):
        logits = classifier(hidden_states)
        return nn.losses.cross_entropy(logits, labels).mean()

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
    model,
    tokenizer,
    examples: list[dict],
    config: TrainingConfig,
) -> dict:
    """
    Evaluate template classification accuracy.

    This is the PRIMARY metric: can we predict the correct template?
    """
    correct = 0
    total = 0
    details = []

    for ex in examples:
        total += 1

        hidden = get_hidden_state(model, tokenizer, ex["nl_input"], config.hidden_layer)
        logits = classifier(hidden[None, :])
        pred_template = int(mx.argmax(logits[0]).item())

        true_template = ex["template_id"]
        is_correct = pred_template == true_template

        if is_correct:
            correct += 1

        details.append({
            "nl": ex["nl_input"],
            "true_template": template_name(TemplateID(true_template)),
            "pred_template": template_name(TemplateID(pred_template)),
            "correct": is_correct,
            "program": ex.get("program_name", "unknown"),
        })

    return {
        "accuracy": correct / total if total > 0 else 0,
        "correct": correct,
        "total": total,
        "details": details,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("AST-Based IR Synthesis - Template Classifier Training")
    print("=" * 60)

    # Load model
    print("\n1. Loading base model (frozen)...")
    from chuk_lazarus.models_v2.loader import load_model

    loaded = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = loaded.model
    tokenizer = loaded.tokenizer
    model.freeze()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    mx.eval(model.parameters())
    print("   Model loaded and frozen.")

    # Load datasets
    print("\n2. Loading datasets...")
    results_dir = Path(__file__).parent / "results"
    train_examples = load_dataset(results_dir / "train_dataset.json")
    test_examples = load_dataset(results_dir / "test_dataset.json")
    print(f"   Train: {len(train_examples)} examples")
    print(f"   Test: {len(test_examples)} examples (Collatz - held out)")

    # Show template distribution
    print("\n3. Template distribution:")
    for name, examples in [("Train", train_examples), ("Test", test_examples)]:
        counts = {}
        for ex in examples:
            tid = ex["template_id"]
            counts[tid] = counts.get(tid, 0) + 1
        print(f"   {name}:")
        for tid, count in sorted(counts.items()):
            print(f"     {template_name(TemplateID(tid))}: {count}")

    # Get hidden dimension
    hidden_dim = model.model.layers[0].self_attn.q_proj.weight.shape[0]
    print(f"\n4. Hidden dimension: {hidden_dim}")

    # Create classifier
    print("\n5. Creating template classifier...")
    config = TrainingConfig()
    classifier = TemplateClassifier(
        hidden_dim=hidden_dim,
        num_templates=NUM_TEMPLATES,
    )

    optimizer = optim.Adam(learning_rate=config.learning_rate)

    # Training loop
    print("\n6. Training...")
    print("-" * 60)
    print(f"{'Epoch':>6} | {'Loss':>8} | {'Train Acc':>10} | {'Test Acc':>10} | {'Time':>8}")
    print("-" * 60)

    best_test_acc = 0

    for epoch in range(config.epochs):
        start = time.perf_counter()

        train_loss = train_epoch(
            classifier, optimizer, model, tokenizer,
            train_examples, config
        )

        elapsed = time.perf_counter() - start

        if (epoch + 1) % config.eval_every == 0 or epoch == 0 or epoch == config.epochs - 1:
            train_eval = evaluate_classification(
                classifier, model, tokenizer,
                train_examples[:200], config
            )
            test_eval = evaluate_classification(
                classifier, model, tokenizer,
                test_examples, config
            )

            print(f"{epoch+1:6d} | {train_loss:8.4f} | {train_eval['accuracy']:10.1%} | "
                  f"{test_eval['accuracy']:10.1%} | {elapsed:7.1f}s")

            if test_eval['accuracy'] > best_test_acc:
                best_test_acc = test_eval['accuracy']
        else:
            print(f"{epoch+1:6d} | {train_loss:8.4f} | {'--':>10} | {'--':>10} | {elapsed:7.1f}s")

    print("-" * 60)

    # Final evaluation
    print("\n7. Final Evaluation...")

    print("\n   Training set:")
    train_final = evaluate_classification(
        classifier, model, tokenizer,
        train_examples, config
    )
    print(f"   Accuracy: {train_final['correct']}/{train_final['total']} = {train_final['accuracy']:.1%}")

    print("\n   Test set (Collatz - UNSEEN):")
    test_final = evaluate_classification(
        classifier, model, tokenizer,
        test_examples, config
    )
    print(f"   Accuracy: {test_final['correct']}/{test_final['total']} = {test_final['accuracy']:.1%}")

    # Show test predictions
    print("\n   Sample test predictions:")
    for detail in test_final["details"][:10]:
        status = "CORRECT" if detail["correct"] else "WRONG"
        print(f"   [{status}] '{detail['nl'][:45]}...'")
        print(f"           pred: {detail['pred_template']}, true: {detail['true_template']}")

    # Analyze errors
    print("\n8. Error Analysis:")
    errors = [d for d in test_final["details"] if not d["correct"]]
    if errors:
        pred_counts = {}
        for e in errors:
            pred = e["pred_template"]
            pred_counts[pred] = pred_counts.get(pred, 0) + 1
        print(f"   Errors by predicted template:")
        for pred, count in sorted(pred_counts.items(), key=lambda x: -x[1]):
            print(f"     {pred}: {count}")
    else:
        print("   No errors!")

    # Save results
    print("\n9. Saving results...")
    results = {
        "config": {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "hidden_layer": config.hidden_layer,
        },
        "train": {
            "accuracy": train_final["accuracy"],
            "total": train_final["total"],
        },
        "test": {
            "accuracy": test_final["accuracy"],
            "total": test_final["total"],
            "details": test_final["details"],
        },
        "best_test_accuracy": best_test_acc,
    }

    results_path = results_dir / "template_classifier_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"   Saved to: {results_path}")

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"""
Training Accuracy: {train_final['accuracy']:.1%}
Test Accuracy (Collatz): {test_final['accuracy']:.1%}

Comparison to baseline:
  - Original program classifier: 0% on Collatz (no Collatz class)
  - Seq2seq generation: 0% on Collatz (no compositional generalization)
  - Template classifier: {test_final['accuracy']:.1%} on Collatz

{"SUCCESS" if test_final['accuracy'] > 0.5 else "PARTIAL" if test_final['accuracy'] > 0 else "FAILED"}:
  {"The model learned to generalize structure!" if test_final['accuracy'] > 0.5 else "Some generalization but needs improvement" if test_final['accuracy'] > 0 else "No generalization achieved"}
""")
    print("=" * 60)


if __name__ == "__main__":
    main()
