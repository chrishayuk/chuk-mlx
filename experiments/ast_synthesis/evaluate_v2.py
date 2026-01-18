"""
Evaluation v2: With Context-Aware Operand Extraction

This version uses the improved operand extractor that correctly handles
phrases like "reach 1 from" and "from 1 to N".

Expected improvement: 80% → ~94% execution accuracy
"""

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from templates import TemplateID, NUM_TEMPLATES, template_name
from slot_filler import fill_slots, detect_program_hint
from linearize import linearize, execute_ir
from operand_extractor import extract_operands  # NEW: context-aware extractor


# =============================================================================
# VOCABULARY TRANSFER DATA
# =============================================================================

def generate_collatz_vocabulary_examples(n: int = 200) -> list[dict]:
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

        examples.append({
            "nl_input": nl,
            "program_name": "vocabulary_transfer",
            "template_id": int(TemplateID.LOOP_CONDITIONAL_ACCUMULATE),
            "template_name": "LOOP_CONDITIONAL_ACCUMULATE",
            "operands": [num],
            "expected_result": 0,
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


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_full_pipeline(classifier, model, tokenizer, examples, config):
    results = {
        "template_correct": 0,
        "execution_correct": 0,
        "total": 0,
        "details": [],
    }

    for ex in examples:
        results["total"] += 1
        nl = ex["nl_input"]
        expected = ex["expected_result"]
        true_template = TemplateID(ex["template_id"])

        # Step 1: Template classification
        hidden = get_hidden_state(model, tokenizer, nl, config.hidden_layer)
        logits = classifier(hidden[None, :])
        pred_template = TemplateID(int(mx.argmax(logits[0]).item()))

        template_correct = (pred_template == true_template)
        if template_correct:
            results["template_correct"] += 1

        # Step 2: Extract operands (IMPROVED - context-aware)
        operands = extract_operands(nl)

        # Step 3: Detect program hint
        program_hint = detect_program_hint(nl, pred_template)

        if program_hint is None:
            results["details"].append({
                "nl": nl,
                "template_correct": template_correct,
                "execution_correct": False,
                "error": "No program hint",
            })
            continue

        # Step 4: Fill slots
        ast = fill_slots(pred_template, program_hint, operands)

        if ast is None:
            results["details"].append({
                "nl": nl,
                "template_correct": template_correct,
                "execution_correct": False,
                "error": "No slot rules",
            })
            continue

        # Step 5: Linearize and execute
        try:
            ir_opcodes = linearize(ast)
            success, result, error = execute_ir(ir_opcodes, operands)

            execution_correct = success and result == expected

            if execution_correct:
                results["execution_correct"] += 1

            results["details"].append({
                "nl": nl[:50],
                "template_correct": template_correct,
                "pred_template": pred_template.name,
                "true_template": true_template.name,
                "execution_correct": execution_correct,
                "result": result,
                "expected": expected,
                "operands": operands,
            })

        except Exception as e:
            results["details"].append({
                "nl": nl,
                "template_correct": template_correct,
                "execution_correct": False,
                "error": str(e),
            })

    results["template_accuracy"] = results["template_correct"] / results["total"]
    results["execution_accuracy"] = results["execution_correct"] / results["total"]

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("EVALUATION v2: With Context-Aware Operand Extraction")
    print("=" * 70)

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
    vocab_transfer = generate_collatz_vocabulary_examples(200)

    # Combine training data
    train_with_vocab = train_examples + vocab_transfer
    random.shuffle(train_with_vocab)

    print(f"   Training: {len(train_with_vocab)} examples")
    print(f"   Test (Collatz): {len(test_examples)} examples")

    # Train classifier
    print("\n3. Training classifier with vocabulary transfer...")
    config = Config()
    start = time.perf_counter()
    classifier = train_classifier(model, tokenizer, train_with_vocab, config)
    elapsed = time.perf_counter() - start
    print(f"   Training completed in {elapsed:.1f}s")

    # Evaluate training set
    print("\n4. Evaluating training set...")
    train_eval = evaluate_full_pipeline(classifier, model, tokenizer, train_examples[:200], config)
    print(f"   Template Accuracy: {train_eval['template_accuracy']:.1%}")
    print(f"   Execution Accuracy: {train_eval['execution_accuracy']:.1%}")

    # Evaluate test set (Collatz)
    print("\n5. Evaluating test set (Collatz - HELD OUT)...")
    test_eval = evaluate_full_pipeline(classifier, model, tokenizer, test_examples, config)
    print(f"   Template Accuracy: {test_eval['template_accuracy']:.1%}")
    print(f"   Execution Accuracy: {test_eval['execution_accuracy']:.1%}")

    # Show detailed results
    print("\n6. Detailed test results (first 15):")
    print("-" * 70)
    for i, detail in enumerate(test_eval["details"][:15]):
        t_status = "T-OK" if detail.get("template_correct") else "T-ERR"
        e_status = "E-OK" if detail.get("execution_correct") else "E-ERR"
        print(f"   [{t_status}] [{e_status}] {detail.get('nl', '')[:45]}...")
        if detail.get("execution_correct"):
            print(f"      Operands: {detail.get('operands')}, Result: {detail.get('result')} (correct!)")
        elif detail.get("error"):
            print(f"      Error: {detail.get('error')}")
        else:
            print(f"      Operands: {detail.get('operands')}, Result: {detail.get('result')}, Expected: {detail.get('expected')}")

    # Error analysis
    print("\n7. Error Analysis:")
    errors = [d for d in test_eval["details"] if not d.get("execution_correct")]
    if errors:
        print(f"   {len(errors)} errors out of {test_eval['total']} examples")
        template_errors = [d for d in errors if not d.get("template_correct")]
        exec_errors = [d for d in errors if d.get("template_correct")]
        print(f"   - Template classification errors: {len(template_errors)}")
        print(f"   - Execution errors (correct template): {len(exec_errors)}")

        if exec_errors:
            print("\n   Execution errors with correct template:")
            for e in exec_errors[:3]:
                print(f"     NL: {e.get('nl', '')[:40]}...")
                print(f"     Operands: {e.get('operands')}, Result: {e.get('result')}, Expected: {e.get('expected')}")
    else:
        print("   No errors! Perfect execution!")

    # Save results
    print("\n8. Saving results...")
    final_results = {
        "version": "v2_context_aware_extraction",
        "train": {
            "template_accuracy": train_eval["template_accuracy"],
            "execution_accuracy": train_eval["execution_accuracy"],
        },
        "test": {
            "template_accuracy": test_eval["template_accuracy"],
            "execution_accuracy": test_eval["execution_accuracy"],
        },
        "improvement": {
            "v1_execution": 0.80,
            "v2_execution": test_eval["execution_accuracy"],
        }
    }

    results_path = results_dir / "evaluation_v2_results.json"
    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"   Saved to: {results_path}")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    print(f"""
| Version | Operand Extraction | Template Acc | Execution Acc |
|---------|-------------------|--------------|---------------|
| v1      | Naive regex       |        94.0% |         80.0% |
| v2      | Context-aware     | {test_eval['template_accuracy']:>12.1%} |  {test_eval['execution_accuracy']:>12.1%} |

Improvement: {80.0:.1%} → {test_eval['execution_accuracy']:.1%} ({(test_eval['execution_accuracy'] - 0.80) * 100:+.1f}%)
""")
    print("=" * 70)


if __name__ == "__main__":
    main()
