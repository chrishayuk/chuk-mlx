"""
End-to-End Evaluation for AST-Based IR Synthesis

Full pipeline:
  NL Input → Template Classifier → Template ID
           → Operand Extractor → Operands
           → Slot Filler → Filled AST
           → Linearizer → IR Opcodes
           → WASM Compiler → Execute
           → Result

This validates the complete system, not just template classification.
"""

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from templates import TemplateID, NUM_TEMPLATES, template_name
from slot_filler import fill_slots, detect_program_hint
from linearize import linearize, execute_ir


# =============================================================================
# MODEL (same as template_classifier.py)
# =============================================================================

class TemplateClassifier(nn.Module):
    """Classifies NL input into template IDs."""

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
# CONFIGURATION
# =============================================================================

@dataclass
class EvalConfig:
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 20
    hidden_layer: int = 13


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

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


# =============================================================================
# TRAINING
# =============================================================================

def train_classifier(
    model, tokenizer, train_examples: list[dict], config: EvalConfig
) -> TemplateClassifier:
    """Train template classifier."""
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
# END-TO-END EVALUATION
# =============================================================================

@dataclass
class EvalResult:
    """Result of evaluating a single example."""
    nl_input: str
    expected_result: int
    true_template: TemplateID
    pred_template: Optional[TemplateID]
    template_correct: bool
    actual_result: Optional[int]
    execution_correct: bool
    error: Optional[str]


def evaluate_example(
    classifier: TemplateClassifier,
    model, tokenizer,
    example: dict,
    config: EvalConfig,
) -> EvalResult:
    """
    Evaluate a single example through the full pipeline.

    Pipeline:
      1. Extract hidden state from LLM
      2. Classify into template
      3. Extract operands from text
      4. Detect program hint for slot filling
      5. Fill template slots
      6. Linearize AST to IR
      7. Execute and check result
    """
    nl_input = example["nl_input"]
    expected = example["expected_result"]
    true_template = TemplateID(example["template_id"])

    # Step 1-2: Template classification
    hidden = get_hidden_state(model, tokenizer, nl_input, config.hidden_layer)
    logits = classifier(hidden[None, :])
    pred_template_idx = int(mx.argmax(logits[0]).item())
    pred_template = TemplateID(pred_template_idx)

    template_correct = (pred_template == true_template)

    # Step 3: Extract operands
    operands = extract_operands(nl_input)

    # Step 4: Detect program hint
    program_hint = detect_program_hint(nl_input, pred_template)

    if program_hint is None:
        return EvalResult(
            nl_input=nl_input,
            expected_result=expected,
            true_template=true_template,
            pred_template=pred_template,
            template_correct=template_correct,
            actual_result=None,
            execution_correct=False,
            error="No program hint detected",
        )

    # Step 5: Fill template slots
    ast = fill_slots(pred_template, program_hint, operands)

    if ast is None:
        return EvalResult(
            nl_input=nl_input,
            expected_result=expected,
            true_template=true_template,
            pred_template=pred_template,
            template_correct=template_correct,
            actual_result=None,
            execution_correct=False,
            error=f"No slot rules for ({pred_template.name}, {program_hint})",
        )

    # Step 6-7: Linearize and execute
    try:
        ir_opcodes = linearize(ast)
        success, result, error = execute_ir(ir_opcodes, operands)

        if not success:
            return EvalResult(
                nl_input=nl_input,
                expected_result=expected,
                true_template=true_template,
                pred_template=pred_template,
                template_correct=template_correct,
                actual_result=None,
                execution_correct=False,
                error=f"Execution failed: {error}",
            )

        execution_correct = (result == expected)

        return EvalResult(
            nl_input=nl_input,
            expected_result=expected,
            true_template=true_template,
            pred_template=pred_template,
            template_correct=template_correct,
            actual_result=result,
            execution_correct=execution_correct,
            error=None if execution_correct else f"Result mismatch: {result} != {expected}",
        )

    except Exception as e:
        return EvalResult(
            nl_input=nl_input,
            expected_result=expected,
            true_template=true_template,
            pred_template=pred_template,
            template_correct=template_correct,
            actual_result=None,
            execution_correct=False,
            error=str(e),
        )


def evaluate_dataset(
    classifier: TemplateClassifier,
    model, tokenizer,
    examples: list[dict],
    config: EvalConfig,
) -> dict:
    """Evaluate a full dataset."""
    results = []
    template_correct = 0
    execution_correct = 0

    for ex in examples:
        result = evaluate_example(classifier, model, tokenizer, ex, config)
        results.append(result)

        if result.template_correct:
            template_correct += 1
        if result.execution_correct:
            execution_correct += 1

    total = len(results)

    return {
        "template_accuracy": template_correct / total if total > 0 else 0,
        "execution_accuracy": execution_correct / total if total > 0 else 0,
        "template_correct": template_correct,
        "execution_correct": execution_correct,
        "total": total,
        "results": results,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("AST-Based IR Synthesis - End-to-End Evaluation")
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
    print(f"   Train: {len(train_examples)} examples")
    print(f"   Test: {len(test_examples)} examples")

    # Train classifier
    print("\n3. Training template classifier...")
    config = EvalConfig()
    start = time.perf_counter()
    classifier = train_classifier(model, tokenizer, train_examples, config)
    elapsed = time.perf_counter() - start
    print(f"   Training completed in {elapsed:.1f}s")

    # Evaluate training set
    print("\n4. Evaluating training set...")
    train_eval = evaluate_dataset(classifier, model, tokenizer, train_examples[:200], config)
    print(f"   Template Accuracy: {train_eval['template_accuracy']:.1%}")
    print(f"   Execution Accuracy: {train_eval['execution_accuracy']:.1%}")

    # Evaluate test set (Collatz)
    print("\n5. Evaluating test set (Collatz - HELD OUT)...")
    test_eval = evaluate_dataset(classifier, model, tokenizer, test_examples, config)
    print(f"   Template Accuracy: {test_eval['template_accuracy']:.1%}")
    print(f"   Execution Accuracy: {test_eval['execution_accuracy']:.1%}")

    # Show detailed results
    print("\n6. Detailed test results:")
    print("-" * 60)

    for i, result in enumerate(test_eval["results"][:15]):
        status_t = "T-OK" if result.template_correct else "T-ERR"
        status_e = "E-OK" if result.execution_correct else "E-ERR"

        print(f"\n   Example {i+1}: [{status_t}] [{status_e}]")
        print(f"   Input: {result.nl_input[:50]}...")
        print(f"   Template: pred={result.pred_template.name}, true={result.true_template.name}")
        print(f"   Result: actual={result.actual_result}, expected={result.expected_result}")
        if result.error:
            print(f"   Error: {result.error}")

    # Error analysis
    print("\n7. Error Analysis:")
    errors = [r for r in test_eval["results"] if not r.template_correct]
    if errors:
        pred_counts = {}
        for r in errors:
            pred = r.pred_template.name
            pred_counts[pred] = pred_counts.get(pred, 0) + 1
        print(f"   Template errors by predicted class:")
        for pred, count in sorted(pred_counts.items(), key=lambda x: -x[1]):
            print(f"     {pred}: {count}")
    else:
        print("   No template classification errors!")

    exec_errors = [r for r in test_eval["results"] if r.template_correct and not r.execution_correct]
    if exec_errors:
        print(f"\n   Execution errors (correct template):")
        for r in exec_errors[:5]:
            print(f"     {r.nl_input[:40]}...")
            print(f"       {r.error}")

    # Save results
    print("\n8. Saving results...")
    results = {
        "config": {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "hidden_layer": config.hidden_layer,
        },
        "train": {
            "template_accuracy": train_eval["template_accuracy"],
            "execution_accuracy": train_eval["execution_accuracy"],
            "total": train_eval["total"],
        },
        "test": {
            "template_accuracy": test_eval["template_accuracy"],
            "execution_accuracy": test_eval["execution_accuracy"],
            "total": test_eval["total"],
        },
    }

    results_path = results_dir / "end_to_end_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"   Saved to: {results_path}")

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"""
TRAINING SET:
  Template Classification: {train_eval['template_accuracy']:.1%}
  End-to-End Execution:    {train_eval['execution_accuracy']:.1%}

TEST SET (Collatz - UNSEEN):
  Template Classification: {test_eval['template_accuracy']:.1%}
  End-to-End Execution:    {test_eval['execution_accuracy']:.1%}

COMPARISON TO BASELINES:
  | Approach                  | Test Template | Test Execution |
  |---------------------------|---------------|----------------|
  | Program Classifier        | N/A           | 0%             |
  | Seq2seq Generation        | N/A           | 0%             |
  | AST Template Classifier   | {test_eval['template_accuracy']:>12.1%} | {test_eval['execution_accuracy']:>14.1%} |

VERDICT: {"SUCCESS" if test_eval['execution_accuracy'] > 0.5 else "PARTIAL SUCCESS" if test_eval['execution_accuracy'] > 0 else "NEEDS WORK"}
  {"Compositional generalization achieved!" if test_eval['execution_accuracy'] > 0.5 else "Some generalization shown" if test_eval['execution_accuracy'] > 0 else "No generalization"}
""")
    print("=" * 60)


if __name__ == "__main__":
    main()
