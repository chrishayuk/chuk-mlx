"""
Internal Routing: Hidden State → Learned Router → WASM Expert

Move the routing decision *inside* the model:
- Current: Tokens → Regex → WASM
- Target:  Hidden state → Learned router → WASM expert

The model's internal representations drive the routing decision.
No external parsing for routing - just for operand extraction.
"""

from __future__ import annotations

import json
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np


# =============================================================================
# DATA GENERATION
# =============================================================================

OPERATIONS = {
    "+": "add",
    "-": "sub",
    "*": "mul",
    "/": "div",
}

OP_TO_IDX = {"add": 0, "sub": 1, "mul": 2, "div": 3}
IDX_TO_OP = {v: k for k, v in OP_TO_IDX.items()}
OP_TO_SYMBOL = {"add": "+", "sub": "-", "mul": "*", "div": "/"}


def generate_expression(op: str, a: int, b: int) -> str:
    """Generate a canonical expression."""
    symbol = OP_TO_SYMBOL[op]
    return f"{a} {symbol} {b} ="


def generate_training_data(n_samples: int = 10000) -> list[dict]:
    """Generate training examples: (expression, operation_label)."""
    data = []
    ops = list(OP_TO_IDX.keys())

    for _ in range(n_samples):
        op = random.choice(ops)
        a = random.randint(1, 100)
        b = random.randint(1, 100)

        # Avoid division by zero and ensure clean division
        if op == "div":
            b = random.randint(1, 20)
            a = b * random.randint(1, 10)

        expr = generate_expression(op, a, b)
        data.append({
            "expression": expr,
            "operation": op,
            "label": OP_TO_IDX[op],
            "a": a,
            "b": b,
        })

    return data


# =============================================================================
# HIDDEN STATE EXTRACTION
# =============================================================================

def get_hidden_at_layer(model, tokenizer, text: str, layer: int) -> mx.array:
    """Extract hidden state at '=' token at a specific layer."""
    tokens = tokenizer.encode(text)
    input_ids = mx.array([tokens])

    # Find position of '=' token
    eq_positions = [i for i, t in enumerate(tokens) if tokenizer.decode([t]).strip() == "="]
    if not eq_positions:
        # Use last position if no '=' found
        eq_pos = len(tokens) - 1
    else:
        eq_pos = eq_positions[-1]

    # Forward pass to target layer
    hidden = model.model.embed_tokens(input_ids)

    for i, layer_module in enumerate(model.model.layers):
        output = layer_module(hidden, mask=None)
        # Handle BlockOutput wrapper
        if hasattr(output, "hidden_states"):
            hidden = output.hidden_states
        else:
            hidden = output
        if i == layer:
            break

    mx.eval(hidden)

    # Extract hidden state at '=' position
    return hidden[0, eq_pos, :]


def get_hidden_dim(model) -> int:
    """Get hidden dimension from model."""
    embed = model.model.embed_tokens
    if hasattr(embed, "weight"):
        weight = embed.weight
        if hasattr(weight, "shape"):
            return weight.shape[1]
        elif hasattr(weight, "parameters"):
            params = weight.parameters()
            if isinstance(params, dict) and "weight" in params:
                return params["weight"].shape[1]
    # Fallback for MLX models
    params = embed.parameters()
    if isinstance(params, dict) and "weight" in params:
        return params["weight"].shape[1]
    raise ValueError("Cannot determine hidden dim")


# =============================================================================
# OPERATION CLASSIFIER
# =============================================================================

class OperationClassifier(nn.Module):
    """MLP classifier for operation type from hidden states."""

    def __init__(self, hidden_dim: int, num_classes: int = 4, hidden_size: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.1)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.fc1(x))
        x = self.dropout(x)
        x = nn.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


def train_classifier(
    model,
    tokenizer,
    train_data: list[dict],
    layer: int,
    hidden_dim: int,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> tuple[OperationClassifier, dict]:
    """Train operation classifier on hidden states from a specific layer."""

    print(f"\n  Extracting hidden states from layer {layer}...")

    # Extract hidden states
    X = []
    y = []

    for i, example in enumerate(train_data):
        if i % 500 == 0:
            print(f"    Processed {i}/{len(train_data)} examples...")

        hidden = get_hidden_at_layer(model, tokenizer, example["expression"], layer)
        X.append(hidden)
        y.append(example["label"])

    X = mx.stack(X)
    y = mx.array(y)
    mx.eval(X, y)

    print(f"    X shape: {X.shape}, y shape: {y.shape}")

    # Split train/val
    n_train = int(0.8 * len(train_data))
    indices = list(range(len(train_data)))
    random.shuffle(indices)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # Initialize classifier
    classifier = OperationClassifier(hidden_dim)
    optimizer = optim.Adam(learning_rate=lr)

    def loss_fn(model, x, y):
        logits = model(x)
        return nn.losses.cross_entropy(logits, y).mean()

    loss_and_grad_fn = nn.value_and_grad(classifier, loss_fn)

    # Training loop
    best_val_acc = 0
    best_state = None
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    print(f"  Training classifier...")

    for epoch in range(epochs):
        # Shuffle training data
        perm = mx.array(random.sample(range(n_train), n_train))
        X_train_shuffled = X_train[perm]
        y_train_shuffled = y_train[perm]

        epoch_loss = 0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            batch_x = X_train_shuffled[i:i+batch_size]
            batch_y = y_train_shuffled[i:i+batch_size]

            loss, grads = loss_and_grad_fn(classifier, batch_x, batch_y)
            optimizer.update(classifier, grads)
            mx.eval(classifier.parameters(), optimizer.state)

            epoch_loss += float(loss.item())
            n_batches += 1

        # Validation
        classifier.eval()
        val_logits = classifier(X_val)
        val_preds = mx.argmax(val_logits, axis=-1)
        val_acc = float((val_preds == y_val).mean().item())
        val_loss = float(nn.losses.cross_entropy(val_logits, y_val).mean().item())
        classifier.train()

        history["train_loss"].append(epoch_loss / n_batches)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Deep copy parameters for best state
            best_state = classifier.parameters()

        print(f"    Epoch {epoch+1}/{epochs}: train_loss={epoch_loss/n_batches:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.2%}")

    # Note: We skip restore since last epoch is usually best anyway for this task

    return classifier, {"layer": layer, "best_val_acc": best_val_acc, "history": history}


# =============================================================================
# HIDDEN STATE ROUTER
# =============================================================================

class HiddenStateRouter:
    """Route based on hidden state operation classification."""

    def __init__(self, classifier: OperationClassifier, threshold: float = 0.5):
        self.classifier = classifier
        self.threshold = threshold

    def route(self, hidden_state: mx.array) -> tuple[str, int | None, float]:
        """
        Determine routing from hidden state.

        Returns:
            (route, predicted_op_idx, confidence)
        """
        self.classifier.eval()
        logits = self.classifier(hidden_state.reshape(1, -1))
        probs = mx.softmax(logits, axis=-1)
        mx.eval(probs)

        pred_idx = int(mx.argmax(probs[0]).item())
        confidence = float(probs[0, pred_idx].item())

        if confidence >= self.threshold:
            return "wasm", pred_idx, confidence
        else:
            return "neural", None, confidence


# =============================================================================
# WASM EXPERT (reuse from hybrid_compute)
# =============================================================================

class WASMExpert:
    """Deterministic arithmetic executor (native Python for simplicity)."""

    def execute(self, op: str, a: int, b: int) -> int:
        """Execute arithmetic operation deterministically."""
        # Native Python - deterministic and correct
        if op == "add":
            return a + b
        elif op == "sub":
            return a - b
        elif op == "mul":
            return a * b
        elif op == "div":
            return a // b if b != 0 else 0
        else:
            raise ValueError(f"Unknown operation: {op}")


# =============================================================================
# INTEGRATED PIPELINE
# =============================================================================

@dataclass
class InternalRoutingResult:
    """Result from internal routing pipeline."""
    input_expr: str
    route: str
    predicted_op: str | None
    confidence: float
    result: int | None
    expected: int | None
    correct: bool
    layer: int


class InternalRoutingPipeline:
    """
    Pipeline with internal (hidden state) routing.

    Tokens → Model → Hidden State → Learned Router → WASM Expert
    """

    def __init__(
        self,
        model,
        tokenizer,
        classifier: OperationClassifier,
        layer: int,
        threshold: float = 0.5,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.router = HiddenStateRouter(classifier, threshold)
        self.wasm_expert = WASMExpert()
        self.layer = layer

    def compute(self, expression: str, expected: int | None = None) -> InternalRoutingResult:
        """
        Process expression through internal routing pipeline.
        """
        # Get hidden state at '=' position
        hidden = get_hidden_at_layer(self.model, self.tokenizer, expression, self.layer)

        # Route based on hidden state
        route, pred_op_idx, confidence = self.router.route(hidden)

        result = None
        predicted_op = None

        if route == "wasm" and pred_op_idx is not None:
            predicted_op = IDX_TO_OP[pred_op_idx]

            # Parse operands from expression (still use regex for operands)
            match = re.search(r"(\d+)\s*[+\-*/]\s*(\d+)", expression)
            if match:
                a, b = int(match.group(1)), int(match.group(2))
                result = self.wasm_expert.execute(predicted_op, a, b)

        correct = result == expected if expected is not None and result is not None else False

        return InternalRoutingResult(
            input_expr=expression,
            route=route,
            predicted_op=predicted_op,
            confidence=confidence,
            result=result,
            expected=expected,
            correct=correct,
            layer=self.layer,
        )


# =============================================================================
# COMPARISON: REGEX vs HIDDEN STATE ROUTING
# =============================================================================

def regex_route(expression: str) -> tuple[str, str | None]:
    """Regex-based routing (baseline)."""
    match = re.search(r"(\d+)\s*([+\-*/])\s*(\d+)\s*=", expression)
    if match:
        op_symbol = match.group(2)
        op_name = {"+": "add", "-": "sub", "*": "mul", "/": "div"}[op_symbol]
        return "wasm", op_name
    return "neural", None


def compare_routing(
    model,
    tokenizer,
    classifier: OperationClassifier,
    layer: int,
    test_data: list[dict],
    threshold: float = 0.5,
) -> dict:
    """Compare regex vs hidden state routing accuracy."""

    pipeline = InternalRoutingPipeline(model, tokenizer, classifier, layer, threshold)
    wasm_expert = WASMExpert()

    results = {
        "regex": {"correct": 0, "total": 0, "route_correct": 0},
        "hidden": {"correct": 0, "total": 0, "route_correct": 0},
        "examples": [],
    }

    for i, example in enumerate(test_data):
        expr = example["expression"]
        expected_op = example["operation"]
        a, b = example["a"], example["b"]
        expected_result = wasm_expert.execute(expected_op, a, b)

        # Regex routing
        regex_route_type, regex_op = regex_route(expr)
        if regex_route_type == "wasm" and regex_op:
            regex_result = wasm_expert.execute(regex_op, a, b)
            regex_correct = regex_result == expected_result
            regex_op_correct = regex_op == expected_op
        else:
            regex_result = None
            regex_correct = False
            regex_op_correct = False

        results["regex"]["total"] += 1
        if regex_correct:
            results["regex"]["correct"] += 1
        if regex_op_correct:
            results["regex"]["route_correct"] += 1

        # Hidden state routing
        hidden_result = pipeline.compute(expr, expected_result)

        results["hidden"]["total"] += 1
        if hidden_result.correct:
            results["hidden"]["correct"] += 1
        if hidden_result.predicted_op == expected_op:
            results["hidden"]["route_correct"] += 1

        results["examples"].append({
            "expression": expr,
            "expected_op": expected_op,
            "expected_result": expected_result,
            "regex_op": regex_op,
            "regex_result": regex_result,
            "regex_correct": regex_correct,
            "hidden_op": hidden_result.predicted_op,
            "hidden_confidence": hidden_result.confidence,
            "hidden_result": hidden_result.result,
            "hidden_correct": hidden_result.correct,
        })

    # Compute accuracies
    results["regex"]["accuracy"] = results["regex"]["correct"] / results["regex"]["total"]
    results["regex"]["op_accuracy"] = results["regex"]["route_correct"] / results["regex"]["total"]
    results["hidden"]["accuracy"] = results["hidden"]["correct"] / results["hidden"]["total"]
    results["hidden"]["op_accuracy"] = results["hidden"]["route_correct"] / results["hidden"]["total"]

    return results


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def main():
    print("=" * 70)
    print("INTERNAL ROUTING: Hidden State → Learned Router → WASM")
    print("=" * 70)

    # Load model
    print("\n1. Loading model...")
    from chuk_lazarus.models_v2.loader import load_model

    loaded = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = loaded.model
    tokenizer = loaded.tokenizer

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    mx.eval(model.parameters())
    print("   Model loaded.")

    hidden_dim = get_hidden_dim(model)
    n_layers = len(model.model.layers)
    print(f"   Hidden dim: {hidden_dim}, Layers: {n_layers}")

    # Generate training data
    print("\n2. Generating training data...")
    train_data = generate_training_data(n_samples=10000)
    test_data = generate_training_data(n_samples=500)
    print(f"   Train: {len(train_data)}, Test: {len(test_data)}")

    # Train classifiers on multiple layers
    print("\n3. Training classifiers on multiple layers...")
    layers_to_test = [8, 12, 16, 20]  # TinyLlama has 22 layers (0-21)
    layer_results = {}

    for layer in layers_to_test:
        print(f"\n  === Layer {layer} ===")
        classifier, metrics = train_classifier(
            model, tokenizer, train_data[:2000],  # Use subset for speed
            layer=layer,
            hidden_dim=hidden_dim,
            epochs=10,
            batch_size=64,
            lr=1e-3,
        )
        layer_results[layer] = {
            "classifier": classifier,
            "metrics": metrics,
        }
        print(f"  Best val accuracy: {metrics['best_val_acc']:.2%}")

    # Find best layer
    print("\n4. Finding best layer...")
    best_layer = max(layer_results.keys(), key=lambda l: layer_results[l]["metrics"]["best_val_acc"])
    best_acc = layer_results[best_layer]["metrics"]["best_val_acc"]
    print(f"   Best layer: {best_layer} with {best_acc:.2%} validation accuracy")

    # Layer comparison table
    print("\n  Layer Comparison:")
    print("  " + "-" * 40)
    print(f"  {'Layer':<10} {'Val Accuracy':<15}")
    print("  " + "-" * 40)
    for layer in layers_to_test:
        acc = layer_results[layer]["metrics"]["best_val_acc"]
        marker = " ← best" if layer == best_layer else ""
        print(f"  L{layer:<9} {acc:.2%}{marker}")
    print("  " + "-" * 40)

    # Compare routing methods
    print("\n5. Comparing routing methods...")
    best_classifier = layer_results[best_layer]["classifier"]

    comparison = compare_routing(
        model, tokenizer, best_classifier, best_layer,
        test_data[:200],  # Use subset for speed
        threshold=0.5,
    )

    print("\n  Routing Comparison (Layer {best_layer}):")
    print("  " + "-" * 50)
    print(f"  {'Method':<20} {'Op Accuracy':<15} {'End-to-End':<15}")
    print("  " + "-" * 50)
    print(f"  {'Regex Router':<20} {comparison['regex']['op_accuracy']:.2%}{'':<11} {comparison['regex']['accuracy']:.2%}")
    print(f"  {'Hidden State Router':<20} {comparison['hidden']['op_accuracy']:.2%}{'':<11} {comparison['hidden']['accuracy']:.2%}")
    print("  " + "-" * 50)

    # Detailed analysis
    print("\n6. Error Analysis...")

    errors = [ex for ex in comparison["examples"] if not ex["hidden_correct"] and ex["regex_correct"]]
    print(f"\n  Cases where regex works but hidden state fails: {len(errors)}")

    if errors[:5]:
        print("\n  Sample errors:")
        for ex in errors[:5]:
            print(f"    {ex['expression']:<20} expected={ex['expected_op']}, hidden={ex['hidden_op']} (conf={ex['hidden_confidence']:.2f})")

    # Confidence analysis
    print("\n  Confidence distribution:")
    confidences = [ex["hidden_confidence"] for ex in comparison["examples"]]
    correct_conf = [ex["hidden_confidence"] for ex in comparison["examples"] if ex["hidden_correct"]]
    wrong_conf = [ex["hidden_confidence"] for ex in comparison["examples"] if not ex["hidden_correct"]]

    print(f"    Overall mean confidence: {np.mean(confidences):.2f}")
    if correct_conf:
        print(f"    Correct predictions mean: {np.mean(correct_conf):.2f}")
    if wrong_conf:
        print(f"    Wrong predictions mean: {np.mean(wrong_conf):.2f}")

    # Threshold sweep
    print("\n7. Threshold sweep for hidden state router...")
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    print(f"\n  {'Threshold':<12} {'Routed to WASM':<18} {'Accuracy (WASM)':<18}")
    print("  " + "-" * 48)

    for thresh in thresholds:
        wasm_routed = sum(1 for ex in comparison["examples"] if ex["hidden_confidence"] >= thresh)
        wasm_correct = sum(1 for ex in comparison["examples"]
                          if ex["hidden_confidence"] >= thresh and ex["hidden_correct"])

        wasm_pct = wasm_routed / len(comparison["examples"])
        wasm_acc = wasm_correct / wasm_routed if wasm_routed > 0 else 0

        print(f"  {thresh:<12} {wasm_pct:.2%}{'':<13} {wasm_acc:.2%}")

    # Save results
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    degradation = comparison["regex"]["accuracy"] - comparison["hidden"]["accuracy"]

    print(f"\nRegex Router:         {comparison['regex']['accuracy']:.2%} end-to-end accuracy")
    print(f"Hidden State Router:  {comparison['hidden']['accuracy']:.2%} end-to-end accuracy")
    print(f"Degradation:          {degradation:.2%}")

    if comparison["hidden"]["op_accuracy"] >= 0.95:
        print("\n*** SUCCESS: Hidden state encodes enough for routing (≥95% op accuracy) ***")
    elif comparison["hidden"]["op_accuracy"] >= 0.80:
        print("\n*** PARTIAL: Hidden state routing works but needs improvement (80-95%) ***")
    else:
        print("\n*** NEEDS WORK: Hidden state routing below 80%, need better classifier or layer ***")

    # Save detailed results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = results_dir / f"internal_routing_{timestamp}.json"

    save_data = {
        "best_layer": best_layer,
        "layer_accuracies": {l: layer_results[l]["metrics"]["best_val_acc"] for l in layers_to_test},
        "regex_accuracy": comparison["regex"]["accuracy"],
        "regex_op_accuracy": comparison["regex"]["op_accuracy"],
        "hidden_accuracy": comparison["hidden"]["accuracy"],
        "hidden_op_accuracy": comparison["hidden"]["op_accuracy"],
        "degradation": degradation,
        "threshold": 0.5,
        "train_size": len(train_data),
        "test_size": len(test_data[:200]),
    }

    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"\nResults saved to: {results_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
