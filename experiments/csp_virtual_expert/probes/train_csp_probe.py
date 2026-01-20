"""
CSP Detection Probe Training.

Experiment 1: Train a linear probe on hidden states to detect CSP problems.
Experiment 2: Train a multi-class probe for CSP subtype classification.

Usage:
    python -m experiments.csp_virtual_expert.probes.train_csp_probe --model openai/gpt-oss-20b

Requirements:
    - chuk_lazarus with model loading support
    - scikit-learn for probe training
    - mlx for hidden state extraction
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Conditional imports for flexibility
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


@dataclass
class ProbeResult:
    """Result from probe training."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    classification_report: str
    confusion_matrix: np.ndarray
    probe_weights: np.ndarray | None = None
    probe_bias: float | None = None


@dataclass
class HiddenStateData:
    """Hidden state data for a single prompt."""

    prompt: str
    category: str
    is_csp: bool
    hidden_state: np.ndarray
    layer: int


def extract_hidden_states(
    model,
    tokenizer,
    prompts: list[tuple[str, str, bool]],
    layer: int = 13,
) -> list[HiddenStateData]:
    """
    Extract hidden states from model for given prompts.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        prompts: List of (prompt, category, is_csp) tuples
        layer: Layer to extract from

    Returns:
        List of HiddenStateData objects
    """
    if not MLX_AVAILABLE:
        raise ImportError("MLX not available. Install with: pip install mlx")

    results = []

    for prompt, category, is_csp in prompts:
        # Tokenize
        input_ids = tokenizer.encode(prompt, return_tensors="mlx")

        # Get hidden states at specified layer
        hidden = _get_hidden_at_layer(model, input_ids, layer)

        # Take last token position
        last_token_hidden = hidden[0, -1, :].tolist()

        results.append(HiddenStateData(
            prompt=prompt,
            category=category,
            is_csp=is_csp,
            hidden_state=np.array(last_token_hidden),
            layer=layer,
        ))

    return results


def _get_hidden_at_layer(model, input_ids, layer: int):
    """
    Extract hidden state at a specific layer.

    This is model-architecture specific and may need adjustment.
    """
    # Embed tokens
    hidden = model.model.embed_tokens(input_ids)

    # Pass through layers up to target
    for i, layer_module in enumerate(model.model.layers):
        hidden = layer_module(hidden, mask=None)
        # Handle different output formats (some return tuples)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        if i == layer:
            break

    mx.eval(hidden)
    return hidden


def train_csp_detection_probe(
    hidden_states: list[HiddenStateData],
    test_size: float = 0.2,
    random_seed: int = 42,
) -> tuple[Any, ProbeResult]:
    """
    Train a binary probe to detect CSP vs non-CSP problems.

    Experiment 1 in the CSP Virtual Expert design.

    Args:
        hidden_states: List of HiddenStateData with is_csp labels
        test_size: Fraction for test set
        random_seed: Random seed for reproducibility

    Returns:
        (probe, ProbeResult) tuple
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn not available. Install with: pip install scikit-learn")

    # Prepare data
    X = np.vstack([h.hidden_state for h in hidden_states])
    y = np.array([1 if h.is_csp else 0 for h in hidden_states])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=y
    )

    # Train probe
    probe = LogisticRegression(max_iter=1000, random_state=random_seed)
    probe.fit(X_train, y_train)

    # Evaluate
    y_pred = probe.predict(X_test)

    report = classification_report(y_test, y_pred, target_names=["non-CSP", "CSP"])
    cm = confusion_matrix(y_test, y_pred)

    # Compute metrics
    from sklearn.metrics import precision_score, recall_score, f1_score

    result = ProbeResult(
        accuracy=probe.score(X_test, y_test),
        precision=precision_score(y_test, y_pred),
        recall=recall_score(y_test, y_pred),
        f1=f1_score(y_test, y_pred),
        classification_report=report,
        confusion_matrix=cm,
        probe_weights=probe.coef_[0],
        probe_bias=probe.intercept_[0],
    )

    return probe, result


def train_subtype_probe(
    hidden_states: list[HiddenStateData],
    test_size: float = 0.2,
    random_seed: int = 42,
) -> tuple[Any, ProbeResult]:
    """
    Train a multi-class probe to classify CSP subtypes.

    Experiment 2 in the CSP Virtual Expert design.

    Args:
        hidden_states: List of HiddenStateData (CSP only) with category labels
        test_size: Fraction for test set
        random_seed: Random seed for reproducibility

    Returns:
        (probe, ProbeResult) tuple
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn not available")

    # Filter to CSP only
    csp_states = [h for h in hidden_states if h.is_csp]

    # Prepare data
    X = np.vstack([h.hidden_state for h in csp_states])
    y = np.array([h.category for h in csp_states])

    # Get unique categories
    categories = sorted(set(y))

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=y
    )

    # Train multi-class probe
    probe = LogisticRegression(
        multi_class="multinomial",
        max_iter=1000,
        random_state=random_seed,
    )
    probe.fit(X_train, y_train)

    # Evaluate
    y_pred = probe.predict(X_test)

    report = classification_report(y_test, y_pred, target_names=categories)
    cm = confusion_matrix(y_test, y_pred, labels=categories)

    from sklearn.metrics import precision_score, recall_score, f1_score

    result = ProbeResult(
        accuracy=probe.score(X_test, y_test),
        precision=precision_score(y_test, y_pred, average="weighted"),
        recall=recall_score(y_test, y_pred, average="weighted"),
        f1=f1_score(y_test, y_pred, average="weighted"),
        classification_report=report,
        confusion_matrix=cm,
    )

    return probe, result


def evaluate_probe(
    probe,
    hidden_states: list[HiddenStateData],
    is_binary: bool = True,
) -> dict[str, Any]:
    """
    Evaluate a trained probe on new data.

    Args:
        probe: Trained sklearn probe
        hidden_states: Test data
        is_binary: Whether this is binary (CSP/non-CSP) or multi-class

    Returns:
        Dictionary of evaluation metrics
    """
    X = np.vstack([h.hidden_state for h in hidden_states])

    if is_binary:
        y_true = np.array([1 if h.is_csp else 0 for h in hidden_states])
    else:
        y_true = np.array([h.category for h in hidden_states])

    y_pred = probe.predict(X)
    y_proba = probe.predict_proba(X)

    return {
        "accuracy": (y_pred == y_true).mean(),
        "predictions": y_pred.tolist(),
        "probabilities": y_proba.tolist(),
        "true_labels": y_true.tolist(),
    }


def sweep_layers(
    model,
    tokenizer,
    prompts: list[tuple[str, str, bool]],
    layers: list[int] = [2, 4, 8, 12, 13, 15, 18, 21],
) -> dict[int, float]:
    """
    Sweep across layers to find best probe location.

    Args:
        model: Loaded model
        tokenizer: Tokenizer
        prompts: Training prompts
        layers: Layers to try

    Returns:
        Dictionary mapping layer -> accuracy
    """
    results = {}

    for layer in layers:
        print(f"Probing layer {layer}...")

        hidden_states = extract_hidden_states(model, tokenizer, prompts, layer=layer)
        probe, result = train_csp_detection_probe(hidden_states)

        results[layer] = result.accuracy
        print(f"  Accuracy: {result.accuracy:.2%}")

    return results


def save_probe(probe, result: ProbeResult, path: str | Path):
    """Save trained probe and results."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump({"probe": probe, "result": result}, f)

    # Also save human-readable metrics
    metrics_path = path.with_suffix(".json")
    with open(metrics_path, "w") as f:
        json.dump({
            "accuracy": result.accuracy,
            "precision": result.precision,
            "recall": result.recall,
            "f1": result.f1,
            "report": result.classification_report,
        }, f, indent=2)


def load_probe(path: str | Path) -> tuple[Any, ProbeResult]:
    """Load trained probe and results."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["probe"], data["result"]


# =============================================================================
# MAIN - Run probe training experiments
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train CSP detection probes")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-7b",
                       help="Model to probe")
    parser.add_argument("--layer", type=int, default=13,
                       help="Layer to probe")
    parser.add_argument("--sweep", action="store_true",
                       help="Sweep across multiple layers")
    parser.add_argument("--output", type=str, default="probes/csp_probe.pkl",
                       help="Output path for probe")
    parser.add_argument("--dry-run", action="store_true",
                       help="Test with synthetic data (no model required)")
    args = parser.parse_args()

    print("CSP Detection Probe Training")
    print("=" * 60)

    if args.dry_run:
        print("\nDry run mode - using synthetic data")
        # Generate synthetic hidden states for testing
        np.random.seed(42)
        hidden_dim = 2048

        from ..data.prompts import get_all_csp_prompts, get_all_non_csp_prompts

        csp_prompts = get_all_csp_prompts()
        non_csp_prompts = get_all_non_csp_prompts()

        hidden_states = []

        # CSP prompts cluster around one centroid
        csp_center = np.random.randn(hidden_dim)
        for prompt, category in csp_prompts:
            hidden = csp_center + 0.5 * np.random.randn(hidden_dim)
            hidden_states.append(HiddenStateData(
                prompt=prompt,
                category=category,
                is_csp=True,
                hidden_state=hidden,
                layer=args.layer,
            ))

        # Non-CSP prompts cluster around different centroid
        non_csp_center = np.random.randn(hidden_dim)
        for prompt, category in non_csp_prompts:
            hidden = non_csp_center + 0.5 * np.random.randn(hidden_dim)
            hidden_states.append(HiddenStateData(
                prompt=prompt,
                category=category,
                is_csp=False,
                hidden_state=hidden,
                layer=args.layer,
            ))

        print(f"Generated {len(hidden_states)} synthetic hidden states")

    else:
        # Load model and extract real hidden states
        print(f"\nLoading model: {args.model}")

        try:
            from chuk_lazarus.inference import load_model
            model, tokenizer, config = load_model(args.model)
        except ImportError:
            print("chuk_lazarus not available. Use --dry-run for testing.")
            return

        from ..data.prompts import iter_prompts

        prompts = list(iter_prompts())
        print(f"Extracting hidden states for {len(prompts)} prompts...")

        if args.sweep:
            layers = [2, 4, 8, 12, 13, 15, 18, 21]
            results = sweep_layers(model, tokenizer, prompts, layers)

            print("\nLayer sweep results:")
            for layer, acc in sorted(results.items(), key=lambda x: -x[1]):
                print(f"  Layer {layer}: {acc:.2%}")

            best_layer = max(results, key=results.get)
            print(f"\nBest layer: {best_layer} ({results[best_layer]:.2%})")
            return

        hidden_states = extract_hidden_states(
            model, tokenizer, prompts, layer=args.layer
        )

    # Train binary CSP detection probe (Experiment 1)
    print(f"\n--- Experiment 1: CSP Detection Probe (Layer {args.layer}) ---")
    probe, result = train_csp_detection_probe(hidden_states)

    print(f"\nResults:")
    print(f"  Accuracy:  {result.accuracy:.2%}")
    print(f"  Precision: {result.precision:.2%}")
    print(f"  Recall:    {result.recall:.2%}")
    print(f"  F1:        {result.f1:.2%}")
    print(f"\nClassification Report:\n{result.classification_report}")

    # Interpret result
    if result.accuracy >= 0.80:
        print("SUCCESS: Strong CSP gate signal detected!")
    elif result.accuracy >= 0.60:
        print("PARTIAL: Weak signal - consider trying other layers")
    else:
        print("FAILED: CSP not distinctly encoded at this layer")

    # Save probe
    output_path = Path(args.output)
    if not args.dry_run:
        save_probe(probe, result, output_path)
        print(f"\nProbe saved to: {output_path}")

    # Train subtype probe (Experiment 2)
    print(f"\n--- Experiment 2: CSP Subtype Classification ---")
    csp_states = [h for h in hidden_states if h.is_csp]

    if len(set(h.category for h in csp_states)) > 1:
        subtype_probe, subtype_result = train_subtype_probe(hidden_states)

        print(f"\nSubtype Results:")
        print(f"  Accuracy: {subtype_result.accuracy:.2%}")
        print(f"\nClassification Report:\n{subtype_result.classification_report}")

        if subtype_result.accuracy >= 0.70:
            print("SUCCESS: Subtype classification works!")
        else:
            print("PARTIAL: Subtypes not well separated")
    else:
        print("Skipped - not enough subtype diversity")


if __name__ == "__main__":
    main()
