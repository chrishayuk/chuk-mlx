#!/usr/bin/env python
"""
CSP Virtual Expert Experiment Runner.

Runs the full experiment:
1. Load model
2. Extract hidden states for CSP and non-CSP prompts
3. Train CSP detection probe (Experiment 1)
4. Train CSP subtype probe (Experiment 2)
5. Sweep layers to find best probe location
6. Report results

Usage:
    python -m experiments.csp_virtual_expert.run_experiment --model ./gpt-oss-lite-v2/gpt-oss-lite-16exp
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# MLX imports
import mlx.core as mx

# Data
from .data.prompts import get_all_csp_prompts, get_all_non_csp_prompts


@dataclass
class HiddenStateData:
    """Hidden state data for a single prompt."""
    prompt: str
    category: str
    is_csp: bool
    hidden_state: np.ndarray
    layer: int


def extract_hidden_state(model, tokenizer, prompt: str, layer: int) -> np.ndarray:
    """
    Extract hidden state at a specific layer for a prompt.

    Returns the last token position's hidden state.
    """
    # Tokenize
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    # Get embeddings
    if hasattr(model, 'model'):
        # Standard transformer structure
        hidden = model.model.embed_tokens(input_ids)
        layers = model.model.layers
    elif hasattr(model, 'embed_tokens'):
        hidden = model.embed_tokens(input_ids)
        layers = model.layers
    else:
        raise ValueError("Unknown model structure")

    # Pass through layers up to target
    for i, layer_module in enumerate(layers):
        # Forward through layer
        output = layer_module(hidden)

        # Handle different output formats
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        if i == layer:
            break

    # Evaluate and convert to numpy
    mx.eval(hidden)

    # Return last token position
    return np.array(hidden[0, -1, :].tolist())


def collect_hidden_states(
    model,
    tokenizer,
    layer: int,
    max_prompts: int | None = None,
) -> list[HiddenStateData]:
    """
    Collect hidden states for all CSP and non-CSP prompts.
    """
    results = []

    # CSP prompts
    csp_prompts = get_all_csp_prompts()
    if max_prompts:
        csp_prompts = csp_prompts[:max_prompts // 2]

    print(f"  Extracting {len(csp_prompts)} CSP prompts...")
    for i, (prompt, category) in enumerate(csp_prompts):
        if (i + 1) % 10 == 0:
            print(f"    {i + 1}/{len(csp_prompts)}")
        hidden = extract_hidden_state(model, tokenizer, prompt, layer)
        results.append(HiddenStateData(
            prompt=prompt,
            category=category,
            is_csp=True,
            hidden_state=hidden,
            layer=layer,
        ))

    # Non-CSP prompts
    non_csp_prompts = get_all_non_csp_prompts()
    if max_prompts:
        non_csp_prompts = non_csp_prompts[:max_prompts // 2]

    print(f"  Extracting {len(non_csp_prompts)} non-CSP prompts...")
    for i, (prompt, category) in enumerate(non_csp_prompts):
        if (i + 1) % 10 == 0:
            print(f"    {i + 1}/{len(non_csp_prompts)}")
        hidden = extract_hidden_state(model, tokenizer, prompt, layer)
        results.append(HiddenStateData(
            prompt=prompt,
            category=category,
            is_csp=False,
            hidden_state=hidden,
            layer=layer,
        ))

    return results


def train_binary_probe(hidden_states: list[HiddenStateData]) -> dict[str, Any]:
    """Train binary CSP detection probe."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score, f1_score

    # Prepare data
    X = np.vstack([h.hidden_state for h in hidden_states])
    y = np.array([1 if h.is_csp else 0 for h in hidden_states])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train
    probe = LogisticRegression(max_iter=1000, random_state=42)
    probe.fit(X_train, y_train)

    # Evaluate
    y_pred = probe.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, target_names=["non-CSP", "CSP"]),
        "probe": probe,
    }


def train_subtype_probe(hidden_states: list[HiddenStateData]) -> dict[str, Any]:
    """Train multi-class CSP subtype probe."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score

    # Filter to CSP only
    csp_states = [h for h in hidden_states if h.is_csp]

    # Prepare data
    X = np.vstack([h.hidden_state for h in csp_states])
    y = np.array([h.category for h in csp_states])

    categories = sorted(set(y))

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train
    probe = LogisticRegression(multi_class="multinomial", max_iter=1000, random_state=42)
    probe.fit(X_train, y_train)

    # Evaluate
    y_pred = probe.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, target_names=categories),
        "categories": categories,
        "probe": probe,
    }


def run_layer_sweep(
    model,
    tokenizer,
    layers: list[int],
    max_prompts: int | None = None,
) -> dict[int, float]:
    """Sweep across layers to find best probe location."""
    results = {}

    for layer in layers:
        print(f"\n--- Layer {layer} ---")

        # Collect hidden states
        hidden_states = collect_hidden_states(model, tokenizer, layer, max_prompts)

        # Train probe
        probe_result = train_binary_probe(hidden_states)

        results[layer] = probe_result["accuracy"]
        print(f"  Accuracy: {probe_result['accuracy']:.2%}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run CSP Virtual Expert Experiment")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to model directory")
    parser.add_argument("--layer", type=int, default=13,
                       help="Layer to probe (default: 13)")
    parser.add_argument("--sweep", action="store_true",
                       help="Sweep multiple layers")
    parser.add_argument("--sweep-layers", type=str, default="2,4,8,12,13,15",
                       help="Layers to sweep (comma-separated)")
    parser.add_argument("--max-prompts", type=int, default=None,
                       help="Max prompts per class (for faster testing)")
    parser.add_argument("--output", type=str, default="results/experiment_results.json",
                       help="Output file for results")
    args = parser.parse_args()

    print("=" * 70)
    print("CSP Virtual Expert Experiment")
    print("=" * 70)

    # Load model
    print(f"\n[1] Loading model: {args.model}")
    start = time.time()

    from chuk_lazarus.models_v2.loader import load_model

    loaded = load_model(args.model)
    model = loaded.model
    tokenizer = loaded.tokenizer

    print(f"    Loaded in {time.time() - start:.1f}s")
    print(f"    Family: {loaded.family_type}")

    # Get number of layers
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        n_layers = len(model.model.layers)
    elif hasattr(model, 'layers'):
        n_layers = len(model.layers)
    else:
        n_layers = 32  # default
    print(f"    Layers: {n_layers}")

    results = {
        "model": args.model,
        "n_layers": n_layers,
    }

    if args.sweep:
        # Layer sweep
        print(f"\n[2] Layer Sweep")
        layers = [int(x) for x in args.sweep_layers.split(",")]
        layers = [l for l in layers if l < n_layers]  # Filter valid layers

        sweep_results = run_layer_sweep(model, tokenizer, layers, args.max_prompts)

        print("\n" + "=" * 70)
        print("Layer Sweep Results")
        print("=" * 70)
        for layer, acc in sorted(sweep_results.items(), key=lambda x: -x[1]):
            bar = "#" * int(acc * 50)
            print(f"  Layer {layer:2d}: {acc:.2%} {bar}")

        best_layer = max(sweep_results, key=sweep_results.get)
        print(f"\n  Best layer: {best_layer} ({sweep_results[best_layer]:.2%})")

        results["layer_sweep"] = {str(k): v for k, v in sweep_results.items()}
        results["best_layer"] = best_layer

        # Use best layer for detailed analysis
        args.layer = best_layer

    # Experiment 1: CSP Detection
    print(f"\n[3] Experiment 1: CSP Detection Probe (Layer {args.layer})")
    print("-" * 50)

    hidden_states = collect_hidden_states(model, tokenizer, args.layer, args.max_prompts)
    print(f"    Collected {len(hidden_states)} hidden states")

    binary_result = train_binary_probe(hidden_states)

    print(f"\n    Results:")
    print(f"    Accuracy: {binary_result['accuracy']:.2%}")
    print(f"    F1 Score: {binary_result['f1']:.2%}")
    print(f"\n{binary_result['report']}")

    # Interpret
    if binary_result['accuracy'] >= 0.80:
        print("    ==> SUCCESS: Strong CSP gate signal detected!")
        exp1_status = "success"
    elif binary_result['accuracy'] >= 0.60:
        print("    ==> PARTIAL: Weak signal - consider other layers")
        exp1_status = "partial"
    else:
        print("    ==> FAILED: CSP not distinctly encoded")
        exp1_status = "failed"

    results["experiment_1"] = {
        "layer": args.layer,
        "accuracy": binary_result['accuracy'],
        "f1": binary_result['f1'],
        "status": exp1_status,
    }

    # Experiment 2: Subtype Classification
    print(f"\n[4] Experiment 2: CSP Subtype Classification")
    print("-" * 50)

    subtype_result = train_subtype_probe(hidden_states)

    print(f"\n    Results:")
    print(f"    Accuracy: {subtype_result['accuracy']:.2%}")
    print(f"    Categories: {subtype_result['categories']}")
    print(f"\n{subtype_result['report']}")

    # Interpret
    if subtype_result['accuracy'] >= 0.70:
        print("    ==> SUCCESS: Subtypes well separated!")
        exp2_status = "success"
    elif subtype_result['accuracy'] >= 0.50:
        print("    ==> PARTIAL: Some subtype clustering")
        exp2_status = "partial"
    else:
        print("    ==> LIMITED: Subtypes not well separated")
        exp2_status = "limited"

    results["experiment_2"] = {
        "accuracy": subtype_result['accuracy'],
        "categories": subtype_result['categories'],
        "status": exp2_status,
    }

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[5] Results saved to: {output_path}")

    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"  Model: {args.model}")
    print(f"  Probe Layer: {args.layer}")
    print(f"  Exp 1 (CSP Detection):  {binary_result['accuracy']:.1%} - {exp1_status.upper()}")
    print(f"  Exp 2 (Subtype Class):  {subtype_result['accuracy']:.1%} - {exp2_status.upper()}")

    if binary_result['accuracy'] >= 0.80:
        print("\n  ==> Hypothesis SUPPORTED: CSP gate detected at layer {args.layer}")
    else:
        print(f"\n  ==> Consider running --sweep to find better layer")


if __name__ == "__main__":
    main()
