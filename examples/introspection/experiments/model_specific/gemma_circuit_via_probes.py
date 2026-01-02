#!/usr/bin/env python3
"""
Gemma Circuit Identification via Probes

Uses the existing chuk_lazarus.introspection.circuit infrastructure
to identify arithmetic circuits in Gemma.

Usage:
    uv run python examples/introspection/experiments/model_specific/gemma_circuit_via_probes.py
"""

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from chuk_lazarus.introspection.circuit import (
    create_arithmetic_dataset,
    ActivationCollector,
    CollectorConfig,
)


def run_circuit_analysis():
    """Run circuit analysis using the built-in tools."""
    print("=" * 70)
    print("GEMMA CIRCUIT IDENTIFICATION VIA PROBES")
    print("=" * 70)

    model_id = "mlx-community/gemma-3-4b-it-bf16"

    # Create arithmetic dataset
    print("\n1. Creating arithmetic dataset...")
    dataset = create_arithmetic_dataset()
    print(f"   Total prompts: {len(dataset.prompts)}")
    print(f"   Arithmetic: {sum(1 for p in dataset.prompts if p.label == 1)}")
    print(f"   Non-arithmetic: {sum(1 for p in dataset.prompts if p.label == 0)}")

    # Load collector
    print(f"\n2. Loading model: {model_id}...")
    collector = ActivationCollector.from_pretrained(model_id)
    print(f"   Layers: {collector.num_layers}")
    print(f"   Hidden size: {collector.hidden_size}")

    # Collect activations from key layers
    print("\n3. Collecting activations...")
    config = CollectorConfig(
        layers=[0, 4, 8, 12, 16, 20, 24, 28, 32, 33],  # Key layers
        capture_hidden_states=True,
        capture_attention_weights=False,  # Skip for speed
    )

    activations = collector.collect(dataset, config, progress=True)
    print(f"   Samples collected: {len(activations)}")
    print(f"   Layers captured: {activations.captured_layers}")

    # Run layer-by-layer probing
    print("\n4. Running layer-by-layer probes...")
    print(f"\n{'Layer':<8} {'Accuracy':<12} {'Interpretation'}")
    print("-" * 40)

    layer_results = {}

    for layer in activations.captured_layers:
        # Get hidden states for this layer
        X = []
        y = []

        for i, prompt in enumerate(dataset.prompts):
            # Get hidden state at last token
            # hidden_states is dict[layer, mx.array] where array is [num_samples, hidden_size]
            layer_hidden = activations.hidden_states.get(layer)
            if layer_hidden is not None:
                hidden = np.array(layer_hidden[i].tolist())
                X.append(hidden)
                y.append(prompt.label)

        if not X:
            continue

        X = np.array(X)
        y = np.array(y)

        # Split train/test
        n_test = max(1, len(X) // 5)
        X_train, X_test = X[:-n_test], X[-n_test:]
        y_train, y_test = y[:-n_test], y[-n_test:]

        # Train probe
        probe = LogisticRegression(max_iter=1000)
        probe.fit(X_train, y_train)

        # Evaluate
        y_pred = probe.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Interpretation
        if accuracy >= 0.95:
            interp = "FULLY ENCODED"
        elif accuracy >= 0.85:
            interp = "MOSTLY ENCODED"
        elif accuracy >= 0.70:
            interp = "EMERGING"
        else:
            interp = "NOT YET"

        print(f"L{layer:<7} {accuracy:>10.1%}   {interp}")
        layer_results[layer] = {
            "accuracy": accuracy,
            "interpretation": interp,
            "probe_weights": probe.coef_[0],
        }

    # Find key neurons
    print("\n5. Finding key arithmetic neurons...")
    print("\nTop neurons by probe weight at computation layers:")

    neuron_importance = defaultdict(list)

    for layer in [20, 24, 28]:
        if layer not in layer_results:
            continue

        weights = layer_results[layer]["probe_weights"]
        top_indices = np.argsort(np.abs(weights))[-20:][::-1]

        print(f"\n  Layer {layer} (accuracy={layer_results[layer]['accuracy']:.1%}):")
        print(f"  {'Neuron':<10} {'Weight':<12} {'Direction'}")
        print("  " + "-" * 35)

        for idx in top_indices[:10]:
            weight = weights[idx]
            direction = "ARITHMETIC+" if weight > 0 else "ARITHMETIC-"
            print(f"  {idx:<10} {weight:>+10.4f}   {direction}")

            neuron_importance[idx].append((layer, abs(weight)))

    # Find neurons important across multiple layers
    print("\n6. Neurons important across multiple layers:")
    print(f"\n{'Neuron':<10} {'Layers':<20} {'Total Importance'}")
    print("-" * 50)

    multi_layer_neurons = [
        (idx, layers) for idx, layers in neuron_importance.items()
        if len(layers) >= 2
    ]
    multi_layer_neurons.sort(key=lambda x: -sum(w for _, w in x[1]))

    for idx, layer_weights in multi_layer_neurons[:20]:
        layers_str = ", ".join(f"L{l}" for l, w in layer_weights)
        total_importance = sum(w for _, w in layer_weights)
        print(f"{idx:<10} {layers_str:<20} {total_importance:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("CIRCUIT SUMMARY")
    print("=" * 70)

    # Find emergence layer
    emergence_layer = None
    for layer in sorted(layer_results.keys()):
        if layer_results[layer]["accuracy"] >= 0.70:
            emergence_layer = layer
            break

    # Find peak layer
    peak_layer = max(layer_results.keys(), key=lambda l: layer_results[l]["accuracy"])

    print(f"\nArithmetic classification:")
    print(f"  Emergence layer (>70%): L{emergence_layer}")
    print(f"  Peak accuracy: L{peak_layer} ({layer_results[peak_layer]['accuracy']:.1%})")

    print(f"\nCircuit structure:")
    print(f"  Early layers (L0-L12): Context encoding")
    if emergence_layer and emergence_layer <= 16:
        print(f"  L{emergence_layer}: Arithmetic detection emerges")
    print(f"  L20-L28: Computation phase")
    print(f"  L{peak_layer}: Peak classification")

    print(f"\nKey neurons (cross-layer):")
    for idx, layer_weights in multi_layer_neurons[:5]:
        layers_str = ", ".join(f"L{l}" for l, w in layer_weights)
        print(f"  Neuron {idx}: active in {layers_str}")

    # Save results
    results = {
        "model": model_id,
        "layer_accuracies": {
            int(layer): {
                "accuracy": float(r["accuracy"]),
                "interpretation": r["interpretation"],
            }
            for layer, r in layer_results.items()
        },
        "emergence_layer": int(emergence_layer) if emergence_layer else None,
        "peak_layer": int(peak_layer),
        "key_neurons": [
            {"neuron": int(idx), "layers": [int(l) for l, w in lw], "importance": float(sum(w for _, w in lw))}
            for idx, lw in multi_layer_neurons[:20]
        ],
    }

    output_path = Path("gemma_discovery_cache/circuit_via_probes.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    run_circuit_analysis()
