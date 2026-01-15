#!/usr/bin/env python3
"""
Retrieval Circuit Analysis

Research question: At which layer do queries with the SAME answer become similar?

If "3*4=" and "2*6=" (both → 12) suddenly become similar at layer N,
that's where the model transitions from query encoding to value retrieval.

This reveals the boundary between:
- Query encoding (layers 0-N): representations differ by input
- Value retrieval (layers N+): representations converge by output
"""

import json
from dataclasses import dataclass

from chuk_lazarus.introspection.layer_analysis import LayerAnalyzer


@dataclass
class ConvergenceResult:
    """Result showing where queries converge by answer."""

    answer: str
    queries: list[str]
    layer_similarities: dict[int, float]  # layer -> avg pairwise similarity
    convergence_layer: int | None  # layer where similarity jumps
    convergence_delta: float  # size of the jump


def analyze_retrieval_circuit(
    model_id: str = "openai/gpt-oss-20b",
    layers: list[int] | None = None,
) -> dict[str, ConvergenceResult]:
    """
    Find where queries with same answers converge in representation space.

    Groups multiplication facts by their product, then measures
    pairwise similarity within each group at each layer.
    """
    # Create test groups: queries with same answer
    answer_groups = {
        "12": ["3*4=", "2*6=", "4*3=", "6*2="],
        "24": ["3*8=", "4*6=", "8*3=", "6*4="],
        "36": ["4*9=", "6*6=", "9*4="],
        "56": ["7*8=", "8*7="],
        "42": ["6*7=", "7*6="],
        "18": ["2*9=", "3*6=", "9*2=", "6*3="],
    }

    # Also include control group: same first operand, different answers
    control_groups = {
        "7x_row": ["7*2=", "7*3=", "7*4=", "7*5="],  # 14, 21, 28, 35
        "3x_row": ["3*2=", "3*4=", "3*5=", "3*7="],  # 6, 12, 15, 21
    }

    print(f"Loading model: {model_id}")
    analyzer = LayerAnalyzer.from_pretrained(model_id)

    if layers is None:
        # Analyze all layers
        n = analyzer.num_layers
        layers = list(range(0, n, 2)) + [n - 1]
        layers = sorted(set(layers))

    print(f"\nAnalyzing layers: {layers}")

    results = {}

    # Analyze answer-grouped queries
    print("\n" + "=" * 70)
    print("SAME-ANSWER CONVERGENCE ANALYSIS")
    print("=" * 70)

    for answer, queries in answer_groups.items():
        print(f"\nAnalyzing answer={answer}: {queries}")

        result = analyzer.analyze_representations(
            prompts=queries,
            layers=layers,
            labels=[f"→{answer}"] * len(queries),
        )

        # Compute average pairwise similarity at each layer
        layer_sims = {}
        for layer_idx in layers:
            rep_result = result.representations[layer_idx]
            sim_matrix = rep_result.similarity_matrix

            # Average of upper triangle (excluding diagonal)
            n = len(queries)
            total = 0.0
            count = 0
            for i in range(n):
                for j in range(i + 1, n):
                    total += sim_matrix[i][j]
                    count += 1

            layer_sims[layer_idx] = total / count if count > 0 else 0.0

        # Find convergence layer (biggest jump in similarity)
        sorted_layers = sorted(layer_sims.keys())
        max_delta = 0.0
        convergence_layer = None

        for i in range(1, len(sorted_layers)):
            prev_layer = sorted_layers[i - 1]
            curr_layer = sorted_layers[i]
            delta = layer_sims[curr_layer] - layer_sims[prev_layer]
            if delta > max_delta:
                max_delta = delta
                convergence_layer = curr_layer

        results[answer] = ConvergenceResult(
            answer=answer,
            queries=queries,
            layer_similarities=layer_sims,
            convergence_layer=convergence_layer,
            convergence_delta=max_delta,
        )

        print("  Layer similarities:")
        for layer in sorted_layers:
            marker = " ← CONVERGENCE" if layer == convergence_layer else ""
            print(f"    L{layer:2d}: {layer_sims[layer]:.4f}{marker}")

    # Analyze control groups (should NOT converge)
    print("\n" + "=" * 70)
    print("CONTROL: SAME-ROW (different answers)")
    print("=" * 70)

    for name, queries in control_groups.items():
        print(f"\nAnalyzing {name}: {queries}")

        result = analyzer.analyze_representations(
            prompts=queries,
            layers=layers,
        )

        layer_sims = {}
        for layer_idx in layers:
            rep_result = result.representations[layer_idx]
            sim_matrix = rep_result.similarity_matrix

            n = len(queries)
            total = 0.0
            count = 0
            for i in range(n):
                for j in range(i + 1, n):
                    total += sim_matrix[i][j]
                    count += 1

            layer_sims[layer_idx] = total / count if count > 0 else 0.0

        print("  Layer similarities:")
        for layer in sorted(layer_sims.keys()):
            print(f"    L{layer:2d}: {layer_sims[layer]:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Where does retrieval happen?")
    print("=" * 70)

    convergence_layers = [r.convergence_layer for r in results.values() if r.convergence_layer]
    if convergence_layers:
        from collections import Counter

        layer_counts = Counter(convergence_layers)
        print("\nConvergence layer distribution:")
        for layer, count in sorted(layer_counts.items()):
            print(f"  Layer {layer}: {count} answer groups")

        avg_convergence = sum(convergence_layers) / len(convergence_layers)
        print(f"\nAverage convergence layer: {avg_convergence:.1f}")
        print(f"This is ~{100 * avg_convergence / analyzer.num_layers:.0f}% through the model")

    return results


def main():
    results = analyze_retrieval_circuit(
        model_id="openai/gpt-oss-20b",
        layers=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 23],
    )

    # Save results
    output = {
        answer: {
            "queries": r.queries,
            "layer_similarities": {str(k): v for k, v in r.layer_similarities.items()},
            "convergence_layer": r.convergence_layer,
            "convergence_delta": r.convergence_delta,
        }
        for answer, r in results.items()
    }

    with open("retrieval_circuit_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\nResults saved to retrieval_circuit_results.json")


if __name__ == "__main__":
    main()
