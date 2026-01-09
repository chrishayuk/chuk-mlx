"""Neuron and direction analysis commands for introspection CLI.

Commands for analyzing individual neuron activations, comparing direction
vectors, and extracting operand directions. This module is a thin CLI wrapper
- all business logic is in NeuronAnalysisService.
"""

from __future__ import annotations

import asyncio
import json
from argparse import Namespace

from ._types import (
    DirectionComparisonConfig,
    DirectionComparisonResult,
    DirectionPairSimilarity,
    NeuronAnalysisConfig,
    parse_layers_string,
)


def introspect_neurons(args: Namespace) -> None:
    """Analyze individual neuron activations across prompts.

    Shows how specific neurons fire across different prompts, useful for
    understanding what individual neurons encode after running a probe.

    Supports single layer (--layer) or multiple layers (--layers) for
    cross-layer neuron tracking.
    """
    asyncio.run(_async_introspect_neurons(args))


async def _async_introspect_neurons(args: Namespace) -> None:
    """Async implementation of neuron analysis."""
    from ....introspection.steering.neuron_service import (
        DiscoveredNeuron,
        NeuronAnalysisService,
    )

    config = NeuronAnalysisConfig.from_args(args)

    # Parse layers
    if config.layers:
        layers_to_analyze = parse_layers_string(config.layers)
    elif config.layer is not None:
        layers_to_analyze = [config.layer]
    else:
        print("ERROR: Must specify --layer or --layers")
        return

    print(f"Loading model: {config.model}")
    print(f"  Analyzing layers: {layers_to_analyze}")

    # Parse prompts
    if config.prompts.startswith("@"):
        with open(config.prompts[1:]) as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        prompts = [p.strip() for p in config.prompts.split("|")]

    # Parse labels
    labels = None
    if config.labels:
        labels = [lbl.strip() for lbl in config.labels.split("|")]
        if len(labels) != len(prompts):
            print(f"Warning: {len(labels)} labels for {len(prompts)} prompts, ignoring labels")
            labels = None

    # Determine neuron source
    neurons: list[int] = []
    neuron_weights: dict[int, float] = {}
    neuron_stats: dict[int, DiscoveredNeuron] = {}

    # Infer auto-discover mode
    auto_discover = config.auto_discover
    if labels and not config.neurons and not config.from_direction:
        auto_discover = True

    if config.from_direction:
        # Load from direction file using service
        neurons, neuron_weights, metadata = NeuronAnalysisService.load_neurons_from_direction(
            config.from_direction, config.top_k
        )
        print(f"  Loaded top {config.top_k} neurons from: {config.from_direction}")
        if "positive_label" in metadata:
            print(
                f"  Direction: {metadata.get('negative_label', 'neg')} -> {metadata['positive_label']}"
            )

    elif auto_discover:
        if not labels:
            print("ERROR: --auto-discover requires --labels to group prompts")
            return

        discover_layer = layers_to_analyze[0]
        print(f"\nAuto-discovering discriminative neurons at layer {discover_layer}...")

        discovered = await NeuronAnalysisService.auto_discover_neurons(
            model=config.model,
            prompts=prompts,
            labels=labels,
            layer=discover_layer,
            top_k=config.top_k,
        )

        neurons = [n.idx for n in discovered]
        neuron_stats = {n.idx: n for n in discovered}

        print(f"\n  Top {config.top_k} discriminative neurons:")
        print(f"  {'Neuron':>8} {'Separation':>12} {'Range':>10} {'Best Pair'}")
        print("  " + "-" * 60)
        for n in discovered:
            pair_str = f"{n.best_pair[0]} vs {n.best_pair[1]}" if n.best_pair else "N/A"
            print(f"  {n.idx:>8} {n.separation:>12.3f} {n.mean_range:>10.1f} {pair_str}")

    elif config.neurons:
        neurons = [int(n.strip()) for n in config.neurons.split(",")]
        print(f"  Analyzing {len(neurons)} neurons: {neurons}")

    else:
        print("ERROR: Must specify --neurons, --from-direction, or --auto-discover")
        return

    # Parse neuron names
    neuron_names: dict[int, str] = {}
    if config.neuron_names:
        names_list = [n.strip() for n in config.neuron_names.split("|")]
        if len(names_list) == len(neurons):
            neuron_names = {neurons[i]: names_list[i] for i in range(len(neurons))}
            print(f"  Neuron names: {neuron_names}")

    # Parse steering config
    steer_config = None
    if config.steer:
        import numpy as np

        steer_arg = config.steer
        if ":" in steer_arg:
            steer_file, steer_coef = steer_arg.split(":")
            steer_coef = float(steer_coef)
        else:
            steer_file = steer_arg
            steer_coef = config.strength or 1.0

        steer_data = np.load(steer_file, allow_pickle=True)
        steer_config = {
            "direction": steer_data["direction"],
            "layer": int(steer_data["layer"]),
            "coefficient": steer_coef,
        }
        print(f"  Steering: {steer_file} @ layer {steer_config['layer']} with coef {steer_coef}")

    # Analyze neurons
    steer_msg = " (with steering)" if steer_config else ""
    print(
        f"\nCollecting activations for {len(prompts)} prompts across {len(layers_to_analyze)} layers{steer_msg}..."
    )

    results = await NeuronAnalysisService.analyze_neurons(
        model=config.model,
        prompts=prompts,
        neurons=neurons,
        layers=layers_to_analyze,
        steer_config=steer_config,
    )

    # Print results
    _print_neuron_results(
        results=results,
        neurons=neurons,
        prompts=prompts,
        labels=labels,
        neuron_names=neuron_names,
        neuron_weights=neuron_weights,
        neuron_stats={k: v.model_dump() for k, v in neuron_stats.items()},
    )

    # Save if requested
    if config.output:
        output_data = {
            "model_id": config.model,
            "layers": layers_to_analyze,
            "neurons": neurons,
            "neuron_names": neuron_names if neuron_names else None,
            "prompts": prompts,
            "labels": labels,
            "by_layer": {
                layer: [r.model_dump() for r in layer_results]
                for layer, layer_results in results.items()
            },
            "neuron_weights": neuron_weights,
            "auto_discovered": auto_discover,
        }
        with open(config.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {config.output}")


def _print_neuron_results(
    results: dict,
    neurons: list[int],
    prompts: list[str],
    labels: list[str] | None,
    neuron_names: dict[int, str],
    neuron_weights: dict[int, float],
    neuron_stats: dict[int, dict],
) -> None:
    """Print neuron analysis results."""

    layers = list(results.keys())

    # Multi-layer mode: show cross-layer comparison first
    if len(layers) > 1:
        print(f"\n{'=' * 80}")
        print("CROSS-LAYER NEURON TRACKING")
        print(f"{'=' * 80}")

        for neuron in neurons:
            neuron_title = neuron_names.get(neuron, f"Neuron {neuron}")
            print(f"\n--- {neuron_title} (N{neuron}) across layers ---")

            header = f"{'Prompt':<20} |"
            for layer in layers:
                header += f" L{layer:>2} |"
            if labels:
                header += " Label"
            print(header)
            print("-" * len(header))

            # Get values for this neuron across all layers
            for i, prompt in enumerate(prompts):
                short_prompt = prompt[:18] + ".." if len(prompt) > 20 else prompt
                row = f"{short_prompt:<20} |"

                for layer in layers:
                    layer_results = results[layer]
                    neuron_result = next((r for r in layer_results if r.neuron_idx == neuron), None)
                    if neuron_result:
                        row += f" {neuron_result.mean_val:+4.0f} |"
                    else:
                        row += "  N/A |"

                if labels and i < len(labels):
                    row += f" {labels[i]}"

                print(row)

    # Per-layer detailed analysis
    for layer in layers:
        print(f"\n{'=' * 80}")
        print(f"NEURON ACTIVATION MAP AT LAYER {layer}")
        print(f"{'=' * 80}")

        layer_results = results[layer]

        # Header
        header = f"{'Prompt':<20} |"
        for n in neurons:
            if n in neuron_names:
                name = neuron_names[n][:6]
                header += f" {name:>6} |"
            else:
                header += f" N{n:>5} |"
        if labels:
            header += " Label"
        print(header)
        print("-" * len(header))

        # Stats per neuron
        print(f"\n--- Layer {layer} Statistics ---")
        for neuron_result in layer_results:
            n = neuron_result.neuron_idx
            extra_str = ""

            if n in neuron_weights:
                w = neuron_weights[n]
                direction_str = "-> POSITIVE detector" if w > 0 else "-> NEGATIVE detector"
                extra_str = f" (weight: {w:+.3f}) {direction_str}"

            if n in neuron_stats:
                sep = neuron_stats[n].get("separation", 0)
                pair = neuron_stats[n].get("best_pair")
                pair_str = f"{pair[0]} vs {pair[1]}" if pair else ""
                extra_str = f" (separation: {sep:.3f}) {pair_str}"

            name_str = f" [{neuron_names[n]}]" if n in neuron_names else ""
            print(
                f"Neuron {n:4d}{name_str}: min={neuron_result.min_val:+7.1f}, "
                f"max={neuron_result.max_val:+7.1f}, mean={neuron_result.mean_val:+7.1f}, "
                f"std={neuron_result.std_val:6.1f}{extra_str}"
            )


def introspect_directions(args: Namespace) -> None:
    """Compare multiple direction vectors for orthogonality.

    Loads saved direction vectors (from 'introspect probe --save-direction')
    and computes the cosine similarity matrix between all pairs.

    Orthogonal directions (cosine ~ 0) indicate independent features.
    """
    asyncio.run(_async_introspect_directions(args))


async def _async_introspect_directions(args: Namespace) -> None:
    """Async implementation of direction comparison."""
    from pathlib import Path

    import numpy as np

    config = DirectionComparisonConfig.from_args(args)

    if len(config.files) < 2:
        print("ERROR: Need at least 2 direction files to compare")
        return

    # Load all direction vectors
    directions = []
    names = []
    metadata = []

    print("Loading direction vectors...")
    for fpath in config.files:
        path = Path(fpath)
        if not path.exists():
            print(f"  ERROR: File not found: {fpath}")
            return

        data = np.load(fpath, allow_pickle=True)
        direction = data["direction"]

        # Get name from file or metadata
        if "label_positive" in data and "label_negative" in data:
            name = f"{data['label_negative']}->{data['label_positive']}"
        else:
            name = path.stem

        layer = int(data["layer"]) if "layer" in data else "?"
        accuracy = float(data["accuracy"]) if "accuracy" in data else None

        directions.append(direction)
        names.append(name)
        metadata.append(
            {
                "file": str(path),
                "name": name,
                "layer": layer,
                "dim": len(direction),
                "accuracy": accuracy,
            }
        )

        acc_str = f", acc={accuracy:.1%}" if accuracy else ""
        print(f"  {name}: layer={layer}, dim={len(direction)}{acc_str}")

    # Compute similarity matrix
    n = len(directions)
    pairs = []
    off_diag = []

    for i in range(n):
        for j in range(i + 1, n):
            if len(directions[i]) == len(directions[j]):
                d_i = directions[i] / (np.linalg.norm(directions[i]) + 1e-8)
                d_j = directions[j] / (np.linalg.norm(directions[j]) + 1e-8)
                sim = float(np.dot(d_i, d_j))
            else:
                sim = float("nan")

            off_diag.append(sim)
            pairs.append(
                DirectionPairSimilarity(
                    name_a=names[i],
                    name_b=names[j],
                    cosine_similarity=sim,
                    orthogonal=abs(sim) < config.threshold if not np.isnan(sim) else False,
                )
            )

    # Create result
    valid_sims = [s for s in off_diag if not np.isnan(s)]
    result = DirectionComparisonResult(
        files=config.files,
        names=names,
        pairs=pairs,
        mean_abs_similarity=float(np.mean([abs(s) for s in valid_sims])) if valid_sims else 0.0,
    )

    # Print results
    _print_direction_comparison(result, config.threshold)

    # Save if requested
    if config.output:
        output_data = result.model_dump()
        output_data["metadata"] = metadata
        with open(config.output, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\nResults saved to: {config.output}")


def _print_direction_comparison(result: DirectionComparisonResult, threshold: float) -> None:
    """Print direction comparison results."""
    print(f"\n{'=' * 80}")
    print("COSINE SIMILARITY MATRIX")
    print(f"{'=' * 80}")
    print(f"(Threshold for 'orthogonal': |cos| < {threshold})")

    # Summary
    orthogonal_pairs = [p for p in result.pairs if p.orthogonal]
    aligned_pairs = [p for p in result.pairs if abs(p.cosine_similarity) > 0.5]

    print(f"\nTotal pairs: {len(result.pairs)}")
    print(f"Orthogonal (|cos| < {threshold}): {len(orthogonal_pairs)}")
    print(f"Aligned (|cos| > 0.5): {len(aligned_pairs)}")

    if orthogonal_pairs:
        print("\nOrthogonal pairs (independent dimensions):")
        for p in sorted(orthogonal_pairs, key=lambda x: abs(x.cosine_similarity)):
            print(f"  {p.name_a} orthogonal to {p.name_b} (cos = {p.cosine_similarity:+.3f})")

    if aligned_pairs:
        print("\nAligned pairs (potentially redundant):")
        for p in sorted(aligned_pairs, key=lambda x: -abs(x.cosine_similarity)):
            print(f"  {p.name_a} aligned with {p.name_b} (cos = {p.cosine_similarity:+.3f})")

    print(f"\nMean |cosine similarity|: {result.mean_abs_similarity:.3f}")

    # Assessment
    if result.mean_abs_similarity < threshold:
        print("Assessment: Directions are largely ORTHOGONAL (independent features)")
    elif result.mean_abs_similarity < 0.3:
        print("Assessment: Directions are mostly INDEPENDENT with some correlation")
    elif result.mean_abs_similarity < 0.5:
        print("Assessment: Directions show MODERATE correlation")
    else:
        print("Assessment: Directions are HIGHLY correlated (may be redundant)")


def introspect_operand_directions(args: Namespace) -> None:
    """Extract operand directions (A_d and B_d) to analyze operand encoding.

    This is useful for understanding if a model uses compositional encoding
    where operand A and B are encoded in separate orthogonal subspaces.
    """
    asyncio.run(_async_introspect_operand_directions(args))


async def _async_introspect_operand_directions(args: Namespace) -> None:
    """Async implementation of operand direction extraction."""
    import mlx.core as mx
    import numpy as np

    from ....introspection import CaptureConfig, ModelHooks, PositionSelection
    from ....introspection.ablation import AblationStudy

    print(f"Loading model: {args.model}")
    study = AblationStudy.from_pretrained(args.model)
    model = study.adapter.model
    tokenizer = study.adapter.tokenizer
    model_config = study.adapter.config

    # Parse digits
    if args.digits:
        digits = [int(d.strip()) for d in args.digits.split(",")]
    else:
        digits = list(range(2, 10))

    # Parse layers
    if args.layers:
        layers = [int(layer.strip()) for layer in args.layers.split(",")]
    else:
        num_layers = study.adapter.num_layers
        layers = sorted(
            {
                int(num_layers * 0.25),
                int(num_layers * 0.5),
                int(num_layers * 0.6),
                int(num_layers * 0.75),
            }
        )

    op = args.operation or "*"

    print(f"Using digits: {digits}")
    print(f"Analyzing layers: {layers}")

    def get_activation(prompt: str, layer: int) -> np.ndarray:
        """Get last-token hidden state."""
        hooks = ModelHooks(model, model_config=model_config)
        hooks.configure(
            CaptureConfig(
                layers=[layer],
                capture_hidden_states=True,
                positions=PositionSelection.LAST,
            )
        )
        input_ids = tokenizer.encode(prompt, return_tensors="np")
        hooks.forward(mx.array(input_ids))
        h = hooks.state.hidden_states[layer][0, 0, :]
        return np.array(h.astype(mx.float32), copy=False)

    def cosine_sim(v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute cosine similarity."""
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8))

    results_by_layer = {}

    for layer in layers:
        print(f"\n{'=' * 70}")
        print(f"LAYER {layer}")
        print(f"{'=' * 70}")

        # Extract A_d directions (fixed B)
        fixed_b = 5 if 5 in digits else digits[len(digits) // 2]
        A_directions = {a: get_activation(f"{a}{op}{fixed_b}=", layer) for a in digits}

        # Extract B_d directions (fixed A)
        fixed_a = 5 if 5 in digits else digits[len(digits) // 2]
        B_directions = {b: get_activation(f"{fixed_a}{op}{b}=", layer) for b in digits}

        # Compute similarities
        a_vs_a = [
            cosine_sim(A_directions[a1], A_directions[a2])
            for i, a1 in enumerate(digits)
            for a2 in digits[i + 1 :]
        ]
        b_vs_b = [
            cosine_sim(B_directions[b1], B_directions[b2])
            for i, b1 in enumerate(digits)
            for b2 in digits[i + 1 :]
        ]
        a_vs_b_cross = [
            cosine_sim(A_directions[a], B_directions[b]) for a in digits for b in digits if a != b
        ]
        a_vs_b_same = [cosine_sim(A_directions[d], B_directions[d]) for d in digits]

        print("\n--- Orthogonality Analysis ---")
        print(f"A_i vs A_j: {np.mean(a_vs_a):.3f} +/- {np.std(a_vs_a):.3f}")
        print(f"B_i vs B_j: {np.mean(b_vs_b):.3f} +/- {np.std(b_vs_b):.3f}")
        print(f"A_i vs B_j (cross): {np.mean(a_vs_b_cross):.3f} +/- {np.std(a_vs_b_cross):.3f}")
        print(f"A_i vs B_i (same): {np.mean(a_vs_b_same):.3f} +/- {np.std(a_vs_b_same):.3f}")

        results_by_layer[layer] = {
            "a_vs_a_mean": float(np.mean(a_vs_a)),
            "a_vs_a_std": float(np.std(a_vs_a)),
            "b_vs_b_mean": float(np.mean(b_vs_b)),
            "b_vs_b_std": float(np.std(b_vs_b)),
            "a_vs_b_cross_mean": float(np.mean(a_vs_b_cross)),
            "a_vs_b_cross_std": float(np.std(a_vs_b_cross)),
            "a_vs_b_same_mean": float(np.mean(a_vs_b_same)),
            "a_vs_b_same_std": float(np.std(a_vs_b_same)),
        }

    # Save if requested
    if args.output:
        output_data = {
            "model": args.model,
            "operation": op,
            "digits": digits,
            "layers": layers,
            "results_by_layer": results_by_layer,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


__all__ = [
    "introspect_neurons",
    "introspect_directions",
    "introspect_operand_directions",
]
