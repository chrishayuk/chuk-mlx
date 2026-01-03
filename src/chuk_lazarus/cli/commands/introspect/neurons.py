"""Neuron and direction analysis commands for introspection CLI.

Commands for analyzing individual neuron activations, comparing direction
vectors, and extracting operand directions.
"""

import sys
from pathlib import Path


def introspect_neurons(args):
    """Analyze individual neuron activations across prompts.

    Shows how specific neurons fire across different prompts, useful for
    understanding what individual neurons encode after running a probe.

    Supports single layer (--layer) or multiple layers (--layers) for
    cross-layer neuron tracking.
    """
    import json

    import mlx.core as mx
    import numpy as np

    from ....introspection import CaptureConfig, ModelHooks, PositionSelection
    from ....introspection.ablation import AblationStudy

    # Parse layers - support both --layer and --layers
    if args.layers:
        layers_to_analyze = [int(layer.strip()) for layer in args.layers.split(",")]
    elif args.layer is not None:
        layers_to_analyze = [args.layer]
    else:
        print("ERROR: Must specify --layer or --layers")
        return

    print(f"Loading model: {args.model}")
    study = AblationStudy.from_pretrained(args.model)
    model = study.adapter.model
    tokenizer = study.adapter.tokenizer
    config = study.adapter.config

    print(f"  Analyzing layers: {layers_to_analyze}")

    # Parse steering config if provided
    steer_config = None
    if getattr(args, "steer", None):
        steer_arg = args.steer
        # Support both 'file.npz:coef' format and separate --strength flag
        if ":" in steer_arg:
            steer_parts = steer_arg.split(":")
            steer_file, steer_coef = steer_parts[0], float(steer_parts[1])
        else:
            steer_file = steer_arg
            steer_coef = getattr(args, "strength", None) or 1.0

        steer_data = np.load(steer_file, allow_pickle=True)
        steer_config = {
            "direction": steer_data["direction"],
            "layer": int(steer_data["layer"]),
            "coefficient": steer_coef,
            "file": steer_file,
        }
        if "label_positive" in steer_data:
            steer_config["positive"] = str(steer_data["label_positive"])
            steer_config["negative"] = str(steer_data["label_negative"])

        print(
            f"  Steering: {steer_file} @ layer {steer_config['layer']} with coefficient {steer_coef}"
        )

    # Parse prompts
    if args.prompts.startswith("@"):
        with open(args.prompts[1:]) as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        prompts = [p.strip() for p in args.prompts.split("|")]

    # Parse labels if provided
    if args.labels:
        labels = [lbl.strip() for lbl in args.labels.split("|")]
        if len(labels) != len(prompts):
            print(f"Warning: {len(labels)} labels for {len(prompts)} prompts, ignoring labels")
            labels = None
    else:
        labels = None

    # Get neurons to analyze
    neurons = []
    neuron_weights = {}
    neuron_stats = {}  # For auto-discover stats

    # Infer auto-discover if labels are provided but no explicit neuron source
    auto_discover = getattr(args, "auto_discover", False)
    if labels and not args.neurons and not args.from_direction:
        auto_discover = True

    if args.from_direction:
        # Load from saved direction file
        data = np.load(args.from_direction)
        direction = data["direction"]
        top_k = args.top_k

        # Get top neurons by absolute weight
        top_indices = np.argsort(np.abs(direction))[-top_k:][::-1]
        neurons = [int(i) for i in top_indices]
        neuron_weights = {int(i): float(direction[i]) for i in top_indices}

        print(f"  Loaded top {top_k} neurons from: {args.from_direction}")
        positive_label = str(data.get("label_positive", "positive"))
        negative_label = str(data.get("label_negative", "negative"))
        print(f"  Direction: {negative_label} -> {positive_label}")

    elif auto_discover:
        # Auto-discover neurons by variance/separation across label groups
        # Use first layer for discovery
        discover_layer = layers_to_analyze[0]
        if not labels:
            print("ERROR: --auto-discover requires --labels to group prompts")
            return

        print(f"\nAuto-discovering discriminative neurons at layer {discover_layer}...")
        print("  Collecting full hidden states for all prompts...")

        # Collect full hidden state for each prompt
        full_activations = []
        for prompt in prompts:
            hooks = ModelHooks(model, model_config=config)
            hooks.configure(
                CaptureConfig(
                    layers=[discover_layer],
                    capture_hidden_states=True,
                    positions=PositionSelection.LAST,
                )
            )

            input_ids = tokenizer.encode(prompt, return_tensors="np")
            hooks.forward(mx.array(input_ids))

            h = hooks.state.hidden_states[discover_layer][0, 0, :]
            h_np = np.array(h.astype(mx.float32), copy=False)
            full_activations.append(h_np)

        full_activations = np.array(full_activations)
        num_neurons = full_activations.shape[1]
        print(f"  Total neurons in layer: {num_neurons}")

        # Group activations by label
        unique_labels_sorted = sorted(set(labels))
        label_groups = {lbl: [] for lbl in unique_labels_sorted}
        for i, lbl in enumerate(labels):
            label_groups[lbl].append(full_activations[i])

        for lbl in unique_labels_sorted:
            label_groups[lbl] = np.array(label_groups[lbl])
            print(f"  Label '{lbl}': {len(label_groups[lbl])} prompts")

        # Calculate separation score for each neuron
        # For multi-class: use max pairwise separation
        # When single samples per group, use range/overall_std as proxy
        single_sample_mode = all(len(label_groups[lbl]) == 1 for lbl in unique_labels_sorted)
        if single_sample_mode:
            print("  Note: Single sample per label - using range-based discrimination")

        neuron_scores = []
        for neuron_idx in range(num_neurons):
            # Get activations for this neuron across all groups
            group_means = []
            group_stds = []
            for lbl in unique_labels_sorted:
                vals = label_groups[lbl][:, neuron_idx]
                group_means.append(np.mean(vals))
                group_stds.append(np.std(vals))

            # Overall std across all prompts (used as normalizer for single-sample mode)
            overall_std = np.std(full_activations[:, neuron_idx])

            # Max pairwise separation (Cohen's d style)
            max_separation = 0.0
            best_pair = None
            for i, lbl1 in enumerate(unique_labels_sorted):
                for j, lbl2 in enumerate(unique_labels_sorted):
                    if i >= j:
                        continue
                    mean_diff = abs(group_means[i] - group_means[j])

                    if single_sample_mode:
                        # With 1 sample per group, use overall_std as normalizer
                        # This finds neurons with large spread across label types
                        if overall_std > 1e-6:
                            separation = mean_diff / overall_std
                        else:
                            separation = 0.0
                    else:
                        # Standard pooled std for multi-sample groups
                        pooled_std = np.sqrt((group_stds[i] ** 2 + group_stds[j] ** 2) / 2)
                        if pooled_std > 1e-6:
                            separation = mean_diff / pooled_std
                        else:
                            separation = 0.0

                    if separation > max_separation:
                        max_separation = separation
                        best_pair = (lbl1, lbl2)

            # Also track the range (max - min across group means)
            mean_range = max(group_means) - min(group_means)

            neuron_scores.append(
                {
                    "idx": neuron_idx,
                    "separation": max_separation,
                    "best_pair": best_pair,
                    "overall_std": overall_std,
                    "mean_range": mean_range,
                    "group_means": {
                        lbl: group_means[i] for i, lbl in enumerate(unique_labels_sorted)
                    },
                }
            )

        # Sort by separation score
        neuron_scores.sort(key=lambda x: -x["separation"])

        # Take top-k
        top_k = args.top_k
        top_neurons = neuron_scores[:top_k]

        neurons = [n["idx"] for n in top_neurons]
        neuron_stats = {n["idx"]: n for n in top_neurons}

        print(f"\n  Top {top_k} discriminative neurons:")
        print(f"  {'Neuron':>8} {'Separation':>12} {'Range':>10} {'Best Pair'}")
        print("  " + "-" * 60)
        for n in top_neurons:
            pair_str = f"{n['best_pair'][0]} vs {n['best_pair'][1]}" if n["best_pair"] else "N/A"
            print(f"  {n['idx']:>8} {n['separation']:>12.3f} {n['mean_range']:>10.1f} {pair_str}")

    elif args.neurons:
        # Parse neuron indices
        neurons = [int(n.strip()) for n in args.neurons.split(",")]
        print(f"  Analyzing {len(neurons)} neurons: {neurons}")

    else:
        print("ERROR: Must specify --neurons, --from-direction, or --auto-discover")
        return

    # Parse neuron names if provided
    neuron_names = {}
    if getattr(args, "neuron_names", None):
        names_list = [n.strip() for n in args.neuron_names.split("|")]
        if len(names_list) != len(neurons):
            print(f"Warning: {len(names_list)} names for {len(neurons)} neurons, ignoring names")
        else:
            neuron_names = {neurons[i]: names_list[i] for i in range(len(neurons))}
            print(f"  Neuron names: {neuron_names}")

    def neuron_label(n: int) -> str:
        """Get display label for a neuron (with name if available)."""
        if n in neuron_names:
            return f"N{n}({neuron_names[n][:8]})"
        return f"N{n}"

    def neuron_header(n: int, width: int = 6) -> str:
        """Get header label for a neuron."""
        if n in neuron_names:
            name = neuron_names[n][:width]
            return f"{name:>{width}}"
        return f"N{n:>{width - 1}}"

    steer_msg = " (with steering)" if steer_config else ""
    print(
        f"\nCollecting activations for {len(prompts)} prompts across {len(layers_to_analyze)} layers{steer_msg}..."
    )

    # Collect activations for ALL layers in one pass per prompt
    # Structure: all_activations[layer][prompt_idx] = hidden_state
    all_activations_by_layer = {layer: [] for layer in layers_to_analyze}

    # If steering, we use ActivationSteering to wrap the model layers
    steerer = None
    if steer_config:
        from ....introspection import ActivationSteering

        steerer = ActivationSteering(model, tokenizer)
        steerer.add_direction(
            steer_config["layer"],
            mx.array(steer_config["direction"]),
        )
        # Wrap the steering layer so forward passes include steering
        steerer._wrap_layer(
            steer_config["layer"],
            steer_config["coefficient"],
        )

    try:
        for prompt in prompts:
            hooks = ModelHooks(model, model_config=config)
            hooks.configure(
                CaptureConfig(
                    layers=layers_to_analyze,
                    capture_hidden_states=True,
                    positions=PositionSelection.LAST,
                )
            )

            input_ids = tokenizer.encode(prompt, return_tensors="np")
            hooks.forward(mx.array(input_ids))

            for layer in layers_to_analyze:
                h = hooks.state.hidden_states[layer][0, 0, :]
                h_np = np.array(h.astype(mx.float32), copy=False)
                all_activations_by_layer[layer].append(h_np)
    finally:
        # Unwrap layers to restore model state
        if steerer:
            steerer._unwrap_layers()

    # Store results for all layers (for JSON output)
    all_layer_results = {}

    # Multi-layer mode: show cross-layer comparison table first
    if len(layers_to_analyze) > 1:
        print(f"\n{'=' * 80}")
        print("CROSS-LAYER NEURON TRACKING")
        print(f"{'=' * 80}")

        # Build cross-layer table: rows are prompts, columns are layers
        for neuron in neurons:
            neuron_title = neuron_names.get(neuron, f"Neuron {neuron}")
            print(f"\n--- {neuron_title} (N{neuron}) across layers ---")

            # Header with layers
            header = f"{'Prompt':<20} |"
            for layer in layers_to_analyze:
                header += f" L{layer:>2} |"
            if labels:
                header += " Label"
            print(header)
            print("-" * len(header))

            # Collect values for this neuron across all layers
            cross_layer_vals = []
            for i, prompt in enumerate(prompts):
                row_vals = []
                for layer in layers_to_analyze:
                    val = all_activations_by_layer[layer][i][neuron]
                    row_vals.append(val)
                cross_layer_vals.append(row_vals)

            cross_layer_matrix = np.array(cross_layer_vals)
            vmin, vmax = cross_layer_matrix.min(), cross_layer_matrix.max()

            # Print rows
            for i, prompt in enumerate(prompts):
                short_prompt = prompt[:18] + ".." if len(prompt) > 20 else prompt
                row = f"{short_prompt:<20} |"

                for j, layer in enumerate(layers_to_analyze):
                    val = cross_layer_matrix[i, j]
                    row += f" {val:+4.0f} |"

                if labels and i < len(labels):
                    row += f" {labels[i]}"

                print(row)

            # Summary stats per layer
            print("-" * len(header))
            row = f"{'mean':<20} |"
            for j in range(len(layers_to_analyze)):
                mean_val = cross_layer_matrix[:, j].mean()
                row += f" {mean_val:+4.0f} |"
            print(row)

            row = f"{'std':<20} |"
            for j in range(len(layers_to_analyze)):
                std_val = cross_layer_matrix[:, j].std()
                row += f" {std_val:4.0f} |"
            print(row)

            row = f"{'range':<20} |"
            for j in range(len(layers_to_analyze)):
                range_val = cross_layer_matrix[:, j].max() - cross_layer_matrix[:, j].min()
                row += f" {range_val:4.0f} |"
            print(row)

    # Now show per-layer detailed analysis
    for layer in layers_to_analyze:
        all_activations = all_activations_by_layer[layer]

        # Build activation matrix
        activation_matrix = np.array([[act[n] for n in neurons] for act in all_activations])

        # Print results as ASCII heatmap
        print(f"\n{'=' * 80}")
        print(f"NEURON ACTIVATION MAP AT LAYER {layer}")
        print(f"{'=' * 80}")

        # Header - use names if available
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

        # Find min/max for heatmap scaling
        vmin, vmax = activation_matrix.min(), activation_matrix.max()

        # Rows
        for i, prompt in enumerate(prompts):
            short_prompt = prompt[:18] + ".." if len(prompt) > 20 else prompt
            row = f"{short_prompt:<20} |"

            for j, n in enumerate(neurons):
                val = activation_matrix[i, j]
                row += f" {val:+6.0f} |"

            if labels and i < len(labels):
                row += f" {labels[i]}"

            print(row)

        print("-" * 80)

        # ASCII heatmap visualization (only for single-layer or first layer to avoid too much output)
        if len(layers_to_analyze) == 1:
            print(f"\n{'=' * 80}")
            print("ASCII HEATMAP (light = low, dark = high)")
            print(f"{'=' * 80}")

            # Normalize for heatmap
            norm_matrix = (activation_matrix - vmin) / (vmax - vmin + 1e-8)

            header = f"{'Prompt':<20} |"
            for n in neurons:
                if n in neuron_names:
                    name = neuron_names[n][:6]
                    header += f" {name:>6} |"
                else:
                    header += f" N{n:>5} |"
            print(header)
            print("-" * len(header))

            heatmap_chars = " .-+*#"
            for i, prompt in enumerate(prompts):
                short_prompt = prompt[:18] + ".." if len(prompt) > 20 else prompt
                row = f"{short_prompt:<20} |"

                for j, n in enumerate(neurons):
                    norm_val = norm_matrix[i, j]
                    char_idx = min(int(norm_val * 4), 4)
                    char = heatmap_chars[char_idx]
                    row += f"  {char * 4}  |"

                if labels and i < len(labels):
                    row += f" {labels[i]}"

                print(row)

        # Neuron statistics
        print(f"\n--- Layer {layer} Statistics ---")

        for j, n in enumerate(neurons):
            vals = activation_matrix[:, j]
            extra_str = ""

            # Show weight from direction file
            if n in neuron_weights:
                w = neuron_weights[n]
                direction_str = "-> POSITIVE detector" if w > 0 else "-> NEGATIVE detector"
                extra_str = f" (weight: {w:+.3f}) {direction_str}"

            # Show separation score from auto-discover
            if n in neuron_stats:
                sep = neuron_stats[n]["separation"]
                pair = neuron_stats[n].get("best_pair")
                pair_str = f"{pair[0]} vs {pair[1]}" if pair else ""
                extra_str = f" (separation: {sep:.3f}) {pair_str}"

            # Include name if available
            name_str = f" [{neuron_names[n]}]" if n in neuron_names else ""
            print(
                f"Neuron {n:4d}{name_str}: min={vals.min():+7.1f}, max={vals.max():+7.1f}, "
                f"mean={vals.mean():+7.1f}, std={vals.std():6.1f}{extra_str}"
            )

        # Correlation with labels if provided (only for single-layer to avoid verbosity)
        if labels and len(layers_to_analyze) == 1:
            print(f"\n{'=' * 80}")
            print("LABEL CORRELATION")
            print(f"{'=' * 80}")

            unique_labels_for_corr = sorted(set(labels))
            for label in unique_labels_for_corr:
                mask = np.array([lbl == label for lbl in labels])
                if mask.sum() > 0:
                    print(f"\n{label}:")
                    for j, n in enumerate(neurons):
                        mean_val = activation_matrix[mask, j].mean()
                        name_str = f" [{neuron_names[n]}]" if n in neuron_names else ""
                        print(f"  Neuron {n:4d}{name_str}: mean={mean_val:+7.1f}")

        # Store for output
        all_layer_results[layer] = {
            "activations": activation_matrix.tolist(),
            "stats": {
                str(n): {
                    "min": float(activation_matrix[:, j].min()),
                    "max": float(activation_matrix[:, j].max()),
                    "mean": float(activation_matrix[:, j].mean()),
                    "std": float(activation_matrix[:, j].std()),
                }
                for j, n in enumerate(neurons)
            },
        }

    # Save if requested
    if args.output:
        output_data = {
            "model_id": args.model,
            "layers": layers_to_analyze,
            "neurons": neurons,
            "neuron_names": neuron_names if neuron_names else None,
            "prompts": prompts,
            "labels": labels,
            "by_layer": all_layer_results,
            "neuron_weights": neuron_weights,
            "auto_discovered": getattr(args, "auto_discover", False),
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


def introspect_directions(args):
    """Compare multiple direction vectors for orthogonality.

    Loads saved direction vectors (from 'introspect probe --save-direction')
    and computes the cosine similarity matrix between all pairs.

    Orthogonal directions (cosine ~ 0) indicate independent features.
    """
    import json
    from pathlib import Path

    import numpy as np

    files = args.files
    threshold = args.threshold

    if len(files) < 2:
        print("ERROR: Need at least 2 direction files to compare")
        return

    # Load all direction vectors
    directions = []
    names = []
    metadata = []

    print("Loading direction vectors...")
    for fpath in files:
        path = Path(fpath)
        if not path.exists():
            print(f"  ERROR: File not found: {fpath}")
            return

        data = np.load(fpath, allow_pickle=True)
        direction = data["direction"]

        # Get name from file or metadata
        if "label_positive" in data and "label_negative" in data:
            pos = str(data["label_positive"])
            neg = str(data["label_negative"])
            name = f"{neg}->{pos}"
        else:
            name = path.stem

        layer = int(data["layer"]) if "layer" in data else "?"
        method = str(data["method"]) if "method" in data else "?"
        accuracy = float(data["accuracy"]) if "accuracy" in data else None

        directions.append(direction)
        names.append(name)
        metadata.append(
            {
                "file": str(path),
                "name": name,
                "layer": layer,
                "method": method,
                "accuracy": accuracy,
                "dim": len(direction),
            }
        )

        acc_str = f", acc={accuracy:.1%}" if accuracy else ""
        print(f"  {name}: layer={layer}, dim={len(direction)}{acc_str}")

    # Check dimensions match
    dims = [len(d) for d in directions]
    if len(set(dims)) > 1:
        print(f"\nWARNING: Dimension mismatch: {dims}")
        print("  Directions from different models/layers may not be comparable")

    # Compute cosine similarity matrix
    n = len(directions)
    similarity = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if dims[i] == dims[j]:
                d_i = directions[i] / (np.linalg.norm(directions[i]) + 1e-8)
                d_j = directions[j] / (np.linalg.norm(directions[j]) + 1e-8)
                similarity[i, j] = np.dot(d_i, d_j)
            else:
                similarity[i, j] = float("nan")

    # Print results
    print(f"\n{'=' * 80}")
    print("COSINE SIMILARITY MATRIX")
    print(f"{'=' * 80}")
    print(f"(Threshold for 'orthogonal': |cos| < {threshold})")
    print()

    # Header
    max_name_len = max(len(n) for n in names)
    col_width = max(8, max_name_len + 2)

    header = " " * (max_name_len + 2)
    for name in names:
        header += f"{name:>{col_width}}"
    print(header)
    print("-" * len(header))

    # Rows
    for i, name in enumerate(names):
        row = f"{name:<{max_name_len}}  "
        for j in range(n):
            val = similarity[i, j]
            if np.isnan(val):
                row += f"{'N/A':>{col_width}}"
            elif i == j:
                row += f"{'1.000':>{col_width}}"
            else:
                row += f"{val:>{col_width}.3f}"
        print(row)

    # ASCII heatmap
    print(f"\n{'=' * 80}")
    print("ORTHOGONALITY HEATMAP")
    print(f"{'=' * 80}")
    print("(# = aligned, + = correlated, - = weak, . = near-orthogonal, space = orthogonal)")
    print()

    header = " " * (max_name_len + 2)
    for name in names:
        short = name[:6] if len(name) > 6 else name
        header += f"{short:>8}"
    print(header)
    print("-" * len(header))

    for i, name in enumerate(names):
        row = f"{name:<{max_name_len}}  "
        for j in range(n):
            val = abs(similarity[i, j])
            if np.isnan(val):
                char = "?"
            elif i == j:
                char = "#"
            elif val > 0.7:
                char = "#"
            elif val > 0.5:
                char = "+"
            elif val > 0.3:
                char = "-"
            elif val > threshold:
                char = "."
            else:
                char = " "
            row += f"{char:>8}"
        print(row)

    # Summary statistics
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")

    # Get off-diagonal elements
    off_diag = []
    for i in range(n):
        for j in range(i + 1, n):
            if not np.isnan(similarity[i, j]):
                off_diag.append((names[i], names[j], similarity[i, j]))

    if off_diag:
        orthogonal_pairs = [(a, b, s) for a, b, s in off_diag if abs(s) < threshold]
        aligned_pairs = [(a, b, s) for a, b, s in off_diag if abs(s) > 0.5]
        correlated_pairs = [(a, b, s) for a, b, s in off_diag if threshold <= abs(s) <= 0.5]

        print(f"\nTotal pairs: {len(off_diag)}")
        print(f"Orthogonal (|cos| < {threshold}): {len(orthogonal_pairs)}")
        print(f"Correlated ({threshold} <= |cos| <= 0.5): {len(correlated_pairs)}")
        print(f"Aligned (|cos| > 0.5): {len(aligned_pairs)}")

        if orthogonal_pairs:
            print("\nOrthogonal pairs (independent dimensions):")
            for a, b, s in sorted(orthogonal_pairs, key=lambda x: abs(x[2])):
                print(f"  {a} orthogonal to {b} (cos = {s:+.3f})")

        if aligned_pairs:
            print("\nAligned pairs (potentially redundant):")
            for a, b, s in sorted(aligned_pairs, key=lambda x: -abs(x[2])):
                print(f"  {a} aligned with {b} (cos = {s:+.3f})")

        # Overall assessment
        mean_abs_sim = np.mean([abs(s) for _, _, s in off_diag])
        print(f"\nMean |cosine similarity|: {mean_abs_sim:.3f}")

        if mean_abs_sim < threshold:
            print("Assessment: Directions are largely ORTHOGONAL (independent features)")
        elif mean_abs_sim < 0.3:
            print("Assessment: Directions are mostly INDEPENDENT with some correlation")
        elif mean_abs_sim < 0.5:
            print("Assessment: Directions show MODERATE correlation")
        else:
            print("Assessment: Directions are HIGHLY correlated (may be redundant)")

    # Save if requested
    if args.output:
        output_data = {
            "files": [str(f) for f in files],
            "names": names,
            "metadata": metadata,
            "similarity_matrix": similarity.tolist(),
            "threshold": threshold,
            "pairs": [
                {"a": a, "b": b, "cosine": s, "orthogonal": abs(s) < threshold}
                for a, b, s in off_diag
            ]
            if off_diag
            else [],
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


def introspect_operand_directions(args):
    """Extract operand directions (A_d and B_d) to analyze how the model encodes operands.

    This is useful for understanding if a model uses compositional encoding (like GPT-OSS)
    where operand A and B are encoded in separate orthogonal subspaces, or holistic encoding
    (like Gemma) where the entire expression is encoded together.

    The command extracts:
    - A_d: direction that encodes the first operand value
    - B_d: direction that encodes the second operand value

    And computes:
    - A_i vs A_j similarity: do different first operand values have distinct directions?
    - A_i vs B_j similarity: are A and B subspaces orthogonal?
    - A_i vs B_i similarity: does digit identity dominate position (A vs B)?
    """
    import json

    import mlx.core as mx
    import numpy as np

    from ....introspection import CaptureConfig, ModelHooks, PositionSelection
    from ....introspection.ablation import AblationStudy

    print(f"Loading model: {args.model}")
    study = AblationStudy.from_pretrained(args.model)
    model = study.adapter.model
    tokenizer = study.adapter.tokenizer
    config = study.adapter.config

    # Parse digits to use
    if args.digits:
        digits = [int(d.strip()) for d in args.digits.split(",")]
    else:
        digits = list(range(2, 10))  # Default: 2-9

    print(f"Using digits: {digits}")

    # Parse layers
    if args.layers:
        layers = [int(layer.strip()) for layer in args.layers.split(",")]
    else:
        # Default: key layers at 25%, 50%, 75%, and specific ones for Gemma-like models
        num_layers = study.adapter.num_layers
        layers = sorted(
            set(
                [
                    int(num_layers * 0.25),
                    int(num_layers * 0.5),
                    int(num_layers * 0.6),
                    int(num_layers * 0.75),
                ]
            )
        )

    print(f"Analyzing layers: {layers}")

    # Operation symbol
    op = args.operation or "*"

    # Collect activations for A_d (first operand) and B_d (second operand)
    # For A_d: fix B, vary A. For B_d: fix A, vary B.
    print(f"\nCollecting activations for {op} operation...")

    def get_activation(prompt, layer):
        """Get last-token hidden state for a prompt at a given layer."""
        hooks = ModelHooks(model, model_config=config)
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

    results_by_layer = {}

    for layer in layers:
        print(f"\n{'=' * 70}")
        print(f"LAYER {layer}")
        print(f"{'=' * 70}")

        # Phase 1: Extract A_d directions (one per digit value)
        # Use fixed B=5 (middle of range), vary A
        fixed_b = 5 if 5 in digits else digits[len(digits) // 2]
        A_directions = {}

        print(f"\nExtracting A_d directions (B fixed at {fixed_b})...")
        for a in digits:
            prompt = f"{a}{op}{fixed_b}="
            h = get_activation(prompt, layer)
            A_directions[a] = h

        # Phase 2: Extract B_d directions (one per digit value)
        # Use fixed A=5, vary B
        fixed_a = 5 if 5 in digits else digits[len(digits) // 2]
        B_directions = {}

        print(f"Extracting B_d directions (A fixed at {fixed_a})...")
        for b in digits:
            prompt = f"{fixed_a}{op}{b}="
            h = get_activation(prompt, layer)
            B_directions[b] = h

        # Phase 3: Compute similarity matrices

        def cosine_sim(v1, v2):
            """Compute cosine similarity between two vectors."""
            dot = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            return float(dot / (norm1 * norm2 + 1e-8))

        # A_i vs A_j: do different A values have distinct directions?
        a_vs_a = []
        for i, a1 in enumerate(digits):
            for a2 in digits[i + 1 :]:
                sim = cosine_sim(A_directions[a1], A_directions[a2])
                a_vs_a.append(sim)

        # B_i vs B_j: do different B values have distinct directions?
        b_vs_b = []
        for i, b1 in enumerate(digits):
            for b2 in digits[i + 1 :]:
                sim = cosine_sim(B_directions[b1], B_directions[b2])
                b_vs_b.append(sim)

        # A_i vs B_j: are A and B subspaces orthogonal?
        a_vs_b_cross = []
        for a in digits:
            for b in digits:
                if a != b:  # Exclude same digit
                    sim = cosine_sim(A_directions[a], B_directions[b])
                    a_vs_b_cross.append(sim)

        # A_i vs B_i: does digit identity dominate role?
        a_vs_b_same = []
        for d in digits:
            sim = cosine_sim(A_directions[d], B_directions[d])
            a_vs_b_same.append(sim)

        # Print results
        print(f"\n--- Orthogonality Analysis ---")
        print(f"A_i vs A_j (diff first operands): {np.mean(a_vs_a):.3f} +/- {np.std(a_vs_a):.3f}")
        print(f"B_i vs B_j (diff second operands): {np.mean(b_vs_b):.3f} +/- {np.std(b_vs_b):.3f}")
        print(f"A_i vs B_j (cross A/B, diff digits): {np.mean(a_vs_b_cross):.3f} +/- {np.std(a_vs_b_cross):.3f}")
        print(f"A_i vs B_i (same digit, diff role): {np.mean(a_vs_b_same):.3f} +/- {np.std(a_vs_b_same):.3f}")

        # Interpretation
        print(f"\n--- Interpretation ---")
        if np.mean(a_vs_a) < 0.5 and np.mean(b_vs_b) < 0.5:
            print("Distinct operand directions (compositional encoding)")
        else:
            print("High A-A and B-B overlap (holistic encoding)")

        if np.mean(a_vs_b_cross) < 0.3:
            print("A and B subspaces are orthogonal")
        else:
            print("A and B subspaces overlap significantly")

        if np.mean(a_vs_b_same) > 0.8:
            print("Digit identity dominates role (same digit similar regardless of position)")

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

    # Summary across layers
    print(f"\n{'=' * 70}")
    print("SUMMARY ACROSS LAYERS")
    print(f"{'=' * 70}")
    print(f"{'Layer':<8} {'A vs A':<12} {'B vs B':<12} {'A vs B (cross)':<14} {'A vs B (same)':<14}")
    print("-" * 60)
    for layer in layers:
        r = results_by_layer[layer]
        print(
            f"L{layer:<7} {r['a_vs_a_mean']:.3f}        {r['b_vs_b_mean']:.3f}        "
            f"{r['a_vs_b_cross_mean']:.3f}          {r['a_vs_b_same_mean']:.3f}"
        )

    # Save results
    if args.output:
        output_data = {
            "model": args.model,
            "operation": op,
            "digits": digits,
            "layers": layers,
            "results_by_layer": results_by_layer,
        }

        # If npz format, also save the actual directions
        if args.output.endswith(".npz"):
            # For last analyzed layer, save the actual direction vectors
            last_layer = layers[-1]
            np.savez(
                args.output,
                model=args.model,
                operation=op,
                digits=np.array(digits),
                layers=np.array(layers),
                A_directions=np.array([A_directions[d] for d in digits]),
                B_directions=np.array([B_directions[d] for d in digits]),
                layer=last_layer,
                results=json.dumps(results_by_layer),
            )
            print(f"\nDirections and results saved to: {args.output}")
        else:
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"\nResults saved to: {args.output}")


__all__ = [
    "introspect_neurons",
    "introspect_directions",
    "introspect_operand_directions",
]
