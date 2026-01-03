"""Activation clustering commands for introspection CLI.

Commands for visualizing activation clusters using PCA and related techniques.
"""

import json


def introspect_activation_cluster(args):
    """Visualize activation clusters using PCA.

    Projects hidden states to 2D to see if different prompt types cluster separately.

    Supports two syntaxes:
    1. Legacy two-class: --class-a "prompts" --class-b "prompts" --label-a X --label-b Y
    2. Multi-class: --prompts "p1|p2|p3" --label L1 --prompts "p4|p5" --label L2 ...
    """
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np

    from ....inference.loader import DType, HFLoader
    from ....introspection import ModelAccessor
    from ....models_v2.families.registry import detect_model_family, get_family_info

    # Parse prompts with labels - support both legacy and new syntax
    prompts = []
    labels = []

    # Check for new multi-class syntax
    if args.prompt_groups and args.labels:
        if len(args.prompt_groups) != len(args.labels):
            print(
                f"ERROR: Number of --prompts ({len(args.prompt_groups)}) must match "
                f"number of --label ({len(args.labels)})"
            )
            return

        for prompt_group, label in zip(args.prompt_groups, args.labels):
            if prompt_group.startswith("@"):
                with open(prompt_group[1:]) as f:
                    group_prompts = [line.strip() for line in f if line.strip()]
            else:
                group_prompts = [p.strip() for p in prompt_group.split("|")]
            prompts.extend(group_prompts)
            labels.extend([label] * len(group_prompts))

    # Fall back to legacy two-class syntax
    elif args.class_a or args.class_b:
        if args.class_a:
            if args.class_a.startswith("@"):
                with open(args.class_a[1:]) as f:
                    class_a_prompts = [line.strip() for line in f if line.strip()]
            else:
                class_a_prompts = [p.strip() for p in args.class_a.split("|")]
            prompts.extend(class_a_prompts)
            labels.extend([args.label_a] * len(class_a_prompts))

        if args.class_b:
            if args.class_b.startswith("@"):
                with open(args.class_b[1:]) as f:
                    class_b_prompts = [line.strip() for line in f if line.strip()]
            else:
                class_b_prompts = [p.strip() for p in args.class_b.split("|")]
            prompts.extend(class_b_prompts)
            labels.extend([args.label_b] * len(class_b_prompts))
    else:
        print("ERROR: Must provide either --prompts/--label pairs or --class-a/--class-b")
        return

    if len(prompts) < 2:
        print("ERROR: Need at least 2 prompts for clustering")
        return

    print(f"Loading model: {args.model}")

    result = HFLoader.download(args.model)
    model_path = result.model_path

    config_path = model_path / "config.json"
    with open(config_path) as f:
        config_data = json.load(f)

    family_type = detect_model_family(config_data)
    if family_type is None:
        raise ValueError(f"Unsupported model: {args.model}")

    family_info = get_family_info(family_type)
    config = family_info.config_class.from_hf_config(config_data)
    model = family_info.model_class(config)

    HFLoader.apply_weights_to_model(model, model_path, config, dtype=DType.BFLOAT16)
    tokenizer = HFLoader.load_tokenizer(model_path)

    num_layers = config.num_hidden_layers
    print(f"  Layers: {num_layers}")

    # Use ModelAccessor for unified model access
    accessor = ModelAccessor(model, config)

    # Parse layers - support single int or comma-separated
    if args.layer is not None:
        if "," in str(args.layer):
            target_layers = [int(layer.strip()) for layer in str(args.layer).split(",")]
        else:
            target_layers = [int(args.layer)]
    else:
        target_layers = [int(num_layers * 0.5)]

    print(f"  Target layer(s): {target_layers}")

    def get_hidden_at_layer(prompt: str, layer: int) -> np.ndarray:
        """Get hidden state at specific layer."""
        input_ids = mx.array(tokenizer.encode(prompt))[None, :]
        layers = accessor.layers
        embed = accessor.embed

        h = embed(input_ids)
        scale = accessor.embedding_scale
        if scale:
            h = h * scale

        seq_len = input_ids.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len).astype(h.dtype)

        for idx, lyr in enumerate(layers):
            try:
                out = lyr(h, mask=mask)
            except TypeError:
                out = lyr(h)
            h = (
                out.hidden_states
                if hasattr(out, "hidden_states")
                else (out[0] if isinstance(out, tuple) else out)
            )
            if idx == layer:
                return np.array(h[0, -1, :].tolist())

        return np.array(h[0, -1, :].tolist())

    # Show what we're clustering
    unique_labels = list(dict.fromkeys(labels))  # Preserve order
    print(f"\nClasses ({len(unique_labels)}):")
    for label in unique_labels:
        count = labels.count(label)
        print(f"  {label}: {count} prompts")

    print(
        f"\nCollecting activations for {len(prompts)} prompts across {len(target_layers)} layer(s)..."
    )

    # Collect activations for all layers at once (more efficient)
    activations_by_layer = {layer: [] for layer in target_layers}

    for prompt in prompts:
        # Get hidden states at all target layers in one forward pass
        for target_layer in target_layers:
            h = get_hidden_at_layer(prompt, target_layer)
            activations_by_layer[target_layer].append(h)

    # PCA import
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print("ERROR: sklearn required. Install with: pip install scikit-learn")
        return

    # Create symbols for each label (use first letter, or A, B, C... if collision)
    symbols = {}
    used_symbols = set()
    fallback_symbols = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    fallback_idx = 0

    for label in unique_labels:
        symbol = label[0].upper()
        if symbol in used_symbols:
            while fallback_idx < len(fallback_symbols):
                symbol = fallback_symbols[fallback_idx]
                fallback_idx += 1
                if symbol not in used_symbols:
                    break
        symbols[label] = symbol
        used_symbols.add(symbol)

    # Process each layer
    all_results = {}
    for target_layer in target_layers:
        X = np.array(activations_by_layer[target_layer])

        pca = PCA(n_components=2)
        projected = pca.fit_transform(X)

        # Compute cluster statistics
        cluster_stats = {}

        for label in unique_labels:
            mask = np.array([lbl == label for lbl in labels])
            points = projected[mask]
            center = np.mean(points, axis=0)
            cluster_stats[label] = {
                "center": center,
                "count": int(np.sum(mask)),
                "points": points,
            }

        # Compute pairwise separations for multi-class
        separations = {}
        for i, l1 in enumerate(unique_labels):
            for l2 in unique_labels[i + 1 :]:
                c1 = cluster_stats[l1]["center"]
                c2 = cluster_stats[l2]["center"]
                sep = float(np.linalg.norm(c1 - c2))
                separations[(l1, l2)] = sep

        # Store results
        all_results[target_layer] = {
            "pca": pca,
            "projected": projected,
            "cluster_stats": cluster_stats,
            "separations": separations,
        }

        # Print results
        print(f"\n{'=' * 70}")
        print(f"ACTIVATION CLUSTERS AT LAYER {target_layer}")
        print(f"{'=' * 70}")
        print(
            f"PCA explained variance: {pca.explained_variance_ratio_[0]:.1%} + {pca.explained_variance_ratio_[1]:.1%}"
        )

        if separations:
            print("\nCluster separations:")
            for (l1, l2), sep in sorted(separations.items(), key=lambda x: -x[1]):
                print(f"  {l1} <-> {l2}: {sep:.2f}")

        print(f"\n{'Label':<15} {'Count':<8} {'Center (PC1, PC2)'}")
        print("-" * 50)
        for label, stats in cluster_stats.items():
            print(
                f"{label:<15} {stats['count']:<8} ({stats['center'][0]:.2f}, {stats['center'][1]:.2f})"
            )

        # ASCII scatter plot
        print(f"\n{'=' * 70}")
        print(f"SCATTER PLOT (ASCII) - Layer {target_layer}")
        print(f"{'=' * 70}")

        # Normalize to grid
        x_min, x_max = projected[:, 0].min(), projected[:, 0].max()
        y_min, y_max = projected[:, 1].min(), projected[:, 1].max()

        grid_width = 60
        grid_height = 20
        grid = [[" " for _ in range(grid_width)] for _ in range(grid_height)]

        for i, (x, y) in enumerate(projected):
            gx = int((x - x_min) / (x_max - x_min + 1e-6) * (grid_width - 1))
            gy = int((y - y_min) / (y_max - y_min + 1e-6) * (grid_height - 1))
            gy = grid_height - 1 - gy  # Flip y
            symbol = symbols.get(labels[i], "?")
            grid[gy][gx] = symbol

        for row in grid:
            print("  " + "".join(row))

        print(f"\n  Legend: {', '.join(f'{s}={lbl}' for lbl, s in symbols.items())}")

        # Save matplotlib plot if requested
        if getattr(args, "save_plot", None):
            try:
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(10, 8))

                # Color palette for multiple classes
                colors = plt.cm.tab10.colors

                for i, label in enumerate(unique_labels):
                    mask = np.array([lbl == label for lbl in labels])
                    points = projected[mask]
                    color = colors[i % len(colors)]
                    ax.scatter(
                        points[:, 0],
                        points[:, 1],
                        c=[color],
                        label=f"{label} (n={int(np.sum(mask))})",
                        alpha=0.7,
                        s=100,
                    )
                    # Mark cluster center
                    center = cluster_stats[label]["center"]
                    ax.scatter(
                        center[0],
                        center[1],
                        c=[color],
                        marker="x",
                        s=200,
                        linewidths=3,
                    )

                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                ax.set_title(f"Activation Clusters - Layer {target_layer}")
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Save plot
                output_path = args.save_plot.replace(".png", f"_layer{target_layer}.png")
                plt.savefig(output_path, dpi=150, bbox_inches="tight")
                print(f"\nPlot saved to: {output_path}")
                plt.close()

            except ImportError:
                print("ERROR: matplotlib required for plotting. Install with: pip install matplotlib")


__all__ = [
    "introspect_activation_cluster",
]
