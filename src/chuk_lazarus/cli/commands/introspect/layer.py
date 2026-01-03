"""Layer analysis commands for introspection CLI.

Commands for layer-by-layer analysis and format sensitivity testing.
"""

import json


def introspect_layer(args):
    """Analyze what specific layers do with representation similarity."""
    from ....introspection import LayerAnalyzer

    print(f"Loading model: {args.model}")
    analyzer = LayerAnalyzer.from_pretrained(args.model)

    # Parse prompts
    if args.prompts.startswith("@"):
        with open(args.prompts[1:]) as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        prompts = [p.strip() for p in args.prompts.split("|")]

    # Parse labels if provided
    labels = None
    if args.labels:
        labels = [lbl.strip() for lbl in args.labels.split(",")]
        if len(labels) != len(prompts):
            print(f"Warning: {len(labels)} labels provided for {len(prompts)} prompts")
            labels = None

    # Parse layers
    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]
    else:
        layers = None  # Use default (key layers)

    print(f"\nAnalyzing {len(prompts)} prompts at layers: {layers or 'auto'}")
    for i, p in enumerate(prompts):
        label_str = f" [{labels[i]}]" if labels else ""
        print(f"  {i + 1}. {p!r}{label_str}")

    # Run representation analysis
    result = analyzer.analyze_representations(
        prompts=prompts,
        layers=layers,
        labels=labels,
        position=-1,  # Last token position
    )

    # Print similarity matrices for each layer
    for layer_idx in result.layers:
        analyzer.print_similarity_matrix(result, layer_idx)

    # If comparing format sensitivity, show summary
    if labels and len(set(labels)) == 2:
        print("\n=== Format Sensitivity Summary ===")
        for layer_idx in result.layers:
            if result.clusters and layer_idx in result.clusters:
                cluster = result.clusters[layer_idx]
                within = cluster.within_cluster_similarity
                between = cluster.between_cluster_similarity
                sep = cluster.separation_score

                print(f"\nLayer {layer_idx}:")
                for label, sim in within.items():
                    print(f"  Within '{label}': {sim:.4f}")
                for (l1, l2), sim in between.items():
                    print(f"  Between '{l1}' <-> '{l2}': {sim:.4f}")
                print(f"  Separation score: {sep:.4f}")

                # Interpretation
                if sep > 0.02:
                    print(f"  -> Layer {layer_idx} DOES distinguish between groups")
                else:
                    print(f"  -> Layer {layer_idx} does NOT distinguish between groups")

    # Run attention analysis if requested
    if args.attention:
        print("\n=== Attention Analysis ===")
        attn_results = analyzer.analyze_attention(
            prompts=prompts,
            layers=layers[:2] if layers and len(layers) > 2 else layers,
        )
        for layer_idx in attn_results:
            analyzer.print_attention_comparison(attn_results, layer_idx, prompts, focus_token=-1)

    # Save if requested
    if args.output:
        output_data = {
            "prompts": prompts,
            "labels": labels,
            "layers": result.layers,
            "similarity_matrices": {
                layer: result.representations[layer].similarity_matrix for layer in result.layers
            },
        }
        if result.clusters:
            output_data["clusters"] = {
                layer: {
                    "within": result.clusters[layer].within_cluster_similarity,
                    "between": {
                        f"{l1}_{l2}": v
                        for (l1, l2), v in result.clusters[layer].between_cluster_similarity.items()
                    },
                    "separation": result.clusters[layer].separation_score,
                }
                for layer in result.clusters
            }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


def introspect_format_sensitivity(args):
    """Quick format sensitivity check (trailing space vs no space)."""
    from ....introspection import analyze_format_sensitivity

    # Parse base prompts (without trailing space)
    if args.prompts.startswith("@"):
        with open(args.prompts[1:]) as f:
            base_prompts = [line.strip().rstrip() for line in f if line.strip()]
    else:
        base_prompts = [p.strip().rstrip() for p in args.prompts.split("|")]

    # Parse layers
    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]
    else:
        layers = None

    print(f"Format sensitivity analysis for {args.model}")
    print(f"Testing {len(base_prompts)} prompts with/without trailing space")

    result = analyze_format_sensitivity(
        model_id=args.model,
        base_prompts=base_prompts,
        layers=layers,
    )

    # Find where format matters
    print("\n=== Where Format Matters ===")
    for layer_idx in result.layers:
        if result.clusters and layer_idx in result.clusters:
            sep = result.clusters[layer_idx].separation_score
            marker = "*" if sep > 0.02 else ""
            print(f"  Layer {layer_idx}: separation = {sep:.4f} {marker}")


__all__ = [
    "introspect_layer",
    "introspect_format_sensitivity",
]
