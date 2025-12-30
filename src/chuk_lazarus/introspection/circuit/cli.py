"""
Standalone CLI for circuit analysis.

This is a separate tool from the main lazarus CLI, focused on
mechanistic interpretability research for tool-calling circuits.

Usage:
    circuit dataset create -o prompts.json --per-tool 25
    circuit collect -m functiongemma-270m -d prompts.json -o activations
    circuit analyze -a activations.safetensors --layer 11
    circuit directions -a activations.safetensors --layer 11 -o directions
    circuit visualize -a activations.safetensors --layer 11 --umap

Installation:
    pip install chuk-lazarus[circuit]

    # Or add entry point in pyproject.toml:
    [project.scripts]
    circuit = "chuk_lazarus.introspection.circuit.cli:main"
"""

import argparse
import json
import sys
from pathlib import Path


def cmd_dataset_create(args):
    """Create a tool-calling prompt dataset."""
    from .dataset import create_tool_calling_dataset

    print(f"Creating dataset with {args.per_tool} prompts per tool category...")

    dataset = create_tool_calling_dataset(
        prompts_per_tool=args.per_tool,
        no_tool_prompts=args.no_tool,
        include_edge_cases=not args.no_edge_cases,
        seed=args.seed,
    )

    # Print summary
    summary = dataset.summary()
    print("\nDataset Summary:")
    print(f"  Total prompts: {summary['total']}")
    print(f"  Tool-calling: {summary['tool_calling']}")
    print(f"  No-tool: {summary['no_tool']}")
    print("\n  By category:")
    for cat, count in summary["by_category"].items():
        print(f"    {cat}: {count}")
    print("\n  By tool:")
    for tool, count in summary["by_tool"].items():
        print(f"    {tool}: {count}")

    # Save
    output_path = Path(args.output)
    dataset.save(output_path)
    print(f"\nSaved to: {output_path}")


def cmd_dataset_show(args):
    """Show contents of a dataset."""
    from .dataset import ToolPromptDataset

    dataset = ToolPromptDataset.load(args.dataset)
    summary = dataset.summary()

    print(f"\nDataset: {dataset.name} (v{dataset.version})")
    print(f"Total prompts: {summary['total']}")
    print(f"Tool-calling: {summary['tool_calling']}")
    print(f"No-tool: {summary['no_tool']}")

    if args.samples:
        print("\nSample prompts:")
        for i, p in enumerate(dataset.sample(min(args.samples, len(dataset)), seed=42)):
            tool_str = f" [{p.expected_tool}]" if p.expected_tool else ""
            print(f"  [{p.category.value}]{tool_str}: {p.text[:60]}...")


def cmd_collect(args):
    """Collect activations from a model."""
    from .collector import ActivationCollector, CollectorConfig
    from .dataset import ToolPromptDataset

    print(f"Loading dataset from {args.dataset}...")
    dataset = ToolPromptDataset.load(args.dataset)
    print(f"  {len(dataset)} prompts")

    # Parse layers
    if args.layers == "all":
        layers = "all"
    elif args.layers == "decision":
        layers = "decision"
    else:
        layers = [int(layer.strip()) for layer in args.layers.split(",")]

    config = CollectorConfig(
        layers=layers,
        capture_hidden_states=True,
        capture_attention_weights=args.attention,
        max_new_tokens=args.generate if args.generate else 0,
    )

    print(f"\nLoading model: {args.model}...")
    collector = ActivationCollector.from_pretrained(args.model)
    print(f"  {collector.num_layers} layers, hidden_size={collector.hidden_size}")

    print("\nCollecting activations...")
    activations = collector.collect(dataset, config, progress=True)

    print("\nCollection complete:")
    print(f"  Samples: {len(activations)}")
    print(f"  Layers captured: {activations.captured_layers}")

    activations.save(args.output, include_outputs=args.generate > 0)


def cmd_analyze(args):
    """Run geometry analysis on collected activations."""
    from .collector import CollectedActivations
    from .geometry import GeometryAnalyzer

    print(f"Loading activations from {args.activations}...")
    activations = CollectedActivations.load(args.activations)
    print(f"  {len(activations)} samples, layers: {activations.captured_layers}")

    analyzer = GeometryAnalyzer(activations)

    # Determine layers to analyze
    if args.layer:
        layers = [args.layer]
    else:
        layers = activations.captured_layers

    results = {}
    for layer in layers:
        print(f"\nAnalyzing layer {layer}...")
        result = analyzer.analyze_layer(layer, include_umap=args.umap)
        results[layer] = result

        # Print results
        if result.pca:
            print(
                f"  PCA: dim@90%={result.pca.intrinsic_dimensionality_90}, dim@95%={result.pca.intrinsic_dimensionality_95}"
            )
            print(f"       var[0]={result.pca.explained_variance_ratio[0]:.2%}")

        if result.binary_probe:
            bp = result.binary_probe
            print(f"  Binary probe: acc={bp.accuracy:.2%}, CV={bp.cv_mean:.2%}±{bp.cv_std:.2%}")

        if result.category_probe:
            cp = result.category_probe
            print(f"  Category probe: acc={cp.accuracy:.2%}")

    if len(layers) > 1:
        print("\n")
        analyzer.print_layer_comparison(results)

    # Save results
    if args.output:
        output = {
            "model_id": activations.model_id,
            "layers": {layer: r.summary() for layer, r in results.items()},
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved analysis to: {args.output}")


def cmd_directions(args):
    """Extract interpretable directions."""
    from .collector import CollectedActivations
    from .directions import DirectionExtractor, DirectionMethod

    print(f"Loading activations from {args.activations}...")
    activations = CollectedActivations.load(args.activations)
    print(f"  {len(activations)} samples, layers: {activations.captured_layers}")

    extractor = DirectionExtractor(activations)

    # Get layer
    layer = args.layer
    if layer is None:
        # Use middle decision layer
        layers = activations.captured_layers
        layer = layers[len(layers) // 2]
        print(f"Using layer {layer} (middle of captured layers)")

    method = DirectionMethod(args.method)

    print(f"\nExtracting directions from layer {layer} using {method.value}...")

    # Tool mode direction
    tool_dir = extractor.extract_tool_mode_direction(layer, method)
    print("\nTool-mode direction:")
    print(f"  Separation score: {tool_dir.separation_score:.3f}")
    print(f"  Classification accuracy: {tool_dir.accuracy:.2%}")
    print(f"  Mean projection (tool): {tool_dir.mean_projection_positive:.3f}")
    print(f"  Mean projection (no-tool): {tool_dir.mean_projection_negative:.3f}")

    # Per-tool directions
    if args.per_tool:
        per_tool = extractor.extract_per_tool_directions(layer)
        print(f"\nPer-tool directions ({len(per_tool)} tools):")
        for name, direction in sorted(per_tool.items(), key=lambda x: -x[1].separation_score):
            print(f"  {name}: separation={direction.separation_score:.3f}")

        # Check orthogonality
        print("\nOrthogonality check (cosine similarities):")
        similarities = extractor.check_orthogonality(per_tool)
        names = list(per_tool.keys())
        for i, name in enumerate(names[:5]):  # Show top 5
            sims = [f"{similarities[i, j]:.2f}" for j in range(min(5, len(names)))]
            print(f"  {name[:15]:<15}: {' '.join(sims)}")

    # Save bundle
    if args.output:
        bundle = extractor.create_bundle(layer, include_per_tool=args.per_tool)
        bundle.save(args.output)


def cmd_visualize(args):
    """Create visualizations of activation geometry."""
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np

    from .collector import CollectedActivations
    from .geometry import GeometryAnalyzer

    print(f"Loading activations from {args.activations}...")
    activations = CollectedActivations.load(args.activations)
    print(f"  {len(activations)} samples")

    analyzer = GeometryAnalyzer(activations)
    layer = args.layer or activations.captured_layers[-1]

    output_dir = Path(args.output) if args.output else Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)

    # PCA variance plot
    if args.pca or args.all:
        print(f"Computing PCA for layer {layer}...")
        pca = analyzer.compute_pca(layer, n_components=min(100, activations.hidden_size))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(pca.explained_variance_ratio[:50], "b-", label="Individual")
        ax1.set_xlabel("Component")
        ax1.set_ylabel("Explained Variance Ratio")
        ax1.set_title(f"Layer {layer} - PCA Explained Variance")
        ax1.legend()

        ax2.plot(pca.cumulative_variance[:50], "r-", label="Cumulative")
        ax2.axhline(y=0.9, color="g", linestyle="--", label="90%")
        ax2.axhline(y=0.95, color="orange", linestyle="--", label="95%")
        ax2.set_xlabel("Number of Components")
        ax2.set_ylabel("Cumulative Variance")
        ax2.set_title(
            f"Intrinsic Dimensionality: 90%@{pca.intrinsic_dimensionality_90}, 95%@{pca.intrinsic_dimensionality_95}"
        )
        ax2.legend()

        plt.tight_layout()
        pca_path = output_dir / f"pca_layer{layer}.png"
        plt.savefig(pca_path, dpi=150)
        plt.close()
        print(f"  Saved: {pca_path}")

    # UMAP plot
    if args.umap or args.all:
        print(f"Computing UMAP for layer {layer}...")
        try:
            umap_result = analyzer.compute_umap(layer)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            # By tool/no-tool
            colors = ["red" if lbl == 1 else "blue" for lbl in umap_result.labels]
            ax1.scatter(
                umap_result.embedding[:, 0], umap_result.embedding[:, 1], c=colors, alpha=0.6, s=30
            )
            ax1.set_title(f"Layer {layer} - UMAP (Tool=Red, No-Tool=Blue)")
            ax1.set_xlabel("UMAP 1")
            ax1.set_ylabel("UMAP 2")

            # By category
            categories = np.unique(umap_result.category_labels)
            cmap = plt.cm.get_cmap("tab10", len(categories))
            cat_to_idx = {cat: i for i, cat in enumerate(categories)}
            colors = [cmap(cat_to_idx[c]) for c in umap_result.category_labels]
            ax2.scatter(
                umap_result.embedding[:, 0], umap_result.embedding[:, 1], c=colors, alpha=0.6, s=30
            )
            ax2.set_title(f"Layer {layer} - UMAP by Category")
            ax2.set_xlabel("UMAP 1")
            ax2.set_ylabel("UMAP 2")

            # Add legend for categories
            for i, cat in enumerate(categories):
                ax2.scatter([], [], c=[cmap(i)], label=cat[:10])
            ax2.legend(loc="upper right", fontsize=8)

            plt.tight_layout()
            umap_path = output_dir / f"umap_layer{layer}.png"
            plt.savefig(umap_path, dpi=150)
            plt.close()
            print(f"  Saved: {umap_path}")
        except ImportError:
            print("  UMAP not available (pip install umap-learn)")

    # Probe accuracy across layers
    if args.probes or args.all:
        print("Computing probes across layers...")
        layers = activations.captured_layers
        binary_accs = []
        cat_accs = []

        for layer in layers:
            bp = analyzer.train_probe(layer)
            binary_accs.append(bp.accuracy)

            from .geometry import ProbeType

            cp = analyzer.train_probe(layer, ProbeType.MULTICLASS)
            cat_accs.append(cp.accuracy)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(layers, binary_accs, "bo-", label="Binary (tool/no-tool)")
        ax.plot(layers, cat_accs, "rs-", label="Category")
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Accuracy")
        ax.set_title("Linear Probe Accuracy by Layer")
        ax.legend()
        ax.set_ylim(0, 1)

        plt.tight_layout()
        probe_path = output_dir / "probe_accuracy.png"
        plt.savefig(probe_path, dpi=150)
        plt.close()
        print(f"  Saved: {probe_path}")

    print(f"\nVisualizations saved to: {output_dir}")


def cmd_steer(args):
    """Demo activation steering (experimental)."""
    print("Activation steering not yet implemented.")
    print("Coming in Phase 11: Causal Intervention")


def cmd_probes(args):
    """Run probe battery for computational stratigraphy."""
    from pathlib import Path

    from .probes import ProbeBattery

    print(f"Loading model: {args.model}...")

    # Load probe datasets
    dataset_dir = None
    if args.datasets:
        dataset_dir = Path(args.datasets)
    else:
        # Use default built-in datasets
        dataset_dir = Path(__file__).parent / "probe_datasets"

    battery = ProbeBattery.from_pretrained(args.model, dataset_dir)
    print(f"  {battery.num_layers} layers")
    print(f"  {len(battery.datasets)} probe datasets loaded")

    # Parse layers
    if args.layers:
        layers = [int(layer.strip()) for layer in args.layers.split(",")]
    else:
        # Default: key layers for stratigraphy
        n = battery.num_layers
        layers = [0, 2, 4, 6, 8, 10, 11, 12, 14, n - 1]
        layers = [layer for layer in layers if layer < n]
        layers = sorted(set(layers))

    print(f"\nProbing layers: {layers}")

    # Filter by category if specified
    categories = None
    if args.category:
        categories = [c.strip() for c in args.category.split(",")]

    # Run probes
    results = battery.run_all_probes(layers=layers, categories=categories, progress=True)

    # Print results
    battery.print_results_table(results)

    # Print stratigraphy
    if not args.no_stratigraphy:
        battery.print_stratigraphy(results, threshold=args.threshold)

    # Save results
    if args.output:
        results.save(args.output)
        print(f"\nResults saved to: {args.output}")


def cmd_probes_init(args):
    """Initialize probe dataset files for customization."""
    from pathlib import Path

    from .probes import save_default_datasets

    output_dir = Path(args.output)
    print(f"Creating default probe datasets in: {output_dir}")
    save_default_datasets(output_dir)
    print("\nEdit these JSON files to customize probes for your research.")


def main():
    """Main entry point for circuit CLI."""
    parser = argparse.ArgumentParser(
        prog="circuit",
        description="Circuit analysis toolkit for mechanistic interpretability",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # === dataset commands ===
    dataset_parser = subparsers.add_parser("dataset", help="Dataset management")
    dataset_subs = dataset_parser.add_subparsers(dest="dataset_cmd")

    # dataset create
    create_parser = dataset_subs.add_parser("create", help="Create prompt dataset")
    create_parser.add_argument("-o", "--output", required=True, help="Output file path")
    create_parser.add_argument("--per-tool", type=int, default=25, help="Prompts per tool category")
    create_parser.add_argument("--no-tool", type=int, default=100, help="No-tool prompts")
    create_parser.add_argument("--no-edge-cases", action="store_true", help="Exclude edge cases")
    create_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # dataset show
    show_parser = dataset_subs.add_parser("show", help="Show dataset contents")
    show_parser.add_argument("dataset", help="Dataset file path")
    show_parser.add_argument("--samples", type=int, default=10, help="Number of samples to show")

    # === collect command ===
    collect_parser = subparsers.add_parser("collect", help="Collect model activations")
    collect_parser.add_argument("-m", "--model", required=True, help="Model ID or path")
    collect_parser.add_argument("-d", "--dataset", required=True, help="Dataset file path")
    collect_parser.add_argument(
        "-o", "--output", required=True, help="Output path (without extension)"
    )
    collect_parser.add_argument(
        "--layers", default="decision", help="Layers to capture (all, decision, or comma-separated)"
    )
    collect_parser.add_argument(
        "--attention", action="store_true", help="Also capture attention weights"
    )
    collect_parser.add_argument(
        "--generate", type=int, default=0, help="Generate N tokens for criterion evaluation"
    )

    # === analyze command ===
    analyze_parser = subparsers.add_parser("analyze", help="Analyze activation geometry")
    analyze_parser.add_argument("-a", "--activations", required=True, help="Activations file path")
    analyze_parser.add_argument(
        "--layer", type=int, help="Specific layer to analyze (default: all)"
    )
    analyze_parser.add_argument("--umap", action="store_true", help="Include UMAP visualization")
    analyze_parser.add_argument("-o", "--output", help="Output JSON path for results")

    # === directions command ===
    dir_parser = subparsers.add_parser("directions", help="Extract interpretable directions")
    dir_parser.add_argument("-a", "--activations", required=True, help="Activations file path")
    dir_parser.add_argument("--layer", type=int, help="Layer to extract from")
    dir_parser.add_argument(
        "--method", default="diff_means", choices=["diff_means", "lda", "probe_weights"]
    )
    dir_parser.add_argument("--per-tool", action="store_true", help="Extract per-tool directions")
    dir_parser.add_argument("-o", "--output", help="Output path for direction bundle")

    # === visualize command ===
    viz_parser = subparsers.add_parser("visualize", help="Create visualizations")
    viz_parser.add_argument("-a", "--activations", required=True, help="Activations file path")
    viz_parser.add_argument("--layer", type=int, help="Layer to visualize")
    viz_parser.add_argument("-o", "--output", help="Output directory")
    viz_parser.add_argument("--pca", action="store_true", help="PCA variance plot")
    viz_parser.add_argument("--umap", action="store_true", help="UMAP scatter plot")
    viz_parser.add_argument("--probes", action="store_true", help="Probe accuracy plot")
    viz_parser.add_argument("--all", action="store_true", help="All visualizations")

    # === steer command (placeholder) ===
    steer_parser = subparsers.add_parser("steer", help="Activation steering (experimental)")
    steer_parser.add_argument("-m", "--model", required=True, help="Model ID")
    steer_parser.add_argument("-d", "--direction", required=True, help="Direction bundle path")
    steer_parser.add_argument("--strength", type=float, default=1.0, help="Steering strength")

    # === probes command ===
    probes_parser = subparsers.add_parser("probes", help="Run probe battery for stratigraphy")
    probes_subs = probes_parser.add_subparsers(dest="probes_cmd")

    # probes run
    probes_run = probes_subs.add_parser("run", help="Run probe battery on model")
    probes_run.add_argument("-m", "--model", required=True, help="Model ID or path")
    probes_run.add_argument("--layers", help="Layers to probe (comma-separated, default: auto)")
    probes_run.add_argument("--datasets", help="Path to probe dataset directory")
    probes_run.add_argument("--category", help="Filter by category (syntactic,semantic,decision)")
    probes_run.add_argument("--threshold", type=float, default=0.75, help="Emergence threshold")
    probes_run.add_argument(
        "--no-stratigraphy", action="store_true", help="Skip stratigraphy output"
    )
    probes_run.add_argument("-o", "--output", help="Output JSON path")

    # probes init
    probes_init = probes_subs.add_parser("init", help="Initialize probe dataset files")
    probes_init.add_argument("-o", "--output", required=True, help="Output directory")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Route to command handlers
    if args.command == "dataset":
        if args.dataset_cmd == "create":
            cmd_dataset_create(args)
        elif args.dataset_cmd == "show":
            cmd_dataset_show(args)
        else:
            dataset_parser.print_help()
    elif args.command == "collect":
        cmd_collect(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "directions":
        cmd_directions(args)
    elif args.command == "visualize":
        cmd_visualize(args)
    elif args.command == "steer":
        cmd_steer(args)
    elif args.command == "probes":
        if args.probes_cmd == "run":
            cmd_probes(args)
        elif args.probes_cmd == "init":
            cmd_probes_init(args)
        else:
            probes_parser.print_help()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
