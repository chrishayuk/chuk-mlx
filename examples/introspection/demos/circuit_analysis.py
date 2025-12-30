#!/usr/bin/env python3
"""
Circuit Analysis Demo: Unpacking Tool-Calling in FunctionGemma

This example demonstrates the Phase 6 workflow for analyzing
tool-calling circuits in FunctionGemma 270M:

1. Create a tool prompt dataset
2. Collect activations at decision layers
3. Run geometry analysis (PCA, probes)
4. Extract interpretable directions
5. Run probe battery for stratigraphy

This is the foundation for understanding WHERE and HOW
tool-calling decisions are made in the model.

Run: uv run python examples/introspection/circuit_analysis.py
"""

import json
from pathlib import Path

# Model to analyze - FunctionGemma vs base Gemma
MODEL_ID = "mlx-community/functiongemma-270m-it-bf16"
BASE_MODEL_ID = "mlx-community/gemma-3-270m-it-bf16"

# Output directory for artifacts
OUTPUT_DIR = Path("circuit_analysis_outputs")


def step1_create_dataset():
    """Step 1: Create a tool prompt dataset."""
    print("\n" + "=" * 60)
    print("STEP 1: Create Tool Prompt Dataset")
    print("=" * 60)

    from chuk_lazarus.introspection.circuit import create_tool_calling_dataset

    dataset = create_tool_calling_dataset(
        prompts_per_tool=15,  # 15 prompts per tool type (weather, calendar, etc.)
        no_tool_prompts=60,   # 60 no-tool prompts
        include_edge_cases=True,
        seed=42,
    )

    summary = dataset.summary()
    print(f"\nDataset created:")
    print(f"  Total prompts: {summary['total']}")
    print(f"  Tool-calling: {summary['tool_calling']}")
    print(f"  No-tool: {summary['no_tool']}")

    print(f"\n  By category:")
    for cat, count in summary["by_category"].items():
        print(f"    {cat}: {count}")

    print(f"\n  Sample prompts:")
    for p in dataset.sample(5, seed=42):
        tool = f"[{p.expected_tool}]" if p.expected_tool else "[no-tool]"
        print(f"    {tool}: {p.text[:50]}...")

    # Save dataset
    OUTPUT_DIR.mkdir(exist_ok=True)
    dataset_path = OUTPUT_DIR / "tool_prompts.json"
    dataset.save(dataset_path)
    print(f"\n  Saved to: {dataset_path}")

    return dataset


def step2_collect_activations(dataset):
    """Step 2: Collect activations from the model."""
    print("\n" + "=" * 60)
    print("STEP 2: Collect Model Activations")
    print("=" * 60)

    from chuk_lazarus.introspection.circuit import (
        ActivationCollector, CollectorConfig
    )

    print(f"\nLoading model: {MODEL_ID}...")
    collector = ActivationCollector.from_pretrained(MODEL_ID)
    print(f"  {collector.num_layers} layers, hidden_size={collector.hidden_size}")

    # Focus on decision layers (L8-L14 for this model)
    config = CollectorConfig(
        layers="decision",  # Auto-selects L8-L14
        capture_hidden_states=True,
        max_new_tokens=0,  # Don't generate for speed
    )

    print(f"\nCollecting activations...")
    activations = collector.collect(dataset, config, progress=True)

    print(f"\nCollection complete:")
    print(f"  Samples: {len(activations)}")
    print(f"  Layers captured: {activations.captured_layers}")
    print(f"  Hidden size: {activations.hidden_size}")

    # Save activations
    acts_path = OUTPUT_DIR / "activations"
    activations.save(acts_path)

    return activations


def step3_geometry_analysis(activations):
    """Step 3: Analyze the geometry of tool-calling activations."""
    print("\n" + "=" * 60)
    print("STEP 3: Geometry Analysis")
    print("=" * 60)

    from chuk_lazarus.introspection.circuit import GeometryAnalyzer, ProbeType

    analyzer = GeometryAnalyzer(activations)

    # Analyze each captured layer
    for layer in activations.captured_layers:
        print(f"\n--- Layer {layer} ---")

        # PCA: How low-rank is the tool space?
        pca = analyzer.compute_pca(layer, n_components=50)
        print(f"  PCA: {pca.intrinsic_dimensionality_90} dims @ 90% variance")
        print(f"       {pca.intrinsic_dimensionality_95} dims @ 95% variance")
        print(f"       Top-1 component explains {pca.explained_variance_ratio[0]:.1%}")

        # Binary probe: Can we separate tool vs no-tool?
        probe = analyzer.train_probe(layer, ProbeType.BINARY)
        print(f"  Binary probe: {probe.accuracy:.1%} accuracy (CV: {probe.cv_mean:.1%}±{probe.cv_std:.1%})")

        # Category probe: Can we separate categories?
        cat_probe = analyzer.train_probe(layer, ProbeType.MULTICLASS)
        print(f"  Category probe: {cat_probe.accuracy:.1%} accuracy")

    # Find the best layer for tool classification
    results = analyzer.compare_layers()
    best_layer = max(
        results.items(),
        key=lambda x: x[1].binary_probe.accuracy if x[1].binary_probe else 0
    )[0]
    print(f"\n  Best layer for binary probe: L{best_layer}")

    # UMAP visualization (if available)
    try:
        print(f"\nComputing UMAP for layer {best_layer}...")
        umap_result = analyzer.compute_umap(best_layer)
        print(f"  UMAP embedding shape: {umap_result.embedding.shape}")

        # Save UMAP coordinates
        import numpy as np
        umap_path = OUTPUT_DIR / f"umap_L{best_layer}.npz"
        np.savez(umap_path,
                 embedding=umap_result.embedding,
                 labels=umap_result.labels,
                 categories=umap_result.category_labels)
        print(f"  Saved UMAP to: {umap_path}")
    except ImportError:
        print("  UMAP not available (pip install umap-learn)")

    return results


def step4_extract_directions(activations):
    """Step 4: Extract interpretable directions."""
    print("\n" + "=" * 60)
    print("STEP 4: Direction Extraction")
    print("=" * 60)

    from chuk_lazarus.introspection.circuit import (
        DirectionExtractor, DirectionMethod
    )

    extractor = DirectionExtractor(activations)

    # Find the decision layer (usually L11-12 for small models)
    # Use the layer with highest probe accuracy
    decision_layer = activations.captured_layers[len(activations.captured_layers) // 2]
    print(f"\nExtracting directions from layer {decision_layer}...")

    # Tool mode direction (separates tool vs no-tool)
    tool_dir = extractor.extract_tool_mode_direction(
        decision_layer,
        method=DirectionMethod.DIFFERENCE_OF_MEANS
    )
    print(f"\nTool-mode direction:")
    print(f"  Separation score: {tool_dir.separation_score:.3f}")
    print(f"  Classification accuracy: {tool_dir.accuracy:.2%}")
    print(f"  Mean projection (tool): {tool_dir.mean_projection_positive:.3f}")
    print(f"  Mean projection (no-tool): {tool_dir.mean_projection_negative:.3f}")

    # Per-tool directions
    per_tool = extractor.extract_per_tool_directions(decision_layer)
    print(f"\nPer-tool directions ({len(per_tool)} tools):")
    for name, direction in sorted(per_tool.items(), key=lambda x: -x[1].separation_score)[:5]:
        print(f"  {name}: separation={direction.separation_score:.3f}")

    # Check orthogonality
    print(f"\nOrthogonality check:")
    similarities = extractor.check_orthogonality(per_tool)
    import numpy as np
    off_diag = similarities[np.triu_indices(len(per_tool), k=1)]
    print(f"  Mean off-diagonal cosine similarity: {off_diag.mean():.3f}")
    print(f"  Max off-diagonal similarity: {off_diag.max():.3f}")

    # Save direction bundle
    bundle = extractor.create_bundle(decision_layer, include_per_tool=True)
    bundle_path = OUTPUT_DIR / f"directions_L{decision_layer}"
    bundle.save(bundle_path)
    print(f"\n  Saved directions to: {bundle_path}")

    return bundle


def step5_run_probe_battery():
    """Step 5: Run full probe battery for stratigraphy."""
    print("\n" + "=" * 60)
    print("STEP 5: Probe Battery (Stratigraphy)")
    print("=" * 60)

    from chuk_lazarus.introspection.circuit import ProbeBattery

    print(f"\nLoading model for probing: {MODEL_ID}...")
    battery = ProbeBattery.from_pretrained(MODEL_ID)
    print(f"  {battery.num_layers} layers")
    print(f"  {len(battery.datasets)} probe datasets")

    # Select layers for stratigraphy analysis
    n = battery.num_layers
    layers = [0, 2, 4, 6, 8, 10, 11, 12, 14, n - 1]
    layers = [l for l in layers if l < n]
    print(f"\nProbing layers: {layers}")

    # Run all probes
    results = battery.run_all_probes(layers=layers, progress=True)

    # Print formatted results
    battery.print_results_table(results)

    # Print stratigraphy
    battery.print_stratigraphy(results, threshold=0.75)

    # Save results
    results_path = OUTPUT_DIR / "probe_battery_results.json"
    results.save(results_path)
    print(f"\nSaved probe results to: {results_path}")

    return results


def main():
    """Run the full circuit analysis pipeline."""
    print("=" * 60)
    print("Circuit Analysis: Unpacking Tool-Calling in FunctionGemma")
    print("=" * 60)
    print(f"\nModel: {MODEL_ID}")
    print(f"Output: {OUTPUT_DIR}")

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Run pipeline
    dataset = step1_create_dataset()
    activations = step2_collect_activations(dataset)
    geometry_results = step3_geometry_analysis(activations)
    directions = step4_extract_directions(activations)
    probe_results = step5_run_probe_battery()

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nArtifacts saved to: {OUTPUT_DIR}")
    print(f"  - tool_prompts.json: Prompt dataset")
    print(f"  - activations.safetensors: Model activations")
    print(f"  - directions_L*.npz: Interpretable directions")
    print(f"  - probe_battery_results.json: Stratigraphy results")

    print("""
Next steps:
1. Examine UMAP plots to visualize tool clusters
2. Compare probe accuracy across layers to find decision boundary
3. Use extracted directions for activation steering experiments
4. Compare FunctionGemma vs base Gemma to see what fine-tuning changed

Key questions to answer:
- At which layer does tool_decision accuracy spike? (That's the decision layer)
- Do syntactic features (imperative, question) emerge before decision?
- Are tool-type directions orthogonal? (Suggests independent circuits)
- What's the intrinsic dimensionality of the tool space? (Low-rank = simple circuit)
""")


if __name__ == "__main__":
    main()
