#!/usr/bin/env python3
"""
Gemma Alignment Circuit Analysis

This script provides tools for analyzing the computation suppression circuits
discovered in Gemma instruct models. Key findings:

- Base model computes arithmetic at layer 22-23 and PRESERVES it (81%)
- Instruct model computes at layer 22-23 but DESTROYS at layer 24 (1%)
- Instruct model REBUILDS partially at layer 31-33 (33%)

This is intentional design for tool delegation - the model suppresses
unreliable internal computation to enable reliable external tool use.

Usage:
    # Run logit lens analysis (already exists)
    uv run python examples/introspection/logit_lens.py \\
        --model mlx-community/gemma-3-4b-it-bf16 \\
        --prompt "6 * 7 =" \\
        --track " 42" \\
        --all-layers

    # Run probe analysis to find the suppression circuit
    uv run python examples/introspection/gemma_alignment_circuits.py \\
        --model mlx-community/gemma-3-4b-it-bf16 \\
        --analysis probes

    # Collect activations for direction extraction
    uv run python examples/introspection/gemma_alignment_circuits.py \\
        --model mlx-community/gemma-3-4b-it-bf16 \\
        --analysis collect

    # Extract suppression direction
    uv run python examples/introspection/gemma_alignment_circuits.py \\
        --model mlx-community/gemma-3-4b-it-bf16 \\
        --analysis directions

    # Compare base vs instruct models
    uv run python examples/introspection/gemma_alignment_circuits.py \\
        --analysis compare
"""

from __future__ import annotations

import argparse
from pathlib import Path


def run_probe_analysis(model_id: str, layers: list[int] | None = None):
    """Run probe analysis to find suppression circuit."""
    from chuk_lazarus.introspection.circuit import (
        ProbeBattery,
        create_arithmetic_probe,
        create_suppression_probe,
        create_factual_consistency_probe,
    )

    print(f"\n{'='*60}")
    print(f"PROBE ANALYSIS: {model_id}")
    print(f"{'='*60}")

    # Load model
    print("\nLoading model...")
    battery = ProbeBattery.from_pretrained(model_id)
    print(f"Model loaded: {battery.num_layers} layers")

    # Add probes
    battery.add_dataset(create_arithmetic_probe())
    battery.add_dataset(create_suppression_probe())
    battery.add_dataset(create_factual_consistency_probe())

    # Run probes
    if layers is None:
        # Focus on layers around the destruction point (L24 for Gemma-3-4b)
        layers = list(range(18, 34, 2))

    print(f"\nProbing layers: {layers}")
    results = battery.run_all_probes(layers=layers)

    # Print results
    battery.print_results_table(results)

    # Find key layers
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)

    for probe_name in results.probes.keys():
        emergence = results.find_emergence_layer(probe_name)
        destruction = results.find_destruction_layer(probe_name)
        if emergence is not None:
            print(f"\n{probe_name}:")
            print(f"  Emerges at: L{emergence}")
            if destruction is not None:
                print(f"  Destroyed at: L{destruction}")

    return results


def collect_activations(model_id: str, output_path: str = "gemma_activations"):
    """Collect activations for direction extraction."""
    from chuk_lazarus.introspection.circuit import (
        ActivationCollector,
        CollectorConfig,
        create_arithmetic_dataset,
    )

    print(f"\n{'='*60}")
    print(f"COLLECTING ACTIVATIONS: {model_id}")
    print(f"{'='*60}")

    # Create dataset
    dataset = create_arithmetic_dataset()
    print(f"\nDataset: {len(dataset)} prompts")
    print(f"  Arithmetic: {len(dataset.get_positive())}")
    print(f"  Non-arithmetic: {len(dataset.get_negative())}")

    # Load collector
    print("\nLoading model...")
    collector = ActivationCollector.from_pretrained(model_id)
    print(f"Model loaded: {collector.num_layers} layers")

    # Collect at key layers
    layers = list(range(18, 34))  # Focus on destruction region
    config = CollectorConfig(
        layers=layers,
        max_new_tokens=0,  # Don't generate, just collect
    )

    print(f"\nCollecting at layers: {layers}")
    activations = collector.collect(dataset, config)

    # Save
    activations.save(output_path)
    print(f"\nSaved to: {output_path}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    summary = activations.summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")

    return activations


def extract_directions(activations_path: str, output_path: str = "gemma_directions"):
    """Extract suppression direction from activations."""
    from chuk_lazarus.introspection.circuit import (
        CollectedActivations,
        DirectionExtractor,
        DirectionMethod,
    )

    print(f"\n{'='*60}")
    print(f"EXTRACTING DIRECTIONS")
    print(f"{'='*60}")

    # Load activations
    print(f"\nLoading activations from: {activations_path}")
    activations = CollectedActivations.load(activations_path)
    print(f"Loaded {len(activations)} samples, {len(activations.captured_layers)} layers")

    # Extract directions
    extractor = DirectionExtractor(activations)
    bundle = extractor.extract_all_layers(method=DirectionMethod.DIFFERENCE_OF_MEANS)

    # Print summary
    extractor.print_summary(bundle)

    # Save
    bundle.save(output_path)
    print(f"\nSaved directions to: {output_path}")

    # Find the best layer for steering
    best_layer = bundle.find_best_layer()
    if best_layer is not None:
        print(f"\nBest layer for steering: L{best_layer}")
        print(f"  Separation: {bundle.directions[best_layer].separation_score:.3f}")
        print(f"  Accuracy: {bundle.directions[best_layer].accuracy:.1%}")

    return bundle


def compare_models():
    """Compare base vs instruct models on arithmetic."""
    import mlx.core as mx
    from chuk_lazarus.introspection.hooks import ModelHooks, CaptureConfig, LayerSelection
    from chuk_lazarus.introspection.logit_lens import LogitLens
    from chuk_lazarus.introspection.ablation import AblationStudy

    models = [
        ("mlx-community/gemma-3-4b-pt-bf16", "Base"),
        ("mlx-community/gemma-3-4b-it-bf16", "Instruct"),
    ]

    prompts = [
        ("6 * 7 =", " 42"),
        ("156 + 287 =", " 443"),
        ("The capital of France is", " Paris"),
    ]

    print(f"\n{'='*70}")
    print("MODEL COMPARISON: Base vs Instruct")
    print(f"{'='*70}")

    for model_id, model_name in models:
        print(f"\n{model_name}: {model_id}")
        print("-" * 60)

        # Load model
        study = AblationStudy.from_pretrained(model_id)
        model = study.adapter.model
        tokenizer = study.adapter.tokenizer
        config = study.adapter.config

        for prompt, track_token in prompts:
            # Set up hooks
            hooks = ModelHooks(model, model_config=config)
            hooks.configure(CaptureConfig(
                layers=LayerSelection.ALL,
                capture_hidden_states=True,
            ))

            # Forward pass
            input_ids = tokenizer.encode(prompt, return_tensors="np")
            input_ids = mx.array(input_ids)
            hooks.forward(input_ids)

            # Logit lens analysis
            lens = LogitLens(hooks, tokenizer)

            # Track the token
            track_id = tokenizer.encode(track_token, add_special_tokens=False)
            if isinstance(track_id, list):
                track_id = track_id[0]

            evolution = lens.track_token(track_token)

            # Find peak and final probabilities
            probs = evolution.probabilities
            layers = evolution.layers
            peak_idx = max(range(len(probs)), key=lambda i: probs[i])
            peak_layer = layers[peak_idx]
            peak_prob = probs[peak_idx]
            final_prob = probs[-1]

            # Find destruction point (if any)
            destruction = None
            for i in range(1, len(probs)):
                if probs[i-1] > 0.5 and probs[i] < 0.2:
                    destruction = layers[i]
                    destruction_from = probs[i-1]
                    destruction_to = probs[i]
                    break

            print(f"\n  Prompt: {prompt!r}")
            print(f"  Track:  {track_token!r}")
            print(f"  Peak:   L{peak_layer} = {peak_prob:.1%}")
            print(f"  Final:  L{layers[-1]} = {final_prob:.1%}")
            if destruction:
                print(f"  Destroyed at L{destruction}: {destruction_from:.1%} -> {destruction_to:.1%}")


def run_steering_experiment(
    model_id: str,
    directions_path: str,
    custom_prompt: str | None = None,
    layers: list[int] | None = None,
    coefficients: list[float] | None = None,
):
    """Run steering experiment to restore arithmetic."""
    from chuk_lazarus.introspection.steering import ActivationSteering, SteeringConfig
    from chuk_lazarus.introspection.circuit import DirectionBundle

    print(f"\n{'='*60}")
    print(f"STEERING EXPERIMENT")
    print(f"{'='*60}")

    # Load directions
    print(f"\nLoading directions from: {directions_path}")
    directions = DirectionBundle.load(directions_path)
    print(f"Loaded directions for {len(directions.directions)} layers")

    # Determine steering layer
    if layers:
        steer_layer = layers[0]  # Use first specified layer
    else:
        steer_layer = directions.find_best_layer()
        if steer_layer is None:
            print("No suitable layer found for steering")
            return

    print(f"Steering at layer: L{steer_layer}")

    # Load steerer
    print(f"\nLoading model: {model_id}")
    steerer = ActivationSteering.from_pretrained(model_id)
    steerer.add_directions(directions)

    # Determine prompts
    if custom_prompt:
        prompts = [custom_prompt]
    else:
        prompts = [
            "6 * 7 =",
            "156 + 287 =",
            "23 * 17 =",
        ]

    # Determine coefficients
    if coefficients is None:
        coefficients = [-1.0, 0.0, 1.0]

    for prompt in prompts:
        print(f"\n{'─'*50}")
        print(f"Prompt: {prompt}")
        print("─" * 50)

        config = SteeringConfig(layers=[steer_layer], max_new_tokens=10)
        steerer.print_comparison(prompt, coefficients=coefficients, config=config)


def run_layer_dynamics(
    model_id: str,
    directions_path: str,
    prompt: str = "6 * 7 =",
    track_token: str = " 42",
    layers: list[int] | None = None,
    coefficients: list[float] | None = None,
):
    """
    Show layer-by-layer dynamics WITH steering.

    This reveals whether steering prevents the L24 destruction.
    """
    from chuk_lazarus.introspection.steering import ActivationSteering, SteeringConfig
    from chuk_lazarus.introspection.circuit import DirectionBundle

    print(f"\n{'='*60}")
    print(f"LAYER DYNAMICS WITH STEERING")
    print(f"{'='*60}")

    # Load directions
    print(f"\nLoading directions from: {directions_path}")
    directions = DirectionBundle.load(directions_path)
    print(f"Loaded directions for {len(directions.directions)} layers")

    # Determine steering layer
    if layers:
        steer_layer = layers[0]
    else:
        steer_layer = directions.find_best_layer()
        if steer_layer is None:
            print("No suitable layer found for steering")
            return

    print(f"Steering at layer: L{steer_layer}")

    # Load steerer
    print(f"\nLoading model: {model_id}")
    steerer = ActivationSteering.from_pretrained(model_id)
    steerer.add_directions(directions)

    # Coefficients
    if coefficients is None:
        coefficients = [-2.0, 0.0, 2.0]

    config = SteeringConfig(layers=[steer_layer])
    steerer.print_layer_dynamics(
        prompt,
        track_token,
        coefficients=coefficients,
        config=config,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Gemma Alignment Circuit Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        default="mlx-community/gemma-3-4b-it-bf16",
        help="Model ID to analyze",
    )
    parser.add_argument(
        "--analysis",
        choices=["probes", "collect", "directions", "compare", "steer", "dynamics"],
        default="probes",
        help="Type of analysis to run",
    )
    parser.add_argument(
        "--activations",
        default="gemma_activations",
        help="Path to activations (for directions/steer)",
    )
    parser.add_argument(
        "--directions",
        default="gemma_directions",
        help="Path to directions (for steer)",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Layers to analyze (comma-separated, e.g., '20,22,24,26')",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt for steering (default: built-in arithmetic prompts)",
    )
    parser.add_argument(
        "--coefficients",
        type=str,
        default="-1.0,0.0,1.0",
        help="Steering coefficients (comma-separated, e.g., '-2.0,-1.0,0.0,1.0,2.0')",
    )
    parser.add_argument(
        "--track",
        type=str,
        default=" 42",
        help="Token to track for dynamics analysis (default: ' 42')",
    )

    args = parser.parse_args()

    # Parse layers
    layers = None
    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]

    # Parse coefficients
    coefficients = [float(x) for x in args.coefficients.split(",")]

    if args.analysis == "probes":
        run_probe_analysis(args.model, layers)
    elif args.analysis == "collect":
        collect_activations(args.model, args.activations)
    elif args.analysis == "directions":
        extract_directions(args.activations, args.directions)
    elif args.analysis == "compare":
        compare_models()
    elif args.analysis == "steer":
        run_steering_experiment(
            args.model,
            args.directions,
            custom_prompt=args.prompt,
            layers=layers,
            coefficients=coefficients,
        )
    elif args.analysis == "dynamics":
        run_layer_dynamics(
            args.model,
            args.directions,
            prompt=args.prompt or "6 * 7 =",
            track_token=args.track,
            layers=layers,
            coefficients=coefficients,
        )


if __name__ == "__main__":
    main()
