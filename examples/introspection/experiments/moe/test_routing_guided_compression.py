#!/usr/bin/env python3
"""
Routing-Guided Expert Compression

Use actual routing patterns on calibration data to identify which experts
are most important and prune the least-used ones.

The idea: Run a calibration dataset through the model, track which experts
are selected, and keep only the most frequently used ones.

Usage:
    uv run python examples/introspection/experiments/moe/test_routing_guided_compression.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))

import mlx.core as mx
from mlx_lm import generate, load

from chuk_lazarus.introspection.moe import (
    ExpertCompressor,
    MoECaptureConfig,
    MoEHooks,
    estimate_model_size,
)


def get_expert_usage(model, tokenizer, prompts: list[str]) -> dict[int, list[int]]:
    """
    Run prompts through model and track which experts are used most at each layer.

    Returns: dict mapping layer_idx -> sorted list of expert indices by usage frequency
    """
    hooks = MoEHooks(model)
    hooks.configure(
        MoECaptureConfig(
            capture_router_logits=False,
            capture_expert_assignments=True,
            capture_routing_weights=False,
        )
    )

    # Accumulate usage counts: layer_idx -> expert_idx -> count
    usage_counts = {layer: {} for layer in hooks.moe_layer_indices}

    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])

        # Forward pass
        hooks.forward(input_ids)

        # Collect expert assignments
        for layer_idx in hooks.moe_layer_indices:
            if layer_idx in hooks.state.selected_experts:
                experts = hooks.state.selected_experts[layer_idx]
                # experts shape: [batch, seq_len, num_experts_per_tok]
                for expert_idx in experts.flatten().tolist():
                    usage_counts[layer_idx][expert_idx] = (
                        usage_counts[layer_idx].get(expert_idx, 0) + 1
                    )

        hooks.state.clear()

    # Sort experts by usage frequency for each layer
    expert_rankings = {}
    for layer_idx, counts in usage_counts.items():
        sorted_experts = sorted(counts.keys(), key=lambda e: counts[e], reverse=True)
        expert_rankings[layer_idx] = sorted_experts

    return expert_rankings


def compress_with_routing_guidance(
    model_path: Path,
    calibration_prompts: list[str],
    target_experts: int,
    test_prompts: list[str],
    baseline_outputs: list[str],
    baseline_size: float,
) -> dict:
    """
    Compress model using routing-guided expert selection.
    """
    # Step 1: Load model and get routing patterns
    print("  Loading model for calibration...")
    model, tokenizer = load(str(model_path))

    print(f"  Running {len(calibration_prompts)} calibration prompts...")
    expert_rankings = get_expert_usage(model, tokenizer, calibration_prompts)

    # Step 2: Apply compression using the ranking
    hooks = MoEHooks(model)
    compressor = ExpertCompressor(model, tokenizer)

    print(f"  Compressing to {target_experts} most-used experts per layer...")
    for layer_idx in hooks.moe_layer_indices:
        # Get the top N experts for this layer
        ranked = expert_rankings.get(layer_idx, list(range(32)))
        kept_experts = sorted(ranked[:target_experts])  # Keep top N, sorted

        # Create custom plan with these experts
        plan = compressor.plan_compression(
            layer_idx, target_experts=target_experts, strategy="aggressive"
        )
        # Override with routing-guided selection
        plan.kept_experts = kept_experts

        compressor.apply_compression(plan, layer_idx, inplace=True)

    mx.eval(model.parameters())

    # Step 3: Test quality
    new_size = estimate_model_size(model)
    reduction = (1 - new_size["total"] / baseline_size) * 100

    quality_scores = []
    outputs = []
    for i, p in enumerate(test_prompts):
        out = generate(model, tokenizer, prompt=p, max_tokens=30, verbose=False)
        outputs.append(out)
        baseline_tokens = set(baseline_outputs[i].split())
        new_tokens = set(out.split())
        overlap = len(baseline_tokens & new_tokens) / max(len(baseline_tokens), 1)
        quality_scores.append(overlap)

    return {
        "size_after": new_size["total"],
        "reduction": reduction,
        "quality": sum(quality_scores) / len(quality_scores),
        "outputs": outputs,
    }


def main():
    print("=" * 70)
    print("Routing-Guided Expert Compression")
    print("=" * 70)
    print("\nIdea: Keep the most frequently routed experts, prune rare ones")

    model_path = (
        Path.home()
        / ".cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee"
    )

    # Get baseline
    print("\nLoading GPT-OSS for baseline...")
    model, tokenizer = load(str(model_path))
    baseline = estimate_model_size(model)
    baseline_size = baseline["total"]
    print(f"Baseline: {baseline_size / 1e9:.2f}B params")

    # Calibration prompts - diverse set to capture routing patterns
    calibration_prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "def hello_world(): print('Hello, World!')",
        "SELECT id, name FROM users WHERE active = true",
        "The Pythagorean theorem states that a^2 + b^2 = c^2",
        "In machine learning, neural networks are",
        "The capital of France is Paris.",
        "import numpy as np\nimport pandas as pd",
        "Write a function to calculate factorial:",
        "HTTP/1.1 200 OK\nContent-Type: application/json",
        "The meaning of life according to philosophy",
        "class Person:\n    def __init__(self, name):",
        "London, New York, Tokyo, and Paris are major cities.",
    ]

    # Test prompts
    test_prompts = [
        "The capital of France is",
        "def fibonacci(n):",
        "Hello, how are you?",
        "SELECT * FROM users WHERE",
    ]

    print("\n--- BASELINE OUTPUTS ---")
    baseline_outputs = []
    for p in test_prompts:
        out = generate(model, tokenizer, prompt=p, max_tokens=30, verbose=False)
        baseline_outputs.append(out)
        print(f"{p}")
        print(f"  -> {out[:60]}...")

    del model, tokenizer

    results = []

    # Test different compression levels with routing guidance
    for target_experts in [16, 12, 8]:
        print(f"\n{'=' * 70}")
        print(f"ROUTING-GUIDED: {target_experts} experts per layer")
        print("=" * 70)

        result = compress_with_routing_guidance(
            model_path,
            calibration_prompts,
            target_experts,
            test_prompts,
            baseline_outputs,
            baseline_size,
        )
        result["name"] = f"Routing-guided {target_experts}"
        results.append(result)

        print(
            f"Size: {baseline_size / 1e9:.2f}B -> {result['size_after'] / 1e9:.2f}B ({result['reduction']:.1f}% reduction)"
        )
        print(f"Quality: {result['quality']:.0%}")
        for i, p in enumerate(test_prompts):
            print(f"  {p[:25]}... -> {result['outputs'][i][:35]}...")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Routing-Guided Compression")
    print("=" * 70)
    print(f"\n{'Configuration':<25} {'Size':<10} {'Reduction':<12} {'Quality':<10}")
    print("-" * 60)

    for r in results:
        print(
            f"{r['name']:<25} {r['size_after'] / 1e9:.2f}B{'':>4} {r['reduction']:.1f}%{'':>6} {r['quality']:.0%}"
        )

    print("-" * 60)
    print("\nThis approach should retain quality better than random/similarity-based")
    print("because it keeps the experts that are actually used for diverse inputs.")
    print("=" * 70)


if __name__ == "__main__":
    main()
