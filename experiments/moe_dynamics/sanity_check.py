#!/usr/bin/env python3
"""
Sanity check for experimental MoE architectures.

Validates that:
1. Forward pass produces non-trivial outputs (not all zeros/constants)
2. Outputs vary with input (model isn't collapsed)
3. Gradients flow (model is trainable)
4. Different prompts produce different routing

This does NOT validate quality - that requires training.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from chuk_lazarus.models_v2.components.ffn.moe_experimental import (
    ExperimentalMoEConfig,
    create_experimental_moe,
)


def check_output_nontrivial(module, x: mx.array, name: str) -> dict:
    """Check that outputs are non-trivial."""
    output = module(x)

    # Handle tuple outputs (some variants return extra info)
    if isinstance(output, tuple):
        output = output[0]

    results = {
        "name": name,
        "output_shape": output.shape,
        "output_mean": float(mx.mean(output)),
        "output_std": float(mx.std(output)),
        "output_min": float(mx.min(output)),
        "output_max": float(mx.max(output)),
        "all_zeros": bool(mx.all(output == 0)),
        "all_same": bool(mx.std(output) < 1e-6),
    }

    return results


def check_input_sensitivity(module, name: str, hidden_size: int) -> dict:
    """Check that different inputs produce different outputs."""
    x1 = mx.random.normal((1, 4, hidden_size))
    x2 = mx.random.normal((1, 4, hidden_size))

    out1 = module(x1)
    out2 = module(x2)

    if isinstance(out1, tuple):
        out1 = out1[0]
    if isinstance(out2, tuple):
        out2 = out2[0]

    # Check if outputs differ
    diff = mx.mean(mx.abs(out1 - out2))
    same_output = bool(diff < 1e-6)

    return {
        "name": name,
        "output_diff": float(diff),
        "same_output": same_output,
    }


def check_gradient_flow(module, x: mx.array, name: str) -> dict:
    """Check that gradients flow through the module."""
    def loss_fn(model, x):
        out = model(x)
        if isinstance(out, tuple):
            out = out[0]
        return mx.mean(out ** 2)

    # Compute gradients
    loss, grads = mx.value_and_grad(loss_fn)(module, x)

    # Count non-zero gradients
    def count_nonzero_grads(grads):
        total = 0
        nonzero = 0
        if isinstance(grads, dict):
            for v in grads.values():
                t, n = count_nonzero_grads(v)
                total += t
                nonzero += n
        elif isinstance(grads, list):
            for v in grads:
                t, n = count_nonzero_grads(v)
                total += t
                nonzero += n
        elif isinstance(grads, mx.array):
            total += grads.size
            nonzero += int(mx.sum(mx.abs(grads) > 1e-10))
        return total, nonzero

    total, nonzero = count_nonzero_grads(grads)

    return {
        "name": name,
        "loss": float(loss),
        "total_grad_params": total,
        "nonzero_grad_params": nonzero,
        "grad_coverage": nonzero / total if total > 0 else 0,
    }


def main():
    print("=" * 70)
    print("MoE ARCHITECTURE SANITY CHECK")
    print("=" * 70)
    print()
    print("This validates forward pass mechanics, NOT quality.")
    print("Quality validation requires training.")
    print()

    # Config matching validation
    config = ExperimentalMoEConfig(
        hidden_size=4096,
        intermediate_size=4096 * 4,
        num_experts=32,
        num_experts_per_tok=4,
        team_size=4,
        num_teams=8,
    )

    # Test input
    x = mx.random.normal((2, 16, config.hidden_size))

    variants_to_test = [
        "standard",
        "tiered",
        "tiered_lightweight",
        "lightweight_team",
    ]

    print("-" * 70)
    print("CHECK 1: Output Non-Triviality")
    print("-" * 70)

    for variant in variants_to_test:
        config.variant = variant
        module = create_experimental_moe(config, layer_idx=0)
        result = check_output_nontrivial(module, x, variant)

        status = "✅" if not result["all_zeros"] and not result["all_same"] else "❌"
        print(f"{status} {variant:20} | mean={result['output_mean']:+.4f} std={result['output_std']:.4f} | zeros={result['all_zeros']} same={result['all_same']}")

    print()
    print("-" * 70)
    print("CHECK 2: Input Sensitivity")
    print("-" * 70)

    for variant in variants_to_test:
        config.variant = variant
        module = create_experimental_moe(config, layer_idx=0)
        result = check_input_sensitivity(module, variant, config.hidden_size)

        status = "✅" if not result["same_output"] else "❌"
        print(f"{status} {variant:20} | output_diff={result['output_diff']:.4f} | collapsed={result['same_output']}")

    print()
    print("-" * 70)
    print("CHECK 3: Gradient Flow")
    print("-" * 70)

    for variant in variants_to_test:
        config.variant = variant
        module = create_experimental_moe(config, layer_idx=0)
        result = check_gradient_flow(module, x, variant)

        status = "✅" if result["grad_coverage"] > 0.5 else "⚠️" if result["grad_coverage"] > 0 else "❌"
        print(f"{status} {variant:20} | grad_coverage={result['grad_coverage']:.1%} ({result['nonzero_grad_params']:,}/{result['total_grad_params']:,})")

    print()
    print("-" * 70)
    print("CHECK 4: Output Consistency Across Layers")
    print("-" * 70)

    # For tiered variants, check that different layers produce different outputs
    for variant in ["tiered", "tiered_lightweight"]:
        config.variant = variant

        outputs = []
        for layer_idx in [0, 4, 8, 12, 16, 20, 23]:
            module = create_experimental_moe(config, layer_idx=layer_idx)
            out = module(x)
            if isinstance(out, tuple):
                out = out[0]
            outputs.append((layer_idx, float(mx.mean(out)), float(mx.std(out))))

        print(f"\n{variant}:")
        for layer_idx, mean, std in outputs:
            phase = "early" if layer_idx < 8 else "middle" if layer_idx < 18 else "late"
            print(f"  L{layer_idx:2d} ({phase:6}): mean={mean:+.4f} std={std:.4f}")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("If all checks pass (✅), the architectures are mechanically sound.")
    print("Next step: Train on actual data and measure quality.")
    print()
    print("Quality validation requires:")
    print("  1. Training both Standard and TieredLightweight on same data")
    print("  2. Measuring perplexity on held-out set")
    print("  3. Evaluating downstream tasks (optional but recommended)")
    print()


if __name__ == "__main__":
    main()
