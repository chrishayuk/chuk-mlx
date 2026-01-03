#!/usr/bin/env python3
"""
Test MoE Expert Compression Quality

This script:
1. Loads GPT-OSS 20B
2. Gets baseline outputs for test prompts
3. Applies compression to one layer
4. Compares outputs before/after compression

Usage:
    uv run python examples/introspection/experiments/moe/test_compression_quality.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))

import mlx.core as mx
from mlx_lm import load, generate

from chuk_lazarus.introspection.moe import (
    ExpertCompressor,
    MoEHooks,
    detect_moe_architecture,
    get_moe_layer_info,
)


def generate_text(model, tokenizer, prompt: str, max_tokens: int = 50) -> str:
    """Generate text from a prompt."""
    return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)


def main():
    print("=" * 70)
    print("MoE Expert Compression Quality Test - GPT-OSS 20B")
    print("=" * 70)

    # Load GPT-OSS
    print("\nLoading GPT-OSS 20B...")
    model_path = Path.home() / ".cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee"
    model, tokenizer = load(str(model_path))

    # Test prompts
    test_prompts = [
        "The capital of France is",
        "def fibonacci(n):",
        "Hello, how are you today?",
    ]

    # Get baseline outputs
    print("\n" + "-" * 70)
    print("BASELINE OUTPUTS (before compression)")
    print("-" * 70)

    baseline_outputs = []
    for prompt in test_prompts:
        output = generate_text(model, tokenizer, prompt, max_tokens=30)
        baseline_outputs.append(output)
        print(f"\nPrompt: {prompt}")
        print(f"Output: {output}")

    # Get MoE layer info
    hooks = MoEHooks(model)
    moe_layers = hooks.moe_layer_indices
    print(f"\n\nFound {len(moe_layers)} MoE layers")

    # Create compressor and analyze
    compressor = ExpertCompressor(model, tokenizer)

    # Pick middle layer
    layer_idx = moe_layers[len(moe_layers) // 2]
    print(f"\nAnalyzing layer {layer_idx} for compression...")

    # Get compression plan
    analysis = compressor.analyze_compression_potential(layer_idx, test_prompts)
    print(f"  Merge candidates: {len(analysis.get('merge_candidates', []))}")
    print(f"  Prune candidates: {len(analysis.get('prune_candidates', []))}")

    # Plan conservative compression (just merge the most similar pair)
    plan = compressor.plan_compression(layer_idx, strategy="balanced")
    print(f"\nCompression plan:")
    print(f"  Original: {plan.original_num_experts} experts")
    print(f"  Target: {plan.target_num_experts} experts")
    print(f"  Merges: {len(plan.merges)}")
    print(f"  Pruned: {len(plan.pruned_experts)}")

    if plan.merges or plan.pruned_experts:
        # Apply compression
        print("\n" + "-" * 70)
        print("APPLYING COMPRESSION...")
        print("-" * 70)

        try:
            compressed_config = compressor.apply_compression(plan, layer_idx, inplace=True)
            print(f"Compression applied!")
            print(f"  Original experts: {compressed_config.original_num_experts}")
            print(f"  Compressed to: {compressed_config.compressed_num_experts} experts")
            mx.eval(model.parameters())

            # Get post-compression outputs
            print("\n" + "-" * 70)
            print("POST-COMPRESSION OUTPUTS")
            print("-" * 70)

            post_outputs = []
            for prompt in test_prompts:
                output = generate_text(model, tokenizer, prompt, max_tokens=30)
                post_outputs.append(output)
                print(f"\nPrompt: {prompt}")
                print(f"Output: {output}")

            # Compare
            print("\n" + "-" * 70)
            print("COMPARISON")
            print("-" * 70)

            for i, prompt in enumerate(test_prompts):
                print(f"\nPrompt: {prompt}")
                print(f"  Before: {baseline_outputs[i][:80]}...")
                print(f"  After:  {post_outputs[i][:80]}...")

                # Simple similarity check
                baseline_tokens = set(baseline_outputs[i].split())
                post_tokens = set(post_outputs[i].split())
                overlap = len(baseline_tokens & post_tokens) / max(len(baseline_tokens), 1)
                print(f"  Token overlap: {overlap:.1%}")

        except Exception as e:
            print(f"Error applying compression: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\nNo compression needed for this layer")

    print("\n" + "=" * 70)
    print("Quality test complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
