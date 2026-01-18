#!/usr/bin/env python3
"""
Build GPT-OSS-Lite Minimal: 4 experts per layer, fits on 8GB machines.

Based on co-activation analysis:
- 87% of experts are cold (< 1% activation)
- Only 1-6 hot experts per layer on typical prompts
- Dominant expert gets 30-55% routing weight
- k=4 means only 4 experts used per token anyway

Result: 96 experts (vs 768) = 87.5% reduction
"""

import json
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


def analyze_hot_experts(model, tokenizer, num_experts_to_keep: int = 4) -> dict[int, list[int]]:
    """Find the top-k hot experts per layer using diverse prompts."""

    # Diverse prompts covering different capabilities
    test_prompts = [
        # Language
        "The capital of France is",
        "Once upon a time in a land",
        "Hello, my name is",
        "The opposite of hot is",

        # Math (important!)
        "2 + 2 =",
        "10 - 3 =",
        "5 * 6 =",
        "127 * 89 =",

        # Code
        "def fibonacci(n):",
        "import numpy as np",
        "SELECT * FROM users WHERE",
        "for i in range(10):",

        # Reasoning
        "If all cats are mammals, then",
        "The pattern 2, 4, 8, 16 continues as",

        # Knowledge
        "Water boils at",
        "Einstein developed the theory of",
    ]

    layers = model.model.layers
    hidden_size = model.args.hidden_size
    activation_counts = defaultdict(lambda: defaultdict(int))

    print(f"Analyzing expert activations across {len(test_prompts)} prompts...")

    for prompt in test_prompts:
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])

        # Forward pass tracking activations
        h = model.model.embed_tokens(input_ids)
        batch_size, seq_len, _ = h.shape
        mask = mx.triu(mx.full((seq_len, seq_len), float('-inf'), dtype=h.dtype), k=1)

        for layer_idx, layer in enumerate(layers):
            # Attention
            normed = layer.input_layernorm(h)
            attn_out = layer.self_attn(normed, mask)
            h = h + attn_out

            # Get router decisions
            normed = layer.post_attention_layernorm(h)
            x_flat = normed.reshape(-1, hidden_size)
            logits = layer.mlp.router(x_flat)

            # Top-k selection (k=4)
            top_k_indices = mx.argsort(logits, axis=-1)[:, -4:]
            mx.eval(top_k_indices)

            # Count activations
            for tok_idx in range(seq_len):
                for k in range(4):
                    expert_idx = int(top_k_indices[tok_idx, k])
                    activation_counts[layer_idx][expert_idx] += 1

            # Continue forward pass
            h = h + layer.mlp(normed)

    # Select top-k hot experts per layer
    hot_experts = {}
    print("\nHot experts per layer:")
    for layer_idx in range(len(layers)):
        counts = activation_counts[layer_idx]
        # Sort by activation count, take top-k
        sorted_experts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        hot = [exp_idx for exp_idx, _ in sorted_experts[:num_experts_to_keep]]

        # If we don't have enough, fill with remaining experts
        if len(hot) < num_experts_to_keep:
            all_experts = set(range(32))
            remaining = list(all_experts - set(hot))
            hot.extend(remaining[:num_experts_to_keep - len(hot)])

        hot_experts[layer_idx] = sorted(hot)  # Sort for consistency

        top_counts = [counts[e] for e in hot_experts[layer_idx]]
        total = sum(counts.values())
        coverage = sum(top_counts) / total if total > 0 else 0
        print(f"  Layer {layer_idx:2d}: {hot_experts[layer_idx]} (coverage: {coverage:.1%})")

    return hot_experts


def extract_lite_weights(model, hot_experts: dict[int, list[int]]) -> dict[str, mx.array]:
    """Extract only the hot expert weights."""

    lite_weights = {}

    # Non-MoE components (unchanged)
    print("\nExtracting non-MoE weights...")
    lite_weights['model.embed_tokens.weight'] = model.model.embed_tokens.weight
    lite_weights['model.norm.weight'] = model.model.norm.weight
    lite_weights['lm_head.weight'] = model.lm_head.weight

    print("Extracting layer weights...")
    for layer_idx in range(24):
        layer = model.model.layers[layer_idx]
        prefix = f'model.layers.{layer_idx}'

        # Attention (unchanged)
        attn = layer.self_attn
        for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            proj = getattr(attn, proj_name)
            if hasattr(proj, 'weight'):
                lite_weights[f'{prefix}.self_attn.{proj_name}.weight'] = proj.weight
            if hasattr(proj, 'scales'):
                lite_weights[f'{prefix}.self_attn.{proj_name}.scales'] = proj.scales
            # Quantization biases (mxfp4)
            if hasattr(proj, 'biases') and proj.biases is not None:
                lite_weights[f'{prefix}.self_attn.{proj_name}.biases'] = proj.biases
            # Linear layer bias (separate from quantization biases)
            if hasattr(proj, 'bias') and proj.bias is not None:
                lite_weights[f'{prefix}.self_attn.{proj_name}.bias'] = proj.bias

        # Attention sinks (important for long-context attention)
        if hasattr(attn, 'sinks') and attn.sinks is not None:
            lite_weights[f'{prefix}.self_attn.sinks'] = attn.sinks

        # Layer norms
        lite_weights[f'{prefix}.input_layernorm.weight'] = layer.input_layernorm.weight
        lite_weights[f'{prefix}.post_attention_layernorm.weight'] = layer.post_attention_layernorm.weight

        # MoE - extract only hot experts
        moe = layer.mlp
        hot = hot_experts[layer_idx]
        num_hot = len(hot)

        # Router weights - only for hot experts
        router_weight = moe.router.weight  # (32, hidden)
        lite_weights[f'{prefix}.mlp.router.weight'] = router_weight[mx.array(hot)]

        # Router bias if present
        if hasattr(moe.router, 'bias') and moe.router.bias is not None:
            router_bias = moe.router.bias  # (32,)
            lite_weights[f'{prefix}.mlp.router.bias'] = router_bias[mx.array(hot)]

        # Expert weights - stack hot experts
        experts = moe.experts

        for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
            proj = getattr(experts, proj_name)

            # Handle quantized weights
            if hasattr(proj, 'weight'):
                # Weight shape: (num_experts, ...) - select hot experts
                w = proj.weight
                lite_weights[f'{prefix}.mlp.experts.{proj_name}.weight'] = w[mx.array(hot)]

            if hasattr(proj, 'scales') and proj.scales is not None:
                s = proj.scales
                lite_weights[f'{prefix}.mlp.experts.{proj_name}.scales'] = s[mx.array(hot)]

            if hasattr(proj, 'biases') and proj.biases is not None:
                b = proj.biases
                lite_weights[f'{prefix}.mlp.experts.{proj_name}.biases'] = b[mx.array(hot)]

            # Output bias (not quantization bias)
            if hasattr(proj, 'bias') and proj.bias is not None:
                bias = proj.bias
                lite_weights[f'{prefix}.mlp.experts.{proj_name}.bias'] = bias[mx.array(hot)]

        print(f"  Layer {layer_idx}: {num_hot} experts extracted")

    return lite_weights


def build_lite(num_experts: int = 4, output_dir: str = "./gpt-oss-lite-minimal"):
    """Build the minimal lite model."""

    from mlx_lm import load

    print("=" * 60)
    print("GPT-OSS-LITE MINIMAL BUILD")
    print(f"Target: {num_experts} experts per layer")
    print("=" * 60)

    # Load full model
    print("\nLoading GPT-OSS-20B...")
    model, tokenizer = load("openai/gpt-oss-20b")

    # Analyze hot experts
    hot_experts = analyze_hot_experts(model, tokenizer, num_experts)

    # Extract weights
    lite_weights = extract_lite_weights(model, hot_experts)

    # Evaluate weights
    mx.eval(lite_weights)

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print(f"\nSaving to {output_path}...")

    # Save weights as safetensors
    mx.save_safetensors(str(output_path / "model.safetensors"), lite_weights)

    # Build config
    config = {
        "model_type": "gpt_oss_lite_minimal",
        "architectures": ["GPTOSSLiteForCausalLM"],
        "hidden_size": 2880,
        "intermediate_size": 2880,
        "num_hidden_layers": 24,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "vocab_size": 201088,
        "num_local_experts": num_experts,
        "num_experts_per_tok": min(4, num_experts),  # Use all if <= 4
        "rope_scaling": {
            "type": "yarn",
            "factor": 10.0,
            "original_max_position_embeddings": 16384
        },
        "hot_experts_by_layer": {str(k): v for k, v in hot_experts.items()},
        "source_model": "openai/gpt-oss-20b",
        "total_experts": num_experts * 24,
        "original_experts": 768,
        "reduction_percent": f"{(1 - num_experts * 24 / 768) * 100:.1f}%"
    }

    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Copy tokenizer files
    tokenizer.save_pretrained(str(output_path))

    # Calculate sizes
    total_bytes = sum(w.nbytes for w in lite_weights.values())
    fp16_gb = total_bytes / 1e9
    q4_estimate = fp16_gb / 4  # Rough Q4 estimate

    print("\n" + "=" * 60)
    print("BUILD COMPLETE")
    print("=" * 60)
    print(f"Output: {output_path}")
    print(f"Experts: {num_experts} per layer × 24 = {num_experts * 24} total")
    print(f"Original: 768 experts")
    print(f"Reduction: {(1 - num_experts * 24 / 768) * 100:.1f}%")
    print(f"Size (fp16): {fp16_gb:.2f} GB")
    print(f"Size (Q4 est): ~{q4_estimate:.1f} GB")
    print(f"Target RAM: ~{q4_estimate * 1.3:.1f} GB")
    print("=" * 60)

    return hot_experts, lite_weights


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build GPT-OSS-Lite Minimal")
    parser.add_argument("--experts", type=int, default=4,
                        help="Number of experts to keep per layer (default: 4)")
    parser.add_argument("--output", type=str, default="./gpt-oss-lite-minimal",
                        help="Output directory")
    args = parser.parse_args()

    build_lite(num_experts=args.experts, output_dir=args.output)
