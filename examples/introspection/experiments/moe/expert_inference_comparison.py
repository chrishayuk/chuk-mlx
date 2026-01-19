"""Compare inference quality between original and compressed expert.

Tests whether compression affects actual model outputs, not just weight reconstruction.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from sklearn.utils.extmath import randomized_svd

from chuk_lazarus.introspection.moe.detector import detect_moe_architecture, get_moe_layers
from chuk_lazarus.introspection.moe.moe_type import MoETypeService


def create_compressed_expert_fn(weights: dict, ranks: dict):
    """Create a function that applies compressed expert.

    Returns function that mimics expert forward pass using compressed weights.
    """
    # Compress each projection
    compressed = {}
    for proj in ["gate", "up", "down"]:
        w = np.array(weights[proj].astype(mx.float32))
        all_w = np.array(weights[f"{proj}_all"].astype(mx.float32))
        base = np.mean(all_w, axis=0)
        delta = w - base

        rank = ranks[proj]
        U, S, Vh = randomized_svd(delta, n_components=rank, n_iter=3, random_state=42)
        U_scaled = U @ np.diag(S)

        compressed[f"{proj}_base"] = mx.array(base)
        compressed[f"{proj}_U"] = mx.array(U_scaled)
        compressed[f"{proj}_V"] = mx.array(Vh)

    def forward(x: mx.array) -> mx.array:
        """Compressed expert forward pass."""
        # Gate: efficient low-rank
        gate_out = x @ compressed["gate_base"].T
        gate_out = gate_out + (x @ compressed["gate_V"].T) @ compressed["gate_U"].T
        if "gate_bias" in weights and weights["gate_bias"] is not None:
            gate_out = gate_out + weights["gate_bias"]

        # Up: efficient low-rank
        up_out = x @ compressed["up_base"].T
        up_out = up_out + (x @ compressed["up_V"].T) @ compressed["up_U"].T
        if "up_bias" in weights and weights["up_bias"] is not None:
            up_out = up_out + weights["up_bias"]

        # SwiGLU
        hidden = nn.silu(gate_out) * up_out

        # Down: efficient low-rank
        out = hidden @ compressed["down_base"].T
        out = out + (hidden @ compressed["down_V"].T) @ compressed["down_U"].T
        if "down_bias" in weights and weights["down_bias"] is not None:
            out = out + weights["down_bias"]

        return out

    return forward


def original_expert_forward(x: mx.array, weights: dict) -> mx.array:
    """Original expert forward pass."""
    gate_out = x @ weights["gate"].T
    if "gate_bias" in weights and weights["gate_bias"] is not None:
        gate_out = gate_out + weights["gate_bias"]

    up_out = x @ weights["up"].T
    if "up_bias" in weights and weights["up_bias"] is not None:
        up_out = up_out + weights["up_bias"]

    hidden = nn.silu(gate_out) * up_out

    out = hidden @ weights["down"].T
    if "down_bias" in weights and weights["down_bias"] is not None:
        out = out + weights["down_bias"]

    return out


def compare_outputs(original: mx.array, compressed: mx.array) -> dict:
    """Compare original and compressed outputs."""
    mx.eval(original, compressed)

    diff = original - compressed
    mse = float(mx.mean(diff * diff))
    rmse = float(mx.sqrt(mx.mean(diff * diff)))

    orig_norm = float(mx.sqrt(mx.mean(original * original)))
    rel_error = rmse / (orig_norm + 1e-10)

    # Cosine similarity
    orig_flat = original.flatten()
    comp_flat = compressed.flatten()
    cosine_sim = float(
        mx.sum(orig_flat * comp_flat) /
        (mx.sqrt(mx.sum(orig_flat * orig_flat)) * mx.sqrt(mx.sum(comp_flat * comp_flat)) + 1e-10)
    )

    # Check if top-k predictions match
    orig_topk = mx.argsort(original[0, 0], axis=-1)[-10:]  # Top 10
    comp_topk = mx.argsort(compressed[0, 0], axis=-1)[-10:]
    mx.eval(orig_topk, comp_topk)
    topk_match = len(set(orig_topk.tolist()) & set(comp_topk.tolist()))

    return {
        "rel_error": rel_error,
        "rmse": rmse,
        "cosine_sim": cosine_sim,
        "top10_match": topk_match,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compare expert inference quality")
    parser.add_argument("--model", "-m", default="openai/gpt-oss-20b")
    parser.add_argument("--layer", "-l", type=int, default=0)
    parser.add_argument("--expert", "-e", type=int, default=0)
    parser.add_argument("--batch-size", "-b", type=int, default=1)
    parser.add_argument("--seq-len", "-s", type=int, default=128)

    args = parser.parse_args()

    print("=" * 70)
    print("EXPERT INFERENCE COMPARISON")
    print("=" * 70)

    # Load model and extract weights
    print(f"\nLoading model: {args.model}")
    model = MoETypeService._load_model(args.model)

    moe_layers = get_moe_layers(model)
    layer_idx = args.layer if args.layer < len(moe_layers) else moe_layers[0]
    experts = MoETypeService._get_experts(model, layer_idx)
    arch = detect_moe_architecture(model)
    gate_w, up_w, down_w, num_experts = MoETypeService._extract_weights(experts, arch)
    gate_bias, up_bias, down_bias = MoETypeService._extract_biases(experts)

    weights = {
        "gate": gate_w[args.expert],
        "up": up_w[args.expert],
        "down": down_w[args.expert],
        "gate_all": gate_w,
        "up_all": up_w,
        "down_all": down_w,
    }
    if gate_bias is not None:
        weights["gate_bias"] = gate_bias[args.expert]
    if up_bias is not None:
        weights["up_bias"] = up_bias[args.expert]
    if down_bias is not None:
        weights["down_bias"] = down_bias[args.expert]

    hidden_dim = gate_w.shape[2]
    print(f"Expert {args.expert}, layer {layer_idx}")
    print(f"Hidden dim: {hidden_dim}")

    # Create test input
    test_input = mx.random.normal((args.batch_size, args.seq_len, hidden_dim))
    mx.eval(test_input)
    print(f"Test input: {test_input.shape}")

    # Run original expert
    print("\nRunning original expert...")
    original_output = original_expert_forward(test_input, weights)
    mx.eval(original_output)
    print(f"Original output: {original_output.shape}")

    # Test various compression levels
    print("\n" + "=" * 70)
    print("COMPRESSION LEVEL COMPARISON")
    print("=" * 70)
    print(f"{'Ranks':^20} | {'Rel Error':^10} | {'Cosine Sim':^10} | {'Compression':^12}")
    print("-" * 70)

    # Different rank configurations
    rank_configs = [
        {"gate": 2, "up": 64, "down": 32},      # Very aggressive
        {"gate": 8, "up": 128, "down": 64},     # Aggressive
        {"gate": 32, "up": 256, "down": 128},   # Moderate
        {"gate": 64, "up": 512, "down": 256},   # Conservative
        {"gate": 128, "up": 1024, "down": 512}, # Very conservative
        {"gate": 256, "up": 1500, "down": 700}, # Near lossless
    ]

    for ranks in rank_configs:
        # Calculate compression ratio
        out_dim = gate_w.shape[1]
        in_dim = gate_w.shape[2]
        original_params = 3 * out_dim * in_dim
        compressed_params = (
            ranks["gate"] * (out_dim + in_dim) +
            ranks["up"] * (out_dim + in_dim) +
            ranks["down"] * (out_dim + in_dim)
        )
        comp_ratio = original_params / compressed_params

        try:
            # Create compressed expert
            compressed_fn = create_compressed_expert_fn(weights, ranks)
            compressed_output = compressed_fn(test_input)
            mx.eval(compressed_output)

            # Compare
            metrics = compare_outputs(original_output, compressed_output)

            rank_str = f"g={ranks['gate']},u={ranks['up']},d={ranks['down']}"
            print(f"{rank_str:^20} | {metrics['rel_error']:^10.4f} | {metrics['cosine_sim']:^10.6f} | {comp_ratio:^12.1f}x")

        except Exception as e:
            print(f"Error with ranks {ranks}: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
For acceptable inference quality:
- Relative error < 0.10 (10%) typically needed
- Cosine similarity > 0.99 ideal

Key findings:
- Lower compression = better quality (expected tradeoff)
- Check if 'acceptable' quality achieves useful compression
""")


if __name__ == "__main__":
    main()
