"""Prove SVD compression works on a single expert.

This test:
1. Extracts one expert's weights (gate, up, down)
2. Compresses using SVD (base + U @ V)
3. Verifies reconstruction error is acceptable (<1%)
4. Runs inference through original vs compressed expert
5. Compares output differences
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from sklearn.utils.extmath import randomized_svd

from chuk_lazarus.introspection.moe.detector import detect_moe_architecture, get_moe_layers
from chuk_lazarus.introspection.moe.moe_type import MoETypeService


def extract_single_expert_weights(model, layer_idx: int, expert_idx: int):
    """Extract weights for a single expert.

    Returns:
        dict with keys: gate, up, down - each is (out_dim, in_dim) weights
        and gate_bias, up_bias, down_bias if available
    """
    experts = MoETypeService._get_experts(model, layer_idx)
    architecture = detect_moe_architecture(model)

    # Get all expert weights
    gate_w, up_w, down_w, num_experts = MoETypeService._extract_weights(experts, architecture)

    # Get biases
    gate_bias, up_bias, down_bias = MoETypeService._extract_biases(experts)

    result = {
        "gate": gate_w[expert_idx],
        "up": up_w[expert_idx],
        "down": down_w[expert_idx],
        "gate_all": gate_w,  # Keep all for computing base
        "up_all": up_w,
        "down_all": down_w,
        "num_experts": num_experts,
    }

    if gate_bias is not None:
        result["gate_bias"] = gate_bias[expert_idx]
    if up_bias is not None:
        result["up_bias"] = up_bias[expert_idx]
    if down_bias is not None:
        result["down_bias"] = down_bias[expert_idx]

    return result


def compress_weight(weight: mx.array, all_weights: mx.array, rank: int):
    """Compress a single expert weight using SVD.

    Args:
        weight: Expert weight (out_dim, in_dim)
        all_weights: All expert weights (num_experts, out_dim, in_dim) for computing base
        rank: SVD truncation rank

    Returns:
        dict with base, U, V for reconstruction
    """
    # Compute base (mean of all experts)
    base = mx.mean(all_weights, axis=0)
    mx.eval(base)

    # Compute delta
    delta = weight - base
    mx.eval(delta)
    delta_np = np.array(delta.astype(mx.float32))

    # SVD truncation using randomized SVD (faster)
    U, S, Vh = randomized_svd(delta_np, n_components=rank, n_iter=2, random_state=42)

    # Absorb singular values into U
    U_scaled = U @ np.diag(S)

    return {
        "base": base,
        "U": mx.array(U_scaled),  # (out_dim, rank)
        "V": mx.array(Vh),         # (rank, in_dim)
        "rank": rank,
    }


def reconstruct_weight(compressed: dict) -> mx.array:
    """Reconstruct weight from compressed form."""
    base = compressed["base"]
    U = compressed["U"]
    V = compressed["V"]

    # expert = base + U @ V
    return base + U @ V


def compute_reconstruction_error(original: mx.array, reconstructed: mx.array) -> dict:
    """Compute reconstruction error metrics."""
    mx.eval(original, reconstructed)

    diff = original - reconstructed
    mse = float(mx.mean(diff * diff))
    rmse = float(mx.sqrt(mx.mean(diff * diff)))

    # Relative error
    orig_norm = float(mx.sqrt(mx.mean(original * original)))
    rel_error = rmse / (orig_norm + 1e-10)

    # Max absolute error
    max_abs_error = float(mx.max(mx.abs(diff)))

    return {
        "mse": mse,
        "rmse": rmse,
        "relative_error": rel_error,
        "max_abs_error": max_abs_error,
    }


def run_expert_forward(x: mx.array, gate_w: mx.array, up_w: mx.array, down_w: mx.array,
                       gate_bias=None, up_bias=None, down_bias=None) -> mx.array:
    """Run forward pass through an expert (gate/up/down projections).

    Uses SiLU activation: output = down(silu(gate(x)) * up(x))
    """
    # Gate projection
    gate_out = x @ gate_w.T
    if gate_bias is not None:
        gate_out = gate_out + gate_bias

    # Up projection
    up_out = x @ up_w.T
    if up_bias is not None:
        up_out = up_out + up_bias

    # SiLU activation and combine
    hidden = nn.silu(gate_out) * up_out

    # Down projection
    out = hidden @ down_w.T
    if down_bias is not None:
        out = out + down_bias

    return out


def run_compressed_forward(x: mx.array,
                           gate_comp: dict, up_comp: dict, down_comp: dict,
                           gate_bias=None, up_bias=None, down_bias=None) -> mx.array:
    """Run forward pass using compressed expert weights.

    Efficient computation using low-rank factors:
    y = x @ (base + U @ V).T + bias
      = x @ base.T + (x @ V.T) @ U.T + bias
    """
    # Gate: efficient low-rank
    gate_base, gate_U, gate_V = gate_comp["base"], gate_comp["U"], gate_comp["V"]
    gate_out = x @ gate_base.T + (x @ gate_V.T) @ gate_U.T
    if gate_bias is not None:
        gate_out = gate_out + gate_bias

    # Up: efficient low-rank
    up_base, up_U, up_V = up_comp["base"], up_comp["U"], up_comp["V"]
    up_out = x @ up_base.T + (x @ up_V.T) @ up_U.T
    if up_bias is not None:
        up_out = up_out + up_bias

    # SiLU and combine
    hidden = nn.silu(gate_out) * up_out

    # Down: efficient low-rank
    down_base, down_U, down_V = down_comp["base"], down_comp["U"], down_comp["V"]
    out = hidden @ down_base.T + (hidden @ down_V.T) @ down_U.T
    if down_bias is not None:
        out = out + down_bias

    return out


def auto_select_rank(all_weights: mx.array, variance_threshold: float = 0.95) -> int:
    """Auto-select rank based on SVD analysis."""
    base = mx.mean(all_weights, axis=0)
    delta = all_weights[0] - base
    mx.eval(delta)
    delta_np = np.array(delta.astype(mx.float32))

    _, S, _ = np.linalg.svd(delta_np, full_matrices=False)
    total = np.sum(S**2)
    if total == 0:
        return 1
    cumsum = np.cumsum(S**2) / total
    return max(1, int(np.searchsorted(cumsum, variance_threshold) + 1))


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Single expert compression test")
    parser.add_argument("--model", "-m", default="openai/gpt-oss-20b",
                        help="Model ID to test")
    parser.add_argument("--layer", "-l", type=int, default=None,
                        help="Layer index (default: first MoE layer)")
    parser.add_argument("--expert", "-e", type=int, default=0,
                        help="Expert index to test (default: 0)")
    parser.add_argument("--gate-rank", type=int, default=None,
                        help="Rank for gate projection (default: auto)")
    parser.add_argument("--up-rank", type=int, default=None,
                        help="Rank for up projection (default: auto)")
    parser.add_argument("--down-rank", type=int, default=None,
                        help="Rank for down projection (default: auto)")
    parser.add_argument("--batch-size", "-b", type=int, default=1,
                        help="Batch size for inference test")
    parser.add_argument("--seq-len", "-s", type=int, default=10,
                        help="Sequence length for inference test")

    args = parser.parse_args()

    print("=" * 70)
    print("SINGLE EXPERT COMPRESSION TEST")
    print("=" * 70)

    # Load model
    print(f"\n1. Loading model: {args.model}")
    model = MoETypeService._load_model(args.model)

    # Get MoE layer info
    moe_layers = get_moe_layers(model)
    if not moe_layers:
        print("ERROR: No MoE layers found!")
        return

    layer_idx = args.layer if args.layer is not None else moe_layers[0]
    print(f"   MoE layers: {moe_layers}")
    print(f"   Testing layer: {layer_idx}, expert: {args.expert}")

    # Extract weights
    print(f"\n2. Extracting expert {args.expert} weights...")
    weights = extract_single_expert_weights(model, layer_idx, args.expert)

    print(f"   Gate shape: {weights['gate'].shape}")
    print(f"   Up shape:   {weights['up'].shape}")
    print(f"   Down shape: {weights['down'].shape}")
    print(f"   Num experts: {weights['num_experts']}")

    # Auto-select ranks if not provided
    print("\n3. Determining compression ranks...")
    gate_rank = args.gate_rank or auto_select_rank(weights["gate_all"])
    up_rank = args.up_rank or auto_select_rank(weights["up_all"])
    down_rank = args.down_rank or auto_select_rank(weights["down_all"])

    print(f"   Gate rank: {gate_rank} (max: {min(weights['gate'].shape)})")
    print(f"   Up rank:   {up_rank} (max: {min(weights['up'].shape)})")
    print(f"   Down rank: {down_rank} (max: {min(weights['down'].shape)})")

    # Compress each projection
    print("\n4. Compressing weights using SVD...")
    gate_comp = compress_weight(weights["gate"], weights["gate_all"], gate_rank)
    up_comp = compress_weight(weights["up"], weights["up_all"], up_rank)
    down_comp = compress_weight(weights["down"], weights["down_all"], down_rank)

    # Calculate compression ratio
    out_dim, in_dim = weights["gate"].shape
    original_params = 3 * out_dim * in_dim  # gate + up + down
    compressed_params = 3 * out_dim * in_dim  # bases
    compressed_params += gate_rank * (out_dim + in_dim)  # gate U, V
    compressed_params += up_rank * (out_dim + in_dim)    # up U, V

    out_dim_d, in_dim_d = weights["down"].shape
    compressed_params += down_rank * (out_dim_d + in_dim_d)  # down U, V

    # Per-expert compression (excluding shared base)
    per_expert_original = 3 * out_dim * in_dim
    per_expert_compressed = (
        gate_rank * (out_dim + in_dim) +
        up_rank * (out_dim + in_dim) +
        down_rank * (out_dim_d + in_dim_d)
    )

    print(f"   Original params (per expert): {per_expert_original:,}")
    print(f"   Compressed delta params:      {per_expert_compressed:,}")
    print(f"   Compression ratio:            {per_expert_original / per_expert_compressed:.1f}x")

    # Verify reconstruction
    print("\n5. Verifying weight reconstruction...")

    gate_recon = reconstruct_weight(gate_comp)
    up_recon = reconstruct_weight(up_comp)
    down_recon = reconstruct_weight(down_comp)

    gate_err = compute_reconstruction_error(weights["gate"], gate_recon)
    up_err = compute_reconstruction_error(weights["up"], up_recon)
    down_err = compute_reconstruction_error(weights["down"], down_recon)

    print(f"   Gate:  rel_error={gate_err['relative_error']:.6f}, RMSE={gate_err['rmse']:.6f}")
    print(f"   Up:    rel_error={up_err['relative_error']:.6f}, RMSE={up_err['rmse']:.6f}")
    print(f"   Down:  rel_error={down_err['relative_error']:.6f}, RMSE={down_err['rmse']:.6f}")

    # Run inference comparison
    print("\n6. Comparing inference output...")

    # Create test input
    hidden_dim = weights["gate"].shape[1]  # in_dim
    test_input = mx.random.normal((args.batch_size, args.seq_len, hidden_dim))
    mx.eval(test_input)

    # Get biases
    gate_bias = weights.get("gate_bias")
    up_bias = weights.get("up_bias")
    down_bias = weights.get("down_bias")

    print(f"   Input shape: {test_input.shape}")
    print(f"   Hidden dim: {hidden_dim}")
    print(f"   Has biases: gate={gate_bias is not None}, up={up_bias is not None}, down={down_bias is not None}")

    # Original forward pass
    original_output = run_expert_forward(
        test_input, weights["gate"], weights["up"], weights["down"],
        gate_bias, up_bias, down_bias
    )
    mx.eval(original_output)

    # Compressed forward pass
    compressed_output = run_compressed_forward(
        test_input, gate_comp, up_comp, down_comp,
        gate_bias, up_bias, down_bias
    )
    mx.eval(compressed_output)

    # Compare outputs
    output_err = compute_reconstruction_error(original_output, compressed_output)

    print("\n   OUTPUT COMPARISON:")
    print(f"   Output shape: {original_output.shape}")
    print(f"   Relative error: {output_err['relative_error']:.8f}")
    print(f"   RMSE:           {output_err['rmse']:.8f}")
    print(f"   Max abs error:  {output_err['max_abs_error']:.8f}")

    # Compute per-token error
    orig_flat = original_output.reshape(-1)
    comp_flat = compressed_output.reshape(-1)
    per_elem_diff = mx.abs(orig_flat - comp_flat)
    orig_abs = mx.abs(orig_flat)
    mx.eval(per_elem_diff, orig_abs)

    print("\n   Per-element stats:")
    print(f"   Mean diff:      {float(mx.mean(per_elem_diff)):.8f}")
    print(f"   Mean orig abs:  {float(mx.mean(orig_abs)):.8f}")
    print(f"   Max diff:       {float(mx.max(per_elem_diff)):.8f}")

    # Final verdict
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    max_weight_err = max(gate_err["relative_error"], up_err["relative_error"], down_err["relative_error"])

    weight_pass = max_weight_err < 0.01  # 1% threshold
    output_pass = output_err["relative_error"] < 0.01  # 1% threshold

    print(f"\nWeight reconstruction: {'PASS' if weight_pass else 'FAIL'} (max rel error: {max_weight_err:.6f})")
    print(f"Output comparison:     {'PASS' if output_pass else 'FAIL'} (rel error: {output_err['relative_error']:.8f})")
    print(f"Compression ratio:     {per_expert_original / per_expert_compressed:.1f}x")

    if weight_pass and output_pass:
        print("\n SUCCESS: Compression works on single expert!")
        print(f"          {per_expert_original / per_expert_compressed:.1f}x compression with <1% error")
    else:
        print("\n NEEDS INVESTIGATION: Error exceeds 1% threshold")
        print("   Consider increasing ranks or investigating weight distribution")

    return {
        "weight_pass": weight_pass,
        "output_pass": output_pass,
        "compression_ratio": per_expert_original / per_expert_compressed,
        "max_weight_error": max_weight_err,
        "output_error": output_err["relative_error"],
        "ranks": {"gate": gate_rank, "up": up_rank, "down": down_rank},
    }


if __name__ == "__main__":
    main()
