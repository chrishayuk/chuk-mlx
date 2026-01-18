#!/usr/bin/env python3
"""Compare MoE implementations between lazarus and mlx_lm."""

import sys
sys.path.insert(0, '/Users/christopherhay/chris-source/chuk-mlx/src')

import json
import numpy as np
import mlx.core as mx
from pathlib import Path

MODEL_DIR = Path("/Users/christopherhay/chris-source/chuk-mlx/gpt-oss-lite-v2")


def bf16_to_array(arr: np.ndarray) -> mx.array:
    if arr.dtype == np.dtype('V2'):
        arr_uint16 = arr.view(np.uint16)
        return mx.array(arr_uint16).view(mx.bfloat16)
    return mx.array(arr)


def test_expert_projection():
    """Test a single expert projection."""
    print("=" * 70)
    print("Testing single expert projection")
    print("=" * 70)

    # Load one set of expert weights
    npz = np.load(MODEL_DIR / "weights.npz", allow_pickle=False)

    gate_weight = bf16_to_array(npz["model.layers.0.mlp.experts.gate_proj.weight"])
    gate_scales = bf16_to_array(npz["model.layers.0.mlp.experts.gate_proj.scales"])
    gate_bias = bf16_to_array(npz["model.layers.0.mlp.experts.gate_proj.bias"])

    print(f"gate_weight: {gate_weight.shape}, dtype={gate_weight.dtype}")
    print(f"gate_scales: {gate_scales.shape}, dtype={gate_scales.dtype}")
    print(f"gate_bias: {gate_bias.shape}, dtype={gate_bias.dtype}")

    # Test input
    x = mx.random.normal((1, 2880))  # One token, hidden_size
    expert_idx = 0

    # My implementation
    print("\n--- My implementation ---")
    out_mine = mx.quantized_matmul(
        x,
        gate_weight[expert_idx],
        scales=gate_scales[expert_idx],
        biases=None,
        transpose=True,
        group_size=32,
        bits=4,
        mode="mxfp4",
    )
    out_mine = out_mine + gate_bias[expert_idx]
    mx.eval(out_mine)
    print(f"Output shape: {out_mine.shape}")
    print(f"Output mean: {mx.mean(out_mine).item():.6f}")
    print(f"Output std: {mx.std(out_mine).item():.6f}")

    # mlx_lm's gather_qmm approach
    print("\n--- mlx_lm gather_qmm approach ---")
    # Expand dims as SwitchGLU does
    x_expanded = mx.expand_dims(x, (-2, -3))  # (1, 1, 1, 2880)

    # Single expert index
    indices = mx.array([[expert_idx]])  # (1, 1)

    out_mlx = mx.gather_qmm(
        x_expanded,
        gate_weight,
        gate_scales.astype(mx.uint8),
        None,  # No quantization biases for mxfp4
        rhs_indices=indices,
        transpose=True,
        group_size=32,
        bits=4,
        mode="mxfp4",
    )
    # Add output bias
    out_mlx = out_mlx + mx.expand_dims(gate_bias[indices], -2)
    out_mlx = out_mlx.squeeze()
    mx.eval(out_mlx)
    print(f"Output shape: {out_mlx.shape}")
    print(f"Output mean: {mx.mean(out_mlx).item():.6f}")
    print(f"Output std: {mx.std(out_mlx).item():.6f}")

    # Compare
    print("\n--- Comparison ---")
    if out_mine.shape == out_mlx.shape:
        diff = mx.abs(out_mine - out_mlx)
        print(f"Max difference: {mx.max(diff).item():.8f}")
        print(f"Are equal (atol=1e-3): {mx.allclose(out_mine, out_mlx, atol=1e-3)}")
    else:
        print("Shapes don't match!")


def test_full_moe_layer():
    """Test a full MoE layer computation."""
    print("\n" + "=" * 70)
    print("Testing full MoE layer")
    print("=" * 70)

    from functools import partial

    @partial(mx.compile, shapeless=True)
    def gpt_oss_swiglu(x_linear: mx.array, x_glu: mx.array, alpha: float = 1.702, limit: float = 7.0):
        x_glu = mx.clip(x_glu, a_min=None, a_max=limit)
        x_linear = mx.clip(x_linear, a_min=-limit, a_max=limit)
        glu_scaled = alpha * x_glu
        sig = mx.sigmoid(glu_scaled)
        out_glu = x_glu * sig
        return out_glu * (x_linear + 1)

    # Load weights
    npz = np.load(MODEL_DIR / "weights.npz", allow_pickle=False)

    layer_idx = 0
    prefix = f"model.layers.{layer_idx}.mlp"

    # Load all expert weights
    weights = {}
    for proj in ['gate_proj', 'up_proj', 'down_proj']:
        weights[f'{proj}.weight'] = bf16_to_array(npz[f'{prefix}.experts.{proj}.weight'])
        weights[f'{proj}.scales'] = npz[f'{prefix}.experts.{proj}.scales']
        weights[f'{proj}.bias'] = bf16_to_array(npz[f'{prefix}.experts.{proj}.bias'])

    router_weight = bf16_to_array(npz[f'{prefix}.router.weight'])
    router_bias = bf16_to_array(npz[f'{prefix}.router.bias'])

    # Test input
    mx.random.seed(42)
    x = mx.random.normal((1, 5, 2880))  # batch=1, seq=5, hidden=2880

    print(f"Input shape: {x.shape}")
    print(f"Router weight shape: {router_weight.shape}")
    print(f"Expert gate_proj weight shape: {weights['gate_proj.weight'].shape}")

    # Compute router logits
    x_flat = x.reshape(-1, 2880)  # (5, 2880)
    logits = x_flat @ router_weight.T + router_bias  # (5, 16)
    mx.eval(logits)
    print(f"Router logits shape: {logits.shape}")

    # Top-4 selection
    k = 4
    partitioned = mx.argpartition(logits, kth=-k, axis=-1)
    indices = partitioned[:, -k:]  # (5, 4)
    topk_logits = mx.take_along_axis(logits, indices, axis=-1)
    weights_routing = mx.softmax(topk_logits, axis=-1)  # (5, 4)
    mx.eval(indices, weights_routing)
    print(f"Selected expert indices:\n{indices}")
    print(f"Routing weights:\n{weights_routing}")

    # My naive loop implementation
    print("\n--- My loop implementation ---")
    output_mine = mx.zeros((5, 2880))

    for tok_idx in range(5):
        token_x = x_flat[tok_idx:tok_idx+1]  # (1, 2880)

        for k_idx in range(4):
            exp_idx = int(indices[tok_idx, k_idx])
            w = weights_routing[tok_idx, k_idx]

            # Gate projection
            gate_out = mx.quantized_matmul(
                token_x,
                weights['gate_proj.weight'][exp_idx],
                scales=mx.array(weights['gate_proj.scales'][exp_idx], dtype=mx.uint8),
                biases=None,
                transpose=True,
                group_size=32,
                bits=4,
                mode="mxfp4",
            ) + weights['gate_proj.bias'][exp_idx]

            # Up projection
            up_out = mx.quantized_matmul(
                token_x,
                weights['up_proj.weight'][exp_idx],
                scales=mx.array(weights['up_proj.scales'][exp_idx], dtype=mx.uint8),
                biases=None,
                transpose=True,
                group_size=32,
                bits=4,
                mode="mxfp4",
            ) + weights['up_proj.bias'][exp_idx]

            # Activation
            hidden = gpt_oss_swiglu(up_out, gate_out)

            # Down projection
            expert_out = mx.quantized_matmul(
                hidden,
                weights['down_proj.weight'][exp_idx],
                scales=mx.array(weights['down_proj.scales'][exp_idx], dtype=mx.uint8),
                biases=None,
                transpose=True,
                group_size=32,
                bits=4,
                mode="mxfp4",
            ) + weights['down_proj.bias'][exp_idx]

            output_mine = output_mine.at[tok_idx].add(w * expert_out[0])

    mx.eval(output_mine)
    print(f"Output shape: {output_mine.shape}")
    print(f"Output mean: {mx.mean(output_mine).item():.6f}")
    print(f"Output std: {mx.std(output_mine).item():.6f}")

    # mlx_lm SwitchGLU approach
    print("\n--- mlx_lm SwitchGLU approach ---")
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchGLU

    # Create a custom activation for GPT-OSS
    class GptOssSwiGLU:
        def __call__(self, x, gate):
            return gpt_oss_swiglu(x, gate)

    # Create SwitchGLU manually
    experts_mlx = SwitchGLU(
        input_dims=2880,
        hidden_dims=2880,
        num_experts=16,
        activation=GptOssSwiGLU(),
        bias=False,  # Will set manually
    )

    # Replace with quantized projections
    experts_mlx.gate_proj = QuantizedSwitchLinear(2880, 2880, 16, bias=True, group_size=32, bits=4, mode="mxfp4")
    experts_mlx.up_proj = QuantizedSwitchLinear(2880, 2880, 16, bias=True, group_size=32, bits=4, mode="mxfp4")
    experts_mlx.down_proj = QuantizedSwitchLinear(2880, 2880, 16, bias=True, group_size=32, bits=4, mode="mxfp4")

    # Load weights
    experts_mlx.gate_proj.weight = weights['gate_proj.weight']
    experts_mlx.gate_proj.scales = mx.array(weights['gate_proj.scales'], dtype=mx.uint8)
    experts_mlx.gate_proj.bias = weights['gate_proj.bias']

    experts_mlx.up_proj.weight = weights['up_proj.weight']
    experts_mlx.up_proj.scales = mx.array(weights['up_proj.scales'], dtype=mx.uint8)
    experts_mlx.up_proj.bias = weights['up_proj.bias']

    experts_mlx.down_proj.weight = weights['down_proj.weight']
    experts_mlx.down_proj.scales = mx.array(weights['down_proj.scales'], dtype=mx.uint8)
    experts_mlx.down_proj.bias = weights['down_proj.bias']

    # Run through SwitchGLU
    output_mlx = experts_mlx(x_flat, indices)
    output_mlx = output_mlx * mx.expand_dims(weights_routing, axis=-1)
    output_mlx = output_mlx.sum(axis=-2)

    mx.eval(output_mlx)
    print(f"Output shape: {output_mlx.shape}")
    print(f"Output mean: {mx.mean(output_mlx).item():.6f}")
    print(f"Output std: {mx.std(output_mlx).item():.6f}")

    # Compare
    print("\n--- Comparison ---")
    diff = mx.abs(output_mine - output_mlx)
    print(f"Max difference: {mx.max(diff).item():.6f}")
    print(f"Mean difference: {mx.mean(diff).item():.6f}")


if __name__ == "__main__":
    test_expert_projection()
    test_full_moe_layer()
