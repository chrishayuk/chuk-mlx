#!/usr/bin/env python3
"""Trace through the working mlx_lm model layer by layer."""

import gc
import json
import numpy as np
from pathlib import Path
from functools import partial

import mlx.core as mx
import mlx.nn as nn
from transformers import AutoTokenizer


MODEL_DIR = Path("/Users/christopherhay/chris-source/chuk-mlx/gpt-oss-lite-v2")


@partial(mx.compile, shapeless=True)
def gpt_oss_swiglu(x_linear: mx.array, x_glu: mx.array, alpha: float = 1.702, limit: float = 7.0) -> mx.array:
    x_glu = mx.clip(x_glu, a_min=None, a_max=limit)
    x_linear = mx.clip(x_linear, a_min=-limit, a_max=limit)
    glu_scaled = alpha * x_glu
    sig = mx.sigmoid(glu_scaled)
    out_glu = x_glu * sig
    return out_glu * (x_linear + 1)


class GptOssSwiGLU(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        return gpt_oss_swiglu(x, gate)


def bf16_bytes_to_array(arr):
    arr_uint16 = arr.view(np.uint16)
    return mx.array(arr_uint16).view(mx.bfloat16)


def load_weight_lazy(npz_file, key):
    arr = npz_file[key]
    if arr.dtype == np.dtype('V2'):
        return bf16_bytes_to_array(arr)
    else:
        return mx.array(arr)


def load_gpt_oss_lite_optimized():
    """Load GPT-OSS-Lite with optimized memory usage."""
    from mlx_lm.models.gpt_oss import Model, ModelArgs
    from mlx_lm.models.switch_layers import SwitchGLU, QuantizedSwitchLinear

    # Load config (from 16-expert version)
    with open(MODEL_DIR / "gpt-oss-lite-16exp" / "config.json") as f:
        config = json.load(f)

    print(f"Loading GPT-OSS-Lite (optimized)")
    print(f"Experts per layer: {config['num_local_experts']}")

    # Create model args
    rope_scaling = {
        'beta_fast': 32.0,
        'beta_slow': 1.0,
        'factor': 32.0,
        'original_max_position_embeddings': 4096,
        'rope_type': 'yarn',
        'truncate': False,
    }

    args = ModelArgs(
        hidden_size=config["hidden_size"],
        intermediate_size=config.get("intermediate_size", config["hidden_size"]),
        num_hidden_layers=24,
        num_attention_heads=config["num_attention_heads"],
        num_key_value_heads=config.get("num_key_value_heads", 8),
        vocab_size=config["vocab_size"],
        num_local_experts=16,
        num_experts_per_tok=config.get("num_experts_per_tok", 4),
        rope_theta=config.get("rope_theta", 150000.0),
        rms_norm_eps=config.get("rms_norm_eps", 1e-5),
        rope_scaling=rope_scaling,
    )

    # Create model structure
    model = Model(args)

    # Patch expert structures
    for layer_idx in range(24):
        num_experts = config["num_local_experts"]
        layer = model.model.layers[layer_idx]

        # Create new router
        layer.mlp.router = nn.Linear(args.hidden_size, num_experts, bias=True)

        # Create new SwitchGLU
        new_experts = SwitchGLU(
            args.hidden_size,
            args.intermediate_size,
            num_experts,
            activation=GptOssSwiGLU(),
            bias=False,
        )
        new_experts.gate_proj = QuantizedSwitchLinear(
            args.hidden_size, args.intermediate_size, num_experts,
            bias=True, group_size=32, bits=4, mode="mxfp4"
        )
        new_experts.up_proj = QuantizedSwitchLinear(
            args.hidden_size, args.intermediate_size, num_experts,
            bias=True, group_size=32, bits=4, mode="mxfp4"
        )
        new_experts.down_proj = QuantizedSwitchLinear(
            args.intermediate_size, args.hidden_size, num_experts,
            bias=True, group_size=32, bits=4, mode="mxfp4"
        )
        layer.mlp.experts = new_experts
        layer.mlp.num_local_experts = num_experts

    gc.collect()

    # Load weights
    weights_npz = np.load(MODEL_DIR / "weights.npz", allow_pickle=False)

    for key in weights_npz.files:
        value = load_weight_lazy(weights_npz, key)
        parts = key.split('.')
        obj = model
        try:
            for part in parts[:-1]:
                if part.isdigit():
                    obj = obj[int(part)]
                else:
                    obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
        except (AttributeError, IndexError, TypeError):
            pass

    weights_npz.close()
    mx.eval(model.parameters())

    return model, args


def trace_layer_by_layer():
    """Trace through model layer by layer."""
    print("=" * 70)
    print("Layer by layer trace - Working mlx_lm model")
    print("=" * 70)

    model, args = load_gpt_oss_lite_optimized()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR / "gpt-oss-lite-16exp")

    prompt = "2 + 2 ="
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    print(f"\nPrompt: '{prompt}'")
    print(f"Tokens: {tokens}")

    # Get embeddings
    x = model.model.embed_tokens(input_ids)
    mx.eval(x)
    print(f"\nEmbeddings - shape: {x.shape}, std: {mx.std(x).item():.4f}")

    # Check layer types
    layer_types = model.model.layer_types
    print(f"Layer types: {layer_types[:4]}... (alternating)")

    # Create masks using mlx_lm's approach
    from mlx_lm.models.base import create_attention_mask

    # Trace through layers
    for i, (layer, layer_type) in enumerate(zip(model.model.layers, layer_types)):
        if layer_type == "full_attention":
            mask = create_attention_mask(x, None)
        else:
            mask = create_attention_mask(x, None, window_size=model.model.window_size)

        x = layer(x, mask, None)
        mx.eval(x)

        std = mx.std(x).item()
        mean = mx.mean(x).item()
        has_nan = mx.any(mx.isnan(x)).item()
        print(f"Layer {i:2d} ({layer_type[:4]}): std={std:.4f}, mean={mean:.4f}, nan={has_nan}")

        if has_nan:
            print("  NaN detected! Stopping.")
            break

    # Final norm
    if not mx.any(mx.isnan(x)).item():
        x = model.model.norm(x)
        mx.eval(x)
        print(f"Final norm: std={mx.std(x).item():.4f}")

    # Test full forward pass
    print("\n--- Full forward pass ---")
    logits = model(input_ids)
    mx.eval(logits)

    next_token = mx.argmax(logits[:, -1, :], axis=-1).item()
    print(f"Predicted next token: {next_token} = '{tokenizer.decode([next_token])}'")


if __name__ == "__main__":
    trace_layer_by_layer()
