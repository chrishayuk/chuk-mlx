#!/usr/bin/env python3
"""
Memory-optimized GPT-OSS-Lite loader.

Optimizations:
1. Stream weights from npz file instead of loading all at once
2. Use gc.collect() between layer loads
3. Create model structure lazily
"""

import gc
import json
import numpy as np
from pathlib import Path
from functools import partial

import mlx.core as mx
import mlx.nn as nn


# =============================================================================
# GPT-OSS Custom SwiGLU Activation (same as lite_loader.py)
# =============================================================================

@partial(mx.compile, shapeless=True)
def gpt_oss_swiglu(x_linear: mx.array, x_glu: mx.array, alpha: float = 1.702, limit: float = 7.0) -> mx.array:
    """GPT-OSS custom SwiGLU activation."""
    x_glu = mx.clip(x_glu, a_min=None, a_max=limit)
    x_linear = mx.clip(x_linear, a_min=-limit, a_max=limit)
    glu_scaled = alpha * x_glu
    sig = mx.sigmoid(glu_scaled)
    out_glu = x_glu * sig
    return out_glu * (x_linear + 1)


class GptOssSwiGLU(nn.Module):
    """GPT-OSS custom SwiGLU activation module."""
    def __init__(self):
        super().__init__()

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        return gpt_oss_swiglu(x, gate)


def bf16_bytes_to_array(arr):
    """Convert bfloat16 bytes (V2 dtype) to mlx array."""
    arr_uint16 = arr.view(np.uint16)
    return mx.array(arr_uint16).view(mx.bfloat16)


def load_weight_lazy(npz_file, key):
    """Load a single weight from npz file."""
    arr = npz_file[key]
    if arr.dtype == np.dtype('V2'):
        return bf16_bytes_to_array(arr)
    else:
        return mx.array(arr)


def load_gpt_oss_lite_optimized(model_path: str = "."):
    """
    Load GPT-OSS-Lite with optimized memory usage.

    Strategy:
    1. Create model structure with minimal memory
    2. Load weights layer by layer
    3. Free memory between layers
    """
    from mlx_lm.models.gpt_oss import Model, ModelArgs
    from mlx_lm.models.switch_layers import SwitchGLU, QuantizedSwitchLinear

    path = Path(model_path)

    # Load config
    with open(path / "config.json") as f:
        config = json.load(f)

    print(f"Loading GPT-OSS-Lite (optimized) from {path}")
    print(f"Expert configuration: {sum(config['experts_per_layer'].values())} experts total")

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
    print("Creating model structure...")
    model = Model(args)

    # Clear any cached memory
    gc.collect()

    # Patch expert structures first (before loading weights)
    print("Patching expert structures...")
    for layer_idx in range(24):
        num_experts = config["experts_per_layer"][str(layer_idx)]
        layer = model.model.layers[layer_idx]

        # Create new router with correct output size
        layer.mlp.router = nn.Linear(args.hidden_size, num_experts, bias=True)

        # Create new SwitchGLU with correct expert count
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

    # Load weights with streaming approach
    print("Loading weights (streaming)...")

    # Open npz file - use allow_pickle=False for safety
    weights_npz = np.load(path / "weights.npz", allow_pickle=False)

    # Group keys by layer for efficient loading
    layer_keys = {i: [] for i in range(24)}
    other_keys = []

    for key in weights_npz.files:
        layer_match = None
        for i in range(24):
            if f'layers.{i}.' in key:
                layer_match = i
                break

        if layer_match is not None:
            layer_keys[layer_match].append(key)
        else:
            other_keys.append(key)

    # Load non-layer weights first (embeddings, norms, lm_head)
    print("  Loading embeddings and head...")
    for key in other_keys:
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

    mx.eval(model.model.embed_tokens.parameters())
    gc.collect()

    # Load layer weights one at a time
    for layer_idx in range(24):
        keys = layer_keys[layer_idx]
        if not keys:
            continue

        print(f"  Loading layer {layer_idx}... ({len(keys)} tensors)")

        for key in keys:
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

        # Evaluate layer parameters to ensure they're loaded
        mx.eval(model.model.layers[layer_idx].parameters())

        # Clean up
        gc.collect()

    # Close npz file
    weights_npz.close()

    print("Model ready")

    # Report memory
    try:
        active = mx.get_active_memory() / 1024 / 1024
        peak = mx.get_peak_memory() / 1024 / 1024
        print(f"Memory: active={active:.0f}MB, peak={peak:.0f}MB")
    except:
        pass

    return model, args


def main():
    """Test the optimized lite loader."""
    from transformers import AutoTokenizer
    import time

    print("=" * 70)
    print("GPT-OSS-LITE Optimized Loader")
    print("=" * 70)
    print()

    # Reset memory tracking
    try:
        mx.reset_peak_memory()
    except:
        pass

    model, args = load_gpt_oss_lite_optimized(".")
    tokenizer = AutoTokenizer.from_pretrained(".")

    prompts = [
        "The capital of France is",
        "def fibonacci(n):",
        "2 + 2 =",
    ]

    print()
    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])

        start = time.time()
        for _ in range(30):
            logits = model(input_ids)
            next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
            input_ids = mx.concatenate([input_ids, next_token], axis=1)
            mx.eval(input_ids)

        elapsed = time.time() - start
        output = tokenizer.decode(input_ids[0].tolist())

        print(f"Prompt: '{prompt}'")
        print(f"Output ({elapsed:.2f}s): {output[:100]}...")
        print()

    # Final memory report
    try:
        active = mx.get_active_memory() / 1024 / 1024
        peak = mx.get_peak_memory() / 1024 / 1024
        print(f"Final memory: active={active:.0f}MB, peak={peak:.0f}MB")
    except:
        pass


if __name__ == "__main__":
    main()
