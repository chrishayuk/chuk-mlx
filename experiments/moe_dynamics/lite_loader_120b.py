#!/usr/bin/env python3
"""
GPT-OSS-120B-Lite Loader

Memory-optimized loader for compressed GPT-OSS-120B with variable expert counts.

Architecture:
- 36 MoE layers (vs 24 in 20B)
- Variable experts per layer (32 early/late, 48 middle)
- MXFP4 quantization
- k=4 top-k routing

Usage:
    python lite_loader_120b.py --model ./gpt-oss-120b-lite-conservative
    python lite_loader_120b.py --model ./gpt-oss-120b-lite-conservative --prompt "127 * 89 = "
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from functools import partial

import numpy as np
import mlx.core as mx
import mlx.nn as nn


# =============================================================================
# GPT-OSS Custom Activation
# =============================================================================

@partial(mx.compile, shapeless=True)
def gpt_oss_swiglu(x_linear: mx.array, x_glu: mx.array, alpha: float = 1.702, limit: float = 7.0) -> mx.array:
    """GPT-OSS custom SwiGLU activation with asymmetric clamping."""
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


# =============================================================================
# Weight Loading Utilities
# =============================================================================

def bf16_bytes_to_array(arr: np.ndarray) -> mx.array:
    """Convert bfloat16 bytes (V2 dtype) to mlx array."""
    arr_uint16 = arr.view(np.uint16)
    return mx.array(arr_uint16).view(mx.bfloat16)


def load_weight_lazy(npz_file, key: str) -> mx.array:
    """Load a single weight from npz file with dtype handling."""
    arr = npz_file[key]
    if arr.dtype == np.dtype('V2'):
        return bf16_bytes_to_array(arr)
    else:
        return mx.array(arr)


def set_nested_attr(obj, key: str, value):
    """Set a nested attribute from a dotted key string."""
    parts = key.split('.')
    for part in parts[:-1]:
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


# =============================================================================
# Model Loading
# =============================================================================

def load_gpt_oss_120b_lite(model_path: str) -> tuple:
    """
    Load GPT-OSS-120B-Lite with optimized memory usage.

    Args:
        model_path: Path to the lite model directory

    Returns:
        Tuple of (model, tokenizer, config)
    """
    from mlx_lm.models.gpt_oss import Model, ModelArgs
    from mlx_lm.models.switch_layers import SwitchGLU, QuantizedSwitchLinear
    from transformers import AutoTokenizer

    path = Path(model_path)

    # Load config
    with open(path / "config.json") as f:
        config = json.load(f)

    num_layers = config["num_hidden_layers"]
    experts_per_layer = config["experts_per_layer"]
    total_experts = sum(int(v) for v in experts_per_layer.values())

    print("=" * 70)
    print("GPT-OSS-120B-Lite Loader")
    print("=" * 70)
    print()
    print(f"Model path: {path}")
    print(f"Layers: {num_layers}")
    print(f"Total experts: {total_experts} (vs 4,608 original)")
    print(f"Compression: {100 * (1 - total_experts / 4608):.1f}%")
    print()

    # Create model args
    # Use the original 120B specs but we'll patch the experts
    rope_scaling = {
        'beta_fast': 32.0,
        'beta_slow': 1.0,
        'factor': 32.0,
        'original_max_position_embeddings': 4096,
        'rope_type': 'yarn',
        'truncate': False,
    }

    # Start with max experts, we'll patch down
    max_experts = max(int(v) for v in experts_per_layer.values())

    args = ModelArgs(
        hidden_size=config["hidden_size"],
        intermediate_size=config.get("intermediate_size", config["hidden_size"]),
        num_hidden_layers=num_layers,
        num_attention_heads=config["num_attention_heads"],
        num_key_value_heads=config.get("num_key_value_heads", 8),
        vocab_size=config["vocab_size"],
        num_local_experts=max_experts,  # Will be patched per layer
        num_experts_per_tok=config.get("num_experts_per_tok", 4),
        rope_theta=config.get("rope_theta", 150000.0),
        rms_norm_eps=config.get("rms_norm_eps", 1e-5),
        rope_scaling=rope_scaling,
    )

    print("Creating model structure...")
    model = Model(args)
    gc.collect()

    # Patch expert structures for each layer
    print("Patching expert structures per layer...")
    for layer_idx in range(num_layers):
        num_experts = int(experts_per_layer[str(layer_idx)])
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

        if layer_idx % 12 == 0:
            print(f"  Layer {layer_idx}: {num_experts} experts")

    gc.collect()

    # Load weights
    print()
    print("Loading weights...")

    # Try different weight file names
    weight_files = [
        "weights.mlx.npz",
        "weights.npz",
        "model.safetensors",
    ]

    weights_path = None
    for wf in weight_files:
        if (path / wf).exists():
            weights_path = path / wf
            break

    if weights_path is None:
        raise FileNotFoundError(f"No weight file found in {path}")

    print(f"  Weight file: {weights_path.name}")

    if weights_path.suffix == ".npz":
        weights_npz = np.load(weights_path, allow_pickle=False)

        # Group keys by layer
        layer_keys = {i: [] for i in range(num_layers)}
        other_keys = []

        for key in weights_npz.files:
            layer_match = None
            for i in range(num_layers):
                if f'layers.{i}.' in key:
                    layer_match = i
                    break

            if layer_match is not None:
                layer_keys[layer_match].append(key)
            else:
                other_keys.append(key)

        # Load non-layer weights first
        print("  Loading embeddings and head...")
        for key in other_keys:
            try:
                value = load_weight_lazy(weights_npz, key)
                set_nested_attr(model, key, value)
            except (AttributeError, IndexError, TypeError) as e:
                pass

        mx.eval(model.model.embed_tokens.parameters())
        gc.collect()

        # Load layers
        for layer_idx in range(num_layers):
            keys = layer_keys[layer_idx]
            if not keys:
                continue

            if layer_idx % 6 == 0:
                print(f"  Loading layers {layer_idx}-{min(layer_idx+5, num_layers-1)}...")

            for key in keys:
                try:
                    value = load_weight_lazy(weights_npz, key)
                    set_nested_attr(model, key, value)
                except (AttributeError, IndexError, TypeError):
                    pass

            mx.eval(model.model.layers[layer_idx].parameters())
            gc.collect()

        weights_npz.close()

    else:
        # Safetensors loading
        from mlx.utils import load as mlx_load
        weights = mlx_load(str(weights_path))
        model.load_weights(list(weights.items()))

    # Load tokenizer
    print()
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(path))

    # Report memory
    print()
    try:
        active = mx.get_active_memory() / 1024 / 1024 / 1024
        peak = mx.get_peak_memory() / 1024 / 1024 / 1024
        print(f"Memory: active={active:.1f}GB, peak={peak:.1f}GB")
    except:
        pass

    print("Model ready!")
    print()

    return model, tokenizer, config


# =============================================================================
# Inference
# =============================================================================

def generate(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 50,
    temperature: float = 0.0,
) -> str:
    """Generate text from a prompt."""
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    generated = []

    for _ in range(max_tokens):
        logits = model(input_ids)

        if temperature == 0:
            next_token = mx.argmax(logits[:, -1, :], axis=-1)
        else:
            logits = logits[:, -1, :] / temperature
            probs = mx.softmax(logits, axis=-1)
            next_token = mx.random.categorical(probs)

        next_token_id = int(next_token.item())
        generated.append(next_token_id)

        if next_token_id == tokenizer.eos_token_id:
            break

        input_ids = mx.concatenate([input_ids, next_token[None, :]], axis=1)
        mx.eval(input_ids)

    return tokenizer.decode(generated)


def run_inference_tests(model, tokenizer, config):
    """Run inference tests on the lite model."""
    print("=" * 70)
    print("Inference Tests")
    print("=" * 70)
    print()

    test_prompts = [
        ("Math", "127 * 89 = "),
        ("Math", "2 + 2 = "),
        ("Code", "def fibonacci(n):"),
        ("Language", "The capital of France is"),
        ("Reasoning", "If all cats are mammals, then"),
    ]

    for category, prompt in test_prompts:
        print(f"[{category}] Prompt: '{prompt}'")

        start = time.time()
        response = generate(model, tokenizer, prompt, max_tokens=30)
        elapsed = time.time() - start

        print(f"  Response ({elapsed:.2f}s): {response[:80]}...")
        print()

    # Performance test
    print("-" * 70)
    print("Performance Test (30 tokens)")
    print("-" * 70)

    prompt = "The quick brown fox"
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    # Warmup
    _ = model(input_ids)
    mx.eval(_)

    # Timed run
    start = time.time()
    for _ in range(30):
        logits = model(input_ids)
        next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        input_ids = mx.concatenate([input_ids, next_token], axis=1)
        mx.eval(input_ids)

    elapsed = time.time() - start
    tokens_per_sec = 30 / elapsed

    print(f"  Time: {elapsed:.2f}s")
    print(f"  Speed: {tokens_per_sec:.1f} tokens/sec")
    print()


def main():
    parser = argparse.ArgumentParser(description="GPT-OSS-120B-Lite Loader")
    parser.add_argument(
        "--model",
        type=str,
        default="./gpt-oss-120b-lite-conservative",
        help="Path to lite model directory",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt to run (optional)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Max tokens to generate",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run inference tests",
    )

    args = parser.parse_args()

    # Reset memory tracking
    try:
        mx.reset_peak_memory()
    except:
        pass

    # Load model
    model, tokenizer, config = load_gpt_oss_120b_lite(args.model)

    if args.prompt:
        # Single prompt mode
        print("=" * 70)
        print(f"Prompt: {args.prompt}")
        print("=" * 70)
        print()

        start = time.time()
        response = generate(model, tokenizer, args.prompt, max_tokens=args.max_tokens)
        elapsed = time.time() - start

        print(f"Response ({elapsed:.2f}s):")
        print(response)
        print()

    elif args.test:
        # Run inference tests
        run_inference_tests(model, tokenizer, config)

    else:
        # Interactive mode
        print("=" * 70)
        print("Interactive Mode (type 'quit' to exit)")
        print("=" * 70)
        print()

        while True:
            try:
                prompt = input("Prompt: ").strip()
                if prompt.lower() in ['quit', 'exit', 'q']:
                    break
                if not prompt:
                    continue

                response = generate(model, tokenizer, prompt, max_tokens=args.max_tokens)
                print(f"Response: {response}")
                print()

            except KeyboardInterrupt:
                print("\nExiting...")
                break

    # Final memory report
    try:
        active = mx.get_active_memory() / 1024 / 1024 / 1024
        peak = mx.get_peak_memory() / 1024 / 1024 / 1024
        print(f"Final memory: active={active:.1f}GB, peak={peak:.1f}GB")
    except:
        pass


if __name__ == "__main__":
    main()
