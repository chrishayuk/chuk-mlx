#!/usr/bin/env python3
"""Test the lite model with mlx_lm generate function."""

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


def load_gpt_oss_lite():
    """Load GPT-OSS-Lite."""
    from mlx_lm.models.gpt_oss import Model, ModelArgs
    from mlx_lm.models.switch_layers import SwitchGLU, QuantizedSwitchLinear

    with open(MODEL_DIR / "gpt-oss-lite-16exp" / "config.json") as f:
        config = json.load(f)

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

    model = Model(args)

    for layer_idx in range(24):
        num_experts = config["num_local_experts"]
        layer = model.model.layers[layer_idx]

        layer.mlp.router = nn.Linear(args.hidden_size, num_experts, bias=True)

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


def test_with_generate():
    print("=" * 70)
    print("Testing lite model with mlx_lm generate")
    print("=" * 70)

    model, args = load_gpt_oss_lite()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR / "gpt-oss-lite-16exp")

    from mlx_lm import generate

    prompts = [
        "2 + 2 =",
        "The capital of France is",
        "def fibonacci(n):",
    ]

    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        result = generate(model, tokenizer, prompt=prompt, max_tokens=20, verbose=False)
        print(f"Generated: '{result}'")


def test_direct_call():
    print("\n" + "=" * 70)
    print("Testing lite model with direct call")
    print("=" * 70)

    model, args = load_gpt_oss_lite()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR / "gpt-oss-lite-16exp")

    prompt = "2 + 2 ="
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    print(f"Prompt: '{prompt}'")
    print(f"Tokens: {tokens}")

    # Direct model call (no cache)
    logits = model(input_ids, cache=None)
    mx.eval(logits)

    next_token = mx.argmax(logits[:, -1, :], axis=-1).item()
    print(f"Direct call prediction: {next_token} = '{tokenizer.decode([next_token])}'")

    # With cache
    from mlx_lm.models.cache import make_prompt_cache

    prompt_cache = make_prompt_cache(model)
    logits_with_cache = model(input_ids, cache=prompt_cache)
    mx.eval(logits_with_cache)

    next_token_cache = mx.argmax(logits_with_cache[:, -1, :], axis=-1).item()
    print(f"With cache prediction: {next_token_cache} = '{tokenizer.decode([next_token_cache])}'")


if __name__ == "__main__":
    test_with_generate()
    test_direct_call()
