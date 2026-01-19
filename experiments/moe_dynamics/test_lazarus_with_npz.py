#!/usr/bin/env python3
"""Test the lazarus gpt_oss_lite model with weights from npz file (includes biases/sinks)."""

import sys
sys.path.insert(0, '/Users/christopherhay/chris-source/chuk-mlx/src')

import json
import numpy as np
import mlx.core as mx
from pathlib import Path
from transformers import AutoTokenizer

MODEL_DIR = Path("/Users/christopherhay/chris-source/chuk-mlx/gpt-oss-lite-v2")
NPZ_PATH = MODEL_DIR / "weights.npz"
CONFIG_PATH = MODEL_DIR / "gpt-oss-lite-16exp" / "config.json"


def bf16_to_array(arr: np.ndarray) -> mx.array:
    """Convert bfloat16 bytes (V2 dtype) to mlx array."""
    if arr.dtype == np.dtype('V2'):
        arr_uint16 = arr.view(np.uint16)
        return mx.array(arr_uint16).view(mx.bfloat16)
    return mx.array(arr)


def load_lazarus_model_with_npz():
    """Load the model using lazarus with weights from npz."""
    from chuk_lazarus.models_v2.families.gpt_oss_lite.config import GptOssLiteConfig
    from chuk_lazarus.models_v2.families.gpt_oss_lite.model import GptOssLiteForCausalLM

    # Load config
    with open(CONFIG_PATH) as f:
        hf_config = json.load(f)

    config = GptOssLiteConfig.from_hf_config(hf_config)
    print(f"Config: hidden_size={config.hidden_size}, num_layers={config.num_hidden_layers}")
    print(f"Config: num_experts={config.num_local_experts}, num_experts_per_tok={config.num_experts_per_tok}")

    # Create model
    model = GptOssLiteForCausalLM(config)

    # Load weights from npz file
    print(f"\nLoading weights from {NPZ_PATH}...")
    npz = np.load(NPZ_PATH, allow_pickle=False)

    loaded = 0
    skipped = 0
    failed = []

    for key in npz.files:
        try:
            value = bf16_to_array(npz[key])

            parts = key.split(".")
            obj = model
            for part in parts[:-1]:
                if part.isdigit():
                    obj = obj[int(part)]
                else:
                    obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
            loaded += 1
        except (AttributeError, IndexError, TypeError) as e:
            # Check if this is an expected skip (experts with more indices than we have)
            if 'mlp.experts' in key:
                # Get layer index and expert index from key
                # e.g., model.layers.0.mlp.experts.gate_proj.weight -> check if expert count matches
                skipped += 1
            else:
                failed.append((key, str(e)))

    npz.close()

    print(f"Loaded {loaded} weights")
    print(f"Skipped {skipped} expert weights (expected - different expert count)")
    if failed:
        print(f"Failed {len(failed)} weights:")
        for k, e in failed[:10]:
            print(f"  {k}: {e}")

    mx.eval(model.parameters())

    # Verify biases are loaded
    print("\n--- Verifying bias loading ---")
    q_bias = model.model.layers[0].self_attn.q_proj.bias
    sinks = model.model.layers[0].self_attn.sinks
    print(f"Layer 0 q_proj.bias: mean_abs={float(mx.mean(mx.abs(q_bias))):.6f}")
    print(f"Layer 0 sinks: mean_abs={float(mx.mean(mx.abs(sinks))):.6f}")

    return model, config


def test_basic_forward():
    """Test basic forward pass."""
    print("=" * 70)
    print("Testing lazarus gpt_oss_lite with npz weights")
    print("=" * 70)

    model, config = load_lazarus_model_with_npz()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "openai/gpt-oss-20b",
        trust_remote_code=True
    )

    prompt = "2 + 2 ="
    tokens = tokenizer.encode(prompt)
    print(f"\nPrompt: '{prompt}'")
    print(f"Tokens: {tokens}")

    input_ids = mx.array([tokens])
    print(f"Input shape: {input_ids.shape}")

    # Forward pass
    print("\n--- Forward pass ---")
    output = model(input_ids)
    mx.eval(output.logits)
    print(f"Logits shape: {output.logits.shape}")

    # Check for NaN/Inf
    has_nan = mx.any(mx.isnan(output.logits)).item()
    has_inf = mx.any(mx.isinf(output.logits)).item()
    print(f"Has NaN: {has_nan}, Has Inf: {has_inf}")

    if has_nan or has_inf:
        print("ERROR: NaN or Inf in logits!")
        return

    # Get prediction
    last_logits = output.logits[:, -1, :]
    next_token = mx.argmax(last_logits, axis=-1).item()
    decoded = tokenizer.decode([next_token])
    print(f"\nPredicted next token: {next_token} = '{decoded}'")

    # Check top-5 predictions
    print("\nTop 5 predictions:")
    top5_indices = mx.argsort(last_logits, axis=-1)[:, -5:]
    top5_logits = mx.take_along_axis(last_logits, top5_indices, axis=-1)
    probs = mx.softmax(top5_logits, axis=-1)

    for i in range(5):
        idx = top5_indices[0, -(i+1)].item()
        prob = probs[0, -(i+1)].item()
        tok = tokenizer.decode([idx])
        print(f"  {i+1}. token={idx} '{tok}' prob={prob:.4f}")

    # Evaluate correctness
    if decoded.strip() == '4':
        print("\n✓ CORRECT! Model predicted '4' for '2 + 2 ='")
    else:
        print(f"\n✗ INCORRECT. Expected '4', got '{decoded}'")


def test_layer_by_layer():
    """Trace through model layer by layer."""
    print("\n" + "=" * 70)
    print("Layer by layer trace")
    print("=" * 70)

    model, config = load_lazarus_model_with_npz()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "openai/gpt-oss-20b",
        trust_remote_code=True
    )

    prompt = "2 + 2 ="
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    # Get embeddings
    embeddings = model.model.embed_tokens(input_ids)
    print(f"Embeddings - shape: {embeddings.shape}, std: {mx.std(embeddings).item():.4f}")

    # Create mask
    mask = "causal"

    # Trace through layers
    x = embeddings
    for i, layer in enumerate(model.model.layers):
        x, _ = layer(x, mask=mask, cache=None)
        mx.eval(x)
        std = mx.std(x).item()
        mean = mx.mean(x).item()
        has_nan = mx.any(mx.isnan(x)).item()
        print(f"Layer {i:2d}: std={std:.4f}, mean={mean:.4f}, nan={has_nan}")

        if has_nan:
            print("  NaN detected! Stopping.")
            break

    # Final norm
    if not mx.any(mx.isnan(x)).item():
        x = model.model.norm(x)
        mx.eval(x)
        print(f"Final norm: std={mx.std(x).item():.4f}")


if __name__ == "__main__":
    test_basic_forward()
    test_layer_by_layer()
