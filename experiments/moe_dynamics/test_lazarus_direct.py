#!/usr/bin/env python3
"""Test the lazarus gpt_oss_lite model directly."""

import sys
sys.path.insert(0, '/Users/christopherhay/chris-source/chuk-mlx/src')

import json
import mlx.core as mx
from pathlib import Path
from transformers import AutoTokenizer

MODEL_PATH = Path("/Users/christopherhay/chris-source/chuk-mlx/gpt-oss-lite-v2/gpt-oss-lite-16exp")

def load_lazarus_model():
    """Load the model using lazarus."""
    from chuk_lazarus.models_v2.families.gpt_oss_lite.config import GptOssLiteConfig
    from chuk_lazarus.models_v2.families.gpt_oss_lite.model import GptOssLiteForCausalLM

    # Load config
    with open(MODEL_PATH / "config.json") as f:
        hf_config = json.load(f)

    config = GptOssLiteConfig.from_hf_config(hf_config)
    print(f"Config: hidden_size={config.hidden_size}, num_layers={config.num_hidden_layers}")
    print(f"Config: num_experts={config.num_local_experts}, num_experts_per_tok={config.num_experts_per_tok}")

    # Create model
    model = GptOssLiteForCausalLM(config)

    # Load weights
    weights = mx.load(str(MODEL_PATH / "model.safetensors"))
    loaded, failed = model.load_weights(weights)
    print(f"Loaded {loaded} weights, {len(failed)} failed")
    if failed:
        print(f"Failed keys: {failed[:5]}...")
    mx.eval(model.parameters())

    return model, config


def test_basic_forward():
    """Test basic forward pass."""
    print("=" * 70)
    print("Testing lazarus gpt_oss_lite")
    print("=" * 70)

    model, config = load_lazarus_model()

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


def test_layer_by_layer():
    """Trace through model layer by layer."""
    print("\n" + "=" * 70)
    print("Layer by layer trace")
    print("=" * 70)

    model, config = load_lazarus_model()

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
    seq_len = input_ids.shape[1]
    mask = mx.triu(mx.full((seq_len, seq_len), float("-inf")), k=1)

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


def compare_with_mlx_lm_direct():
    """Compare model behavior with a direct mlx_lm gpt_oss load."""
    print("\n" + "=" * 70)
    print("Comparing with mlx_lm weights loading")
    print("=" * 70)

    # Load lazarus model
    model_lazarus, config = load_lazarus_model()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "openai/gpt-oss-20b",
        trust_remote_code=True
    )

    prompt = "2 + 2 ="
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    # Test lazarus
    print("\n--- Lazarus model ---")
    out_lazarus = model_lazarus(input_ids)
    mx.eval(out_lazarus.logits)
    next_lazarus = mx.argmax(out_lazarus.logits[:, -1, :], axis=-1).item()
    print(f"Prediction: {next_lazarus} = '{tokenizer.decode([next_lazarus])}'")

    # Check embedding weights match
    print("\n--- Weight comparison ---")
    weights = mx.load(str(MODEL_PATH / "model.safetensors"))

    embed_weight = weights["model.embed_tokens.weight"]
    model_embed = model_lazarus.model.embed_tokens.weight
    print(f"Embedding weights match: {mx.allclose(embed_weight, model_embed)}")

    # Check first layer norm
    norm_weight = weights["model.layers.0.input_layernorm.weight"]
    model_norm = model_lazarus.model.layers[0].input_layernorm.weight
    print(f"Layer 0 input norm weights match: {mx.allclose(norm_weight, model_norm)}")

    # Check attention
    q_weight = weights["model.layers.0.self_attn.q_proj.weight"]
    model_q = model_lazarus.model.layers[0].self_attn.q_proj.weight
    print(f"Layer 0 q_proj weights match: {mx.allclose(q_weight, model_q)}")


if __name__ == "__main__":
    test_basic_forward()
    test_layer_by_layer()
    compare_with_mlx_lm_direct()
