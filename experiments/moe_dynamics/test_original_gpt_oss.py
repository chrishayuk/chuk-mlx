#!/usr/bin/env python3
"""Test the original gpt_oss model from mlx_lm."""

import mlx.core as mx
from mlx_lm import load, generate


def test_original_model():
    print("=" * 70)
    print("Testing original gpt-oss model from mlx_lm")
    print("=" * 70)

    # Load the original model
    print("Loading openai/gpt-oss-20b...")
    model, tokenizer = load("openai/gpt-oss-20b")

    # Test prompt
    prompt = "2 + 2 ="
    print(f"\nPrompt: '{prompt}'")

    # Generate
    result = generate(model, tokenizer, prompt=prompt, max_tokens=10, verbose=False)
    print(f"Generated: '{result}'")

    # Also test layer by layer
    print("\n--- Layer by layer trace ---")
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    x = model.model.embed_tokens(input_ids)
    mx.eval(x)
    print(f"Embeddings std: {mx.std(x).item():.4f}")

    layer_types = model.model.layer_types
    from mlx_lm.models.base import create_attention_mask

    for i, (layer, layer_type) in enumerate(zip(model.model.layers, layer_types)):
        if layer_type == "full_attention":
            mask = create_attention_mask(x, None)
        else:
            mask = create_attention_mask(x, None, window_size=model.model.window_size)

        x = layer(x, mask, None)
        mx.eval(x)
        std = mx.std(x).item()
        print(f"Layer {i:2d} ({layer_type[:4]}): std={std:.4f}")

        # Early stop if exploding
        if std > 1000:
            print("  Explosion detected, stopping trace")
            break

    # Final norm
    x = model.model.norm(x)
    mx.eval(x)
    print(f"Final norm: std={mx.std(x).item():.4f}")


if __name__ == "__main__":
    test_original_model()
