#!/usr/bin/env python3
"""Debug script to trace exactly what mlx_lm generate does vs direct calls."""

import mlx.core as mx
from pathlib import Path
from transformers import AutoTokenizer

# Path to the full gpt_oss model (mlx_lm supports this)
MODEL_PATH = "mlx-community/GPT-4o-mini-gpt-oss-moe-4bit"

def test_mlx_lm_flow():
    """Trace the exact flow that mlx_lm.generate uses."""
    print("=" * 70)
    print("Testing mlx_lm flow")
    print("=" * 70)

    # Load model using mlx_lm
    from mlx_lm import load
    from mlx_lm.models import cache
    from mlx_lm.models.base import create_attention_mask

    model, tokenizer = load(str(MODEL_PATH))

    # Tokenize prompt
    prompt = "2 + 2 ="
    tokens = tokenizer.encode(prompt)
    print(f"Prompt: '{prompt}'")
    print(f"Tokens: {tokens}")

    input_ids = mx.array([tokens])
    print(f"Input shape: {input_ids.shape}")

    # Create cache like generate_step does
    prompt_cache = cache.make_prompt_cache(model)
    print(f"\nCache type: {type(prompt_cache)}")
    print(f"Num cache layers: {len(prompt_cache)}")
    print(f"First cache type: {type(prompt_cache[0])}")

    # Call model WITH cache (like mlx_lm does)
    print("\n--- Model call WITH cache ---")
    logits_with_cache = model(input_ids, cache=prompt_cache)
    mx.eval(logits_with_cache)

    # Get next token
    next_logits = logits_with_cache[:, -1, :]
    next_token_with_cache = mx.argmax(next_logits, axis=-1).item()
    decoded_with_cache = tokenizer.decode([next_token_with_cache])
    print(f"Next token with cache: {next_token_with_cache} = '{decoded_with_cache}'")

    # Check cache state after first call
    print(f"\nCache[0] offset after first call: {prompt_cache[0].offset}")

    # Call model WITHOUT cache (like direct call)
    print("\n--- Model call WITHOUT cache ---")
    logits_no_cache = model(input_ids, cache=None)
    mx.eval(logits_no_cache)

    next_logits_no = logits_no_cache[:, -1, :]
    next_token_no_cache = mx.argmax(next_logits_no, axis=-1).item()
    decoded_no_cache = tokenizer.decode([next_token_no_cache])
    print(f"Next token without cache: {next_token_no_cache} = '{decoded_no_cache}'")

    # Let's trace deeper - what mask does the model see?
    print("\n--- Tracing mask creation ---")
    x_dummy = mx.zeros((1, len(tokens), model.args.hidden_size))

    # With cache (before call - offset=0)
    fresh_cache = cache.make_prompt_cache(model)
    mask_with_fresh_cache = create_attention_mask(x_dummy, fresh_cache[0])
    print(f"Mask with fresh cache (offset=0): {mask_with_fresh_cache}")

    # With cache after call
    mask_with_used_cache = create_attention_mask(x_dummy, prompt_cache[0])
    print(f"Mask with used cache (offset={prompt_cache[0].offset}): {mask_with_used_cache}")

    # Without cache
    mask_no_cache = create_attention_mask(x_dummy, None)
    print(f"Mask without cache: {mask_no_cache}")


def test_hidden_states_flow():
    """Compare hidden states through layers with and without cache."""
    print("\n" + "=" * 70)
    print("Testing hidden states flow")
    print("=" * 70)

    from mlx_lm import load
    from mlx_lm.models import cache

    model, tokenizer = load(str(MODEL_PATH))

    prompt = "2 + 2 ="
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    # Get embeddings
    embeddings = model.model.embed_tokens(input_ids)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embeddings std: {mx.std(embeddings).item():.4f}")

    # Trace through with cache
    print("\n--- WITH cache ---")
    prompt_cache = cache.make_prompt_cache(model)
    x = embeddings

    # Check layer types for this model
    if hasattr(model.model, 'layer_types'):
        print(f"Layer types: {model.model.layer_types[:4]}...")
        swa_idx = model.model.swa_idx if hasattr(model.model, 'swa_idx') else 0
        ga_idx = model.model.ga_idx if hasattr(model.model, 'ga_idx') else 0
        print(f"SWA index: {swa_idx}, GA index: {ga_idx}")

    for i, (layer, c) in enumerate(zip(model.model.layers[:5], prompt_cache[:5])):
        # Get appropriate mask
        layer_type = model.model.layer_types[i] if hasattr(model.model, 'layer_types') else "full"

        from mlx_lm.models.base import create_attention_mask
        if layer_type == "full_attention":
            mask = create_attention_mask(x, c)
        else:  # sliding_attention
            mask = create_attention_mask(x, c, window_size=model.model.window_size)

        x = layer(x, mask, c)
        mx.eval(x)
        print(f"Layer {i} ({layer_type}) - std: {mx.std(x).item():.4f}, mean: {mx.mean(x).item():.4f}")

    # Trace through WITHOUT cache
    print("\n--- WITHOUT cache ---")
    x = embeddings
    for i, layer in enumerate(model.model.layers[:5]):
        layer_type = model.model.layer_types[i] if hasattr(model.model, 'layer_types') else "full"

        from mlx_lm.models.base import create_attention_mask
        if layer_type == "full_attention":
            mask = create_attention_mask(x, None)
        else:  # sliding_attention
            mask = create_attention_mask(x, None, window_size=model.model.window_size)

        x = layer(x, mask, None)
        mx.eval(x)
        print(f"Layer {i} ({layer_type}) - std: {mx.std(x).item():.4f}, mean: {mx.mean(x).item():.4f}")


def test_generation_step_by_step():
    """Simulate generate_step to see what's happening."""
    print("\n" + "=" * 70)
    print("Simulating generate_step")
    print("=" * 70)

    from mlx_lm import load
    from mlx_lm.models import cache

    model, tokenizer = load(str(MODEL_PATH))

    prompt = "2 + 2 ="
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array(tokens)  # 1D for generate_step

    # Create cache
    prompt_cache = cache.make_prompt_cache(model)

    # Prefill pass (like generate_step does)
    print("Prefill pass...")
    logits = model(input_ids[None], cache=prompt_cache)
    mx.eval([c.state for c in prompt_cache])

    # Get first generated token
    last_logits = logits[:, -1, :]
    first_token = mx.argmax(last_logits, axis=-1)
    mx.eval(first_token)
    print(f"First generated token: {first_token.item()} = '{tokenizer.decode([first_token.item()])}'")

    # Second token
    logits2 = model(first_token[None], cache=prompt_cache)
    second_token = mx.argmax(logits2[:, -1, :], axis=-1)
    mx.eval(second_token)
    print(f"Second generated token: {second_token.item()} = '{tokenizer.decode([second_token.item()])}'")

    # Third token
    logits3 = model(second_token[None], cache=prompt_cache)
    third_token = mx.argmax(logits3[:, -1, :], axis=-1)
    mx.eval(third_token)
    print(f"Third generated token: {third_token.item()} = '{tokenizer.decode([third_token.item()])}'")


if __name__ == "__main__":
    test_mlx_lm_flow()
    test_hidden_states_flow()
    test_generation_step_by_step()
