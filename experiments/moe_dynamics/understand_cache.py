#!/usr/bin/env python3
"""Understand how mlx_lm cache and mask generation works."""

import mlx.core as mx
from mlx_lm.models.cache import KVCache, create_attention_mask, RotatingKVCache
from mlx_lm.models.base import create_attention_mask as create_mask_base, create_causal_mask

def test_mask_creation():
    """Test mask creation with different cache states."""
    print("=" * 70)
    print("Testing mask creation")
    print("=" * 70)

    # Simulate hidden states: batch=1, seq_len=5, hidden_dim=128
    h = mx.zeros((1, 5, 128))
    seq_len = h.shape[1]

    # Case 1: No cache at all
    print("\n--- Case 1: No cache ---")
    mask = create_mask_base(h, cache=None)
    print(f"Mask type: {type(mask)}, value: {mask}")

    # Case 2: Fresh KVCache (offset=0)
    print("\n--- Case 2: Fresh KVCache (offset=0) ---")
    fresh_cache = KVCache()
    mask = create_mask_base(h, cache=fresh_cache)
    print(f"Cache offset: {fresh_cache.offset}")
    print(f"Mask type: {type(mask)}, value: {mask}")

    # The mask from cache's make_mask
    cache_mask = fresh_cache.make_mask(seq_len, return_array=False, window_size=None)
    print(f"Cache make_mask result: {cache_mask}")

    # Case 3: KVCache after processing 5 tokens (offset=5)
    print("\n--- Case 3: KVCache after 5 tokens (offset=5) ---")
    used_cache = KVCache()
    # Simulate having processed tokens - add dummy k/v
    k_dummy = mx.zeros((1, 8, 5, 64))  # batch, n_kv_heads, seq, head_dim
    v_dummy = mx.zeros((1, 8, 5, 64))
    used_cache.update_and_fetch(k_dummy, v_dummy)
    print(f"Cache offset after update: {used_cache.offset}")

    mask = create_mask_base(h, cache=used_cache)
    print(f"Mask for 5 new tokens with offset=5: {mask}")

    # Case 4: Single token generation (N=1) with cache
    print("\n--- Case 4: Single token (N=1) with cache ---")
    h_single = mx.zeros((1, 1, 128))
    mask = create_mask_base(h_single, cache=used_cache)
    print(f"Mask for single token: {mask}")

    # Case 5: What create_causal_mask actually produces
    print("\n--- Case 5: Explicit causal masks ---")
    mask_5_offset0 = create_causal_mask(5, offset=0)
    print(f"Causal mask N=5, offset=0:\n{mask_5_offset0}")

    mask_5_offset5 = create_causal_mask(5, offset=5)
    print(f"\nCausal mask N=5, offset=5:\n{mask_5_offset5}")


def test_attention_flow():
    """Trace how attention uses the mask."""
    print("\n" + "=" * 70)
    print("Testing attention flow")
    print("=" * 70)

    # Simulate attention computation
    batch = 1
    n_heads = 4
    seq_len = 5
    head_dim = 32

    # Random Q, K, V
    mx.random.seed(42)
    q = mx.random.normal((batch, n_heads, seq_len, head_dim))
    k = mx.random.normal((batch, n_heads, seq_len, head_dim))
    v = mx.random.normal((batch, n_heads, seq_len, head_dim))

    scale = head_dim ** -0.5

    # Test 1: With "causal" string mask
    print("\n--- Attention with mask='causal' ---")
    out1 = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")
    mx.eval(out1)
    print(f"Output shape: {out1.shape}")
    print(f"Output mean: {mx.mean(out1).item():.6f}")
    print(f"Output std: {mx.std(out1).item():.6f}")

    # Test 2: With explicit causal mask (additive, -inf)
    print("\n--- Attention with explicit causal mask ---")
    mask_explicit = mx.triu(mx.full((seq_len, seq_len), float("-inf")), k=1)
    out2 = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask_explicit)
    mx.eval(out2)
    print(f"Output shape: {out2.shape}")
    print(f"Output mean: {mx.mean(out2).item():.6f}")
    print(f"Output std: {mx.std(out2).item():.6f}")

    # Test 3: With boolean mask
    print("\n--- Attention with boolean mask ---")
    mask_bool = mx.triu(mx.ones((seq_len, seq_len), dtype=mx.bool_), k=1)
    mask_bool = ~mask_bool  # Invert: True means attend
    out3 = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask_bool)
    mx.eval(out3)
    print(f"Output shape: {out3.shape}")
    print(f"Output mean: {mx.mean(out3).item():.6f}")
    print(f"Output std: {mx.std(out3).item():.6f}")

    # Test 4: No mask at all
    print("\n--- Attention with no mask ---")
    out4 = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=None)
    mx.eval(out4)
    print(f"Output shape: {out4.shape}")
    print(f"Output mean: {mx.mean(out4).item():.6f}")
    print(f"Output std: {mx.std(out4).item():.6f}")

    # Compare outputs
    print("\n--- Comparing outputs ---")
    print(f"out1 == out2 (causal vs explicit): {mx.allclose(out1, out2, atol=1e-5)}")
    print(f"out1 == out3 (causal vs bool): {mx.allclose(out1, out3, atol=1e-5)}")


def test_what_lazarus_does():
    """Show what the current lazarus implementation does."""
    print("\n" + "=" * 70)
    print("What lazarus currently does")
    print("=" * 70)

    seq_len = 5

    # Current lazarus mask creation:
    # elif cache is None and seq_len > 1:
    #     mask = mx.triu(mx.full((seq_len, seq_len), float("-inf")), k=1)
    # else:
    #     mask = None

    print("\n--- Lazarus mask creation ---")
    print("Case: cache=None, seq_len=5")
    mask_lazarus = mx.triu(mx.full((seq_len, seq_len), float("-inf")), k=1)
    print(f"Mask:\n{mask_lazarus}")

    # What mlx_lm does:
    # mask = "causal" (the string)
    print("\n--- mlx_lm mask creation ---")
    print("mask = 'causal' (string)")

    # Both should produce the same attention output
    mx.random.seed(42)
    q = mx.random.normal((1, 4, seq_len, 32))
    k = mx.random.normal((1, 4, seq_len, 32))
    v = mx.random.normal((1, 4, seq_len, 32))
    scale = 32 ** -0.5

    out_lazarus = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask_lazarus)
    out_mlx = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")
    mx.eval(out_lazarus, out_mlx)

    print(f"\nOutputs equal: {mx.allclose(out_lazarus, out_mlx, atol=1e-5)}")
    print(f"Max difference: {mx.max(mx.abs(out_lazarus - out_mlx)).item()}")


if __name__ == "__main__":
    test_mask_creation()
    test_attention_flow()
    test_what_lazarus_does()
