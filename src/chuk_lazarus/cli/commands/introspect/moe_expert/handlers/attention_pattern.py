"""Handler for 'attention-pattern' action - show what each position attends to.

This is the foundation for understanding attention→routing relationship.

The key insight:
  - The router is a LINEAR function over the hidden state
  - The hidden state comes from attention + residual
  - So: router(attention(input)) → expert

If we understand what attention encodes, we understand what the router "sees".

Note: Since mx.fast.scaled_dot_product_attention doesn't return attention weights
(it's optimized to not store them), we compute attention weights manually by
capturing Q, K values and computing softmax(Q @ K.T / sqrt(d_k)).
"""

from __future__ import annotations

import asyncio
import math
from argparse import Namespace

import mlx.core as mx

from ......introspection.moe import ExpertRouter


def handle_attention_pattern(args: Namespace) -> None:
    """Handle the 'attention-pattern' action - show attention weights for a position.

    Shows what tokens each position attends to, which is the foundation
    for understanding how attention drives expert routing.

    Example:
        lazarus introspect moe-expert attention-pattern -m openai/gpt-oss-20b \
            -p "King is to queen" --position 2 --layer 11
    """
    asyncio.run(_async_attention_pattern(args))


async def _async_attention_pattern(args: Namespace) -> None:
    """Async implementation of attention-pattern handler."""
    model_id = args.model
    prompt = getattr(args, "prompt", None) or "King is to queen as man is to woman"
    position = getattr(args, "position", None)  # None means last position
    layer = getattr(args, "layer", None)
    head = getattr(args, "head", None)
    top_k = getattr(args, "top_k", 5)

    print()
    print("=" * 70)
    print("ATTENTION PATTERN ANALYSIS")
    print("=" * 70)
    print()
    print("=" * 70)
    print("WHAT THIS SHOWS")
    print("=" * 70)
    print()
    print("  Each position in a sequence attends to previous positions.")
    print("  The attention weights determine how much information flows")
    print("  from each source position to the query position.")
    print()
    print("  Attention weights are computed as:")
    print("    attention = softmax(Q @ K.T / sqrt(d_k))")
    print()
    print("  The resulting hidden state is:")
    print("    h = attention @ V + residual")
    print()
    print("  The router then reads this hidden state to select experts.")
    print()
    print("=" * 70)
    print("EXPERIMENT")
    print("=" * 70)
    print()
    print(f"  Model: {model_id}")
    print(f"  Prompt: \"{prompt}\"")
    print()
    print(f"  Loading model...")

    async with await ExpertRouter.from_pretrained(model_id) as router:
        info = router.info
        moe_layers = list(info.moe_layers)
        total_layers = info.total_layers

        # Determine which layer to analyze
        if layer is not None:
            target_layer = layer
        else:
            # Use middle MoE layer by default
            target_layer = moe_layers[len(moe_layers) // 2]

        print(f"  Using layer {target_layer} (of {total_layers} total)")
        print()

        # Tokenize
        input_ids = mx.array(router.tokenizer.encode(prompt))[None, :]
        tokens = [router.tokenizer.decode([t]) for t in input_ids[0].tolist()]

        print("  Tokens:")
        for i, tok in enumerate(tokens):
            print(f"    [{i}] \"{tok}\"")
        print()

        # Determine query position
        if position is None:
            query_pos = len(tokens) - 1
        elif position < 0:
            query_pos = len(tokens) + position
        else:
            query_pos = min(position, len(tokens) - 1)

        print(f"  Analyzing position {query_pos}: \"{tokens[query_pos]}\"")
        if head is not None:
            print(f"  Using head {head} only")
        else:
            print("  Averaging across all heads")
        print()

        # Capture attention weights by patching the attention layer
        print("  Running forward pass to capture attention...")

        # Storage for captured Q, K
        captured_qk: dict[int, tuple[mx.array, mx.array]] = {}

        # Get the attention layer for the target block
        target_block = router._model.model.layers[target_layer]
        attn = target_block.self_attn
        attn_class = type(attn)
        original_call = attn_class.__call__

        def patched_attn_call(attn_self, x, mask=None, cache=None):
            """Patch to capture Q and K."""
            batch, seq_len, _ = x.shape

            # Project Q, K
            q = attn_self.q_proj(x)
            k = attn_self.k_proj(x)

            # Reshape to (batch, seq_len, num_heads, head_dim)
            q = q.reshape(batch, seq_len, attn_self.num_heads, attn_self.head_dim)
            k = k.reshape(batch, seq_len, attn_self.num_kv_heads, attn_self.head_dim)

            # Transpose to (batch, num_heads, seq_len, head_dim)
            q = q.transpose(0, 2, 1, 3)
            k = k.transpose(0, 2, 1, 3)

            # Apply RoPE
            if cache is not None:
                q = attn_self.rope(q, offset=cache[0].shape[2])
                k = attn_self.rope(k, offset=cache[0].shape[2])
            else:
                q = attn_self.rope(q)
                k = attn_self.rope(k)

            # Store the Q, K for later analysis
            captured_qk[target_layer] = (q, k)

            # Call original (which will recompute Q, K but that's fine for correctness)
            return original_call(attn_self, x, mask=mask, cache=cache)

        try:
            # Patch attention
            attn_class.__call__ = patched_attn_call

            # Run forward pass
            router._model(input_ids)

        finally:
            # Restore original
            attn_class.__call__ = original_call

        if target_layer not in captured_qk:
            print("  [ERROR] Could not capture attention for this layer")
            return

        q, k = captured_qk[target_layer]
        # q: (batch, num_heads, seq_len, head_dim)
        # k: (batch, num_kv_heads, seq_len, head_dim)

        # Handle GQA: expand K to match Q's head count
        num_heads = q.shape[1]
        num_kv_heads = k.shape[1]
        if num_kv_heads < num_heads:
            # Repeat K for GQA
            repeat_factor = num_heads // num_kv_heads
            k = mx.repeat(k, repeat_factor, axis=1)

        # Compute attention scores: (batch, num_heads, seq_len, seq_len)
        head_dim = q.shape[-1]
        scale = 1.0 / math.sqrt(head_dim)
        attn_scores = (q @ k.transpose(0, 1, 3, 2)) * scale

        # Apply causal mask
        seq_len = attn_scores.shape[-1]
        causal_mask = mx.triu(mx.full((seq_len, seq_len), -1e9), k=1)
        attn_scores = attn_scores + causal_mask

        # Softmax
        attn_weights = mx.softmax(attn_scores, axis=-1)  # (batch, num_heads, seq_len, seq_len)

        # Get weights for query position
        query_attn = attn_weights[0, :, query_pos, :]  # (num_heads, seq_len)

        # Aggregate across heads or select specific head
        if head is not None:
            head_idx = min(head, num_heads - 1)
            attn_for_pos = query_attn[head_idx]  # (seq_len,)
        else:
            # Mean across heads
            attn_for_pos = mx.mean(query_attn, axis=0)  # (seq_len,)

        # Convert to list and get top-k
        attn_list = attn_for_pos.tolist()
        indexed = list(enumerate(attn_list))
        sorted_attn = sorted(indexed, key=lambda x: x[1], reverse=True)

        print()
        print("=" * 70)
        print("ATTENTION WEIGHTS")
        print("=" * 70)
        print()
        print(f"  Position {query_pos}: \"{tokens[query_pos]}\"")
        print()
        print("  Top attended positions:")
        print()

        for pos_idx, weight in sorted_attn[:top_k]:
            tok = tokens[pos_idx] if pos_idx < len(tokens) else "?"
            # Create bar visualization
            bar_len = int(weight * 40)
            bar = "█" * bar_len + "░" * (40 - bar_len)
            marker = " (self)" if pos_idx == query_pos else ""
            print(f"    {weight:.3f} [{bar}] \"{tok}\"{marker}")

        print()

        # Show self-attention separately if not in top-k
        self_attn = attn_list[query_pos]
        in_top_k = any(pos_idx == query_pos for pos_idx, _ in sorted_attn[:top_k])
        if not in_top_k:
            print(f"  Self-attention (position {query_pos}): {self_attn:.3f}")
            print()

        # Also capture router weights to show the routing decision
        print("=" * 70)
        print("ROUTING DECISION (for comparison)")
        print("=" * 70)
        print()

        weights_list = await router.capture_router_weights(prompt, layers=[target_layer])
        if weights_list and weights_list[0].positions:
            pos_weights = weights_list[0].positions[query_pos]
            experts = pos_weights.expert_indices[:4]
            expert_weights = pos_weights.weights[:4]

            print(f"  Token \"{tokens[query_pos]}\" at layer {target_layer}:")
            print()
            for exp, w in zip(experts, expert_weights):
                bar_len = int(w * 40)
                bar = "█" * bar_len + "░" * (40 - bar_len)
                print(f"    E{exp:02d} {w:.3f} [{bar}]")
            print()

        print("=" * 70)
        print("KEY INSIGHT")
        print("=" * 70)
        print()
        print("  The attention pattern shows WHERE information flows FROM.")
        print("  The hidden state at each position is a WEIGHTED SUM of values")
        print("  from attended positions, plus the residual.")
        print()
        print("  The router reads this hidden state to select experts.")
        print("  So: attention → hidden state → router → expert selection")
        print()
        print("  Different attention patterns → different hidden states")
        print("  Different hidden states → different expert selections")
        print()
        print("=" * 70)
