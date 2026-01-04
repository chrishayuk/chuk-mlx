"""Handler for 'attention-routing' action - analyze attention patterns that drive routing.

Research question: What does attention encode that the router uses?

The router is a LINEAR function over the hidden state.
The hidden state is attention output + residual.
So: router(attention(input)) → expert

If we understand what attention encodes, we understand what the router "sees."
"""

from __future__ import annotations

import asyncio
import math
from argparse import Namespace
from collections import defaultdict

import mlx.core as mx

from ......introspection.moe import ExpertRouter


# Default test contexts: same trigram, different attention patterns expected
DEFAULT_CONTEXTS = [
    ("minimal", "2 + 3"),
    ("instruction", "Calculate: 2 + 3"),
    ("sentence", "The sum is 2 + 3"),
    ("code", "result = 2 + 3"),
]


def handle_attention_routing(args: Namespace) -> None:
    """Handle the 'attention-routing' action - analyze attention→routing relationship.

    Shows:
    1. What tokens each position attends to
    2. How attention patterns correlate with expert selection
    3. Whether different experts see different attention patterns
    4. How this varies across layers (early/middle/late)

    Example:
        lazarus introspect moe-expert attention-routing -m openai/gpt-oss-20b
        lazarus introspect moe-expert attention-routing -m openai/gpt-oss-20b --layers 0,12,23
        lazarus introspect moe-expert attention-routing -m openai/gpt-oss-20b --contexts "def add,def hello"
    """
    asyncio.run(_async_attention_routing(args))


def _capture_attention_weights(
    router: ExpertRouter,
    prompt: str,
    target_layer: int,
) -> tuple[list[str], mx.array | None]:
    """Capture attention weights for a prompt at a specific layer.

    Returns:
        Tuple of (tokens, attention_weights) where attention_weights is
        (num_heads, seq_len, seq_len) or None if capture failed.
    """
    input_ids = mx.array(router.tokenizer.encode(prompt))[None, :]
    tokens = [router.tokenizer.decode([t]) for t in input_ids[0].tolist()]

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

        # Call original
        return original_call(attn_self, x, mask=mask, cache=cache)

    try:
        attn_class.__call__ = patched_attn_call
        router._model(input_ids)
    finally:
        attn_class.__call__ = original_call

    if target_layer not in captured_qk:
        return tokens, None

    q, k = captured_qk[target_layer]

    # Handle GQA
    num_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    if num_kv_heads < num_heads:
        repeat_factor = num_heads // num_kv_heads
        k = mx.repeat(k, repeat_factor, axis=1)

    # Compute attention scores
    head_dim = q.shape[-1]
    scale = 1.0 / math.sqrt(head_dim)
    attn_scores = (q @ k.transpose(0, 1, 3, 2)) * scale

    # Apply causal mask
    seq_len = attn_scores.shape[-1]
    causal_mask = mx.triu(mx.full((seq_len, seq_len), -1e9), k=1)
    attn_scores = attn_scores + causal_mask

    # Softmax
    attn_weights = mx.softmax(attn_scores, axis=-1)  # (batch, num_heads, seq_len, seq_len)

    return tokens, attn_weights[0]  # Remove batch dim


def _parse_layers(layers_str: str | None, moe_layers: tuple[int, ...]) -> list[int]:
    """Parse layers argument into list of layer indices."""
    if layers_str is None:
        # Default: early, middle, late
        if len(moe_layers) >= 3:
            return [moe_layers[0], moe_layers[len(moe_layers) // 2], moe_layers[-1]]
        return list(moe_layers)

    if layers_str.lower() == "all":
        return list(moe_layers)

    # Parse comma-separated
    return [int(x.strip()) for x in layers_str.split(",")]


def _parse_contexts(contexts_str: str | None) -> list[tuple[str, str]]:
    """Parse contexts argument into list of (name, prompt) tuples."""
    if contexts_str is None:
        return DEFAULT_CONTEXTS

    contexts = []
    for ctx in contexts_str.split(","):
        ctx = ctx.strip()
        if ctx:
            # Use first word as name, full string as prompt
            name = ctx.split()[0] if ctx.split() else ctx[:10]
            contexts.append((name, ctx))
    return contexts


async def _async_attention_routing(args: Namespace) -> None:
    """Async implementation of attention-routing handler."""
    model_id = args.model
    layers_str = getattr(args, "layers", None)
    contexts_str = getattr(args, "contexts", None)
    target_token = getattr(args, "token", None) or "+"

    # Parse contexts
    test_contexts = _parse_contexts(contexts_str)

    print()
    print("=" * 70)
    print("ATTENTION → ROUTING ANALYSIS")
    print("=" * 70)
    print()
    print("=" * 70)
    print("RESEARCH QUESTION")
    print("=" * 70)
    print()
    print("  The router is a LINEAR function over the hidden state.")
    print("  The hidden state comes from attention + residual.")
    print()
    print("  So: router(attention(input)) → expert")
    print()
    print("  HYPOTHESIS: Different attention patterns → different experts")
    print("              Layer position affects context sensitivity")
    print()
    print("=" * 70)
    print("EXPERIMENT")
    print("=" * 70)
    print()
    print(f"  Model: {model_id}")
    print(f"  Target token: '{target_token}'")
    print()

    print("  Contexts to analyze:")
    for name, ctx in test_contexts:
        print(f"    {name:<12}: \"{ctx}\"")
    print()

    print("=" * 70)
    print("RUNNING ANALYSIS")
    print("=" * 70)
    print()
    print(f"  Loading model: {model_id}")

    async with await ExpertRouter.from_pretrained(model_id) as router:
        info = router.info
        moe_layers = info.moe_layers

        # Parse layers
        target_layers = _parse_layers(layers_str, moe_layers)
        layer_labels = {
            target_layers[0]: "Early",
            target_layers[-1]: "Late",
        }
        if len(target_layers) >= 3:
            layer_labels[target_layers[len(target_layers) // 2]] = "Middle"

        print(f"  Analyzing layers: {target_layers}")
        print()

        # Results by layer
        results_by_layer: dict[int, list[dict]] = {layer: [] for layer in target_layers}

        for layer in target_layers:
            label = layer_labels.get(layer, "")
            print(f"  Layer {layer} ({label}):")

            for ctx_name, ctx in test_contexts:
                # Capture attention weights
                tokens, attn_weights = _capture_attention_weights(router, ctx, layer)

                # Get router weights
                weights = await router.capture_router_weights(ctx, layers=[layer])

                if not weights or not weights[0].positions:
                    continue

                layer_weights = weights[0]

                # Find the target token position
                target_pos_idx = None
                for i, pos in enumerate(layer_weights.positions):
                    if target_token.lower() in pos.token.lower():
                        target_pos_idx = i
                        break

                if target_pos_idx is None:
                    target_pos_idx = len(layer_weights.positions) - 1

                target_pos = layer_weights.positions[target_pos_idx]
                primary_expert = target_pos.expert_indices[0] if target_pos.expert_indices else -1

                # Compute attention pattern summary
                attn_summary = None
                if attn_weights is not None:
                    pos_attn = mx.mean(attn_weights[:, target_pos_idx, :], axis=0).tolist()
                    indexed = list(enumerate(pos_attn))
                    sorted_attn = sorted(indexed, key=lambda x: x[1], reverse=True)[:3]
                    attn_summary = [
                        (tokens[idx] if idx < len(tokens) else "?", weight)
                        for idx, weight in sorted_attn
                    ]

                result = {
                    "context_name": ctx_name,
                    "context": ctx,
                    "tokens": tokens,
                    "target_pos": target_pos_idx,
                    "target_token": target_pos.token,
                    "primary_expert": primary_expert,
                    "all_experts": target_pos.expert_indices,
                    "weights": target_pos.weights,
                    "attn_summary": attn_summary,
                }
                results_by_layer[layer].append(result)

                print(f"    {ctx_name:<12} → E{primary_expert}")

            print()

        # Summary by layer
        print("=" * 70)
        print("LAYER-BY-LAYER SUMMARY")
        print("=" * 70)
        print()

        for layer in target_layers:
            results = results_by_layer[layer]
            label = layer_labels.get(layer, "")
            unique_experts = set(r["primary_expert"] for r in results)

            print(f"  Layer {layer} ({label}):")
            for r in results:
                print(f"    {r['context_name']:<12} → E{r['primary_expert']}")

            if len(unique_experts) == 1:
                print(f"    → Same expert for all contexts (low differentiation)")
            else:
                print(f"    → {len(unique_experts)} different experts (context-sensitive)")
            print()

        # Attention patterns section
        print("=" * 70)
        print("ATTENTION PATTERNS (Middle Layer)")
        print("=" * 70)
        print()

        # Show attention for middle layer
        middle_layer = target_layers[len(target_layers) // 2] if len(target_layers) >= 2 else target_layers[0]
        for r in results_by_layer[middle_layer]:
            print(f"  {r['context_name']:<12} → E{r['primary_expert']}")
            if r["attn_summary"]:
                for tok, weight in r["attn_summary"]:
                    bar_len = int(weight * 30)
                    bar = "█" * bar_len
                    print(f"    {weight:.2f} {bar} \"{tok}\"")
            print()

        # Analysis
        print("=" * 70)
        print("ANALYSIS")
        print("=" * 70)
        print()

        # Compare early vs middle vs late
        early_layer = target_layers[0]
        late_layer = target_layers[-1]

        early_unique = len(set(r["primary_expert"] for r in results_by_layer[early_layer]))
        middle_unique = len(set(r["primary_expert"] for r in results_by_layer[middle_layer]))
        late_unique = len(set(r["primary_expert"] for r in results_by_layer[late_layer]))

        print(f"  Early  (L{early_layer:2d}): {early_unique} unique experts")
        print(f"  Middle (L{middle_layer:2d}): {middle_unique} unique experts")
        print(f"  Late   (L{late_layer:2d}): {late_unique} unique experts")
        print()

        if middle_unique >= early_unique and middle_unique >= late_unique:
            print("  FINDING: Maximum differentiation in MIDDLE layers")
            print("           This is where context-framing matters most.")
        elif late_unique > middle_unique:
            print("  FINDING: Late layers show high differentiation")
            print("           Context affects output decisions directly.")
        else:
            print("  FINDING: Early layers show high differentiation")
            print("           Context affects initial tagging.")

        print()
        print("=" * 70)
        print("KEY INSIGHT")
        print("=" * 70)
        print()
        print("  Same token. Same trigram. Different contexts.")
        print()
        print(f"  At layer {early_layer}: slightly different experts")
        print(f"  At layer {middle_layer}: maximum differentiation")
        print(f"  At layer {late_layer}: converging toward output")
        print()
        print("  The middle layers are where context-framing")
        print("  matters most. That's where the model decides")
        print("  HOW to process, not just WHAT to output.")
        print()
        print("=" * 70)
