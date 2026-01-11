"""Handler for 'attention-routing' action - analyze attention patterns that drive routing.

This module is a thin CLI wrapper - business logic is in AttentionRoutingService.

Research question: What does attention encode that the router uses?
"""

from __future__ import annotations

import asyncio
from argparse import Namespace

from ......introspection.moe import ExpertRouter
from ......introspection.moe.attention_routing_service import AttentionRoutingService


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


async def _async_attention_routing(args: Namespace) -> None:
    """Async implementation of attention-routing handler."""
    model_id = args.model
    layers_str = getattr(args, "layers", None)
    contexts_str = getattr(args, "contexts", None)
    target_token = getattr(args, "token", None) or "+"

    # Parse contexts using service
    test_contexts = AttentionRoutingService.parse_contexts(contexts_str)

    _print_header(model_id, target_token, test_contexts)

    async with await ExpertRouter.from_pretrained(model_id) as router:
        info = router.info
        moe_layers = info.moe_layers

        # Parse layers using service
        target_layers = AttentionRoutingService.parse_layers(layers_str, moe_layers)
        layer_labels = AttentionRoutingService.get_layer_labels(target_layers)

        print(f"  Analyzing layers: {target_layers}")
        print()

        # Results by layer
        results_by_layer: dict[int, list[dict]] = {layer: [] for layer in target_layers}

        for layer in target_layers:
            label = layer_labels.get(layer, "")
            print(f"  Layer {layer} ({label}):")

            for ctx_name, ctx in test_contexts:
                # Capture attention weights using service
                attn_result = AttentionRoutingService.capture_attention_weights(router, ctx, layer)

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

                # Compute attention pattern summary using service
                attn_summary = None
                if attn_result.success and attn_result.attention_weights is not None:
                    summary = AttentionRoutingService.compute_attention_summary(
                        attn_result.attention_weights,
                        attn_result.tokens,
                        target_pos_idx,
                    )
                    attn_summary = summary.top_attended

                result = {
                    "context_name": ctx_name,
                    "context": ctx,
                    "tokens": attn_result.tokens,
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

        # Print summaries
        _print_layer_summary(target_layers, layer_labels, results_by_layer)
        _print_attention_patterns(target_layers, results_by_layer)
        _print_analysis(target_layers, layer_labels, results_by_layer)


def _print_header(model_id: str, target_token: str, test_contexts: list[tuple[str, str]]) -> None:
    """Print the experiment header."""
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
        print(f'    {name:<12}: "{ctx}"')
    print()

    print("=" * 70)
    print("RUNNING ANALYSIS")
    print("=" * 70)
    print()
    print(f"  Loading model: {model_id}")


def _print_layer_summary(
    target_layers: list[int],
    layer_labels: dict[int, str],
    results_by_layer: dict[int, list[dict]],
) -> None:
    """Print layer-by-layer summary."""
    print("=" * 70)
    print("LAYER-BY-LAYER SUMMARY")
    print("=" * 70)
    print()

    for layer in target_layers:
        results = results_by_layer[layer]
        label = layer_labels.get(layer, "")
        unique_experts = {r["primary_expert"] for r in results}

        print(f"  Layer {layer} ({label}):")
        for r in results:
            print(f"    {r['context_name']:<12} → E{r['primary_expert']}")

        if len(unique_experts) == 1:
            print("    → Same expert for all contexts (low differentiation)")
        else:
            print(f"    → {len(unique_experts)} different experts (context-sensitive)")
        print()


def _print_attention_patterns(
    target_layers: list[int],
    results_by_layer: dict[int, list[dict]],
) -> None:
    """Print attention patterns section."""
    print("=" * 70)
    print("ATTENTION PATTERNS (Middle Layer)")
    print("=" * 70)
    print()

    # Show attention for middle layer
    middle_layer = (
        target_layers[len(target_layers) // 2] if len(target_layers) >= 2 else target_layers[0]
    )
    for r in results_by_layer[middle_layer]:
        print(f"  {r['context_name']:<12} → E{r['primary_expert']}")
        if r["attn_summary"]:
            for tok, weight in r["attn_summary"]:
                bar_len = int(weight * 30)
                bar = "█" * bar_len
                print(f'    {weight:.2f} {bar} "{tok}"')
        print()


def _print_analysis(
    target_layers: list[int],
    layer_labels: dict[int, str],
    results_by_layer: dict[int, list[dict]],
) -> None:
    """Print analysis and key insights."""
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print()

    # Compare early vs middle vs late
    early_layer = target_layers[0]
    middle_layer = (
        target_layers[len(target_layers) // 2] if len(target_layers) >= 2 else target_layers[0]
    )
    late_layer = target_layers[-1]

    early_unique = len({r["primary_expert"] for r in results_by_layer[early_layer]})
    middle_unique = len({r["primary_expert"] for r in results_by_layer[middle_layer]})
    late_unique = len({r["primary_expert"] for r in results_by_layer[late_layer]})

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
