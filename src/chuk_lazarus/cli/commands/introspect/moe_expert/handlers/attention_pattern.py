"""Handler for 'attention-pattern' action - show what each position attends to.

This is the foundation for understanding attention→routing relationship.
This module is a thin CLI wrapper - business logic is in MoEAnalysisService.
"""

from __future__ import annotations

import asyncio
from argparse import Namespace

from ......introspection.moe import ExpertRouter, MoEAnalysisService
from .._types import AttentionPatternConfig


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
    config = AttentionPatternConfig.from_args(args)

    _print_header(config)

    async with await ExpertRouter.from_pretrained(config.model) as router:
        info = router.info
        moe_layers = list(info.moe_layers)
        total_layers = info.total_layers

        # Determine which layer to analyze
        if config.layer is not None:
            target_layer = config.layer
        else:
            target_layer = moe_layers[len(moe_layers) // 2]

        print(f"  Using layer {target_layer} (of {total_layers} total)")
        print()

        # Tokenize
        tokens = [router.tokenizer.decode([t]) for t in router.tokenizer.encode(config.prompt)]

        print("  Tokens:")
        for i, tok in enumerate(tokens):
            print(f'    [{i}] "{tok}"')
        print()

        # Determine query position
        if config.position is None:
            query_pos = len(tokens) - 1
        elif config.position < 0:
            query_pos = len(tokens) + config.position
        else:
            query_pos = min(config.position, len(tokens) - 1)

        print(f'  Analyzing position {query_pos}: "{tokens[query_pos]}"')
        if config.head is not None:
            print(f"  Using head {config.head} only")
        else:
            print("  Averaging across all heads")
        print()

        # Capture attention weights using the service
        print("  Running forward pass to capture attention...")

        result = await MoEAnalysisService.capture_attention_weights(
            model=config.model,
            prompt=config.prompt,
            layer=target_layer,
            query_position=query_pos,
            head=config.head,
            top_k=config.top_k,
        )

        # Print attention weights
        _print_attention_weights(result, tokens)

        # Also capture router weights to show the routing decision
        print("=" * 70)
        print("ROUTING DECISION (for comparison)")
        print("=" * 70)
        print()

        weights_list = await router.capture_router_weights(config.prompt, layers=[target_layer])
        if weights_list and weights_list[0].positions:
            pos_weights = weights_list[0].positions[result.query_position]
            experts = pos_weights.expert_indices[:4]
            expert_weights = pos_weights.weights[:4]

            print(f'  Token "{result.query_token}" at layer {target_layer}:')
            print()
            for exp, w in zip(experts, expert_weights):
                bar_len = int(w * 40)
                bar = "█" * bar_len + "░" * (40 - bar_len)
                print(f"    E{exp:02d} {w:.3f} [{bar}]")
            print()

        _print_insight()


def _print_header(config: AttentionPatternConfig) -> None:
    """Print the explanation header."""
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
    print(f"  Model: {config.model}")
    print(f'  Prompt: "{config.prompt}"')
    print()
    print("  Loading model...")


def _print_attention_weights(result, tokens: list[str]) -> None:
    """Print attention weight results."""
    print()
    print("=" * 70)
    print("ATTENTION WEIGHTS")
    print("=" * 70)
    print()
    print(f'  Position {result.query_position}: "{result.query_token}"')
    print()
    print("  Top attended positions:")
    print()

    for pos_idx, weight in result.attention_weights:
        tok = tokens[pos_idx] if pos_idx < len(tokens) else "?"
        bar_len = int(weight * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        marker = " (self)" if pos_idx == result.query_position else ""
        print(f'    {weight:.3f} [{bar}] "{tok}"{marker}')

    print()

    # Show self-attention separately if not in top-k
    in_top_k = any(pos_idx == result.query_position for pos_idx, _ in result.attention_weights)
    if not in_top_k:
        print(f"  Self-attention (position {result.query_position}): {result.self_attention:.3f}")
        print()


def _print_insight() -> None:
    """Print key insight section."""
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
