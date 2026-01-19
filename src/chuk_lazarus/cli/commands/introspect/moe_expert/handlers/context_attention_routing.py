"""Handler for context-aware attention-routing correlation analysis.

Tests the hypothesis: "Attention drives routing, but context-dependently."

Instead of predicting routing from attention features alone (which fails
because it ignores context), this experiment measures whether CHANGES in
attention patterns correlate with CHANGES in routing decisions.

If attention drives routing, contexts with similar attention patterns
should have similar routing decisions.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argparse import Namespace

logger = logging.getLogger(__name__)


def handle_context_attention_routing(args: Namespace) -> None:
    """Handle context-aware attention-routing correlation analysis.

    Args:
        args: Command arguments with model, token, etc.
    """
    asyncio.run(_async_handle_context_attention_routing(args))


async def _async_handle_context_attention_routing(args: Namespace) -> dict:
    """Async implementation of context-attention-routing analysis."""
    from chuk_lazarus.introspection.moe import ExpertRouter
    from chuk_lazarus.introspection.moe.context_attention_routing_service import (
        ContextAttentionRoutingService,
        print_context_attention_routing_analysis,
    )

    model_id = args.model
    target_token = getattr(args, "token", None) or "127"
    layers_arg = getattr(args, "layers", None)
    contexts_arg = getattr(args, "contexts", None)

    print(f"Loading model: {model_id}")
    router = await ExpertRouter.from_pretrained(model_id)

    service = ContextAttentionRoutingService(router)

    # Parse layers
    layers = None
    if layers_arg:
        if isinstance(layers_arg, str):
            layers = [int(x.strip()) for x in layers_arg.split(",")]
        elif isinstance(layers_arg, list):
            layers = layers_arg

    # Parse contexts
    contexts = None
    if contexts_arg:
        # Format: "name1:prompt1,name2:prompt2" or just "prompt1,prompt2"
        contexts = []
        for ctx in contexts_arg.split("|"):
            ctx = ctx.strip()
            if ":" in ctx:
                name, prompt = ctx.split(":", 1)
                contexts.append((name.strip(), prompt.strip()))
            else:
                # Use first word as name
                name = ctx.split()[0] if ctx.split() else ctx[:10]
                contexts.append((name, ctx))

    print(f"Analyzing context-attention-routing correlation...")
    print(f"  Target token: '{target_token}'")
    print()

    analysis = await service.analyze(
        target_token=target_token,
        contexts=contexts,
        layers=layers,
        model_id=model_id,
    )

    # Print report
    print_context_attention_routing_analysis(analysis)

    # Return summary
    return {
        "model": model_id,
        "target_token": target_token,
        "num_contexts": analysis.num_contexts,
        "overall_correlation": analysis.overall_correlation,
        "best_layer": analysis.best_layer,
        "hypothesis_supported": analysis.hypothesis_supported,
        "layer_correlations": [
            {
                "layer": lc.layer,
                "correlation": lc.attention_routing_correlation,
                "unique_experts": lc.num_unique_experts,
            }
            for lc in analysis.layer_correlations
        ],
    }


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Context-aware attention-routing correlation analysis"
    )
    parser.add_argument("-m", "--model", required=True, help="Model ID")
    parser.add_argument(
        "-t", "--token", default="127",
        help="Target token to analyze (default: 127)"
    )
    parser.add_argument(
        "-l", "--layers",
        help="Comma-separated layer indices to analyze"
    )
    parser.add_argument(
        "-c", "--contexts",
        help="Pipe-separated contexts: 'name:prompt|name:prompt' or 'prompt|prompt'"
    )

    args = parser.parse_args()
    result = asyncio.run(_async_handle_context_attention_routing(args))
    return result


if __name__ == "__main__":
    main()
