"""Handler for 'divergence' action - analyze layer divergence."""

from __future__ import annotations

import asyncio
from argparse import Namespace

from ......introspection.moe import ExpertRouter
from ..formatters import format_header


def handle_divergence(args: Namespace) -> None:
    """Handle the 'divergence' action - analyze routing divergence across layers.

    Args:
        args: Parsed CLI arguments. Required:
            - model: Model ID
            - prompt: Input prompt

    Example:
        lazarus introspect moe-expert divergence -m openai/gpt-oss-20b -p "Hello world"
    """
    asyncio.run(_async_divergence(args))


async def _async_divergence(args: Namespace) -> None:
    """Async implementation of divergence handler."""
    if not hasattr(args, "prompt") or args.prompt is None:
        print("Error: --prompt/-p is required for divergence action")
        return

    model_id = args.model
    prompt = args.prompt

    print(f"Loading model: {model_id}")

    async with await ExpertRouter.from_pretrained(model_id) as router:
        print(format_header("LAYER DIVERGENCE ANALYSIS"))
        print(f"Model: {model_id}")
        print(f"Prompt: {prompt}")
        print()

        weights = await router.capture_router_weights(prompt)

        # Compare adjacent layers
        print("Adjacent layer agreement:")
        print("-" * 50)

        for i in range(len(weights) - 1):
            layer_a = weights[i]
            layer_b = weights[i + 1]

            agreements = 0
            total = min(len(layer_a.positions), len(layer_b.positions))

            for pos_a, pos_b in zip(layer_a.positions, layer_b.positions):
                if pos_a.expert_indices and pos_b.expert_indices:
                    if pos_a.expert_indices[0] == pos_b.expert_indices[0]:
                        agreements += 1

            agreement_rate = agreements / total if total > 0 else 0
            bar = "#" * int(agreement_rate * 30)
            print(f"  L{layer_a.layer_idx} -> L{layer_b.layer_idx}: {agreement_rate:.1%} {bar}")

        print("=" * 70)
