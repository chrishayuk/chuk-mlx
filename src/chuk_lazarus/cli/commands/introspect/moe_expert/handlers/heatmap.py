"""Handler for 'heatmap' action - generate routing heatmap visualization."""

from __future__ import annotations

import asyncio
from argparse import Namespace

from ......introspection.moe import ExpertRouter
from ......introspection.moe.visualization import (
    routing_heatmap_ascii,
    save_routing_heatmap,
)
from ..formatters import format_header


def handle_heatmap(args: Namespace) -> None:
    """Handle the 'heatmap' action - generate routing heatmap visualization.

    Args:
        args: Parsed CLI arguments. Required:
            - model: Model ID
            - prompt: Prompt to analyze

    Example:
        lazarus introspect moe-expert heatmap -m openai/gpt-oss-20b -p "def fibonacci(n):"
        lazarus introspect moe-expert heatmap -m openai/gpt-oss-20b -p "Hello world" --output heatmap.png
    """
    asyncio.run(_async_heatmap(args))


async def _async_heatmap(args: Namespace) -> None:
    """Async implementation of heatmap handler."""
    model_id = args.model
    prompt = getattr(args, "prompt", "Hello, how are you?")
    layer = getattr(args, "layer", None)
    output_path = getattr(args, "output", None)
    ascii_mode = getattr(args, "ascii", False)

    print(f"Loading model: {model_id}")

    async with await ExpertRouter.from_pretrained(model_id) as router:
        info = router.info

        print(format_header("ROUTING HEATMAP"))
        print(f"Model: {model_id}")
        print(f"Prompt: {prompt!r}")
        print(f"Experts: {info.num_experts}")
        print()

        # Capture router weights
        all_weights = await router.capture_router_weights(prompt)

        if not all_weights:
            print("No routing data captured")
            return

        # Determine target layer
        target_layer = layer if layer is not None else info.moe_layers[0]
        layer_weights = next(
            (w for w in all_weights if w.layer_idx == target_layer), all_weights[0]
        )

        if ascii_mode or output_path is None:
            # Print ASCII heatmap
            ascii_output = routing_heatmap_ascii(layer_weights, info.num_experts)
            print(ascii_output)
        else:
            # Save matplotlib heatmap
            try:
                save_routing_heatmap(
                    layer_weights,
                    info.num_experts,
                    path=output_path,
                    title=f"Expert Routing - {model_id}",
                )
                print(f"Heatmap saved to: {output_path}")
            except ImportError:
                print("matplotlib not installed. Using ASCII mode:")
                ascii_output = routing_heatmap_ascii(layer_weights, info.num_experts)
                print(ascii_output)

        # Show summary stats
        print()
        print(f"Layer {layer_weights.layer_idx}:")
        print(f"  Tokens: {len(layer_weights.positions)}")

        # Count expert activations
        expert_counts: dict[int, int] = {}
        for pos in layer_weights.positions:
            for exp_idx in pos.expert_indices:
                expert_counts[exp_idx] = expert_counts.get(exp_idx, 0) + 1

        print("  Top activated experts:")
        for exp_idx, count in sorted(expert_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"    Expert {exp_idx}: {count} tokens")

        print("=" * 70)
