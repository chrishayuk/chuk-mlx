"""Handler for 'entropy' action - analyze routing entropy."""

from __future__ import annotations

import asyncio
import math
from argparse import Namespace

from ......introspection.moe import ExpertRouter
from ..formatters import format_entropy_analysis


def handle_entropy(args: Namespace) -> None:
    """Handle the 'entropy' action - analyze routing entropy across layers.

    Args:
        args: Parsed CLI arguments. Required:
            - model: Model ID
            - prompt: Input prompt

    Example:
        lazarus introspect moe-expert entropy -m openai/gpt-oss-20b -p "Hello world"
    """
    asyncio.run(_async_entropy(args))


async def _async_entropy(args: Namespace) -> None:
    """Async implementation of entropy handler."""
    if not hasattr(args, "prompt") or args.prompt is None:
        print("Error: --prompt/-p is required for entropy action")
        return

    model_id = args.model
    prompt = args.prompt

    print(f"Loading model: {model_id}")

    async with await ExpertRouter.from_pretrained(model_id) as router:
        info = router.info
        max_entropy = math.log(info.num_experts)

        weights = await router.capture_router_weights(prompt)

        entropies: list[tuple[int, float, float]] = []

        for layer_weights in weights:
            layer_entropies: list[float] = []
            for pos in layer_weights.positions:
                # Calculate entropy from weights
                entropy = 0.0
                for w in pos.weights:
                    if w > 0:
                        entropy -= w * math.log(w + 1e-10)
                layer_entropies.append(entropy)

            mean_ent = sum(layer_entropies) / len(layer_entropies) if layer_entropies else 0
            norm_ent = mean_ent / max_entropy if max_entropy > 0 else 0
            entropies.append((layer_weights.layer_idx, mean_ent, norm_ent))

        output = format_entropy_analysis(entropies, model_id, prompt)
        print(output)
