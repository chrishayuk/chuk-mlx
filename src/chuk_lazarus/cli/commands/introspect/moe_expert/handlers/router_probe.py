"""Handler for 'router-probe' action - probe router inputs."""

from __future__ import annotations

import asyncio
from argparse import Namespace

from ......introspection.datasets import get_context_tests
from ......introspection.moe import ExpertRouter
from ..formatters import format_header


def handle_router_probe(args: Namespace) -> None:
    """Handle the 'router-probe' action - probe router input decomposition.

    Args:
        args: Parsed CLI arguments. Required:
            - model: Model ID

    Example:
        lazarus introspect moe-expert router-probe -m openai/gpt-oss-20b
    """
    asyncio.run(_async_router_probe(args))


async def _async_router_probe(args: Namespace) -> None:
    """Async implementation of router_probe handler."""
    model_id = args.model
    layer = getattr(args, "layer", None)

    print(f"Loading model: {model_id}")

    # Load test prompts from JSON
    test_data = get_context_tests()

    async with await ExpertRouter.from_pretrained(model_id) as router:
        info = router.info

        print(format_header("ROUTER INPUT DECOMPOSITION"))
        print(f"Model: {model_id}")
        print()

        target_layer = layer if layer is not None else info.moe_layers[0]

        print(f"Testing router inputs at layer {target_layer}:")
        print("-" * 60)
        print(f"{'Prompt':<20} {'Context':<10} {'Top Experts':<30}")
        print("-" * 60)

        for test in test_data.tests:
            weights = await router.capture_router_weights(test.prompt, layers=[target_layer])

            if weights and weights[0].positions:
                # Get last position (target token)
                last_pos = weights[0].positions[-1]
                experts = ", ".join(
                    f"E{e}({w:.2f})"
                    for e, w in zip(last_pos.expert_indices[:3], last_pos.weights[:3])
                )
                print(f"{test.prompt:<20} {test.context_type:<10} {experts}")

        print("=" * 70)
