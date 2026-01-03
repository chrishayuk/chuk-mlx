"""Handler for 'context-test' action - test context independence."""

from __future__ import annotations

import asyncio
from argparse import Namespace

from ......introspection.datasets import get_context_tests
from ......introspection.moe import ExpertRouter
from ..formatters import format_header


def handle_context_test(args: Namespace) -> None:
    """Handle the 'context-test' action - test if routing is context-independent.

    Args:
        args: Parsed CLI arguments. Required:
            - model: Model ID

    Example:
        lazarus introspect moe-expert context-test -m openai/gpt-oss-20b
    """
    asyncio.run(_async_context_test(args))


async def _async_context_test(args: Namespace) -> None:
    """Async implementation of context_test handler."""
    model_id = args.model
    layer = getattr(args, "layer", None)

    print(f"Loading model: {model_id}")

    # Load test data from JSON
    test_data = get_context_tests()

    async with await ExpertRouter.from_pretrained(model_id) as router:
        info = router.info

        print(format_header("CONTEXT INDEPENDENCE TEST"))
        print(f"Model: {model_id}")
        print(f"Target token: '{test_data.target_token}'")
        print()

        target_layer = layer if layer is not None else info.moe_layers[0]

        results: dict[str, list[tuple[int, ...]]] = {}

        for test in test_data.tests:
            weights = await router.capture_router_weights(test.prompt, layers=[target_layer])

            if weights and weights[0].positions:
                # Get routing for last token (target)
                last_pos = weights[0].positions[-1]
                experts = last_pos.expert_indices

                if test.context_type not in results:
                    results[test.context_type] = []
                results[test.context_type].append(experts)

        print(f"Routing for '{test_data.target_token}' by context (layer {target_layer}):")
        print("-" * 50)

        for context_type, expert_lists in results.items():
            all_same = all(e == expert_lists[0] for e in expert_lists)
            status = "CONSISTENT" if all_same else "VARIES"
            examples = [f"E{e[0]}" if e else "?" for e in expert_lists[:3]]
            print(f"  {context_type:<12}: [{status}] {', '.join(examples)}")

        # Overall verdict
        all_contexts_same = all(all(e == lists[0] for e in lists) for lists in results.values())

        print()
        if all_contexts_same:
            print("Verdict: Routing is CONTEXT-INDEPENDENT for this token")
        else:
            print("Verdict: Routing is CONTEXT-DEPENDENT")

        print("=" * 70)
