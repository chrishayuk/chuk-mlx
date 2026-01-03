"""Handler for 'pattern-discovery' action - discover expert patterns."""

from __future__ import annotations

import asyncio
from argparse import Namespace
from collections import Counter

from ......introspection.datasets import get_pattern_discovery_prompts
from ......introspection.moe import ExpertRouter
from ..formatters import format_header


def handle_pattern_discovery(args: Namespace) -> None:
    """Handle the 'pattern-discovery' action - discover expert activation patterns.

    Args:
        args: Parsed CLI arguments. Required:
            - model: Model ID

    Example:
        lazarus introspect moe-expert pattern-discovery -m openai/gpt-oss-20b
    """
    asyncio.run(_async_pattern_discovery(args))


async def _async_pattern_discovery(args: Namespace) -> None:
    """Async implementation of pattern_discovery handler."""
    model_id = args.model
    layer = getattr(args, "layer", None)

    print(f"Loading model: {model_id}")

    # Load pattern test data from JSON
    patterns_data = get_pattern_discovery_prompts()

    async with await ExpertRouter.from_pretrained(model_id) as router:
        info = router.info

        print(format_header("EXPERT PATTERN DISCOVERY"))
        print(f"Model: {model_id}")
        print()

        target_layer = layer if layer is not None else info.moe_layers[0]

        # Track which experts activate for each pattern category
        category_experts: dict[str, Counter[int]] = {}

        for category_name in patterns_data.get_category_names():
            category = patterns_data.get_category(category_name)
            if not category:
                continue

            expert_counts: Counter[int] = Counter()

            for prompt in category.prompts:
                try:
                    weights = await router.capture_router_weights(prompt, layers=[target_layer])
                    if weights and weights[0].positions:
                        for pos in weights[0].positions:
                            for exp in pos.expert_indices:
                                expert_counts[exp] += 1
                except Exception:
                    continue

            category_experts[category_name] = expert_counts

        print(f"Pattern-Expert associations (layer {target_layer}):")
        print("-" * 60)

        for category, counts in category_experts.items():
            if not counts:
                continue
            top_experts = counts.most_common(5)
            total = sum(counts.values())
            experts_str = ", ".join(f"E{e}({c / total:.0%})" for e, c in top_experts)
            print(f"  {category:<18}: {experts_str}")

        # Find specialist experts (strongly associated with one pattern)
        print()
        print("Potential specialist experts:")
        print("-" * 40)

        expert_specializations: dict[int, list[tuple[str, float]]] = {}
        for category, counts in category_experts.items():
            total = sum(counts.values())
            for exp, count in counts.items():
                rate = count / total if total > 0 else 0
                if rate > 0.3:  # More than 30% of activations
                    if exp not in expert_specializations:
                        expert_specializations[exp] = []
                    expert_specializations[exp].append((category, rate))

        for exp, specs in sorted(expert_specializations.items()):
            specs_str = ", ".join(f"{cat}({rate:.0%})" for cat, rate in specs)
            print(f"  E{exp:2d}: {specs_str}")

        print("=" * 70)
