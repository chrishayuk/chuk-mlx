"""Handler for 'role' action - analyze layer-specific roles."""

from __future__ import annotations

import asyncio
from argparse import Namespace

from ......introspection.moe import ExpertRouter, PromptCategoryGroup, get_prompts_by_group
from ..formatters import format_header


def handle_role(args: Namespace) -> None:
    """Handle the 'role' action - analyze layer-specific expert roles.

    Args:
        args: Parsed CLI arguments. Required:
            - model: Model ID

    Example:
        lazarus introspect moe-expert role -m openai/gpt-oss-20b
    """
    asyncio.run(_async_role(args))


async def _async_role(args: Namespace) -> None:
    """Async implementation of role handler."""
    model_id = args.model
    layer = getattr(args, "layer", None)

    print(f"Loading model: {model_id}")

    async with await ExpertRouter.from_pretrained(model_id) as router:
        info = router.info

        print(format_header("LAYER ROLE ANALYSIS"))
        print(f"Model: {model_id}")
        print()

        target_layer = layer if layer is not None else info.moe_layers[0]

        # Analyze which experts activate for different categories
        category_experts: dict[str, dict[int, int]] = {}

        for group in [
            PromptCategoryGroup.CODE,
            PromptCategoryGroup.MATH,
            PromptCategoryGroup.FACTS,
        ]:
            category_prompts = get_prompts_by_group(group)
            expert_counts: dict[int, int] = {}

            for cat_prompts in category_prompts:
                for prompt in cat_prompts.prompts[:5]:  # Sample 5 per category
                    try:
                        weights = await router.capture_router_weights(prompt, layers=[target_layer])
                        if weights and weights[0].positions:
                            for pos in weights[0].positions:
                                for exp in pos.expert_indices:
                                    expert_counts[exp] = expert_counts.get(exp, 0) + 1
                    except Exception:
                        continue

            category_experts[group.value] = expert_counts

        print(f"Expert activation by category (layer {target_layer}):")
        print("-" * 50)

        for category, counts in category_experts.items():
            top_experts = sorted(counts.items(), key=lambda x: -x[1])[:5]
            experts_str = ", ".join(f"E{e}({c})" for e, c in top_experts)
            print(f"  {category:<10}: {experts_str}")

        print("=" * 70)
