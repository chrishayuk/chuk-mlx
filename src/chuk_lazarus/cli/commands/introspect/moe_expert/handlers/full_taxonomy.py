"""Handler for 'full-taxonomy' action - generate full expert taxonomy."""

from __future__ import annotations

import asyncio
from argparse import Namespace
from collections import Counter

from ......introspection.moe import (
    ExpertCategory,
    ExpertIdentity,
    ExpertRole,
    ExpertRouter,
    ExpertTaxonomy,
    get_prompts_flat,
)
from ..formatters import format_taxonomy


def handle_full_taxonomy(args: Namespace) -> None:
    """Handle the 'full-taxonomy' action - generate complete expert taxonomy.

    Args:
        args: Parsed CLI arguments. Required:
            - model: Model ID

    Example:
        lazarus introspect moe-expert full-taxonomy -m openai/gpt-oss-20b
    """
    asyncio.run(_async_full_taxonomy(args))


async def _async_full_taxonomy(args: Namespace) -> None:
    """Async implementation of full_taxonomy handler."""
    model_id = args.model
    verbose = getattr(args, "verbose", False)
    num_prompts = getattr(args, "num_prompts", 100)

    print(f"Loading model: {model_id}")

    async with await ExpertRouter.from_pretrained(model_id) as router:
        info = router.info

        print(
            f"Generating taxonomy for {info.num_experts} experts across {len(info.moe_layers)} layers..."
        )
        print(f"Using {num_prompts} sample prompts...")

        # Get sample prompts
        prompts_with_cats = get_prompts_flat()[:num_prompts]

        # Track expert activations by category
        expert_categories: dict[
            tuple[int, int], Counter[str]
        ] = {}  # (layer, expert) -> category counts
        expert_tokens: dict[tuple[int, int], list[str]] = {}  # (layer, expert) -> tokens

        for cat, prompt in prompts_with_cats:
            try:
                weights = await router.capture_router_weights(prompt)

                for layer_weights in weights:
                    layer_idx = layer_weights.layer_idx
                    for pos in layer_weights.positions:
                        for exp in pos.expert_indices:
                            key = (layer_idx, exp)
                            if key not in expert_categories:
                                expert_categories[key] = Counter()
                                expert_tokens[key] = []
                            expert_categories[key][cat.value] += 1
                            if pos.token and len(expert_tokens[key]) < 10:
                                expert_tokens[key].append(pos.token)
            except Exception:
                continue

        # Build expert identities
        identities: list[ExpertIdentity] = []

        for (layer_idx, expert_idx), cat_counts in expert_categories.items():
            if not cat_counts:
                continue

            total = sum(cat_counts.values())
            top_cat = cat_counts.most_common(1)[0][0] if cat_counts else "unknown"
            confidence = cat_counts[top_cat] / total if total > 0 else 0

            # Determine role based on distribution
            if confidence > 0.5:
                role = ExpertRole.SPECIALIST
            elif len(cat_counts) > 5 and confidence < 0.3:
                role = ExpertRole.GENERALIST
            else:
                role = ExpertRole.GENERALIST

            # Map to ExpertCategory
            try:
                primary = ExpertCategory(top_cat)
            except ValueError:
                primary = ExpertCategory.UNKNOWN

            key = (layer_idx, expert_idx)
            tokens = tuple(expert_tokens.get(key, []))

            identities.append(
                ExpertIdentity(
                    expert_idx=expert_idx,
                    layer_idx=layer_idx,
                    primary_category=primary,
                    role=role,
                    confidence=confidence,
                    activation_rate=total / num_prompts,
                    top_tokens=tokens,
                )
            )

        # Build taxonomy
        taxonomy = ExpertTaxonomy(
            model_id=model_id,
            num_layers=info.total_layers,
            num_experts=info.num_experts,
            expert_identities=tuple(identities),
            patterns=(),  # Would need more analysis
            layer_analyses=(),  # Would need more analysis
        )

        output = format_taxonomy(taxonomy, verbose=verbose)
        print(output)
