"""Handler for 'moe-type-compare' action - compare MoE types between models."""

from __future__ import annotations

import asyncio
from argparse import Namespace

from ......introspection.moe import MoETypeService
from ..formatters import format_moe_type_comparison


def handle_moe_type_compare(args: Namespace) -> None:
    """Handle the 'moe-type-compare' action - compare MoE types between two models.

    Args:
        args: Parsed CLI arguments. Required:
            - model: First model ID
            - compare_model: Second model ID

    Example:
        lazarus introspect moe-expert moe-type-compare \\
            -m openai/gpt-oss-20b \\
            -c allenai/OLMoE-1B-7B-0924
    """
    asyncio.run(_async_moe_type_compare(args))


async def _async_moe_type_compare(args: Namespace) -> None:
    """Async implementation of moe-type-compare handler."""
    model1: str = args.model
    model2: str | None = getattr(args, "compare_model", None)

    if not model2:
        print("Error: --compare-model/-c is required for moe-type-compare")
        return

    print("Comparing MoE types...")
    print(f"  Model 1: {model1}")
    print(f"  Model 2: {model2}")
    print()

    # Run analyses sequentially to avoid tqdm threading issues during download
    print(f"Analyzing {model1}...")
    result1 = await MoETypeService.analyze(model1)

    print(f"\nAnalyzing {model2}...")
    result2 = await MoETypeService.analyze(model2)

    print(format_moe_type_comparison(result1, result2))
