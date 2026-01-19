"""Handler for 'moe-type-analyze' action - detect pseudo vs native MoE."""

from __future__ import annotations

import asyncio
from argparse import Namespace
from pathlib import Path

from ......introspection.moe import MoETypeService
from ..formatters import format_moe_type_result, format_orthogonality_ascii


def handle_moe_type_analyze(args: Namespace) -> None:
    """Handle the 'moe-type-analyze' action - analyze MoE type (pseudo vs native).

    Args:
        args: Parsed CLI arguments. Required:
            - model: Model ID
        Optional:
            - layer: Specific layer to analyze
            - visualize: Show orthogonality heatmap visualization
            - output: Path to save JSON result

    Example:
        lazarus introspect moe-expert moe-type-analyze -m openai/gpt-oss-20b
        lazarus introspect moe-expert moe-type-analyze -m allenai/OLMoE-1B-7B-0924 --layer 0
        lazarus introspect moe-expert moe-type-analyze -m openai/gpt-oss-20b --visualize
    """
    asyncio.run(_async_moe_type_analyze(args))


async def _async_moe_type_analyze(args: Namespace) -> None:
    """Async implementation of moe-type-analyze handler."""
    model_id: str = args.model
    layer: int | None = getattr(args, "layer", None)
    output_path: str | None = getattr(args, "output", None)
    visualize: bool = getattr(args, "visualize", False)

    print(f"Analyzing MoE type: {model_id}")

    result = await MoETypeService.analyze(model_id, layer=layer)

    # Show standard result
    print(format_moe_type_result(result))

    # Show orthogonality visualization if requested
    if visualize:
        print()
        print(format_orthogonality_ascii(result))

    if output_path:
        Path(output_path).write_text(result.model_dump_json(indent=2))
        print(f"\nSaved to: {output_path}")
