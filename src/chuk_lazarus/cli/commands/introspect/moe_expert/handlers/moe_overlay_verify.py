"""Handler for moe-overlay-verify action."""

from __future__ import annotations

import asyncio
from argparse import Namespace
from pathlib import Path


def handle_moe_overlay_verify(args: Namespace) -> None:
    """Verify reconstruction accuracy of overlay representation."""
    asyncio.run(_async_moe_overlay_verify(args))


async def _async_moe_overlay_verify(args: Namespace) -> None:
    from ......introspection.moe import MoECompressionService
    from ..formatters import format_verification_result

    model_id: str = args.model
    layer: int | None = getattr(args, "layer", None)
    gate_rank: int | None = getattr(args, "gate_rank", None)
    up_rank: int | None = getattr(args, "up_rank", None)
    down_rank: int | None = getattr(args, "down_rank", None)
    output_path: str | None = getattr(args, "output", None)

    print(f"Verifying overlay reconstruction: {model_id}")
    if gate_rank or up_rank or down_rank:
        print(f"  Ranks: gate={gate_rank}, up={up_rank}, down={down_rank}")
    else:
        print("  Ranks: auto-selecting from SVD analysis")

    result = await MoECompressionService.verify_reconstruction(
        model_id,
        layer=layer,
        gate_rank=gate_rank,
        up_rank=up_rank,
        down_rank=down_rank,
    )

    print(format_verification_result(result))

    if output_path:
        Path(output_path).write_text(result.model_dump_json(indent=2))
        print(f"\nSaved to: {output_path}")
