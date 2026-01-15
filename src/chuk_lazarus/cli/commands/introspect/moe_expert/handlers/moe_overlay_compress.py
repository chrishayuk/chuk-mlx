"""Handler for moe-overlay-compress action."""

from __future__ import annotations

import asyncio
from argparse import Namespace
from pathlib import Path


def handle_moe_overlay_compress(args: Namespace) -> None:
    """Compress MoE model to overlay format (base + low-rank deltas)."""
    asyncio.run(_async_moe_overlay_compress(args))


async def _async_moe_overlay_compress(args: Namespace) -> None:
    from ......introspection.moe import MoECompressionService

    model_id: str = args.model
    output_path: str = getattr(args, "output", None) or _default_output_path(model_id)
    gate_rank: int | None = getattr(args, "gate_rank", None)
    up_rank: int | None = getattr(args, "up_rank", None)
    down_rank: int | None = getattr(args, "down_rank", None)
    dtype: str = getattr(args, "dtype", "bfloat16")

    print(f"Compressing model: {model_id}")
    print(f"Output: {output_path}")
    if gate_rank or up_rank or down_rank:
        print(f"Ranks: gate={gate_rank}, up={up_rank}, down={down_rank}")
    else:
        print("Ranks: auto-selecting from SVD analysis")
    print()

    result = await MoECompressionService.compress_model(
        model_id,
        output_path,
        gate_rank=gate_rank,
        up_rank=up_rank,
        down_rank=down_rank,
        dtype=dtype,
    )

    _print_compression_result(result)


def _default_output_path(model_id: str) -> str:
    """Generate default output path from model ID."""
    # openai/gpt-oss-20b -> gpt-oss-20b-overlay
    name = model_id.split("/")[-1]
    return f"{name}-overlay"


def _print_compression_result(result) -> None:
    """Format and print compression result."""
    config = result.config

    print("=" * 70)
    print("COMPRESSION COMPLETE")
    print("=" * 70)
    print(f"Output:             {result.output_path}")
    print()
    print("Storage:")
    print(f"  Original:         {result.original_mb:,.1f} MB")
    print(f"  Compressed:       {result.compressed_mb:,.1f} MB")
    print(f"  Ratio:            {result.compression_ratio:.1f}x")
    print(f"  Savings:          {result.original_mb - result.compressed_mb:,.1f} MB")
    print()
    print("Ranks:")
    print(f"  Gate:             {config.gate_rank}")
    print(f"  Up:               {config.up_rank}")
    print(f"  Down:             {config.down_rank}")
    print()
    print("Reconstruction Error:")
    print(f"  Mean MSE:         {result.mean_reconstruction_error:.2e}")
    print(f"  Max MSE:          {result.max_reconstruction_error:.2e}")
    print()
    print("Files:")
    output = Path(result.output_path)
    for f in sorted(output.iterdir()):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name:30} {size_mb:>8.1f} MB")
    print("=" * 70)
