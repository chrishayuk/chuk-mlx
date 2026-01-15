"""Handler for moe-overlay-compress action."""

from __future__ import annotations

import asyncio
from argparse import Namespace
from pathlib import Path

# Quality presets as fractions of hidden dimension
# (gate_fraction, up_fraction, down_fraction)
# These scale automatically based on model size
QUALITY_PRESETS = {
    "fast": (0.001, 0.01, 0.005),  # ~12x compression, fastest
    "balanced": (0.001, 0.02, 0.01),  # ~8x compression, good tradeoff (default)
    "quality": (0.002, 0.04, 0.02),  # ~5x compression, best reconstruction
}

# Minimum ranks to ensure reasonable quality
MIN_RANKS = {"gate": 2, "up": 16, "down": 8}


def handle_moe_overlay_compress(args: Namespace) -> None:
    """Compress MoE model to overlay format (base + low-rank deltas)."""
    asyncio.run(_async_moe_overlay_compress(args))


async def _async_moe_overlay_compress(args: Namespace) -> None:
    from ......introspection.moe import MoECompressionService

    model_id: str = args.model
    output_path: str = getattr(args, "output", None) or _default_output_path(model_id)
    dtype: str = getattr(args, "dtype", "bfloat16")
    resume: bool = getattr(args, "resume", True)

    # Resolve ranks from preset or explicit values
    quality: str = getattr(args, "quality", "balanced")
    gate_rank: int | None = getattr(args, "gate_rank", None)
    up_rank: int | None = getattr(args, "up_rank", None)
    down_rank: int | None = getattr(args, "down_rank", None)

    print(f"Compressing model: {model_id}")
    print(f"Output: {output_path}")

    # If no explicit ranks, compute from model dimensions and preset
    if gate_rank is None and up_rank is None and down_rank is None:
        print("Detecting model dimensions...")
        hidden_dim = await _get_hidden_dim(model_id)

        gate_frac, up_frac, down_frac = QUALITY_PRESETS[quality]
        gate_rank = max(MIN_RANKS["gate"], int(hidden_dim * gate_frac))
        up_rank = max(MIN_RANKS["up"], int(hidden_dim * up_frac))
        down_rank = max(MIN_RANKS["down"], int(hidden_dim * down_frac))
        rank_source = f"quality={quality}"
        print(f"Hidden dim: {hidden_dim}")
    else:
        # Fill in any missing ranks with balanced defaults
        if gate_rank is None or up_rank is None or down_rank is None:
            hidden_dim = await _get_hidden_dim(model_id)
            gate_frac, up_frac, down_frac = QUALITY_PRESETS["balanced"]
            gate_rank = gate_rank or max(MIN_RANKS["gate"], int(hidden_dim * gate_frac))
            up_rank = up_rank or max(MIN_RANKS["up"], int(hidden_dim * up_frac))
            down_rank = down_rank or max(MIN_RANKS["down"], int(hidden_dim * down_frac))
        rank_source = "custom"

    print(f"Quality: {rank_source} (gate={gate_rank}, up={up_rank}, down={down_rank})")
    if resume:
        print("Resume: enabled (will continue from checkpoint if found)")
    print()

    result = await MoECompressionService.compress_model(
        model_id,
        output_path,
        gate_rank=gate_rank,
        up_rank=up_rank,
        down_rank=down_rank,
        dtype=dtype,
        resume=resume,
    )

    _print_compression_result(result)


async def _get_hidden_dim(model_id: str) -> int:
    """Get hidden dimension from model config."""
    import asyncio

    def _load_dim() -> int:
        from ......introspection.moe.moe_type import MoETypeService

        model = MoETypeService._load_model(model_id)
        # Try common config attribute names
        config = getattr(model, "config", None) or getattr(model, "args", None)
        if config:
            for attr in ["hidden_size", "hidden_dim", "d_model", "n_embd"]:
                if hasattr(config, attr):
                    return getattr(config, attr)
        # Fallback: check first layer embedding
        if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            return model.model.embed_tokens.weight.shape[1]
        # Default for unknown models
        return 2048

    return await asyncio.to_thread(_load_dim)


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
