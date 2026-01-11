"""Build length cache command."""

from __future__ import annotations

import logging
from argparse import Namespace

from .._utils import get_sample_id, get_sample_text, load_dataset
from ._types import LengthBuildConfig, LengthBuildResult

logger = logging.getLogger(__name__)


async def data_lengths_build(config: LengthBuildConfig) -> LengthBuildResult:
    """Build a length cache from a dataset.

    Args:
        config: Build configuration.

    Returns:
        Build result with statistics.
    """
    from chuk_lazarus.data.batching import LengthCache
    from chuk_lazarus.utils.tokenizer_loader import load_tokenizer

    logger.info(f"Loading tokenizer: {config.tokenizer}")
    tokenizer = load_tokenizer(config.tokenizer)

    # Compute tokenizer hash for cache invalidation
    try:
        from chuk_lazarus.data.tokenizers.fingerprint import compute_fingerprint

        fp = compute_fingerprint(tokenizer)
        tokenizer_hash = fp.fingerprint
    except Exception:
        tokenizer_hash = "unknown"

    logger.info(f"Loading dataset: {config.dataset}")
    samples = load_dataset(config.dataset)

    samples_processed = 0
    async with LengthCache.create(config.output, tokenizer_hash) as cache:
        for i, sample in enumerate(samples):
            sample_id = get_sample_id(sample, i)
            text = get_sample_text(sample)

            if text:
                token_ids = tokenizer.encode(text, add_special_tokens=True)
                await cache.add(sample_id, len(token_ids))
                samples_processed += 1

            if (i + 1) % 1000 == 0:
                logger.info(f"Processed {i + 1}/{len(samples)} samples")

    return LengthBuildResult(
        dataset=str(config.dataset),
        tokenizer=config.tokenizer,
        samples_processed=samples_processed,
        output_path=config.output,
        tokenizer_hash=tokenizer_hash,
    )


async def data_lengths_build_cmd(args: Namespace) -> None:
    """CLI entry point for data lengths build command.

    Args:
        args: Parsed command line arguments.
    """
    config = LengthBuildConfig.from_args(args)
    result = await data_lengths_build(config)
    print(result.to_display())
