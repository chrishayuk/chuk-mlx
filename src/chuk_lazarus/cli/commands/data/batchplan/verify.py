"""Verify batch plan command."""

from __future__ import annotations

import logging
import sys
from argparse import Namespace

from ._types import BatchPlanVerifyConfig, BatchPlanVerifyResult

logger = logging.getLogger(__name__)


async def data_batchplan_verify(config: BatchPlanVerifyConfig) -> BatchPlanVerifyResult:
    """Verify a batch plan can be reproduced.

    Args:
        config: Verify configuration.

    Returns:
        Verification result.
    """
    from chuk_lazarus.data.batching import (
        BatchingConfig,
        BatchingMode,
        BatchPlanBuilder,
        LengthCache,
        PadPolicy,
        load_batch_plan,
    )

    # Load original plan
    logger.info(f"Loading batch plan: {config.plan}")
    original = load_batch_plan(config.plan)

    # Rebuild from lengths
    logger.info(f"Rebuilding from lengths: {config.lengths}")
    cache = await LengthCache.load(config.lengths)
    lengths = cache.get_all()

    # Recreate config from plan meta
    batching_config = BatchingConfig(
        mode=BatchingMode(original.meta.mode),
        pad_policy=PadPolicy(original.meta.pad_policy),
        token_budget=original.meta.token_budget,
        bucket_edges=tuple(original.meta.bucket_edges),
        overflow_max=original.meta.overflow_max,
        seed=original.meta.seed,
    )

    builder = BatchPlanBuilder(
        lengths=lengths,
        batching_config=batching_config,
        dataset_hash=original.meta.dataset_hash,
        tokenizer_hash=original.meta.tokenizer_hash,
    )

    rebuilt = await builder.build(num_epochs=original.num_epochs)

    # Compare fingerprints
    match = original.fingerprint == rebuilt.fingerprint

    # Detailed comparison if mismatch
    epoch_comparison = []
    if not match:
        for ep in range(original.num_epochs):
            orig_mbs = list(original.iter_epoch(ep))
            rebuilt_mbs = list(rebuilt.iter_epoch(ep))

            if len(orig_mbs) != len(rebuilt_mbs):
                epoch_comparison.append(
                    {
                        "epoch": ep,
                        "count_differs": True,
                        "original_count": len(orig_mbs),
                        "rebuilt_count": len(rebuilt_mbs),
                    }
                )
            else:
                matches = sum(1 for o, r in zip(orig_mbs, rebuilt_mbs) if o.samples == r.samples)
                epoch_comparison.append(
                    {
                        "epoch": ep,
                        "count_differs": False,
                        "matches": matches,
                        "total": len(orig_mbs),
                    }
                )

    return BatchPlanVerifyResult(
        original_fingerprint=original.fingerprint,
        rebuilt_fingerprint=rebuilt.fingerprint,
        match=match,
        epoch_comparison=epoch_comparison,
    )


async def data_batchplan_verify_cmd(args: Namespace) -> None:
    """CLI entry point for batchplan verify command.

    Args:
        args: Parsed command line arguments.
    """
    config = BatchPlanVerifyConfig.from_args(args)
    result = await data_batchplan_verify(config)
    print(result.to_display())

    if not result.match:
        sys.exit(1)
