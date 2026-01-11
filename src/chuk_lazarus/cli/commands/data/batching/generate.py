"""Generate NPZ batch files command."""

from __future__ import annotations

import logging
from argparse import Namespace

from .._utils import get_sample_id, get_sample_text, load_dataset
from ._types import GenerateConfig, GenerateResult

logger = logging.getLogger(__name__)


async def data_batch_generate(config: GenerateConfig) -> GenerateResult:
    """Generate NPZ batch files from a BatchPlan.

    Args:
        config: Generation configuration.

    Returns:
        Generation result.
    """
    from chuk_lazarus.data.batching import (
        BatchReader,
        BatchWriter,
        load_batch_plan,
    )
    from chuk_lazarus.utils.tokenizer_loader import load_tokenizer

    # Load batch plan
    logger.info(f"Loading batch plan: {config.plan}")
    plan = load_batch_plan(config.plan)

    # Load tokenizer
    logger.info(f"Loading tokenizer: {config.tokenizer}")
    tokenizer = load_tokenizer(config.tokenizer)

    # Load dataset
    logger.info(f"Loading dataset: {config.dataset}")
    raw_samples = load_dataset(config.dataset)

    # Tokenize samples
    logger.info("Tokenizing samples...")
    samples = {}
    for i, sample in enumerate(raw_samples):
        sample_id = get_sample_id(sample, i)
        text = get_sample_text(sample)

        if text:
            input_ids = tokenizer.encode(text, add_special_tokens=True)
            loss_mask = [1] * len(input_ids)
            samples[sample_id] = {
                "input_ids": input_ids,
                "loss_mask": loss_mask,
            }

        if (i + 1) % 1000 == 0:
            logger.info(f"Tokenized {i + 1}/{len(raw_samples)} samples")

    # Create writer
    logger.info(f"Writing batches to: {config.output}")

    writer = BatchWriter(
        plan=plan,
        samples=samples,
        output_dir=config.output,
        pad_id=tokenizer.pad_token_id or 0,
    )

    # Write batches
    files = writer.write_all()

    # Verify
    reader = BatchReader(config.output)

    return GenerateResult(
        batch_plan=str(config.plan),
        dataset=str(config.dataset),
        output_dir=config.output,
        num_files=len(files),
        num_epochs=reader.num_epochs,
        fingerprint=reader.fingerprint,
    )


async def data_batch_generate_cmd(args: Namespace) -> None:
    """CLI entry point for batch generate command.

    Args:
        args: Parsed command line arguments.
    """
    config = GenerateConfig.from_args(args)
    result = await data_batch_generate(config)
    print(result.to_display())
