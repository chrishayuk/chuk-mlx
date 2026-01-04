"""Inference command handler.

This module provides the async inference command implementation.
"""

from __future__ import annotations

import logging
from argparse import Namespace
from typing import TYPE_CHECKING

from ._types import (
    GenerationResult,
    InferenceConfig,
    InferenceResult,
    InputMode,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


async def run_inference(config: InferenceConfig) -> InferenceResult:
    """Run inference on a model.

    Args:
        config: Inference configuration

    Returns:
        InferenceResult with all generations
    """
    from ....models import load_model

    logger.info(f"Loading model: {config.model}")
    model = load_model(config.model)

    if config.adapter:
        logger.info(f"Loading adapter: {config.adapter}")
        model.load_adapter(config.adapter)

    # Determine prompts based on input mode
    prompts = _get_prompts(config)

    generations: list[GenerationResult] = []
    for prompt in prompts:
        response = model.generate(
            prompt,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )
        generations.append(
            GenerationResult(
                prompt=prompt,
                response=response,
                tokens_generated=len(response.split()),  # Approximate
            )
        )

    return InferenceResult(
        generations=generations,
        model=config.model,
        adapter=config.adapter,
    )


def _get_prompts(config: InferenceConfig) -> list[str]:
    """Get prompts based on input mode.

    Args:
        config: Inference configuration

    Returns:
        List of prompts
    """
    if config.input_mode == InputMode.SINGLE:
        return [config.prompt] if config.prompt else []
    elif config.input_mode == InputMode.FILE:
        if config.prompt_file:
            with open(config.prompt_file) as f:
                return [line.strip() for line in f if line.strip()]
        return []
    else:
        # Interactive mode
        prompts = []
        print("Enter prompts (Ctrl+D to finish):")
        try:
            while True:
                prompt = input("> ")
                if prompt:
                    prompts.append(prompt)
        except EOFError:
            pass
        return prompts


async def run_inference_cmd(args: Namespace) -> None:
    """CLI entry point for inference command.

    Args:
        args: Parsed command-line arguments
    """
    config = InferenceConfig.from_args(args)
    result = await run_inference(config)
    print(result.to_display())


__all__ = [
    "run_inference",
    "run_inference_cmd",
]
