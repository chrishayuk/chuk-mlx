"""Inference command handler.

This module provides the async inference command implementation.
The CLI command is a thin wrapper that delegates to UnifiedPipeline.
"""

from __future__ import annotations

from argparse import Namespace

from ._types import GenerationResult, InferenceConfig, InferenceResult, InputMode


async def run_inference_cmd(args: Namespace) -> None:
    """CLI entry point for inference command.

    This is a thin wrapper that:
    1. Converts CLI args to InferenceConfig
    2. Uses UnifiedPipeline for generation
    3. Prints the result

    Args:
        args: Parsed command-line arguments
    """
    from ....inference import UnifiedPipeline

    # Convert CLI args to config
    config = InferenceConfig.from_args(args)

    # Load the model (quiet by default for CLI)
    pipeline = UnifiedPipeline.from_pretrained(config.model, verbose=False)

    # Collect prompts based on input mode
    prompts: list[str] = []
    if config.input_mode == InputMode.SINGLE:
        prompts = [config.prompt]
    elif config.input_mode == InputMode.FILE:
        with open(config.prompt_file) as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        # Interactive mode - single prompt from stdin
        print("Enter prompt (Ctrl+D to finish):")
        import sys

        prompts = [sys.stdin.read().strip()]

    # Generate responses
    generations = []
    for prompt in prompts:
        if config.chat:
            # Use chat mode with template
            response = pipeline.chat(
                prompt,
                system_message=config.system,
                max_new_tokens=config.max_tokens,
                temperature=config.temperature,
            )
        else:
            # Raw generation without template
            response = pipeline.generate(
                prompt,
                max_new_tokens=config.max_tokens,
                temperature=config.temperature,
            )
        generations.append(
            GenerationResult(
                prompt=prompt,
                response=response.text,
                tokens_generated=response.stats.output_tokens,
            )
        )

    # Build result
    result = InferenceResult(
        generations=generations,
        model=config.model,
        adapter=config.adapter,
    )

    # Print result
    print(result.to_display())


__all__ = [
    "run_inference_cmd",
]
