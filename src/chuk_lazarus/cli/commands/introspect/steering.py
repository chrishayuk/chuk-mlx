"""Activation steering commands for introspection CLI.

Commands for extracting and applying activation steering directions.
This module is a thin CLI wrapper - all business logic is in SteeringService.
"""

from __future__ import annotations

import asyncio
import json
from argparse import Namespace
from pathlib import Path

from ._types import SteeringConfig, SteeringExtractionResult, SteeringGenerationResult
from ._utils import parse_prompts


def introspect_steer(args: Namespace) -> None:
    """Apply activation steering to manipulate model behavior.

    Supports three modes:
    1. Extract direction: Compute steering direction from contrastive prompts
    2. Apply direction: Load pre-computed direction and steer generation
    3. Compare: Show outputs at different steering coefficients
    """
    asyncio.run(_async_introspect_steer(args))


async def _async_introspect_steer(args: Namespace) -> None:
    """Async implementation of steering command."""
    from ....introspection.steering import SteeringService

    config = SteeringConfig.from_args(args)

    # Mode 1: Extract direction from contrastive prompts
    if config.extract:
        if not config.positive or not config.negative:
            raise ValueError("--extract requires --positive and --negative prompts")

        print(f"Loading model: {config.model}")
        print(f"\nExtracting direction...")
        print(f"  Positive: {config.positive!r}")
        print(f"  Negative: {config.negative!r}")

        result = await SteeringService.extract_direction(
            model=config.model,
            positive_prompt=config.positive,
            negative_prompt=config.negative,
            layer=config.layer,
        )

        # Display result
        extraction_result = SteeringExtractionResult(
            layer=result.layer,
            norm=result.norm,
            cosine_similarity=result.cosine_similarity,
            separation=result.separation,
            output_path=config.output,
        )
        print(extraction_result.to_display())

        # Save direction
        if config.output:
            SteeringService.save_direction(
                result=result,
                output_path=config.output,
                model_id=config.model,
            )

        return

    # Mode 2 & 3: Apply steering or compare
    print(f"Loading model: {config.model}")

    # Load direction - from file, neuron, or contrastive prompts
    direction, layer, metadata = await _get_direction(config)

    # Parse prompts
    prompts = parse_prompts(config.prompts)

    # Mode: Compare coefficients
    if config.compare:
        coefficients = [float(c) for c in config.compare.split(",")]
        print(f"\nComparing steering at coefficients: {coefficients}")

        for prompt in prompts:
            result = await SteeringService.compare_coefficients(
                model=config.model,
                prompt=prompt,
                direction=direction,
                layer=layer,
                coefficients=coefficients,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
            )

            print(f"\n{'=' * 70}")
            print(f"Prompt: {prompt!r}")
            print(f"{'=' * 70}")

            for coef, output in sorted(result.results.items()):
                direction_label = (
                    "-> positive" if coef > 0 else "<- negative" if coef < 0 else "neutral"
                )
                print(f"\n  Coef {coef:+.1f} ({direction_label}):")
                print(f"    {output!r}")

    # Mode: Single coefficient generation
    else:
        print(f"\nSteering at layer {layer} with coefficient {config.coefficient}")

        results = await SteeringService.generate_with_steering(
            model=config.model,
            prompts=prompts,
            direction=direction,
            layer=layer,
            coefficient=config.coefficient,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            name=config.name,
            positive_label=config.positive_label,
            negative_label=config.negative_label,
        )

        for r in results:
            result = SteeringGenerationResult(
                prompt=r.prompt,
                output=r.output,
                layer=r.layer,
                coefficient=r.coefficient,
            )
            print(result.to_display())

        # Save if requested
        if config.output:
            output_data = [
                {
                    "prompt": r.prompt,
                    "output": r.output,
                    "layer": r.layer,
                    "coefficient": r.coefficient,
                }
                for r in results
            ]
            with open(config.output, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"\nResults saved to: {config.output}")


async def _get_direction(config: SteeringConfig) -> tuple:
    """Get direction from config (file, neuron, or on-the-fly extraction).

    Returns:
        Tuple of (direction, layer, metadata).
    """
    from ....introspection.steering import ActivationSteering, SteeringService

    neuron_idx = config.neuron
    if neuron_idx is not None:
        # Create one-hot direction for single neuron steering
        steerer = ActivationSteering.from_pretrained(config.model)
        layer = config.layer or steerer.num_layers // 2
        hidden_size = steerer.model.config.hidden_size
        direction = SteeringService.create_neuron_direction(hidden_size, neuron_idx)
        print(f"\nSteering neuron {neuron_idx} at layer {layer}")
        print(f"  Hidden size: {hidden_size}")
        return direction, layer, {}

    elif config.direction:
        # Load from file
        direction, layer, metadata = SteeringService.load_direction(config.direction)

        if layer is None:
            layer = config.layer

        print(f"\nLoaded direction from: {config.direction}")
        if "positive_prompt" in metadata:
            print(f"  Positive: {metadata['positive_prompt']}")
        if "negative_prompt" in metadata:
            print(f"  Negative: {metadata['negative_prompt']}")
        print(f"  Layer: {layer}")
        if "norm" in metadata:
            print(f"  Norm: {metadata['norm']:.4f}")

        return direction, layer, metadata

    else:
        # Generate direction on-the-fly from positive/negative
        if not config.positive or not config.negative:
            raise ValueError(
                "Must provide --direction, --neuron, or both --positive and --negative"
            )

        result = await SteeringService.extract_direction(
            model=config.model,
            positive_prompt=config.positive,
            negative_prompt=config.negative,
            layer=config.layer,
        )
        print(f"Using on-the-fly direction from layer {result.layer}")
        return result.direction, result.layer, {}


__all__ = [
    "introspect_steer",
]
