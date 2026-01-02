#!/usr/bin/env python3
"""
Simple Inference Example

Demonstrates the UnifiedPipeline which auto-detects model family.
Works with any supported model without specifying model/config classes!

Usage:
    # Default: TinyLlama
    uv run python examples/inference/simple_inference.py

    # With a specific model (auto-detected family)
    uv run python examples/inference/simple_inference.py --model-id "HuggingFaceTB/SmolLM2-360M-Instruct"

    # Custom prompt
    uv run python examples/inference/simple_inference.py --prompt "Write a haiku about coding"

    # List supported families
    uv run python examples/inference/simple_inference.py --list-families
"""

from __future__ import annotations

import argparse
import sys

from chuk_lazarus.inference import DType, UnifiedPipeline, UnifiedPipelineConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="Simple Inference Example")
    parser.add_argument(
        "--model-id",
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="HuggingFace model ID (any supported family)",
    )
    parser.add_argument(
        "--prompt",
        default="What is the capital of France?",
        help="User prompt",
    )
    parser.add_argument(
        "--system",
        default="You are a helpful assistant.",
        help="System message",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--list-families",
        action="store_true",
        help="List supported model families",
    )
    args = parser.parse_args()

    if args.list_families:
        from chuk_lazarus.models_v2.families import list_model_families

        print("Supported model families:")
        for family in list_model_families():
            print(f"  - {family.value}")
        return 0

    # Configure pipeline
    config = UnifiedPipelineConfig(
        dtype=DType.BFLOAT16,
        default_system_message=args.system,
        default_max_tokens=args.max_tokens,
        default_temperature=args.temperature,
    )

    # Load model - auto-detects family!
    pipeline = UnifiedPipeline.from_pretrained(
        args.model_id,
        pipeline_config=config,
    )

    # Generate
    print("\n" + "=" * 60)
    print(f"Model: {args.model_id} ({pipeline.family_type.value})")
    print(f"User: {args.prompt}")
    print("-" * 60)

    result = pipeline.chat(args.prompt)

    print(f"Assistant: {result.text}")
    print("-" * 60)
    print(f"Stats: {result.stats.summary}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
