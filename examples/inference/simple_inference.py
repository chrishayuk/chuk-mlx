#!/usr/bin/env python3
"""
Simple Inference Example

Demonstrates the simplified inference API that works with any model family.
Compare this ~50 line example to the 400+ line model-specific examples!

Usage:
    # Default: TinyLlama
    uv run python examples/inference/simple_inference.py

    # With a specific model
    uv run python examples/inference/simple_inference.py --model-id "HuggingFaceTB/SmolLM2-360M-Instruct"

    # Custom prompt
    uv run python examples/inference/simple_inference.py --prompt "Write a haiku about coding"
"""

from __future__ import annotations

import argparse

from chuk_lazarus.inference import (
    GenerationConfig,
    InferencePipeline,
    PipelineConfig,
    DType,
)
from chuk_lazarus.models_v2 import LlamaConfig, LlamaForCausalLM


def main():
    parser = argparse.ArgumentParser(description="Simple Inference Example")
    parser.add_argument(
        "--model-id",
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="HuggingFace model ID",
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
    args = parser.parse_args()

    # Configure pipeline
    config = PipelineConfig(
        dtype=DType.BFLOAT16,
        default_system_message=args.system,
        default_max_tokens=args.max_tokens,
        default_temperature=args.temperature,
    )

    # Load model - ONE LINE!
    pipeline = InferencePipeline.from_pretrained(
        args.model_id,
        LlamaForCausalLM,
        LlamaConfig,
        pipeline_config=config,
    )

    # Generate - ONE LINE!
    print("\n" + "=" * 60)
    print(f"User: {args.prompt}")
    print("-" * 60)

    result = pipeline.chat(args.prompt)

    print(f"Assistant: {result.text}")
    print("-" * 60)
    print(f"Stats: {result.stats.summary}")
    print("=" * 60)


if __name__ == "__main__":
    main()
