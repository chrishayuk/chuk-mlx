#!/usr/bin/env python3
"""
Llama Family Inference Example (Simplified)

Demonstrates the new simplified API for Llama-family models.
This replaces the 580+ line examples/models/llama/03_llama_family_inference.py
with a much cleaner implementation.

Supports:
- TinyLlama (1.1B)
- SmolLM2 (135M, 360M, 1.7B)
- Llama 2/3/3.1/3.2
- Mistral 7B

Usage:
    # Default: TinyLlama
    uv run python examples/inference/llama_inference.py

    # SmolLM2 (no auth required)
    uv run python examples/inference/llama_inference.py --model smollm2-360m

    # Llama 3.2 1B (requires HF auth)
    uv run python examples/inference/llama_inference.py --model llama3.2-1b

    # List models
    uv run python examples/inference/llama_inference.py --list
"""

from __future__ import annotations

import argparse
from enum import Enum

from chuk_lazarus.inference import (
    DType,
    InferencePipeline,
    PipelineConfig,
)
from chuk_lazarus.models_v2 import LlamaConfig, LlamaForCausalLM


class LlamaModel(str, Enum):
    """Available Llama-family model presets."""

    # TinyLlama
    TINYLLAMA = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # SmolLM2 (no auth required)
    SMOLLM2_135M = "HuggingFaceTB/SmolLM2-135M-Instruct"
    SMOLLM2_360M = "HuggingFaceTB/SmolLM2-360M-Instruct"
    SMOLLM2_1_7B = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

    # Llama 2 (requires auth)
    LLAMA2_7B = "meta-llama/Llama-2-7b-chat-hf"
    LLAMA2_13B = "meta-llama/Llama-2-13b-chat-hf"

    # Llama 3.2 (requires auth)
    LLAMA3_2_1B = "meta-llama/Llama-3.2-1B-Instruct"
    LLAMA3_2_3B = "meta-llama/Llama-3.2-3B-Instruct"

    # Llama 3.1 (requires auth)
    LLAMA3_1_8B = "meta-llama/Llama-3.1-8B-Instruct"

    # Mistral
    MISTRAL_7B = "mistralai/Mistral-7B-Instruct-v0.3"


# Short aliases for CLI
MODEL_ALIASES = {
    "tinyllama": LlamaModel.TINYLLAMA,
    "smollm2-135m": LlamaModel.SMOLLM2_135M,
    "smollm2-360m": LlamaModel.SMOLLM2_360M,
    "smollm2-1.7b": LlamaModel.SMOLLM2_1_7B,
    "llama2-7b": LlamaModel.LLAMA2_7B,
    "llama2-13b": LlamaModel.LLAMA2_13B,
    "llama3.2-1b": LlamaModel.LLAMA3_2_1B,
    "llama3.2-3b": LlamaModel.LLAMA3_2_3B,
    "llama3.1-8b": LlamaModel.LLAMA3_1_8B,
    "mistral-7b": LlamaModel.MISTRAL_7B,
}


def main():
    parser = argparse.ArgumentParser(
        description="Llama Family Inference (Simplified)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_ALIASES.keys()),
        default="tinyllama",
        help="Model preset",
    )
    parser.add_argument(
        "--model-id",
        help="Custom HuggingFace model ID (overrides --model)",
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
        help="Max tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models",
    )
    args = parser.parse_args()

    # List mode
    if args.list:
        print("Available Llama-family models:\n")
        for alias, model in MODEL_ALIASES.items():
            print(f"  {alias:15} -> {model.value}")
        return

    # Get model ID
    model_id = args.model_id or MODEL_ALIASES[args.model].value

    # Load model with pipeline
    pipeline = InferencePipeline.from_pretrained(
        model_id,
        LlamaForCausalLM,
        LlamaConfig,
        pipeline_config=PipelineConfig(
            dtype=DType.BFLOAT16,
            default_system_message=args.system,
            default_max_tokens=args.max_tokens,
            default_temperature=args.temperature,
        ),
    )

    # Generate
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
