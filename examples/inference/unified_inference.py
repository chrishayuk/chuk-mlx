#!/usr/bin/env python3
"""
Unified inference example.

This script demonstrates the UnifiedPipeline which auto-detects
the model family from HuggingFace config.json.

Usage:
    # Run with a model
    python unified_inference.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0

    # List supported families
    python unified_inference.py --list-families

    # Test with tiny model
    python unified_inference.py --test-tiny

    # Interactive mode
    python unified_inference.py --model distilgpt2 --interactive
"""

from __future__ import annotations

import argparse
import sys
from enum import Enum

from chuk_lazarus.models_v2.families import list_model_families


class ExitCode(int, Enum):
    """Exit codes for the script."""

    SUCCESS = 0
    ERROR = 1
    MODEL_NOT_FOUND = 2


def list_families() -> None:
    """List all supported model families."""
    print("Supported model families:")
    print("=" * 40)
    for family_type in list_model_families():
        print(f"  - {family_type.value}")


def test_tiny() -> int:
    """Test with tiny models to verify the setup works."""
    import mlx.core as mx

    from chuk_lazarus.models_v2.families import gpt2, llama

    print("=" * 60)
    print("Unified Pipeline Tiny Model Tests")
    print("=" * 60)

    # Test Llama
    print("\n1. Testing Llama (tiny)...")
    llama_config = llama.LlamaConfig.tiny()
    llama_model = llama.LlamaForCausalLM(llama_config)
    input_ids = mx.array([[1, 2, 3, 4, 5]])
    output = llama_model(input_ids)
    print(f"   Forward: OK (shape={output.logits.shape})")

    # Test GPT-2
    print("\n2. Testing GPT-2 (tiny)...")
    gpt2_config = gpt2.GPT2Config.tiny()
    gpt2_model = gpt2.GPT2ForCausalLM(gpt2_config)
    output = gpt2_model(input_ids)
    print(f"   Forward: OK (shape={output.logits.shape})")

    # Test family detection
    print("\n3. Testing family detection...")
    from chuk_lazarus.models_v2.families import detect_model_family

    test_cases = [
        {"model_type": "llama"},
        {"model_type": "gpt2"},
        {"model_type": "gemma3_text"},
        {"architectures": ["LlamaForCausalLM"]},
        {"architectures": ["GPT2LMHeadModel"]},
    ]

    for config in test_cases:
        family = detect_model_family(config)
        status = "OK" if family else "FAILED"
        print(f"   {config} -> {family.value if family else 'None'} [{status}]")

    print("\n" + "=" * 60)
    print("SUCCESS! All tests passed.")
    print("=" * 60)

    return ExitCode.SUCCESS


def run_inference(
    model_id: str,
    prompt: str = "Hello, how are you?",
    max_tokens: int = 100,
    temperature: float = 0.7,
    interactive: bool = False,
) -> int:
    """Run inference with a model."""
    from chuk_lazarus.inference import UnifiedPipeline

    try:
        pipeline = UnifiedPipeline.from_pretrained(model_id)
    except ValueError as e:
        print(f"Error loading model: {e}")
        return ExitCode.MODEL_NOT_FOUND

    print()
    print("=" * 60)
    print(f"Model: {model_id}")
    print(f"Family: {pipeline.family_type.value}")
    print("=" * 60)

    if interactive:
        print("\nInteractive mode (type 'quit' to exit)")
        print("-" * 40)
        while True:
            try:
                user_input = input("\nYou: ").strip()
                if user_input.lower() in ("quit", "exit", "q"):
                    break
                if not user_input:
                    continue

                result = pipeline.chat(
                    user_input,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                )
                print(f"\nAssistant: {result.text}")
                print(f"({result.stats.tokens_per_second:.1f} tok/s)")

            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
    else:
        print(f"\nPrompt: {prompt}")
        print("-" * 40)

        result = pipeline.chat(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )

        print(f"\nResponse: {result.text}")
        print("\nStats:")
        print(f"  Tokens generated: {result.stats.tokens_generated}")
        print(f"  Tokens/second: {result.stats.tokens_per_second:.1f}")

    return ExitCode.SUCCESS


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Unified inference pipeline example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--model",
        type=str,
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, how are you?",
        help="Prompt to use for inference",
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
        "--interactive",
        action="store_true",
        help="Run in interactive chat mode",
    )
    parser.add_argument(
        "--list-families",
        action="store_true",
        help="List supported model families",
    )
    parser.add_argument(
        "--test-tiny",
        action="store_true",
        help="Run tests with tiny models",
    )

    args = parser.parse_args()

    if args.list_families:
        list_families()
        return ExitCode.SUCCESS

    if args.test_tiny:
        return test_tiny()

    if not args.model:
        parser.print_help()
        print("\nError: --model is required unless using --list-families or --test-tiny")
        return ExitCode.ERROR

    return run_inference(
        model_id=args.model,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        interactive=args.interactive,
    )


if __name__ == "__main__":
    sys.exit(main())
