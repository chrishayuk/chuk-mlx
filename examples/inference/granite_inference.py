#!/usr/bin/env python3
"""
IBM Granite Inference Example (Simplified)

Demonstrates the simplified API for Granite models.
Supports both Granite 3.x (dense) and 4.x (hybrid Mamba/Transformer).

Usage:
    # Test tiny (no download)
    uv run python examples/inference/granite_inference.py --test-tiny

    # Granite 3.1 2B
    uv run python examples/inference/granite_inference.py --model granite-3.1-2b

    # Granite 4.0 Micro
    uv run python examples/inference/granite_inference.py --model granite-4.0-micro
"""

from __future__ import annotations

import argparse
from enum import Enum

import mlx.core as mx

from chuk_lazarus.inference import (
    DType,
    InferencePipeline,
    PipelineConfig,
)
from chuk_lazarus.models_v2 import (
    GraniteConfig,
    GraniteForCausalLM,
    GraniteHybridConfig,
    GraniteHybridForCausalLM,
    count_parameters,
)


class GraniteModelType(str, Enum):
    """Granite model architecture types."""

    DENSE = "granite"  # Granite 3.x
    HYBRID = "granitemoehybrid"  # Granite 4.x


class GraniteModel(str, Enum):
    """Available Granite model presets."""

    # Granite 3.x (Dense)
    GRANITE_3_1_2B = "ibm-granite/granite-3.1-2b-instruct"
    GRANITE_3_1_8B = "ibm-granite/granite-3.1-8b-instruct"
    GRANITE_3_3_2B = "ibm-granite/granite-3.3-2b-instruct"
    GRANITE_3_3_8B = "ibm-granite/granite-3.3-8b-instruct"

    # Granite 4.x (Hybrid)
    GRANITE_4_0_MICRO = "ibm-granite/granite-4.0-micro"
    GRANITE_4_0_TINY = "ibm-granite/granite-4.0-tiny-preview"


MODEL_ALIASES = {
    "granite-3.1-2b": (GraniteModel.GRANITE_3_1_2B, GraniteModelType.DENSE),
    "granite-3.1-8b": (GraniteModel.GRANITE_3_1_8B, GraniteModelType.DENSE),
    "granite-3.3-2b": (GraniteModel.GRANITE_3_3_2B, GraniteModelType.DENSE),
    "granite-3.3-8b": (GraniteModel.GRANITE_3_3_8B, GraniteModelType.DENSE),
    "granite-4.0-micro": (GraniteModel.GRANITE_4_0_MICRO, GraniteModelType.HYBRID),
    "granite-4.0-tiny": (GraniteModel.GRANITE_4_0_TINY, GraniteModelType.HYBRID),
}


def test_tiny():
    """Test tiny model configurations without downloading."""
    print("=" * 60)
    print("Granite Tiny Model Tests")
    print("=" * 60)

    # Test Granite 3.x
    print("\n1. Testing Granite 3.x (dense)...")
    config3 = GraniteConfig.tiny()
    model3 = GraniteForCausalLM(config3)
    params3 = count_parameters(model3)
    print(f"   {params3.summary()}")

    input_ids = mx.array([[1, 2, 3, 4, 5]])
    output3 = model3(input_ids)
    mx.eval(output3.logits)
    print(f"   Forward: OK (shape={output3.logits.shape})")

    # Test Granite 4.x
    print("\n2. Testing Granite 4.x (hybrid)...")
    config4 = GraniteHybridConfig.tiny()
    model4 = GraniteHybridForCausalLM(config4)
    params4 = count_parameters(model4)
    print(f"   {params4.summary()}")

    output4 = model4(input_ids)
    mx.eval(output4.logits)
    print(f"   Forward: OK (shape={output4.logits.shape})")

    print("\n" + "=" * 60)
    print("SUCCESS! All tiny tests passed.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Granite Inference (Simplified)")
    parser.add_argument(
        "--model",
        choices=list(MODEL_ALIASES.keys()),
        default="granite-3.1-2b",
        help="Model preset",
    )
    parser.add_argument("--model-id", help="Custom HuggingFace model ID")
    parser.add_argument("--test-tiny", action="store_true", help="Run tiny tests")
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
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--list", action="store_true", help="List models")
    args = parser.parse_args()

    if args.test_tiny:
        test_tiny()
        return

    if args.list:
        print("Available Granite models:\n")
        for alias, (model, model_type) in MODEL_ALIASES.items():
            print(f"  {alias:20} -> {model.value} ({model_type.value})")
        return

    # Get model info
    if args.model_id:
        model_id = args.model_id
        # Default to dense for custom models
        model_type = GraniteModelType.DENSE
    else:
        model_enum, model_type = MODEL_ALIASES[args.model]
        model_id = model_enum.value

    # Select appropriate model/config classes
    if model_type == GraniteModelType.HYBRID:
        model_class = GraniteHybridForCausalLM
        config_class = GraniteHybridConfig
    else:
        model_class = GraniteForCausalLM
        config_class = GraniteConfig

    # Load with pipeline
    pipeline = InferencePipeline.from_pretrained(
        model_id,
        model_class,
        config_class,
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
