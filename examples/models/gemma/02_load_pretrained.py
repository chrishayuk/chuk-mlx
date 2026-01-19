#!/usr/bin/env python3
"""
Load Pretrained FunctionGemma Example

This example shows how to load pretrained weights from mlx-community
into our native Gemma implementation.

Requirements:
    pip install huggingface_hub safetensors

Usage:
    python 02_load_pretrained.py
"""

import json
from pathlib import Path

import mlx.core as mx

from chuk_lazarus.models_v2.families.gemma import (
    GemmaConfig,
    GemmaForCausalLM,
    convert_hf_weights,
)


def download_model(model_id: str = "mlx-community/functiongemma-270m-it-4bit") -> Path:
    """Download model from HuggingFace Hub."""
    from huggingface_hub import snapshot_download

    print(f"Downloading {model_id}...")
    path = snapshot_download(
        repo_id=model_id,
        allow_patterns=["*.json", "*.safetensors"],
    )
    return Path(path)


def load_config(model_path: Path) -> GemmaConfig:
    """Load config from model directory."""
    config_path = model_path / "config.json"
    with open(config_path) as f:
        config_dict = json.load(f)

    # Map HuggingFace config to our config
    return GemmaConfig(
        model_type=config_dict.get("model_type", "gemma3_text"),
        vocab_size=config_dict["vocab_size"],
        hidden_size=config_dict["hidden_size"],
        num_hidden_layers=config_dict["num_hidden_layers"],
        num_attention_heads=config_dict["num_attention_heads"],
        num_key_value_heads=config_dict.get(
            "num_key_value_heads", config_dict["num_attention_heads"]
        ),
        intermediate_size=config_dict["intermediate_size"],
        head_dim=config_dict.get("head_dim", 256),
        query_pre_attn_scalar=config_dict.get("query_pre_attn_scalar", 256.0),
        sliding_window=config_dict.get("sliding_window", 512),
        sliding_window_pattern=config_dict.get("_sliding_window_pattern", 6),
        max_position_embeddings=config_dict.get("max_position_embeddings", 32768),
        rope_theta=config_dict.get("rope_theta", 1000000.0),
        rope_local_base_freq=config_dict.get("rope_local_base_freq", 10000.0),
        rms_norm_eps=config_dict.get("rms_norm_eps", 1e-6),
    )


def load_weights(model_path: Path) -> dict:
    """Load weights from safetensors files."""

    weights = {}

    # Find all safetensor files
    safetensor_files = list(model_path.glob("*.safetensors"))

    for sf_path in safetensor_files:
        print(f"Loading {sf_path.name}...")
        file_weights = mx.load(str(sf_path))
        weights.update(file_weights)

    return weights


def main():
    """Load and test pretrained FunctionGemma."""
    print("=" * 60)
    print("Loading Pretrained FunctionGemma")
    print("=" * 60)

    # Download model
    model_path = download_model()
    print(f"Model downloaded to: {model_path}")

    # Load config
    config = load_config(model_path)
    print("\nConfig loaded:")
    print(f"  - hidden_size: {config.hidden_size}")
    print(f"  - num_layers: {config.num_hidden_layers}")
    print(f"  - vocab_size: {config.vocab_size}")
    print(f"  - head_dim: {config.head_dim}")

    # Create model
    print("\nCreating model...")
    model = GemmaForCausalLM(config)

    # Load weights
    print("\nLoading weights...")
    weights = load_weights(model_path)
    print(f"Loaded {len(weights)} weight tensors")

    # Convert and apply weights
    converted_weights = convert_hf_weights(weights)
    model.update(converted_weights)
    print("Weights applied successfully!")

    # Test forward pass
    print("\nTesting forward pass...")
    test_input = mx.array([[1, 2, 3, 4, 5]])
    output = model(test_input)
    print(f"Output shape: {output.logits.shape}")

    # Simple generation test
    print("\nTesting generation...")
    generated = model.generate(
        test_input,
        max_new_tokens=10,
        temperature=0.7,
    )
    print(f"Generated shape: {generated.shape}")
    print(f"Generated tokens: {generated[0].tolist()}")

    print("\n" + "=" * 60)
    print("Model loaded and tested successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
