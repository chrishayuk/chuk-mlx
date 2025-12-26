#!/usr/bin/env python3
"""
StarCoder2 Inference Example

Demonstrates code generation using StarCoder2 models.

Supports:
- StarCoder2 3B (smallest, fastest)
- StarCoder2 7B (balanced)
- StarCoder2 15B (largest, best quality)

Usage:
    # Default: StarCoder2 3B
    uv run python examples/inference/starcoder2_inference.py

    # StarCoder2 7B
    uv run python examples/inference/starcoder2_inference.py --model starcoder2-7b

    # StarCoder2 15B
    uv run python examples/inference/starcoder2_inference.py --model starcoder2-15b

    # Custom prompt
    uv run python examples/inference/starcoder2_inference.py --prompt "def fibonacci(n):"

    # Interactive mode
    uv run python examples/inference/starcoder2_inference.py --interactive

    # List available models
    uv run python examples/inference/starcoder2_inference.py --list
"""

from __future__ import annotations

import argparse
import json
import time
from enum import Enum
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_unflatten

from chuk_lazarus.inference import (
    GenerationConfig,
    HFLoader,
)
from chuk_lazarus.models_v2.families.starcoder2 import StarCoder2Config, StarCoder2ForCausalLM


class StarCoder2Model(str, Enum):
    """Available StarCoder2 model presets."""

    # StarCoder2 family
    STARCODER2_3B = "bigcode/starcoder2-3b"
    STARCODER2_7B = "bigcode/starcoder2-7b"
    STARCODER2_15B = "bigcode/starcoder2-15b"


MODEL_ALIASES = {
    # StarCoder2
    "starcoder2-3b": StarCoder2Model.STARCODER2_3B,
    "starcoder2-7b": StarCoder2Model.STARCODER2_7B,
    "starcoder2-15b": StarCoder2Model.STARCODER2_15B,
    # Shortcuts
    "3b": StarCoder2Model.STARCODER2_3B,
    "7b": StarCoder2Model.STARCODER2_7B,
    "15b": StarCoder2Model.STARCODER2_15B,
}


class StarCoder2Token(int, Enum):
    """Special token IDs for StarCoder2."""

    EOS = 0  # <|endoftext|>
    FIM_PREFIX = 1  # <fim_prefix>
    FIM_SUFFIX = 2  # <fim_suffix>
    FIM_MIDDLE = 3  # <fim_middle>


def load_starcoder2_config(model_path: Path, weights: dict | None = None) -> StarCoder2Config:
    """Load and convert HuggingFace config to StarCoder2Config."""
    config_path = model_path / "config.json"
    with open(config_path) as f:
        hf_config = json.load(f)

    hidden_size = hf_config["hidden_size"]
    head_dim = hf_config.get("head_dim") or (hidden_size // hf_config["num_attention_heads"])

    # Try to get num_attention_heads from config, otherwise infer from weights
    num_attention_heads = hf_config.get("num_attention_heads")
    num_key_value_heads = hf_config.get("num_key_value_heads")

    if weights and (num_attention_heads is None or num_key_value_heads is None):
        # Infer from weight shapes
        for k, v in weights.items():
            if "layers.0" in k and "self_attn.q_proj.weight" in k:
                if num_attention_heads is None:
                    num_attention_heads = v.shape[0] // head_dim
                    print(f"  Inferred num_attention_heads={num_attention_heads}")
            if "layers.0" in k and "self_attn.k_proj.weight" in k:
                if num_key_value_heads is None:
                    num_key_value_heads = v.shape[0] // head_dim
                    print(f"  Inferred num_key_value_heads={num_key_value_heads}")
                break

    # Fallback defaults
    if num_attention_heads is None:
        num_attention_heads = hidden_size // head_dim
    if num_key_value_heads is None:
        num_key_value_heads = num_attention_heads

    # Check if lm_head is present to determine if embeddings are tied
    # If lm_head is missing from weights, embeddings are tied
    has_lm_head = weights is not None and any("lm_head" in k for k in weights.keys())
    tie_embeddings = hf_config.get("tie_word_embeddings", not has_lm_head)

    return StarCoder2Config(
        model_type=hf_config.get("model_type", "starcoder2"),
        vocab_size=hf_config.get("vocab_size", 49152),
        hidden_size=hidden_size,
        num_hidden_layers=hf_config["num_hidden_layers"],
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        intermediate_size=hf_config["intermediate_size"],
        max_position_embeddings=hf_config.get("max_position_embeddings", 16384),
        rope_theta=hf_config.get("rope_theta", 100000.0),
        layer_norm_eps=hf_config.get("norm_epsilon", 1e-5),
        sliding_window=hf_config.get("sliding_window", 4096),
        attention_bias=hf_config.get("use_bias", True),
        mlp_bias=hf_config.get("use_bias", True),
        tie_word_embeddings=tie_embeddings,
    )


def load_starcoder2_weights(model_path: Path) -> dict:
    """Load weights from safetensors and convert names."""
    raw_weights = {}
    for sf_path in sorted(model_path.glob("*.safetensors")):
        print(f"  Loading {sf_path.name}...")
        file_weights = mx.load(str(sf_path))
        raw_weights.update(file_weights)

    # Convert weight names from HuggingFace to our format
    weights = {}
    for name, weight in raw_weights.items():
        new_name = name

        # MLP naming: c_fc -> up_proj, c_proj -> down_proj
        new_name = new_name.replace("mlp.c_fc", "mlp.up_proj")
        new_name = new_name.replace("mlp.c_proj", "mlp.down_proj")

        # Embedding: model.embed_tokens.weight -> model.embed_tokens.weight.weight
        if new_name == "model.embed_tokens.weight":
            new_name = "model.embed_tokens.weight.weight"

        # LM head: lm_head.weight -> lm_head.lm_head.weight (only if present)
        # For tied embeddings, lm_head won't exist in checkpoint
        if new_name == "lm_head.weight":
            new_name = "lm_head.lm_head.weight"

        weights[new_name] = weight

    return weights


def load_starcoder2_model(model_id: str):
    """Load StarCoder2 model, tokenizer, and config."""
    print(f"Loading {model_id}...")
    print("=" * 60)

    # Download
    print("\n1. Downloading model...")
    result = HFLoader.download(model_id)
    print(f"   Path: {result.model_path}")

    # Load weights first (needed to infer config)
    print("\n2. Loading weights...")
    weights = load_starcoder2_weights(result.model_path)
    print(f"   Loaded {len(weights)} tensors")

    # Load config
    print("\n3. Loading configuration...")
    config = load_starcoder2_config(result.model_path, weights)
    print(f"   Layers: {config.num_hidden_layers}, Hidden: {config.hidden_size}")
    print(f"   Heads: {config.num_attention_heads} attn, {config.num_key_value_heads} kv")

    # Load tokenizer
    print("\n4. Loading tokenizer...")
    tokenizer = HFLoader.load_tokenizer(result.model_path)
    print(f"   Vocab size: {len(tokenizer)}")

    # Create model
    print("\n5. Creating model...")
    model = StarCoder2ForCausalLM(config)

    # Apply weights using tree_unflatten
    print("\n6. Applying weights...")
    nested_weights = tree_unflatten(list(weights.items()))
    model.update(nested_weights)
    mx.eval(model.parameters())
    print("   Done!")

    print("\n" + "=" * 60)
    print("Model loaded successfully!")

    return model, tokenizer, config


def generate(
    model: StarCoder2ForCausalLM,
    tokenizer,
    prompt: str,
    config: GenerationConfig | None = None,
) -> tuple[str, float]:
    """Generate code and return (text, tokens_per_sec)."""
    if config is None:
        config = GenerationConfig()

    # Tokenize
    input_ids = tokenizer.encode(prompt, return_tensors="np")
    input_ids = mx.array(input_ids)
    input_length = input_ids.shape[1]

    # Stop tokens
    stop_tokens = [tokenizer.eos_token_id]
    if tokenizer.eos_token_id is None:
        stop_tokens = [StarCoder2Token.EOS.value]

    # Generate
    start = time.time()
    output_ids = model.generate(
        input_ids,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
        stop_tokens=stop_tokens,
    )
    mx.eval(output_ids)
    gen_time = time.time() - start

    # Decode
    new_tokens = output_ids[0, input_length:].tolist()
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    tokens_per_sec = len(new_tokens) / gen_time if gen_time > 0 else 0
    return text, tokens_per_sec


def interactive_loop(model: StarCoder2ForCausalLM, tokenizer, model_config: StarCoder2Config):
    """Interactive code completion loop."""
    print("\n" + "=" * 60)
    print("StarCoder2 Interactive Mode")
    print("=" * 60)
    print("Enter code to complete. Use 'quit' to exit, 'clear' to reset.")
    print("Tip: End with ':' or '(' for better completions")
    print("-" * 60)

    gen_config = GenerationConfig(max_new_tokens=256, temperature=0.2, top_k=50, top_p=0.95)

    while True:
        try:
            # Multi-line input
            print("\nCode (end with empty line):")
            lines = []
            while True:
                try:
                    line = input()
                    if line == "":
                        break
                    lines.append(line)
                except EOFError:
                    break

            user_input = "\n".join(lines)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        if user_input.lower() == "clear":
            print("Ready for new input.")
            continue

        # Check context length
        input_ids = tokenizer.encode(user_input, return_tensors="np")
        if input_ids.shape[1] > model_config.max_position_embeddings - 256:
            print("Warning: Input too long, please shorten.")
            continue

        # Generate
        print("\nCompleting...")
        response, tps = generate(model, tokenizer, user_input, gen_config)
        print("\n" + "-" * 40)
        print(user_input + response)
        print("-" * 40)
        print(f"[{tps:.1f} tok/s]")


def test_tiny():
    """Test tiny model config."""
    print("=" * 60)
    print("StarCoder2 Tiny Model Test")
    print("=" * 60)

    config = StarCoder2Config.tiny()
    model = StarCoder2ForCausalLM(config)

    input_ids = mx.array([[1, 2, 3, 4, 5]])
    output = model(input_ids)
    mx.eval(output.logits)
    print(f"Forward: OK (shape={output.logits.shape})")

    gen = model.generate(input_ids, max_new_tokens=5)
    mx.eval(gen)
    print(f"Generate: OK (shape={gen.shape})")

    print("\nSUCCESS!")


def main():
    parser = argparse.ArgumentParser(description="StarCoder2 Code Completion")
    parser.add_argument(
        "--model",
        choices=list(MODEL_ALIASES.keys()),
        default="starcoder2-3b",
        help="Model preset",
    )
    parser.add_argument("--model-id", help="Custom HuggingFace model ID")
    parser.add_argument("--test-tiny", action="store_true", help="Run tiny test")
    parser.add_argument(
        "--prompt",
        default="def fibonacci(n):",
        help="Code prompt to complete",
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--list", action="store_true", help="List models")
    args = parser.parse_args()

    if args.test_tiny:
        test_tiny()
        return

    if args.list:
        print("Available StarCoder2 models:\n")
        for alias, model in MODEL_ALIASES.items():
            print(f"  {alias:15} -> {model.value}")
        return

    # Get model ID
    model_id = args.model_id or MODEL_ALIASES[args.model].value

    # Load model
    model, tokenizer, config = load_starcoder2_model(model_id)

    if args.interactive:
        interactive_loop(model, tokenizer, config)
        return

    # Single prompt mode
    print("\n" + "=" * 60)
    print(f"Prompt: {args.prompt}")
    print("-" * 60)

    gen_config = GenerationConfig(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=50,
        top_p=0.95,
    )

    response, tps = generate(model, tokenizer, args.prompt, gen_config)

    print(f"{args.prompt}{response}")
    print("-" * 60)
    print(f"Speed: {tps:.1f} tokens/sec")
    print("=" * 60)


if __name__ == "__main__":
    main()
