#!/usr/bin/env python3
"""
Gemma Inference Example (Simplified)

Demonstrates the simplified API for Gemma text models.
For vision/multimodal, see examples/models/gemma/04_gemma3_vision_inference.py.

Supports:
- Gemma 3 270M (smallest, fastest)
- FunctionGemma 270M (function calling optimized)
- Gemma 3 1B (text-only)
- Gemma 3 4B (multimodal - text-only mode here)
- Gemma 3 12B
- Gemma 3 27B

Usage:
    # Default: Gemma 3 1B
    uv run python examples/inference/gemma_inference.py

    # Gemma 3 270M (smallest)
    uv run python examples/inference/gemma_inference.py --model gemma3-270m

    # FunctionGemma 270M (function calling)
    uv run python examples/inference/gemma_inference.py --model functiongemma

    # Gemma 3 4B
    uv run python examples/inference/gemma_inference.py --model gemma3-4b

    # Custom prompt
    uv run python examples/inference/gemma_inference.py --prompt "Write a haiku about MLX"

    # Interactive chat mode
    uv run python examples/inference/gemma_inference.py --chat

    # List available models
    uv run python examples/inference/gemma_inference.py --list
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
    ChatHistory,
    DType,
    GenerationConfig,
    HFLoader,
    Role,
)
from chuk_lazarus.models_v2.families.gemma import GemmaConfig, GemmaForCausalLM


class GemmaModel(str, Enum):
    """Available Gemma model presets."""

    # Gemma 3 270M (smallest, fastest)
    GEMMA3_270M = "mlx-community/gemma-3-270m-it-bf16"

    # FunctionGemma (270M - function calling optimized)
    FUNCTIONGEMMA_270M = "mlx-community/functiongemma-270m-it-bf16"

    # Gemma 3 family
    GEMMA3_1B = "mlx-community/gemma-3-1b-it-bf16"
    GEMMA3_4B = "mlx-community/gemma-3-4b-it-bf16"
    GEMMA3_12B = "mlx-community/gemma-3-12b-it-bf16"
    GEMMA3_27B = "mlx-community/gemma-3-27b-it-bf16"


MODEL_ALIASES = {
    # Gemma 3 270M
    "gemma3-270m": GemmaModel.GEMMA3_270M,
    # FunctionGemma
    "functiongemma": GemmaModel.FUNCTIONGEMMA_270M,
    "functiongemma-270m": GemmaModel.FUNCTIONGEMMA_270M,
    # Gemma 3
    "gemma3-1b": GemmaModel.GEMMA3_1B,
    "gemma3-4b": GemmaModel.GEMMA3_4B,
    "gemma3-12b": GemmaModel.GEMMA3_12B,
    "gemma3-27b": GemmaModel.GEMMA3_27B,
}


class GemmaToken(int, Enum):
    """Special token IDs for Gemma 3."""

    END_OF_TURN = 106


def load_gemma_config(model_path: Path, weights: dict | None = None) -> GemmaConfig:
    """Load and convert HuggingFace config to GemmaConfig.

    Handles both text-only (1B) and multimodal (4B+) config formats.
    """
    config_path = model_path / "config.json"
    with open(config_path) as f:
        hf_config = json.load(f)

    # Handle multimodal models with nested text_config
    if "text_config" in hf_config:
        text_config = hf_config["text_config"]
        model_type = text_config.get(
            "model_type", hf_config.get("model_type", "gemma3_text")
        )
    else:
        text_config = hf_config
        model_type = hf_config.get("model_type", "gemma3_text")

    hidden_size = text_config["hidden_size"]
    head_dim = text_config.get("head_dim", 256)

    # Try to get num_attention_heads from config, otherwise infer from weights
    num_attention_heads = text_config.get("num_attention_heads")
    num_key_value_heads = text_config.get("num_key_value_heads")

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

    return GemmaConfig(
        model_type=model_type,
        vocab_size=text_config.get("vocab_size", 262144),
        hidden_size=hidden_size,
        num_hidden_layers=text_config["num_hidden_layers"],
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        intermediate_size=text_config["intermediate_size"],
        head_dim=head_dim,
        query_pre_attn_scalar=text_config.get("query_pre_attn_scalar", 256.0),
        sliding_window=text_config.get("sliding_window", 512),
        sliding_window_pattern=text_config.get(
            "sliding_window_pattern", text_config.get("_sliding_window_pattern", 6)
        ),
        max_position_embeddings=text_config.get("max_position_embeddings", 32768),
        rope_theta=text_config.get("rope_theta", 1000000.0),
        rope_local_base_freq=text_config.get("rope_local_base_freq", 10000.0),
        rms_norm_eps=text_config.get("rms_norm_eps", 1e-6),
    )


def load_gemma_weights(model_path: Path, text_only: bool = True) -> dict:
    """Load weights from safetensors, optionally filtering vision weights."""
    weights = {}
    for sf_path in sorted(model_path.glob("*.safetensors")):
        print(f"  Loading {sf_path.name}...")
        file_weights = mx.load(str(sf_path))
        weights.update(file_weights)

    if text_only:
        # Filter vision weights for multimodal models
        text_weights = {}
        skipped = 0
        for k, v in weights.items():
            if any(prefix in k for prefix in ["vision_tower", "multi_modal_projector"]):
                skipped += 1
                continue
            # Rename language_model.* to model.* for compatibility
            if k.startswith("language_model."):
                k = k.replace("language_model.", "", 1)
            text_weights[k] = v
        if skipped > 0:
            print(f"  Filtered {skipped} vision tensors (text-only mode)")
        return text_weights

    return weights


def load_gemma_model(model_id: str):
    """Load Gemma model, tokenizer, and config."""
    print(f"Loading {model_id}...")
    print("=" * 60)

    # Download
    print("\n1. Downloading model...")
    result = HFLoader.download(model_id)
    print(f"   Path: {result.model_path}")

    # Load weights first (needed to infer config for multimodal models)
    print("\n2. Loading weights...")
    weights = load_gemma_weights(result.model_path)
    print(f"   Loaded {len(weights)} tensors")

    # Load config
    print("\n3. Loading configuration...")
    config = load_gemma_config(result.model_path, weights)
    print(f"   Layers: {config.num_hidden_layers}, Hidden: {config.hidden_size}")
    print(f"   Heads: {config.num_attention_heads} attn, {config.num_key_value_heads} kv")

    # Load tokenizer
    print("\n4. Loading tokenizer...")
    tokenizer = HFLoader.load_tokenizer(result.model_path)
    print(f"   Vocab size: {len(tokenizer)}")

    # Create model
    print("\n5. Creating model...")
    model = GemmaForCausalLM(config)

    # Apply weights using tree_unflatten (Gemma convention)
    print("\n6. Applying weights...")
    nested_weights = tree_unflatten(list(weights.items()))
    model.update(nested_weights)
    mx.eval(model.parameters())
    print("   Done!")

    print("\n" + "=" * 60)
    print("Model loaded successfully!")

    return model, tokenizer, config


def generate(
    model: GemmaForCausalLM,
    tokenizer,
    prompt: str,
    config: GenerationConfig | None = None,
) -> tuple[str, float]:
    """Generate text and return (text, tokens_per_sec)."""
    if config is None:
        config = GenerationConfig()

    # Tokenize
    input_ids = tokenizer.encode(prompt, return_tensors="np")
    input_ids = mx.array(input_ids)
    input_length = input_ids.shape[1]

    # Stop tokens
    stop_tokens = [tokenizer.eos_token_id, GemmaToken.END_OF_TURN.value]

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
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    tokens_per_sec = len(new_tokens) / gen_time if gen_time > 0 else 0
    return text, tokens_per_sec


def format_gemma_prompt(user_message: str, system_message: str | None = None) -> str:
    """Format prompt using Gemma 3 chat template."""
    if system_message:
        return f"<bos><start_of_turn>user\n{system_message}\n\n{user_message}<end_of_turn>\n<start_of_turn>model\n"
    return f"<bos><start_of_turn>user\n{user_message}<end_of_turn>\n<start_of_turn>model\n"


def chat_loop(model: GemmaForCausalLM, tokenizer, model_config: GemmaConfig):
    """Interactive chat loop."""
    print("\n" + "=" * 60)
    print("Gemma 3 Chat")
    print("=" * 60)
    print("Type 'quit' to exit, 'clear' to reset conversation")
    print("-" * 60)

    history = ChatHistory()
    gen_config = GenerationConfig(max_new_tokens=512, temperature=0.7, top_k=40, top_p=0.95)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        if user_input.lower() == "clear":
            history.clear()
            print("Conversation cleared.")
            continue

        # Add to history
        history.add_user(user_input)

        # Build conversation
        conv_text = ""
        for msg in history.messages:
            role = "user" if msg.role == Role.USER else "model"
            conv_text += f"<start_of_turn>{role}\n{msg.content}<end_of_turn>\n"

        prompt = f"<bos>{conv_text}<start_of_turn>model\n"

        # Check context length
        input_ids = tokenizer.encode(prompt, return_tensors="np")
        if input_ids.shape[1] > model_config.max_position_embeddings - 256:
            print("Warning: Context too long, truncating history...")
            history.messages = history.messages[-4:]
            continue

        # Generate
        print("\nGemma: ", end="", flush=True)
        response, tps = generate(model, tokenizer, prompt, gen_config)
        print(response)
        print(f"  [{tps:.1f} tok/s]")

        # Add response to history
        history.add_assistant(response)


def test_tiny():
    """Test tiny model config."""
    print("=" * 60)
    print("Gemma Tiny Model Test")
    print("=" * 60)

    config = GemmaConfig.tiny()
    model = GemmaForCausalLM(config)

    input_ids = mx.array([[1, 2, 3, 4, 5]])
    output = model(input_ids)
    mx.eval(output.logits)
    print(f"Forward: OK (shape={output.logits.shape})")

    gen = model.generate(input_ids, max_new_tokens=5)
    mx.eval(gen)
    print(f"Generate: OK (shape={gen.shape})")

    print("\nSUCCESS!")


def main():
    parser = argparse.ArgumentParser(description="Gemma 3 Inference (Simplified)")
    parser.add_argument(
        "--model",
        choices=list(MODEL_ALIASES.keys()),
        default="gemma3-1b",
        help="Model preset",
    )
    parser.add_argument("--model-id", help="Custom HuggingFace model ID")
    parser.add_argument("--test-tiny", action="store_true", help="Run tiny test")
    parser.add_argument(
        "--prompt",
        default="What is the capital of France?",
        help="User prompt",
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--chat", action="store_true", help="Interactive chat mode")
    parser.add_argument("--list", action="store_true", help="List models")
    args = parser.parse_args()

    if args.test_tiny:
        test_tiny()
        return

    if args.list:
        print("Available Gemma 3 models:\n")
        for alias, model in MODEL_ALIASES.items():
            print(f"  {alias:12} -> {model.value}")
        return

    # Get model ID
    model_id = args.model_id or MODEL_ALIASES[args.model].value

    # Load model
    model, tokenizer, config = load_gemma_model(model_id)

    if args.chat:
        chat_loop(model, tokenizer, config)
        return

    # Single prompt mode
    print("\n" + "=" * 60)
    print(f"User: {args.prompt}")
    print("-" * 60)

    prompt = format_gemma_prompt(args.prompt)
    gen_config = GenerationConfig(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=40,
        top_p=0.95,
    )

    response, tps = generate(model, tokenizer, prompt, gen_config)

    print(f"Gemma: {response}")
    print("-" * 60)
    print(f"Speed: {tps:.1f} tokens/sec")
    print("=" * 60)


if __name__ == "__main__":
    main()
