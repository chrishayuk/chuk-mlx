#!/usr/bin/env python3
"""
Jamba Inference Example

Demonstrates inference with AI21 Labs' Jamba hybrid Mamba-Transformer MoE models.

Jamba combines:
- Mamba SSM layers (7 out of 8) for O(n) complexity
- Attention layers (1 out of 8) for precise recall
- MoE (every 2nd layer) for scaling capacity
- 256K context window

Supports:
- Jamba v0.1 (52B total, ~12B active)
- Jamba 1.5 Mini (52B total, 12B active)
- Jamba 1.5 Large (398B total, 94B active)

Usage:
    # Default: Jamba v0.1
    uv run python examples/inference/jamba_inference.py

    # Custom prompt
    uv run python examples/inference/jamba_inference.py --prompt "Explain quantum computing"

    # Interactive chat mode
    uv run python examples/inference/jamba_inference.py --chat

    # List available models
    uv run python examples/inference/jamba_inference.py --list

    # Test with tiny model (no download)
    uv run python examples/inference/jamba_inference.py --test-tiny

Note: Jamba models are large (52B+ parameters) and require significant memory.
      The full models need authentication from AI21 Labs.
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
    GenerationConfig,
    HFLoader,
    Role,
)
from chuk_lazarus.models_v2.families.jamba import JambaConfig, JambaForCausalLM


class JambaModel(str, Enum):
    """Available Jamba model presets."""

    JAMBA_V0_1 = "ai21labs/Jamba-v0.1"
    JAMBA_1_5_MINI = "ai21labs/AI21-Jamba-1.5-Mini"
    JAMBA_1_5_LARGE = "ai21labs/AI21-Jamba-1.5-Large"


MODEL_ALIASES = {
    "jamba": JambaModel.JAMBA_V0_1,
    "jamba-v0.1": JambaModel.JAMBA_V0_1,
    "jamba-1.5-mini": JambaModel.JAMBA_1_5_MINI,
    "jamba-mini": JambaModel.JAMBA_1_5_MINI,
    "jamba-1.5-large": JambaModel.JAMBA_1_5_LARGE,
    "jamba-large": JambaModel.JAMBA_1_5_LARGE,
}


def load_jamba_config(model_path: Path, weights: dict | None = None) -> JambaConfig:
    """Load and convert HuggingFace config to JambaConfig."""
    config_path = model_path / "config.json"
    with open(config_path) as f:
        hf_config = json.load(f)

    return JambaConfig(
        model_type=hf_config.get("model_type", "jamba"),
        vocab_size=hf_config.get("vocab_size", 65536),
        hidden_size=hf_config["hidden_size"],
        num_hidden_layers=hf_config["num_hidden_layers"],
        num_attention_heads=hf_config["num_attention_heads"],
        num_key_value_heads=hf_config.get("num_key_value_heads", hf_config["num_attention_heads"]),
        intermediate_size=hf_config["intermediate_size"],
        max_position_embeddings=hf_config.get("max_position_embeddings", 262144),
        rms_norm_eps=hf_config.get("rms_norm_eps", 1e-6),
        # Hybrid pattern
        attn_layer_period=hf_config.get("attn_layer_period", 8),
        attn_layer_offset=hf_config.get("attn_layer_offset", 4),
        expert_layer_period=hf_config.get("expert_layer_period", 2),
        expert_layer_offset=hf_config.get("expert_layer_offset", 1),
        # MoE
        num_experts=hf_config.get("num_experts", 16),
        num_experts_per_tok=hf_config.get("num_experts_per_tok", 2),
        # Mamba
        mamba_d_state=hf_config.get("mamba_d_state", 16),
        mamba_d_conv=hf_config.get("mamba_d_conv", 4),
        mamba_expand=hf_config.get("mamba_expand", 2),
        mamba_dt_rank=hf_config.get("mamba_dt_rank", 256),
        mamba_conv_bias=hf_config.get("mamba_conv_bias", True),
        mamba_proj_bias=hf_config.get("mamba_proj_bias", False),
        tie_word_embeddings=hf_config.get("tie_word_embeddings", False),
    )


def load_jamba_weights(model_path: Path) -> dict:
    """Load weights from safetensors."""
    weights = {}
    for sf_path in sorted(model_path.glob("*.safetensors")):
        print(f"  Loading {sf_path.name}...")
        file_weights = mx.load(str(sf_path))
        weights.update(file_weights)
    return weights


def load_jamba_model(model_id: str):
    """Load Jamba model, tokenizer, and config."""
    print(f"Loading {model_id}...")
    print("=" * 60)
    print("Note: Jamba models are large and require significant memory.")
    print("=" * 60)

    # Download
    print("\n1. Downloading model...")
    result = HFLoader.download(model_id)
    print(f"   Path: {result.model_path}")

    # Load weights
    print("\n2. Loading weights...")
    weights = load_jamba_weights(result.model_path)
    print(f"   Loaded {len(weights)} tensors")

    # Load config
    print("\n3. Loading configuration...")
    config = load_jamba_config(result.model_path, weights)
    print(f"   Layers: {config.num_hidden_layers}, Hidden: {config.hidden_size}")
    print(f"   Attn period: {config.attn_layer_period}, MoE period: {config.expert_layer_period}")
    print(f"   Experts: {config.num_experts} total, {config.num_experts_per_tok} active")

    # Load tokenizer
    print("\n4. Loading tokenizer...")
    tokenizer = HFLoader.load_tokenizer(result.model_path)
    print(f"   Vocab size: {len(tokenizer)}")

    # Create model
    print("\n5. Creating model...")
    model = JambaForCausalLM(config)

    # Apply weights
    print("\n6. Applying weights...")
    nested_weights = tree_unflatten(list(weights.items()))
    model.update(nested_weights)
    mx.eval(model.parameters())
    print("   Done!")

    print("\n" + "=" * 60)
    print("Model loaded successfully!")

    return model, tokenizer, config


def generate(
    model: JambaForCausalLM,
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
    stop_tokens = [tokenizer.eos_token_id]

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


def chat_loop(model: JambaForCausalLM, tokenizer, model_config: JambaConfig):
    """Interactive chat loop."""
    print("\n" + "=" * 60)
    print("Jamba Chat")
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

        # Build conversation using simple format
        conv_text = ""
        for msg in history.messages:
            role = "User" if msg.role == Role.USER else "Assistant"
            conv_text += f"{role}: {msg.content}\n"
        conv_text += "Assistant:"

        # Check context length
        input_ids = tokenizer.encode(conv_text, return_tensors="np")
        if input_ids.shape[1] > model_config.max_position_embeddings - 512:
            print("Warning: Context too long, truncating history...")
            history.messages = history.messages[-4:]
            continue

        # Generate
        print("\nJamba: ", end="", flush=True)
        response, tps = generate(model, tokenizer, conv_text, gen_config)
        print(response)
        print(f"  [{tps:.1f} tok/s]")

        # Add response to history
        history.add_assistant(response)


def test_tiny():
    """Test tiny model config."""
    print("=" * 60)
    print("Jamba Tiny Model Test")
    print("=" * 60)

    config = JambaConfig.tiny()
    print(f"Config: layers={config.num_hidden_layers}, hidden={config.hidden_size}")
    print(f"Attn period={config.attn_layer_period}, MoE period={config.expert_layer_period}")
    print(f"Experts: {config.num_experts} total, {config.num_experts_per_tok} active")

    # Show layer pattern
    print("\nLayer pattern:")
    for i in range(config.num_hidden_layers):
        layer_type = "ATT" if config.is_attention_layer(i) else "MAM"
        ffn_type = "MoE" if config.is_moe_layer(i) else "FFN"
        print(f"  Layer {i}: {layer_type} + {ffn_type}")

    model = JambaForCausalLM(config)
    print("\nModel created successfully!")

    input_ids = mx.array([[1, 2, 3, 4, 5]])
    output = model(input_ids)
    mx.eval(output.logits)
    print(f"Forward: OK (shape={output.logits.shape})")

    gen = model.generate(input_ids, max_new_tokens=5)
    mx.eval(gen)
    print(f"Generate: OK (shape={gen.shape})")

    print("\nSUCCESS!")


def main():
    parser = argparse.ArgumentParser(description="Jamba Inference")
    parser.add_argument(
        "--model",
        choices=list(MODEL_ALIASES.keys()),
        default="jamba",
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
        print("Available Jamba models:\n")
        for alias, model in MODEL_ALIASES.items():
            print(f"  {alias:18} -> {model.value}")
        print("\nNote: These models require AI21 Labs authentication on HuggingFace.")
        print("Run: huggingface-cli login")
        return

    # Get model ID
    model_id = args.model_id or MODEL_ALIASES[args.model].value

    # Load model
    model, tokenizer, config = load_jamba_model(model_id)

    if args.chat:
        chat_loop(model, tokenizer, config)
        return

    # Single prompt mode
    print("\n" + "=" * 60)
    print(f"User: {args.prompt}")
    print("-" * 60)

    gen_config = GenerationConfig(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=40,
        top_p=0.95,
    )

    response, tps = generate(model, tokenizer, args.prompt, gen_config)

    print(f"Jamba: {response}")
    print("-" * 60)
    print(f"Speed: {tps:.1f} tokens/sec")
    print("=" * 60)


if __name__ == "__main__":
    main()
