#!/usr/bin/env python3
"""
Gemma 3 Inference Example

This example demonstrates how to:
1. Load a pretrained Gemma 3 model from mlx-community
2. Apply chat templates for instruction-following
3. Generate text with various sampling strategies

Supported models from mlx-community (bf16 recommended):
- mlx-community/gemma-3-1b-it-bf16 (1B params, bfloat16)
- mlx-community/gemma-3-4b-it-bf16 (4B params, bfloat16)
- mlx-community/gemma-3-12b-it-bf16 (12B params, bfloat16)
- mlx-community/gemma-3-27b-it-bf16 (27B params, bfloat16)

Note: 4-bit quantized models require additional quantization support.
Use bf16 models for direct loading with this implementation.

Requirements:
    pip install huggingface_hub safetensors transformers

Usage:
    python 03_gemma3_inference.py
    python 03_gemma3_inference.py --model mlx-community/gemma-3-4b-it-bf16
    python 03_gemma3_inference.py --prompt "Explain quantum computing in simple terms"

References:
    - https://huggingface.co/blog/gemma3
    - https://huggingface.co/collections/mlx-community/gemma-3
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_unflatten

from chuk_lazarus.models_v2.families.gemma import (
    GemmaConfig,
    GemmaForCausalLM,
)

# Chat template for Gemma 3 instruction-tuned models
GEMMA3_CHAT_TEMPLATE = """<bos><start_of_turn>user
{prompt}<end_of_turn>
<start_of_turn>model
"""

# Multi-turn chat template
GEMMA3_MULTI_TURN_TEMPLATE = """<bos>{conversation}<start_of_turn>model
"""


def format_turn(role: str, content: str) -> str:
    """Format a single conversation turn."""
    return f"<start_of_turn>{role}\n{content}<end_of_turn>\n"


def download_model(model_id: str) -> Path:
    """Download model from HuggingFace Hub."""
    from huggingface_hub import snapshot_download

    print(f"Downloading {model_id}...")
    path = snapshot_download(
        repo_id=model_id,
        allow_patterns=["*.json", "*.safetensors"],
    )
    return Path(path)


def load_config_from_hf(model_path: Path, weights: dict | None = None) -> GemmaConfig:
    """Load and convert HuggingFace config to GemmaConfig.

    Args:
        model_path: Path to the model directory
        weights: Optional pre-loaded weights dict for inferring missing config values
    """
    config_path = model_path / "config.json"
    with open(config_path) as f:
        hf_config = json.load(f)

    # Handle multimodal models (4B+) which have nested text_config
    # vs text-only models (1B) which have flat config
    if "text_config" in hf_config:
        # Multimodal model - extract text config
        text_config = hf_config["text_config"]
        # Some fields may be in the top-level config
        model_type = text_config.get("model_type", hf_config.get("model_type", "gemma3_text"))
        is_multimodal = True
    else:
        # Text-only model - use config directly
        text_config = hf_config
        model_type = hf_config.get("model_type", "gemma3_text")
        is_multimodal = False

    hidden_size = text_config["hidden_size"]
    head_dim = text_config.get("head_dim", 256)

    # Try to get num_attention_heads from config, otherwise infer from weights
    num_attention_heads = text_config.get("num_attention_heads")
    num_key_value_heads = text_config.get("num_key_value_heads")

    if (num_attention_heads is None or num_key_value_heads is None) and weights is not None:
        # Infer from weight shapes
        # q_proj shape: (num_heads * head_dim, hidden_size)
        # k_proj shape: (num_kv_heads * head_dim, hidden_size)
        q_proj_key = None
        k_proj_key = None
        for k in weights.keys():
            if "layers.0" in k and "self_attn.q_proj.weight" in k:
                q_proj_key = k
            if "layers.0" in k and "self_attn.k_proj.weight" in k:
                k_proj_key = k
            if q_proj_key and k_proj_key:
                break

        if q_proj_key and num_attention_heads is None:
            q_proj_shape = weights[q_proj_key].shape
            num_attention_heads = q_proj_shape[0] // head_dim
            print(f"  Inferred num_attention_heads={num_attention_heads} from weights")

        if k_proj_key and num_key_value_heads is None:
            k_proj_shape = weights[k_proj_key].shape
            num_key_value_heads = k_proj_shape[0] // head_dim
            print(f"  Inferred num_key_value_heads={num_key_value_heads} from weights")

    # Fallback defaults if still None
    if num_attention_heads is None:
        num_attention_heads = hidden_size // head_dim
    if num_key_value_heads is None:
        num_key_value_heads = num_attention_heads

    # Map HuggingFace config fields to our config
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


def load_tokenizer(model_path: Path):
    """Load tokenizer from the model directory."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(model_path))


def load_weights(model_path: Path, text_only: bool = True) -> dict:
    """Load weights from safetensors files.

    Args:
        model_path: Path to the model directory
        text_only: If True, filter out vision-related weights (for multimodal models)
    """
    weights = {}
    for sf_path in model_path.glob("*.safetensors"):
        print(f"  Loading {sf_path.name}...")
        file_weights = mx.load(str(sf_path))
        weights.update(file_weights)

    if text_only:
        # Filter out vision-related weights for text-only inference
        # Multimodal models (4B+) have vision_tower, multi_modal_projector, etc.
        text_weights = {}
        skipped = 0
        for k, v in weights.items():
            if any(prefix in k for prefix in ["vision_tower", "multi_modal_projector"]):
                skipped += 1
                continue
            # Also rename language_model.* to model.* for compatibility
            if k.startswith("language_model."):
                k = k.replace("language_model.", "", 1)
            text_weights[k] = v
        if skipped > 0:
            print(f"  Filtered out {skipped} vision-related tensors (text-only mode)")
        return text_weights

    return weights


def load_gemma3_model(model_id: str) -> tuple[GemmaForCausalLM, any, GemmaConfig]:
    """
    Load a Gemma 3 model from HuggingFace Hub.

    Args:
        model_id: HuggingFace model ID (e.g., "mlx-community/gemma-3-1b-it-bf16")

    Returns:
        Tuple of (model, tokenizer, config)
    """
    # Download model
    model_path = download_model(model_id)

    # Load weights first (needed to infer config for multimodal models)
    print("Loading weights...")
    weights = load_weights(model_path)
    print(f"  Loaded {len(weights)} tensors")

    # Load config (pass weights to infer missing values for multimodal models)
    print("Loading config...")
    config = load_config_from_hf(model_path, weights=weights)
    print(f"  Model: {config.num_hidden_layers} layers, {config.hidden_size} hidden dim")
    print(f"  Heads: {config.num_attention_heads} attention, {config.num_key_value_heads} kv")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = load_tokenizer(model_path)

    # Create model
    print("Creating model...")
    model = GemmaForCausalLM(config)

    # Use tree_unflatten to convert flat weight keys to nested structure
    # This handles the conversion from "model.layers.0.self_attn.q_proj.weight"
    # to the nested dict format that model.update() expects
    nested_weights = tree_unflatten(list(weights.items()))
    model.update(nested_weights)

    return model, tokenizer, config


def generate_response(
    model: GemmaForCausalLM,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_k: int | None = 40,
    top_p: float | None = 0.95,
    system_prompt: str | None = None,
) -> str:
    """
    Generate a response to a prompt.

    Args:
        model: Gemma model
        tokenizer: Tokenizer
        prompt: User prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0 = greedy)
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling parameter
        system_prompt: Optional system prompt

    Returns:
        Generated response text
    """
    # Format prompt with chat template
    if system_prompt:
        formatted = f"<bos><start_of_turn>user\n{system_prompt}\n\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    else:
        formatted = GEMMA3_CHAT_TEMPLATE.format(prompt=prompt)

    # Tokenize
    input_ids = tokenizer.encode(formatted, return_tensors="np")
    input_ids = mx.array(input_ids)

    # Generate
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        stop_tokens=[tokenizer.eos_token_id, 106],  # 106 is <end_of_turn>
    )

    # Decode only the generated part
    generated_ids = output_ids[0, input_ids.shape[1] :].tolist()
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return response.strip()


def chat_loop(model: GemmaForCausalLM, tokenizer, config: GemmaConfig):
    """Interactive chat loop."""
    print("\n" + "=" * 60)
    print("Gemma 3 Chat")
    print("=" * 60)
    print("Type 'quit' to exit, 'clear' to reset conversation")
    print("-" * 60)

    conversation_history = []

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
            conversation_history = []
            print("Conversation cleared.")
            continue

        # Add user turn to history
        conversation_history.append({"role": "user", "content": user_input})

        # Build full conversation
        conv_text = ""
        for turn in conversation_history:
            conv_text += format_turn(turn["role"], turn["content"])

        # Format with template
        full_prompt = f"<bos>{conv_text}<start_of_turn>model\n"

        # Tokenize
        input_ids = tokenizer.encode(full_prompt, return_tensors="np")
        input_ids = mx.array(input_ids)

        # Check context length
        if input_ids.shape[1] > config.max_position_embeddings - 256:
            print("Warning: Context too long, truncating history...")
            conversation_history = conversation_history[-4:]
            continue

        # Generate
        print("\nGemma: ", end="", flush=True)
        output_ids = model.generate(
            input_ids,
            max_new_tokens=512,
            temperature=0.7,
            top_k=40,
            top_p=0.95,
            stop_tokens=[tokenizer.eos_token_id, 106],  # 106 is <end_of_turn>
        )

        # Decode response
        generated_ids = output_ids[0, input_ids.shape[1] :].tolist()
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        print(response)

        # Add assistant response to history
        conversation_history.append({"role": "model", "content": response})


def main():
    parser = argparse.ArgumentParser(description="Gemma 3 Inference Example")
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/gemma-3-1b-it-bf16",
        help="HuggingFace model ID (use bf16 models for direct loading)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt to generate (skips chat mode)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (0 = greedy)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Top-k sampling parameter",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling parameter",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Start interactive chat mode",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Gemma 3 Inference")
    print("=" * 60)
    print(f"Model: {args.model}")
    print("-" * 60)

    # Load model
    model, tokenizer, config = load_gemma3_model(args.model)
    print("\nModel loaded successfully!")

    # Evaluate model to ensure weights are loaded
    mx.eval(model.parameters())

    if args.chat:
        # Interactive chat mode
        chat_loop(model, tokenizer, config)
    elif args.prompt:
        # Single prompt mode
        print(f"\nPrompt: {args.prompt}")
        print("-" * 60)

        response = generate_response(
            model,
            tokenizer,
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )

        print(f"\nResponse:\n{response}")
    else:
        # Demo mode with example prompts
        print("\n" + "=" * 60)
        print("Running demo prompts...")
        print("=" * 60)

        demo_prompts = [
            "What is the capital of France?",
            "Write a haiku about programming.",
            "Explain what makes a good API design in 2-3 sentences.",
        ]

        for prompt in demo_prompts:
            print(f"\n{'=' * 60}")
            print(f"Prompt: {prompt}")
            print("-" * 60)

            response = generate_response(
                model,
                tokenizer,
                prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )

            print(f"Response:\n{response}")

        print("\n" + "=" * 60)
        print("Demo complete!")
        print("Run with --chat for interactive mode or --prompt for single queries")
        print("=" * 60)


if __name__ == "__main__":
    main()
