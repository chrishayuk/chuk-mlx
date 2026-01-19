"""
TinyLlama Inference Example

Demonstrates loading and running inference with TinyLlama using
the models_v2 architecture. This proves out the Llama family implementation
with a real pretrained model from HuggingFace.

This example shows:
- Loading TinyLlama from HuggingFace Hub
- Weight conversion from HF format
- Text generation with KV-cache
- Introspection of the loaded model

Requirements:
    pip install huggingface_hub safetensors transformers

Run with:
    uv run python examples/models/llama/02_tinyllama_inference.py

Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- 1.1B parameters
- Llama 2 architecture
- Fine-tuned for chat
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx

from chuk_lazarus.models_v2 import (
    LlamaConfig,
    LlamaForCausalLM,
    count_parameters,
    print_introspection,
)


def download_model(model_id: str, cache_dir: str | None = None) -> Path:
    """Download model from HuggingFace Hub."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError("huggingface_hub not installed. Run: pip install huggingface_hub")

    print(f"Downloading {model_id}...")
    path = snapshot_download(
        model_id,
        cache_dir=cache_dir,
        allow_patterns=["*.json", "*.safetensors", "*.model"],
    )
    return Path(path)


def load_tokenizer(model_path: Path):
    """Load tokenizer from model path."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError("transformers not installed. Run: pip install transformers")

    return AutoTokenizer.from_pretrained(str(model_path))


def load_weights(
    model_path: Path, dtype: str = "float16", tie_word_embeddings: bool = False
) -> dict:
    """Load and convert weights from safetensors.

    Returns a nested dict structure that MLX model.update() expects.
    Note: MLX uses lists for layer collections, not dicts with numeric keys.

    Args:
        model_path: Path to model directory
        dtype: Target dtype for weights
        tie_word_embeddings: Whether lm_head weights are tied to embeddings
    """
    # Use MLX's native safetensors loading (handles bfloat16 properly)
    import mlx.core as mx

    # Find safetensors file(s)
    safetensor_files = list(model_path.glob("*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors files found in {model_path}")

    # Load all weight files using MLX's native loader
    all_weights = {}
    for sf_path in safetensor_files:
        print(f"  Loading {sf_path.name}...")
        weights = mx.load(str(sf_path))
        all_weights.update(weights)

    # Convert to target dtype
    dtype_map = {
        "float16": mx.float16,
        "float32": mx.float32,
        "bfloat16": mx.bfloat16,
    }
    target_dtype = dtype_map.get(dtype, mx.float16)

    # First pass: convert HF names and collect into a flat dict
    flat_weights: dict[str, mx.array] = {}
    for hf_name, weight in all_weights.items():
        our_name = _convert_weight_name(hf_name, tie_word_embeddings=tie_word_embeddings)
        if our_name is None:
            continue

        # Convert dtype
        if weight.dtype in (mx.float32, mx.float16, mx.bfloat16):
            weight = weight.astype(target_dtype)

        flat_weights[our_name] = weight

    # Second pass: build nested structure, converting numeric keys to list indices
    # First, find the maximum layer index
    max_layer_idx = -1
    for name in flat_weights:
        parts = name.split(".")
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                try:
                    layer_idx = int(parts[i + 1])
                    max_layer_idx = max(max_layer_idx, layer_idx)
                except ValueError:
                    pass

    # Build nested structure
    nested_weights: dict = {}
    for name, weight in flat_weights.items():
        parts = name.split(".")
        current = nested_weights

        i = 0
        while i < len(parts) - 1:
            part = parts[i]

            # Check if this is a list container (layers)
            if part == "layers":
                if part not in current:
                    # Initialize list with empty dicts for each layer
                    current[part] = [{} for _ in range(max_layer_idx + 1)]
                # Next part is the index
                layer_idx = int(parts[i + 1])
                current = current[part][layer_idx]
                i += 2  # Skip both "layers" and the index
            else:
                if part not in current:
                    current[part] = {}
                current = current[part]
                i += 1

        # Set the final value
        current[parts[-1]] = weight

    return nested_weights


def _convert_weight_name(hf_name: str, tie_word_embeddings: bool = False) -> str | None:
    """Convert HuggingFace weight name to our format."""
    import re

    # Direct mappings - note the structure differences:
    # HF: model.embed_tokens.weight -> Ours: model.embed_tokens.weight.weight
    # (because TokenEmbedding wraps nn.Embedding which has its own .weight)
    if hf_name == "model.embed_tokens.weight":
        return "model.embed_tokens.weight.weight"
    if hf_name == "model.norm.weight":
        return "model.norm.weight"
    # lm_head.weight -> lm_head.lm_head.weight (unless tied)
    if hf_name == "lm_head.weight":
        if tie_word_embeddings:
            return None  # Skip - will use tied embeddings
        return "lm_head.lm_head.weight"

    # Layer pattern: model.layers.{i}.{rest}
    layer_pattern = re.compile(r"model\.layers\.(\d+)\.(.*)")
    match = layer_pattern.match(hf_name)
    if match:
        layer_idx = match.group(1)
        rest = match.group(2)
        return f"model.layers.{layer_idx}.{rest}"

    # Unknown weight - skip
    return None


def format_chat_prompt(tokenizer, user_message: str) -> str:
    """Format a message for TinyLlama chat format."""
    # TinyLlama uses the ChatML format
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_message},
    ]

    # Use the tokenizer's chat template if available
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Fallback to manual ChatML format
    return (
        "<|system|>\nYou are a helpful assistant.</s>\n"
        f"<|user|>\n{user_message}</s>\n"
        "<|assistant|>\n"
    )


def generate_text(
    model: LlamaForCausalLM,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    verbose: bool = True,
) -> str:
    """Generate text from the model."""
    # Tokenize input
    input_ids = tokenizer.encode(prompt, return_tensors="np")
    input_ids = mx.array(input_ids)

    if verbose:
        print(f"  Input tokens: {input_ids.shape[1]}")

    # Get stop tokens
    stop_tokens = []
    if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
        stop_tokens.append(tokenizer.eos_token_id)

    # Generate
    start_time = time.time()
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        stop_tokens=stop_tokens,
    )
    mx.eval(output_ids)
    gen_time = time.time() - start_time

    # Decode only the new tokens
    new_tokens = output_ids[0, input_ids.shape[1] :]
    generated_text = tokenizer.decode(new_tokens.tolist(), skip_special_tokens=True)

    if verbose:
        tokens_generated = new_tokens.shape[0]
        tokens_per_sec = tokens_generated / gen_time
        print(f"  Generated {tokens_generated} tokens in {gen_time:.2f}s")
        print(f"  Speed: {tokens_per_sec:.1f} tokens/sec")

    return generated_text


def main():
    parser = argparse.ArgumentParser(description="TinyLlama Inference Example")
    parser.add_argument(
        "--model-id",
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--prompt",
        default="What is the capital of France?",
        help="Prompt to generate from",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Data type for weights",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Cache directory for downloaded models",
    )
    parser.add_argument(
        "--skip-introspection",
        action="store_true",
        help="Skip printing model introspection",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("TinyLlama Inference Example (models_v2)")
    print("=" * 60)

    # 1. Download model
    print("\n1. Downloading model from HuggingFace...")
    model_path = download_model(args.model_id, cache_dir=args.cache_dir)
    print(f"   Model path: {model_path}")

    # 2. Load config
    print("\n2. Loading configuration...")
    config_path = model_path / "config.json"
    with open(config_path) as f:
        config_data = json.load(f)

    config = LlamaConfig(**config_data)
    print(f"   Model type: {config.model_type}")
    print(f"   Vocab size: {config.vocab_size}")
    print(f"   Hidden size: {config.hidden_size}")
    print(f"   Layers: {config.num_hidden_layers}")
    print(f"   Attention heads: {config.num_attention_heads}")
    print(f"   KV heads: {config.num_key_value_heads}")

    # 3. Create model
    print("\n3. Creating LlamaForCausalLM...")
    model = LlamaForCausalLM(config)

    params = count_parameters(model)
    print(f"   {params.summary()}")

    # 4. Load weights
    print("\n4. Loading weights...")
    weights = load_weights(
        model_path,
        dtype=args.dtype,
        tie_word_embeddings=config.tie_word_embeddings,
    )

    def count_tensors(d) -> int:
        """Count tensors in nested dict/list structure."""
        count = 0
        if isinstance(d, dict):
            for v in d.values():
                count += count_tensors(v)
        elif isinstance(d, list):
            for item in d:
                count += count_tensors(item)
        else:
            count += 1
        return count

    print(f"   Loaded {count_tensors(weights)} weight tensors")

    # Update model with weights
    model.update(weights)
    mx.eval(model.parameters())
    print("   Weights loaded successfully!")

    # 5. Optional: print introspection
    if not args.skip_introspection:
        print("\n5. Model Introspection:")
        print_introspection(model, config)

    # 6. Load tokenizer
    print("\n6. Loading tokenizer...")
    tokenizer = load_tokenizer(model_path)
    print(f"   Vocab size: {len(tokenizer)}")

    # 7. Generate text
    print("\n7. Generating text...")
    print("-" * 40)

    # Format as chat prompt
    chat_prompt = format_chat_prompt(tokenizer, args.prompt)
    print(f"User: {args.prompt}\n")
    print("Assistant: ", end="", flush=True)

    response = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=chat_prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        verbose=False,
    )
    print(response)

    print("-" * 40)

    # 8. Interactive mode prompt
    print("\n" + "=" * 60)
    print("SUCCESS! TinyLlama inference works with models_v2")
    print("=" * 60)

    # Bonus: show generation stats
    print("\nGeneration stats (single prompt):")
    _ = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=chat_prompt,
        max_new_tokens=50,
        temperature=args.temperature,
        verbose=True,
    )


if __name__ == "__main__":
    main()
