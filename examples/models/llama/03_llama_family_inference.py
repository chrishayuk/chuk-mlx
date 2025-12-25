"""
Llama Family Inference Example

Demonstrates loading and running inference with various Llama-family models
using the models_v2 architecture. Supports:
- TinyLlama (1.1B)
- SmolLM2 (135M, 360M, 1.7B) - Llama architecture, no auth required
- Llama 2 (7B, 13B) - requires HF access
- Llama 3 / 3.1 / 3.2 (1B, 3B, 8B) - requires HF access
- Mistral (7B) - uses Llama architecture with sliding window attention

Requirements:
    pip install huggingface_hub transformers

Run with:
    # TinyLlama (default - small and fast)
    uv run python examples/models/llama/03_llama_family_inference.py

    # SmolLM2 (no auth required)
    uv run python examples/models/llama/03_llama_family_inference.py --model smollm2-360m

    # Llama 2 7B (requires HF access)
    uv run python examples/models/llama/03_llama_family_inference.py --model llama2-7b

    # Llama 3.2 1B (requires HF access)
    uv run python examples/models/llama/03_llama_family_inference.py --model llama3.2-1b

    # List all available models
    uv run python examples/models/llama/03_llama_family_inference.py --list-models

    # Custom model from HuggingFace
    uv run python examples/models/llama/03_llama_family_inference.py --model-id "meta-llama/Llama-3.2-1B-Instruct"
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import mlx.core as mx

from chuk_lazarus.models_v2 import (
    LlamaConfig,
    LlamaForCausalLM,
    count_parameters,
    print_introspection,
)

# Preset model configurations
MODEL_PRESETS = {
    # TinyLlama - great for testing
    "tinyllama": {
        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "description": "TinyLlama 1.1B Chat - Fast testing model",
    },
    # Original Llama (requires HF access)
    "llama-7b": {
        "model_id": "meta-llama/Llama-2-7b-hf",
        "description": "Original Llama 7B - Base model (not chat)",
    },
    # Llama 2 (requires HF access)
    "llama2-7b": {
        "model_id": "meta-llama/Llama-2-7b-chat-hf",
        "description": "Llama 2 7B Chat - Original Llama 2",
    },
    "llama2-13b": {
        "model_id": "meta-llama/Llama-2-13b-chat-hf",
        "description": "Llama 2 13B Chat - Larger Llama 2",
    },
    # Llama 3.2 - smallest official Llama 3 models
    "llama3.2-1b": {
        "model_id": "meta-llama/Llama-3.2-1B-Instruct",
        "description": "Llama 3.2 1B Instruct - Smallest Llama 3",
    },
    "llama3.2-3b": {
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "description": "Llama 3.2 3B Instruct - Small but capable",
    },
    # Llama 3.1 8B
    "llama3.1-8b": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "description": "Llama 3.1 8B Instruct - Standard size",
    },
    # Mistral 7B - Llama-compatible architecture
    "mistral-7b": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "description": "Mistral 7B v0.3 - Sliding window attention",
    },
    # SmolLM2 - very small models
    "smollm2-135m": {
        "model_id": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "description": "SmolLM2 135M - Tiny model for quick testing",
    },
    "smollm2-360m": {
        "model_id": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "description": "SmolLM2 360M - Small but useful",
    },
    "smollm2-1.7b": {
        "model_id": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "description": "SmolLM2 1.7B - Good balance",
    },
}


def download_model(model_id: str, cache_dir: str | None = None) -> Path:
    """Download model from HuggingFace Hub.

    Prefers sharded safetensors (model-*.safetensors) over consolidated
    to avoid downloading duplicate weights.
    """
    try:
        from huggingface_hub import list_repo_files, snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub not installed. Run: pip install huggingface_hub"
        )

    print(f"Downloading {model_id}...")

    # Check what files are available to avoid downloading duplicates
    try:
        files = list_repo_files(model_id)
        has_sharded = any("model-0" in f and f.endswith(".safetensors") for f in files)
        has_consolidated = any(f == "consolidated.safetensors" for f in files)

        # Build ignore patterns to avoid downloading duplicate weights
        ignore_patterns = []
        if has_sharded and has_consolidated:
            # Prefer sharded files (smaller individual downloads, can resume)
            ignore_patterns.append("consolidated.safetensors")
            print("   (Skipping consolidated.safetensors - using sharded files)")
    except Exception:
        # If we can't list files, just download everything
        ignore_patterns = []

    path = snapshot_download(
        model_id,
        cache_dir=cache_dir,
        allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*"],
        ignore_patterns=ignore_patterns if ignore_patterns else None,
    )
    return Path(path)


def load_tokenizer(model_path: Path):
    """Load tokenizer from model path."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError("transformers not installed. Run: pip install transformers")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_weights(
    model_path: Path,
    dtype: str = "float16",
    tie_word_embeddings: bool = False,
) -> dict:
    """Load and convert weights from safetensors.

    Returns a nested dict structure that MLX model.update() expects.
    """
    # Find safetensors file(s)
    safetensor_files = sorted(model_path.glob("*.safetensors"))
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

        # Convert dtype (only for float types)
        # Note: bfloat16 -> float16 can cause precision loss, prefer bfloat16
        if weight.dtype in (mx.float32, mx.float16, mx.bfloat16):
            weight = weight.astype(target_dtype)

        flat_weights[our_name] = weight

    # Build nested structure
    return _build_nested_weights(flat_weights)


def _convert_weight_name(hf_name: str, tie_word_embeddings: bool = False) -> str | None:
    """Convert HuggingFace weight name to our format."""
    # Embeddings
    if hf_name == "model.embed_tokens.weight":
        return "model.embed_tokens.weight.weight"

    # Final layer norm
    if hf_name == "model.norm.weight":
        return "model.norm.weight"

    # LM head
    if hf_name == "lm_head.weight":
        if tie_word_embeddings:
            return None  # Skip - uses tied embeddings
        return "lm_head.lm_head.weight"

    # Layer pattern: model.layers.{i}.{component}
    layer_match = re.match(r"model\.layers\.(\d+)\.(.*)", hf_name)
    if layer_match:
        layer_idx = layer_match.group(1)
        rest = layer_match.group(2)

        # Skip rotary embeddings - we compute these dynamically
        if "rotary_emb" in rest:
            return None

        return f"model.layers.{layer_idx}.{rest}"

    # Unknown weight - skip with warning for debugging
    # print(f"  [skip] Unknown weight: {hf_name}")
    return None


def _build_nested_weights(flat_weights: dict[str, mx.array]) -> dict:
    """Build nested dict/list structure from flat weight names."""
    # Find maximum layer index
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
    nested: dict = {}
    for name, weight in flat_weights.items():
        parts = name.split(".")
        current = nested

        i = 0
        while i < len(parts) - 1:
            part = parts[i]

            if part == "layers":
                if part not in current:
                    current[part] = [{} for _ in range(max_layer_idx + 1)]
                layer_idx = int(parts[i + 1])
                current = current[part][layer_idx]
                i += 2
            else:
                if part not in current:
                    current[part] = {}
                current = current[part]
                i += 1

        current[parts[-1]] = weight

    return nested


def format_chat_prompt(tokenizer, user_message: str, system_message: str | None = None) -> str:
    """Format a message using the tokenizer's chat template."""
    messages = []

    if system_message:
        messages.append({"role": "system", "content": system_message})

    messages.append({"role": "user", "content": user_message})

    # Use the tokenizer's chat template if available
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass  # Fall back to simple format

    # Simple fallback format
    prompt = ""
    if system_message:
        prompt += f"System: {system_message}\n\n"
    prompt += f"User: {user_message}\n\nAssistant:"
    return prompt


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
        if isinstance(tokenizer.eos_token_id, list):
            stop_tokens.extend(tokenizer.eos_token_id)
        else:
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
    new_tokens = output_ids[0, input_ids.shape[1]:]
    generated_text = tokenizer.decode(new_tokens.tolist(), skip_special_tokens=True)

    if verbose:
        tokens_generated = new_tokens.shape[0]
        tokens_per_sec = tokens_generated / gen_time if gen_time > 0 else 0
        print(f"  Generated {tokens_generated} tokens in {gen_time:.2f}s")
        print(f"  Speed: {tokens_per_sec:.1f} tokens/sec")

    return generated_text


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


def main():
    parser = argparse.ArgumentParser(
        description="Llama Family Inference Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available model presets:
  tinyllama     - TinyLlama 1.1B Chat (fast, good for testing)
  smollm2-135m  - SmolLM2 135M (very fast, limited quality)
  smollm2-360m  - SmolLM2 360M (fast, reasonable quality)
  smollm2-1.7b  - SmolLM2 1.7B (balanced)
  llama3.2-1b   - Llama 3.2 1B Instruct (requires access)
  llama3.2-3b   - Llama 3.2 3B Instruct (requires access)
  llama3.1-8b   - Llama 3.1 8B Instruct (requires access, ~16GB RAM)
  mistral-7b    - Mistral 7B v0.3 (requires ~14GB RAM)

Examples:
  # Quick test with TinyLlama
  python 03_llama_family_inference.py --model tinyllama --prompt "Hello!"

  # Test with SmolLM2 (no auth needed)
  python 03_llama_family_inference.py --model smollm2-360m

  # Custom HuggingFace model
  python 03_llama_family_inference.py --model-id "organization/model-name"
""",
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_PRESETS.keys()),
        default="tinyllama",
        help="Model preset to use",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Custom HuggingFace model ID (overrides --model)",
    )
    parser.add_argument(
        "--prompt",
        default="What is the capital of France?",
        help="Prompt to generate from",
    )
    parser.add_argument(
        "--system",
        default="You are a helpful assistant.",
        help="System message for chat",
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
        default="bfloat16",
        choices=["float16", "float32", "bfloat16"],
        help="Data type for weights (bfloat16 recommended for stability)",
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
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available model presets and exit",
    )

    args = parser.parse_args()

    # List models mode
    if args.list_models:
        print("Available model presets:\n")
        for name, info in MODEL_PRESETS.items():
            print(f"  {name:15} - {info['description']}")
            print(f"                   {info['model_id']}")
            print()
        return

    # Determine model ID
    if args.model_id:
        model_id = args.model_id
        model_name = model_id.split("/")[-1]
    else:
        preset = MODEL_PRESETS[args.model]
        model_id = preset["model_id"]
        model_name = args.model

    print("=" * 60)
    print(f"Llama Family Inference: {model_name}")
    print("=" * 60)

    # 1. Download model
    print("\n1. Downloading model from HuggingFace...")
    try:
        model_path = download_model(model_id, cache_dir=args.cache_dir)
    except Exception as e:
        print(f"   Error: {e}")
        print("\n   Note: Some models require HuggingFace authentication.")
        print("   Run: huggingface-cli login")
        return
    print(f"   Model path: {model_path}")

    # 2. Load config
    print("\n2. Loading configuration...")
    config_path = model_path / "config.json"
    with open(config_path) as f:
        config_data = json.load(f)

    # Handle list-valued token IDs (Llama 3.2 uses lists for eos_token_id)
    for key in ["eos_token_id", "bos_token_id", "pad_token_id"]:
        if key in config_data and isinstance(config_data[key], list):
            # Use first token ID if it's a list
            config_data[key] = config_data[key][0] if config_data[key] else None

    config = LlamaConfig(**config_data)
    print(f"   Model type: {config.model_type}")
    print(f"   Vocab size: {config.vocab_size}")
    print(f"   Hidden size: {config.hidden_size}")
    print(f"   Layers: {config.num_hidden_layers}")
    print(f"   Attention heads: {config.num_attention_heads}")
    print(f"   KV heads: {config.num_key_value_heads}")
    if config.sliding_window:
        print(f"   Sliding window: {config.sliding_window}")

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
    print(f"   Loaded {count_tensors(weights)} weight tensors")

    # Update model with weights
    model.update(weights)
    mx.eval(model.parameters())
    print("   Weights loaded successfully!")

    # 5. Optional introspection
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

    chat_prompt = format_chat_prompt(tokenizer, args.prompt, args.system)
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

    # Success message
    print("\n" + "=" * 60)
    print(f"SUCCESS! {model_name} inference works with models_v2")
    print("=" * 60)

    # Generation stats
    print("\nGeneration stats:")
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
