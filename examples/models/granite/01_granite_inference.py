"""
IBM Granite Inference Example

Demonstrates loading and running inference with IBM Granite models
using the models_v2 architecture. Supports:

Granite 3.x (Dense Transformer):
- Granite 3.0 8B
- Granite 3.1 2B/8B

Granite 4.x (Hybrid Mamba-2/Transformer):
- Granite 4.0 Micro (3B dense)
- Granite 4.0 Tiny (7B total, 1B active) - MoE
- Granite 4.0 Small (32B total, 9B active) - MoE

Requirements:
    pip install huggingface_hub transformers

Run with:
    # Test tiny config (no download needed)
    uv run python examples/models/granite/01_granite_inference.py --test-tiny

    # Granite 3.1 2B (recommended for testing)
    uv run python examples/models/granite/01_granite_inference.py --model granite-3.1-2b

    # Granite 4.0 Micro
    uv run python examples/models/granite/01_granite_inference.py --model granite-4.0-micro

    # List available models
    uv run python examples/models/granite/01_granite_inference.py --list-models
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import mlx.core as mx

from chuk_lazarus.models_v2 import (
    GraniteConfig,
    GraniteForCausalLM,
    GraniteHybridConfig,
    GraniteHybridForCausalLM,
    count_parameters,
)

# Model presets
MODEL_PRESETS = {
    # Granite 3.x (Dense Transformer)
    "granite-3.0-8b": {
        "model_id": "ibm-granite/granite-3.0-8b-instruct",
        "description": "Granite 3.0 8B - Dense transformer",
        "model_type": "granite",
    },
    "granite-3.1-2b": {
        "model_id": "ibm-granite/granite-3.1-2b-instruct",
        "description": "Granite 3.1 2B - Long context (128K)",
        "model_type": "granite",
    },
    "granite-3.1-8b": {
        "model_id": "ibm-granite/granite-3.1-8b-instruct",
        "description": "Granite 3.1 8B - Long context (128K)",
        "model_type": "granite",
    },
    "granite-3.3-2b": {
        "model_id": "ibm-granite/granite-3.3-2b-instruct",
        "description": "Granite 3.3 2B - Latest 3.x",
        "model_type": "granite",
    },
    "granite-3.3-8b": {
        "model_id": "ibm-granite/granite-3.3-8b-instruct",
        "description": "Granite 3.3 8B - Latest 3.x",
        "model_type": "granite",
    },
    # Granite 4.x (Hybrid Mamba-2/Transformer)
    "granite-4.0-micro": {
        "model_id": "ibm-granite/granite-4.0-micro",
        "description": "Granite 4.0 Micro (3B) - Dense hybrid",
        "model_type": "granitemoehybrid",
    },
    "granite-4.0-tiny": {
        "model_id": "ibm-granite/granite-4.0-tiny-preview",
        "description": "Granite 4.0 Tiny (7B/1B) - MoE hybrid",
        "model_type": "granitemoehybrid",
    },
}


def download_model(model_id: str, cache_dir: str | None = None) -> Path:
    """Download model from HuggingFace Hub."""
    try:
        from huggingface_hub import list_repo_files, snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub not installed. Run: pip install huggingface_hub"
        )

    print(f"Downloading {model_id}...")

    try:
        files = list_repo_files(model_id)
        has_sharded = any("model-0" in f and f.endswith(".safetensors") for f in files)
        has_consolidated = any(f == "consolidated.safetensors" for f in files)

        ignore_patterns = []
        if has_sharded and has_consolidated:
            ignore_patterns.append("consolidated.safetensors")
            print("   (Skipping consolidated.safetensors)")
    except Exception:
        ignore_patterns = []

    path = snapshot_download(
        model_id,
        cache_dir=cache_dir,
        allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*"],
        ignore_patterns=ignore_patterns if ignore_patterns else None,
    )
    return Path(path)


def load_tokenizer(model_path: Path):
    """Load tokenizer."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError("transformers not installed. Run: pip install transformers")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_weights(model_path: Path, config, dtype: str = "bfloat16") -> dict:
    """Load and convert weights."""
    safetensor_files = sorted(model_path.glob("*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors in {model_path}")

    all_weights = {}
    for sf_path in safetensor_files:
        print(f"  Loading {sf_path.name}...")
        weights = mx.load(str(sf_path))
        all_weights.update(weights)

    dtype_map = {
        "float16": mx.float16,
        "float32": mx.float32,
        "bfloat16": mx.bfloat16,
    }
    target_dtype = dtype_map.get(dtype, mx.bfloat16)

    # Convert names
    flat_weights: dict[str, mx.array] = {}
    for hf_name, weight in all_weights.items():
        our_name = _convert_weight_name(hf_name, config)
        if our_name is None:
            continue
        if weight.dtype in (mx.float32, mx.float16, mx.bfloat16):
            weight = weight.astype(target_dtype)
        flat_weights[our_name] = weight

    return _build_nested_weights(flat_weights, config)


def _convert_weight_name(hf_name: str, config) -> str | None:
    """Convert HF weight name to our format."""
    # Embeddings
    if hf_name == "model.embed_tokens.weight":
        return "model.embed_tokens.weight.weight"

    # Final norm
    if hf_name == "model.norm.weight":
        return "model.norm.weight"

    # LM head
    if hf_name == "lm_head.weight":
        if config.tie_word_embeddings:
            return None
        return "lm_head.lm_head.weight"

    # Layer pattern
    layer_match = re.match(r"model\.layers\.(\d+)\.(.*)", hf_name)
    if layer_match:
        layer_idx = layer_match.group(1)
        rest = layer_match.group(2)

        # Skip rotary embeddings
        if "rotary_emb" in rest:
            return None

        return f"model.layers.{layer_idx}.{rest}"

    return None


def _build_nested_weights(flat_weights: dict[str, mx.array], config) -> dict:
    """Build nested structure."""
    max_layer_idx = -1
    for name in flat_weights:
        parts = name.split(".")
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                try:
                    max_layer_idx = max(max_layer_idx, int(parts[i + 1]))
                except ValueError:
                    pass

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
    """Format using chat template."""
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": user_message})

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass

    prompt = ""
    if system_message:
        prompt += f"System: {system_message}\n\n"
    prompt += f"User: {user_message}\n\nAssistant:"
    return prompt


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 100, temperature: float = 0.7, verbose: bool = True) -> str:
    """Generate text."""
    input_ids = tokenizer.encode(prompt, return_tensors="np")
    input_ids = mx.array(input_ids)

    if verbose:
        print(f"  Input tokens: {input_ids.shape[1]}")

    stop_tokens = []
    if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
        if isinstance(tokenizer.eos_token_id, list):
            stop_tokens.extend(tokenizer.eos_token_id)
        else:
            stop_tokens.append(tokenizer.eos_token_id)

    start_time = time.time()
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        stop_tokens=stop_tokens,
    )
    mx.eval(output_ids)
    gen_time = time.time() - start_time

    new_tokens = output_ids[0, input_ids.shape[1]:]
    generated_text = tokenizer.decode(new_tokens.tolist(), skip_special_tokens=True)

    if verbose:
        tokens_generated = new_tokens.shape[0]
        tokens_per_sec = tokens_generated / gen_time if gen_time > 0 else 0
        print(f"  Generated {tokens_generated} tokens in {gen_time:.2f}s")
        print(f"  Speed: {tokens_per_sec:.1f} tokens/sec")

    return generated_text


def test_tiny_models():
    """Test tiny configs without downloading."""
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

    gen3 = model3.generate(input_ids, max_new_tokens=5)
    mx.eval(gen3)
    print(f"   Generate: OK (shape={gen3.shape})")

    # Test Granite 4.x dense
    print("\n2. Testing Granite 4.x Hybrid (dense)...")
    config4 = GraniteHybridConfig.tiny()
    model4 = GraniteHybridForCausalLM(config4)
    params4 = count_parameters(model4)
    print(f"   {params4.summary()}")
    print(f"   Layers: {config4.num_mamba_layers} Mamba, {config4.num_attention_layers} Attention")

    output4 = model4(input_ids)
    mx.eval(output4.logits)
    print(f"   Forward: OK (shape={output4.logits.shape})")

    gen4 = model4.generate(input_ids, max_new_tokens=5)
    mx.eval(gen4)
    print(f"   Generate: OK (shape={gen4.shape})")

    # Test Granite 4.x MoE
    print("\n3. Testing Granite 4.x Hybrid + MoE...")
    config4_moe = GraniteHybridConfig.tiny_moe()
    model4_moe = GraniteHybridForCausalLM(config4_moe)
    params4_moe = count_parameters(model4_moe)
    print(f"   {params4_moe.summary()}")
    print(f"   Experts: {config4_moe.num_local_experts} total, {config4_moe.num_experts_per_tok} active")

    output4_moe = model4_moe(input_ids)
    mx.eval(output4_moe.logits)
    print(f"   Forward: OK (shape={output4_moe.logits.shape})")

    print("\n" + "=" * 60)
    print("SUCCESS! All Granite tiny tests passed.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="IBM Granite Inference Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available models:
  Granite 3.x (Dense):
    granite-3.1-2b  - 2B params, 128K context
    granite-3.1-8b  - 8B params, 128K context
    granite-3.3-2b  - Latest 3.x, 2B
    granite-3.3-8b  - Latest 3.x, 8B

  Granite 4.x (Hybrid Mamba-2/Transformer):
    granite-4.0-micro - 3B dense hybrid
    granite-4.0-tiny  - 7B total (1B active) MoE

Examples:
  # Test tiny (no download)
  python 01_granite_inference.py --test-tiny

  # Granite 3.1 2B
  python 01_granite_inference.py --model granite-3.1-2b
""",
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_PRESETS.keys()),
        default="granite-3.1-2b",
        help="Model preset",
    )
    parser.add_argument("--model-id", default=None, help="Custom HuggingFace model ID")
    parser.add_argument("--test-tiny", action="store_true", help="Run tiny tests")
    parser.add_argument("--prompt", default="What is the capital of France?", help="Prompt")
    parser.add_argument("--system", default="You are a helpful assistant.", help="System message")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--list-models", action="store_true", help="List models and exit")

    args = parser.parse_args()

    if args.list_models:
        print("Available Granite models:\n")
        for name, info in MODEL_PRESETS.items():
            print(f"  {name:20} - {info['description']}")
            print(f"                       {info['model_id']}")
            print()
        return

    if args.test_tiny:
        test_tiny_models()
        return

    # Get model info
    if args.model_id:
        model_id = args.model_id
        model_name = model_id.split("/")[-1]
        model_type = "granite"  # Default
    else:
        preset = MODEL_PRESETS[args.model]
        model_id = preset["model_id"]
        model_name = args.model
        model_type = preset["model_type"]

    print("=" * 60)
    print(f"Granite Inference: {model_name}")
    print("=" * 60)

    # Download
    print("\n1. Downloading model...")
    try:
        model_path = download_model(model_id, cache_dir=args.cache_dir)
    except Exception as e:
        print(f"   Error: {e}")
        return
    print(f"   Path: {model_path}")

    # Load config
    print("\n2. Loading configuration...")
    config_path = model_path / "config.json"
    with open(config_path) as f:
        config_data = json.load(f)

    # Handle token IDs
    for key in ["eos_token_id", "bos_token_id", "pad_token_id"]:
        if key in config_data and isinstance(config_data[key], list):
            config_data[key] = config_data[key][0] if config_data[key] else None

    # Create config based on type
    actual_model_type = config_data.get("model_type", model_type)
    if actual_model_type == "granitemoehybrid":
        config = GraniteHybridConfig(**config_data)
        print(f"   Type: Granite 4.x Hybrid")
        print(f"   Mamba layers: {config.num_mamba_layers}, Attention: {config.num_attention_layers}")
        if config.is_moe:
            print(f"   MoE: {config.num_local_experts} experts, {config.num_experts_per_tok} active")
    else:
        config = GraniteConfig(**config_data)
        print(f"   Type: Granite 3.x Dense")

    print(f"   Hidden: {config.hidden_size}, Layers: {config.num_hidden_layers}")

    # Create model
    print("\n3. Creating model...")
    if actual_model_type == "granitemoehybrid":
        model = GraniteHybridForCausalLM(config)
    else:
        model = GraniteForCausalLM(config)
    params = count_parameters(model)
    print(f"   {params.summary()}")

    # Load weights
    print("\n4. Loading weights...")
    weights = load_weights(model_path, config, dtype=args.dtype)
    model.update(weights)
    mx.eval(model.parameters())
    print("   Weights loaded!")

    # Load tokenizer
    print("\n5. Loading tokenizer...")
    tokenizer = load_tokenizer(model_path)
    print(f"   Vocab size: {len(tokenizer)}")

    # Generate
    print("\n6. Generating...")
    print("-" * 40)
    prompt = format_chat_prompt(tokenizer, args.prompt, args.system)
    print(f"User: {args.prompt}\n")
    print("Assistant: ", end="", flush=True)

    response = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        verbose=False,
    )
    print(response)
    print("-" * 40)

    print("\n" + "=" * 60)
    print("SUCCESS! Granite inference complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
