"""
Llama 4 Inference Example

Demonstrates loading and running inference with Llama 4 models
using the models_v2 architecture. Supports:
- Llama 4 Scout (17B active / 109B total)
- Llama 4 Maverick (17B active / 400B total)

Key features:
- MoE (Mixture of Experts) with shared expert
- iRoPE (interleaved RoPE and NoPE layers)
- QK normalization

Requirements:
    pip install huggingface_hub transformers

Note: Llama 4 models are large and require significant memory:
- Scout: ~27GB for BF16 inference
- Maverick: ~100GB for BF16 inference

Run with:
    # Test with tiny config (no download needed)
    uv run python examples/models/llama4/01_llama4_inference.py --test-tiny

    # Llama 4 Scout (requires HF access and ~27GB RAM)
    uv run python examples/models/llama4/01_llama4_inference.py --model llama4-scout

    # Custom model
    uv run python examples/models/llama4/01_llama4_inference.py --model-id "meta-llama/Llama-4-Scout-17B-16E-Instruct"
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import mlx.core as mx

from chuk_lazarus.models_v2 import (
    Llama4ForCausalLM,
    Llama4TextConfig,
    count_parameters,
    print_introspection,
)

# Preset model configurations
MODEL_PRESETS = {
    "llama4-scout": {
        "model_id": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "description": "Llama 4 Scout - 17B active / 109B total MoE",
    },
    "llama4-maverick": {
        "model_id": "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
        "description": "Llama 4 Maverick - 17B active / 400B total MoE",
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

    # Check for sharded vs consolidated files
    try:
        files = list_repo_files(model_id)
        has_sharded = any("model-0" in f and f.endswith(".safetensors") for f in files)
        has_consolidated = any(f == "consolidated.safetensors" for f in files)

        ignore_patterns = []
        if has_sharded and has_consolidated:
            ignore_patterns.append("consolidated.safetensors")
            print("   (Skipping consolidated.safetensors - using sharded files)")
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
    """Load tokenizer from model path."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError("transformers not installed. Run: pip install transformers")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_weights(
    model_path: Path,
    config: Llama4TextConfig,
    dtype: str = "bfloat16",
) -> dict:
    """Load and convert Llama 4 weights from safetensors.

    Llama 4 has a different weight structure due to MoE:
    - model.layers.{i}.feed_forward.router.weight
    - model.layers.{i}.feed_forward.shared_expert.*
    - model.layers.{i}.feed_forward.experts.{e}.*

    This function fuses per-expert weights into SwitchGLU format.
    """
    safetensor_files = sorted(model_path.glob("*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors files found in {model_path}")

    # Convert to target dtype
    dtype_map = {
        "float16": mx.float16,
        "float32": mx.float32,
        "bfloat16": mx.bfloat16,
    }
    target_dtype = dtype_map.get(dtype, mx.bfloat16)

    # Collect expert weights per layer for fusion
    # layer_idx -> proj_type -> expert_idx -> weight
    expert_weights: dict[int, dict[str, dict[int, mx.array]]] = {}
    flat_weights: dict[str, mx.array] = {}

    # Load and process weight files one at a time to reduce memory
    for sf_path in safetensor_files:
        print(f"  Loading {sf_path.name}...")
        weights = mx.load(str(sf_path))

        for hf_name, weight in weights.items():
            # Convert dtype
            if weight.dtype in (mx.float32, mx.float16, mx.bfloat16):
                weight = weight.astype(target_dtype)

            # Check for routed expert weights - need to collect and fuse
            expert_match = re.match(
                r"model\.layers\.(\d+)\.(?:feed_forward|mlp)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight",
                hf_name,
            )
            if expert_match:
                layer_idx = int(expert_match.group(1))
                expert_idx = int(expert_match.group(2))
                proj_type = expert_match.group(3)

                if layer_idx not in expert_weights:
                    expert_weights[layer_idx] = {}
                if proj_type not in expert_weights[layer_idx]:
                    expert_weights[layer_idx][proj_type] = {}

                expert_weights[layer_idx][proj_type][expert_idx] = weight
                continue

            # Convert other weights normally
            our_name = _convert_llama4_weight_name(hf_name, config)
            if our_name is not None:
                flat_weights[our_name] = weight

        # Clear loaded weights to free memory
        del weights
        mx.eval([])  # Trigger cleanup

    # Fuse expert weights into SwitchGLU format: (num_experts, out_dim, in_dim)
    print("  Fusing expert weights...")
    for layer_idx, proj_dict in expert_weights.items():
        for proj_type, experts_dict in proj_dict.items():
            # Stack expert weights in order
            num_experts = len(experts_dict)
            expert_list = [experts_dict[i] for i in range(num_experts)]
            fused = mx.stack(expert_list, axis=0)

            # SwitchLinear expects (num_experts, output_dims, input_dims)
            # HF format is (output_dims, input_dims), so stacking gives correct shape
            our_name = f"model.layers.{layer_idx}.mlp.experts.{proj_type}.weight"
            flat_weights[our_name] = fused

    # Clear expert weights dict
    del expert_weights
    mx.eval([])

    return _build_nested_weights_v2(flat_weights, config)


def _convert_llama4_weight_name(hf_name: str, config: Llama4TextConfig) -> str | None:
    """Convert HuggingFace Llama 4 weight name to our format.

    Note: Routed expert weights (feed_forward.experts.{i}.*) are handled separately
    in load_weights() and fused into SwitchGLU format.
    """
    # Embeddings
    if hf_name == "model.embed_tokens.weight":
        return "model.embed_tokens.weight.weight"

    # Final layer norm
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

        # Skip routed expert weights - handled separately for fusion
        if re.match(r"(?:feed_forward|mlp)\.experts\.\d+\.", rest):
            return None

        # Attention projections (with QK norm)
        if rest.startswith("self_attn."):
            return f"model.layers.{layer_idx}.{rest}"

        # Layer norms
        if rest in ("input_layernorm.weight", "post_attention_layernorm.weight"):
            return f"model.layers.{layer_idx}.{rest}"

        # MoE components - map to our structure
        if rest.startswith("feed_forward.") or rest.startswith("mlp."):
            # Normalize to feed_forward prefix
            if rest.startswith("mlp."):
                rest = rest.replace("mlp.", "feed_forward.", 1)

            # Router
            if rest == "feed_forward.router.weight":
                return f"model.layers.{layer_idx}.mlp.router.weight"

            # Shared expert
            if rest.startswith("feed_forward.shared_expert."):
                sub = rest.replace("feed_forward.shared_expert.", "")
                return f"model.layers.{layer_idx}.mlp.shared_expert.{sub}"

        # Standard MLP (non-MoE fallback)
        if rest.startswith("mlp."):
            return f"model.layers.{layer_idx}.{rest}"

        return f"model.layers.{layer_idx}.{rest}"

    return None


def _build_nested_weights_v2(flat_weights: dict[str, mx.array], config: Llama4TextConfig) -> dict:
    """Build nested dict/list structure from flat weight names.

    V2: Handles fused expert weights (no per-expert indexing needed).
    """
    # Find maximum layer index
    max_layer_idx = -1

    for name in flat_weights:
        parts = name.split(".")
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                try:
                    max_layer_idx = max(max_layer_idx, int(parts[i + 1]))
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

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass

    # Fallback
    prompt = ""
    if system_message:
        prompt += f"System: {system_message}\n\n"
    prompt += f"User: {user_message}\n\nAssistant:"
    return prompt


def generate_text(
    model: Llama4ForCausalLM,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    verbose: bool = True,
) -> str:
    """Generate text from the model."""
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


def test_tiny_model():
    """Test with a tiny Llama 4 config (no download needed)."""
    print("=" * 60)
    print("Llama 4 Tiny Model Test")
    print("=" * 60)

    # Create tiny config
    config = Llama4TextConfig.tiny()
    print(f"\nConfig:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Layers: {config.num_hidden_layers}")
    print(f"  Experts: {config.num_local_experts}")
    print(f"  Experts per token: {config.num_experts_per_tok}")
    print(f"  QK norm: {config.use_qk_norm}")

    # Create model
    print("\nCreating model...")
    model = Llama4ForCausalLM(config)

    params = count_parameters(model)
    print(f"  {params.summary()}")

    # Test forward pass
    print("\nTesting forward pass...")
    input_ids = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    output = model(input_ids)
    mx.eval(output.logits)
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output logits shape: {output.logits.shape}")

    # Test generation
    print("\nTesting generation...")
    output_ids = model.generate(
        input_ids,
        max_new_tokens=10,
        temperature=1.0,
    )
    mx.eval(output_ids)
    print(f"  Generated sequence shape: {output_ids.shape}")

    print("\n" + "=" * 60)
    print("SUCCESS! Llama 4 tiny model test passed.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Llama 4 Inference Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Llama 4 Model Presets:
  llama4-scout    - 17B active / 109B total (16 experts)
  llama4-maverick - 17B active / 400B total (128 experts)

Examples:
  # Test tiny model (no download)
  python 01_llama4_inference.py --test-tiny

  # Run with Scout (requires ~27GB RAM)
  python 01_llama4_inference.py --model llama4-scout
""",
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_PRESETS.keys()),
        default="llama4-scout",
        help="Model preset to use",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Custom HuggingFace model ID",
    )
    parser.add_argument(
        "--test-tiny",
        action="store_true",
        help="Run tiny model test (no download)",
    )
    parser.add_argument(
        "--prompt",
        default="What is the capital of France?",
        help="Prompt to generate from",
    )
    parser.add_argument(
        "--system",
        default="You are a helpful assistant.",
        help="System message",
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
        "--dtype",
        default="bfloat16",
        choices=["float16", "float32", "bfloat16"],
        help="Data type for weights",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Cache directory",
    )

    args = parser.parse_args()

    # Tiny model test
    if args.test_tiny:
        test_tiny_model()
        return

    # Get model ID
    if args.model_id:
        model_id = args.model_id
        model_name = model_id.split("/")[-1]
    else:
        preset = MODEL_PRESETS[args.model]
        model_id = preset["model_id"]
        model_name = args.model

    print("=" * 60)
    print(f"Llama 4 Inference: {model_name}")
    print("=" * 60)

    # Download
    print("\n1. Downloading model...")
    try:
        model_path = download_model(model_id, cache_dir=args.cache_dir)
    except Exception as e:
        print(f"   Error: {e}")
        print("\n   Note: Llama 4 models require HuggingFace authentication.")
        print("   Run: huggingface-cli login")
        return
    print(f"   Path: {model_path}")

    # Load config
    print("\n2. Loading configuration...")
    config_path = model_path / "config.json"
    with open(config_path) as f:
        config_data = json.load(f)

    # Handle list token IDs
    for key in ["eos_token_id", "bos_token_id", "pad_token_id"]:
        if key in config_data and isinstance(config_data[key], list):
            config_data[key] = config_data[key][0] if config_data[key] else None

    config = Llama4TextConfig(**config_data)
    print(f"   Model type: {config.model_type}")
    print(f"   Hidden size: {config.hidden_size}")
    print(f"   Layers: {config.num_hidden_layers}")
    print(f"   Experts: {config.num_local_experts}")

    # Create model
    print("\n3. Creating model...")
    model = Llama4ForCausalLM(config)
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
    print("SUCCESS! Llama 4 inference complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
