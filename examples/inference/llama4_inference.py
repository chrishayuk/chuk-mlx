#!/usr/bin/env python3
"""
Llama 4 Inference Example (Simplified)

Demonstrates the simplified API for Llama 4 MoE models.

Llama 4 key features:
- MoE (Mixture of Experts) with shared expert
- iRoPE (interleaved RoPE and NoPE layers)
- QK normalization

Note: Llama 4 models require significant memory:
- Scout: ~27GB for BF16
- Maverick: ~100GB for BF16

Usage:
    # Test tiny (no download)
    uv run python examples/inference/llama4_inference.py --test-tiny

    # Llama 4 Scout (requires HF auth + ~27GB RAM)
    uv run python examples/inference/llama4_inference.py --model llama4-scout
"""

from __future__ import annotations

import argparse
import json
import re
import time
from enum import Enum
from pathlib import Path

import mlx.core as mx

from chuk_lazarus.inference import (
    DType,
    GenerationConfig,
    HFLoader,
    WeightConverter,
)
from chuk_lazarus.models_v2 import (
    Llama4ForCausalLM,
    Llama4TextConfig,
    count_parameters,
)


class Llama4Model(str, Enum):
    """Available Llama 4 model presets."""

    SCOUT = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    MAVERICK = "meta-llama/Llama-4-Maverick-17B-128E-Instruct"


MODEL_ALIASES = {
    "llama4-scout": Llama4Model.SCOUT,
    "llama4-maverick": Llama4Model.MAVERICK,
}


class Llama4WeightConverter:
    """Weight converter for Llama 4 MoE models.

    Handles the unique structure of Llama 4:
    - Routed expert weights need fusion into SwitchGLU format
    - Shared expert mapping
    - Router weights
    """

    def __init__(self, config: Llama4TextConfig):
        self.config = config
        self.expert_weights: dict[int, dict[str, dict[int, mx.array]]] = {}

    def convert(self, hf_name: str) -> str | None:
        """Convert HuggingFace weight name to framework format."""
        # Embeddings
        if hf_name == "model.embed_tokens.weight":
            return "model.embed_tokens.weight.weight"

        # Final norm
        if hf_name == "model.norm.weight":
            return "model.norm.weight"

        # LM head
        if hf_name == "lm_head.weight":
            if self.config.tie_word_embeddings:
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

            # Skip routed experts - handled separately for fusion
            if re.match(r"(?:feed_forward|mlp)\.experts\.\d+\.", rest):
                return None

            # Attention projections
            if rest.startswith("self_attn."):
                return f"model.layers.{layer_idx}.{rest}"

            # Layer norms
            if rest in ("input_layernorm.weight", "post_attention_layernorm.weight"):
                return f"model.layers.{layer_idx}.{rest}"

            # MoE components
            if rest.startswith("feed_forward.") or rest.startswith("mlp."):
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


def load_llama4_weights(
    model_path: Path,
    config: Llama4TextConfig,
    dtype: DType = DType.BFLOAT16,
) -> dict:
    """Load and fuse Llama 4 MoE weights."""
    safetensor_files = sorted(model_path.glob("*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors in {model_path}")

    target_dtype = dtype.to_mlx()
    converter = Llama4WeightConverter(config)

    # Collect expert weights for fusion
    expert_weights: dict[int, dict[str, dict[int, mx.array]]] = {}
    flat_weights: dict[str, mx.array] = {}

    for sf_path in safetensor_files:
        print(f"  Loading {sf_path.name}...")
        weights = mx.load(str(sf_path))

        for hf_name, weight in weights.items():
            # Convert dtype
            if weight.dtype in (mx.float32, mx.float16, mx.bfloat16):
                weight = weight.astype(target_dtype)

            # Check for routed expert weights
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

            # Convert other weights
            our_name = converter.convert(hf_name)
            if our_name is not None:
                flat_weights[our_name] = weight

        del weights
        mx.eval([])

    # Fuse expert weights into SwitchGLU format
    print("  Fusing expert weights...")
    for layer_idx, proj_dict in expert_weights.items():
        for proj_type, experts_dict in proj_dict.items():
            num_experts = len(experts_dict)
            expert_list = [experts_dict[i] for i in range(num_experts)]
            fused = mx.stack(expert_list, axis=0)

            our_name = f"model.layers.{layer_idx}.mlp.experts.{proj_type}.weight"
            flat_weights[our_name] = fused

    del expert_weights
    mx.eval([])

    return _build_nested_weights(flat_weights, config)


def _build_nested_weights(flat_weights: dict[str, mx.array], config) -> dict:
    """Build nested structure for model.update()."""
    max_layer_idx = config.num_hidden_layers - 1

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


def load_llama4_model(model_id: str):
    """Load Llama 4 model, tokenizer, and config."""
    print(f"Loading {model_id}...")
    print("=" * 60)

    # Download
    print("\n1. Downloading model...")
    result = HFLoader.download(model_id)
    print(f"   Path: {result.model_path}")

    # Load config
    print("\n2. Loading configuration...")
    config_path = result.model_path / "config.json"
    with open(config_path) as f:
        config_data = json.load(f)

    # Handle list token IDs
    for key in ("eos_token_id", "bos_token_id", "pad_token_id"):
        if key in config_data and isinstance(config_data[key], list):
            config_data[key] = config_data[key][0] if config_data[key] else None

    config = Llama4TextConfig(**config_data)
    print(f"   Hidden: {config.hidden_size}, Layers: {config.num_hidden_layers}")
    print(f"   Experts: {config.num_local_experts}, Active: {config.num_experts_per_tok}")

    # Create model
    print("\n3. Creating model...")
    model = Llama4ForCausalLM(config)
    params = count_parameters(model)
    print(f"   {params.summary()}")

    # Load weights with MoE fusion
    print("\n4. Loading weights...")
    weights = load_llama4_weights(result.model_path, config)
    model.update(weights)
    mx.eval(model.parameters())
    print("   Done!")

    # Load tokenizer
    print("\n5. Loading tokenizer...")
    tokenizer = HFLoader.load_tokenizer(result.model_path)
    print(f"   Vocab size: {len(tokenizer)}")

    print("\n" + "=" * 60)
    print("Model loaded successfully!")

    return model, tokenizer, config


def generate(model, tokenizer, prompt: str, config: GenerationConfig | None = None):
    """Generate text from prompt."""
    if config is None:
        config = GenerationConfig()

    # Tokenize
    input_ids = tokenizer.encode(prompt, return_tensors="np")
    input_ids = mx.array(input_ids)
    input_length = input_ids.shape[1]

    # Stop tokens
    stop_tokens = []
    if tokenizer.eos_token_id:
        if isinstance(tokenizer.eos_token_id, list):
            stop_tokens.extend(tokenizer.eos_token_id)
        else:
            stop_tokens.append(tokenizer.eos_token_id)

    # Generate
    start = time.time()
    output_ids = model.generate(
        input_ids,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        stop_tokens=stop_tokens,
    )
    mx.eval(output_ids)
    gen_time = time.time() - start

    # Decode
    new_tokens = output_ids[0, input_length:].tolist()
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    tps = len(new_tokens) / gen_time if gen_time > 0 else 0
    return text, tps


def format_chat(tokenizer, user_message: str, system_message: str | None = None) -> str:
    """Format using tokenizer's chat template."""
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


def test_tiny():
    """Test with tiny config."""
    print("=" * 60)
    print("Llama 4 Tiny Model Test")
    print("=" * 60)

    config = Llama4TextConfig.tiny()
    print(f"\nConfig:")
    print(f"  Hidden: {config.hidden_size}")
    print(f"  Layers: {config.num_hidden_layers}")
    print(f"  Experts: {config.num_local_experts}")

    model = Llama4ForCausalLM(config)
    params = count_parameters(model)
    print(f"  {params.summary()}")

    # Test forward
    input_ids = mx.array([[1, 2, 3, 4, 5]])
    output = model(input_ids)
    mx.eval(output.logits)
    print(f"\nForward: OK (shape={output.logits.shape})")

    # Test generate
    gen = model.generate(input_ids, max_new_tokens=5)
    mx.eval(gen)
    print(f"Generate: OK (shape={gen.shape})")

    print("\n" + "=" * 60)
    print("SUCCESS!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Llama 4 Inference (Simplified)")
    parser.add_argument(
        "--model",
        choices=list(MODEL_ALIASES.keys()),
        default="llama4-scout",
        help="Model preset",
    )
    parser.add_argument("--model-id", help="Custom HuggingFace model ID")
    parser.add_argument("--test-tiny", action="store_true", help="Run tiny test")
    parser.add_argument("--prompt", default="What is the capital of France?")
    parser.add_argument("--system", default="You are a helpful assistant.")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--list", action="store_true", help="List models")
    args = parser.parse_args()

    if args.test_tiny:
        test_tiny()
        return

    if args.list:
        print("Available Llama 4 models:\n")
        for alias, model in MODEL_ALIASES.items():
            print(f"  {alias:18} -> {model.value}")
        return

    # Get model ID
    model_id = args.model_id or MODEL_ALIASES[args.model].value

    # Load model
    model, tokenizer, config = load_llama4_model(model_id)

    # Generate
    print("\n" + "=" * 60)
    print(f"User: {args.prompt}")
    print("-" * 60)

    prompt = format_chat(tokenizer, args.prompt, args.system)
    gen_config = GenerationConfig(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    response, tps = generate(model, tokenizer, prompt, gen_config)

    print(f"Assistant: {response}")
    print("-" * 60)
    print(f"Speed: {tps:.1f} tokens/sec")
    print("=" * 60)


if __name__ == "__main__":
    main()
