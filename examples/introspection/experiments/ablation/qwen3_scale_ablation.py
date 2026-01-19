#!/usr/bin/env python3
"""
Qwen3 Scale Ablation Study

Compare tool-calling layer positions across Qwen3 model scales:
- Qwen3-0.6B (28 layers)
- Qwen3-1.7B (28 layers)
- Qwen3-4B (36 layers)
- Qwen3-8B (36 layers)

Questions:
1. Do tool-calling layers move relatively (stay at ~60-70% depth)?
2. Do they expand (more layers dedicated to tool decisions)?
3. How does Qwen3 compare to Gemma-3?

Run: uv run python examples/introspection/qwen3_scale_ablation.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
from _loader import load_model


@dataclass
class ModelSpec:
    """Specification for a model to test."""

    name: str
    model_id: str
    num_layers: int


# Models to test - Qwen3 family
MODELS = [
    ModelSpec("Qwen3-0.6B", "mlx-community/Qwen3-0.6B-bf16", 28),
    ModelSpec("Qwen3-1.7B", "mlx-community/Qwen3-1.7B-bf16", 28),
    ModelSpec("Qwen3-4B", "Qwen/Qwen3-4B-MLX-bf16", 36),
    # ModelSpec("Qwen3-8B", "Qwen/Qwen3-8B-MLX-bf16", 36),  # Can add if memory allows
]


def load_qwen3_model(model_id: str):
    """Load a Qwen3 model using the unified loader."""
    model, tokenizer, config, _ = load_model(model_id)
    return model, tokenizer, config


def create_tool_prompt(tokenizer) -> str:
    """Create a tool-calling prompt for Qwen3."""
    # Qwen3 supports tool calling natively with its chat template
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string", "description": "City name"}},
                    "required": ["location"],
                },
            },
        }
    ]

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant with access to tools. Use them when appropriate.",
        },
        {"role": "user", "content": "What's the weather in Tokyo?"},
    ]

    try:
        return tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Fallback without tools
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Manual fallback
            return "<|im_start|>system\nYou are a helpful assistant with access to tools.<|im_end|>\n<|im_start|>user\nYou have access to get_weather(location). Use it to get the weather in Tokyo.<|im_end|>\n<|im_start|>assistant\n"


def has_tool_call(text: str) -> bool:
    """Check if output contains tool-calling markers."""
    markers = [
        "get_weather",
        '{"name":',
        '"function_call"',
        "```tool",
        "<tool_call>",
        "tool_call",
        '{"location"',
        "Tokyo",
    ]
    return any(m.lower() in text.lower() for m in markers)


def generate_with_mlp_ablation(
    model: Qwen3ForCausalLM,
    tokenizer,
    input_ids: mx.array,
    ablate_layer: int | None = None,
    max_new_tokens: int = 60,
) -> str:
    """Generate with optional MLP ablation at a single layer."""
    original_weight = None

    if ablate_layer is not None:
        layer = model.model.layers[ablate_layer]
        original_weight = mx.array(layer.mlp.down_proj.weight)
        layer.mlp.down_proj.weight = mx.zeros_like(original_weight)
        mx.eval(layer.mlp.down_proj.weight)

    # Get stop tokens
    eos = tokenizer.eos_token_id
    stop_tokens = eos if isinstance(eos, list) else [eos] if eos else []

    # Add Qwen3-specific stop tokens
    for special in ["<|im_end|>", "<|endoftext|>"]:
        if special in tokenizer.get_vocab():
            stop_tokens.append(tokenizer.get_vocab()[special])

    generated = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        stop_tokens=stop_tokens,
    )

    if original_weight is not None:
        layer = model.model.layers[ablate_layer]
        layer.mlp.down_proj.weight = original_weight
        mx.eval(layer.mlp.down_proj.weight)

    output_ids = generated[0, input_ids.shape[1] :].tolist()
    return tokenizer.decode(output_ids, skip_special_tokens=False)


def run_ablation_sweep(
    model: Qwen3ForCausalLM,
    tokenizer,
    prompt: str,
    num_layers: int,
) -> dict:
    """Run MLP ablation sweep across all layers."""
    input_ids = tokenizer.encode(prompt, return_tensors="np")
    input_ids = mx.array(input_ids)

    # Get original output
    original_output = generate_with_mlp_ablation(model, tokenizer, input_ids, ablate_layer=None)
    original_has_tool = has_tool_call(original_output)

    print(f"  Original output: {original_output[:100]}...")
    print(f"  Has tool call: {original_has_tool}")

    results = {
        "original_output": original_output,
        "original_has_tool": original_has_tool,
        "layers": {},
    }

    for layer in range(num_layers):
        ablated_output = generate_with_mlp_ablation(model, tokenizer, input_ids, ablate_layer=layer)
        ablated_has_tool = has_tool_call(ablated_output)
        is_coherent = len(ablated_output.strip()) > 10 and "!!!!" not in ablated_output

        changed = original_has_tool != ablated_has_tool

        results["layers"][layer] = {
            "output": ablated_output[:200],
            "has_tool": ablated_has_tool,
            "changed": changed,
            "coherent": is_coherent,
        }

        status = "CAUSAL" if changed else "-"
        coherent_str = "coherent" if is_coherent else "BROKEN"
        print(f"    L{layer:2d}: tool={ablated_has_tool}, {status:6s} ({coherent_str})")

    return results


def main():
    print("=" * 80)
    print("Qwen3 Scale Ablation Study")
    print("Comparing tool-calling layers across model scales")
    print("=" * 80)

    all_results = {}

    for spec in MODELS:
        print(f"\n{'─' * 60}")
        print(f"Model: {spec.name}")
        print(f"ID: {spec.model_id}")
        print(f"Layers: {spec.num_layers}")
        print(f"{'─' * 60}")

        try:
            model, tokenizer, config = load_qwen3_model(spec.model_id)

            prompt = create_tool_prompt(tokenizer)
            print(f"\nPrompt length: {len(prompt)} chars")

            results = run_ablation_sweep(model, tokenizer, prompt, spec.num_layers)

            # Find causal layers
            causal_layers = [
                l for l, data in results["layers"].items() if data["changed"] and data["coherent"]
            ]
            causal_broken = [
                l
                for l, data in results["layers"].items()
                if data["changed"] and not data["coherent"]
            ]

            print(f"\n  Causal layers (coherent): {causal_layers}")
            print(f"  Causal layers (broken): {causal_broken}")

            if causal_layers:
                relative_positions = [l / spec.num_layers for l in causal_layers]
                print(f"  Relative positions: {[f'{p:.1%}' for p in relative_positions]}")

            all_results[spec.name] = {
                "model_id": spec.model_id,
                "num_layers": spec.num_layers,
                "original_has_tool": results["original_has_tool"],
                "causal_layers": causal_layers,
                "causal_broken": causal_broken,
                "relative_positions": [l / spec.num_layers for l in causal_layers]
                if causal_layers
                else [],
                "layers": results["layers"],
            }

            # Clean up
            del model
            mx.clear_cache()

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback

            traceback.print_exc()
            all_results[spec.name] = {"error": str(e)}

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Tool-Calling Layers Across Qwen3 Scales")
    print("=" * 80)

    print(f"\n{'Model':<20} {'Layers':<8} {'Tool?':<8} {'Causal Layers':<25} {'Relative Pos':<25}")
    print("-" * 90)

    for name, data in all_results.items():
        if "error" in data:
            print(f"{name:<20} ERROR: {data['error'][:40]}")
            continue

        layers = data["num_layers"]
        has_tool = "Yes" if data["original_has_tool"] else "No"
        causal = str(data["causal_layers"]) if data["causal_layers"] else "None"
        relative = (
            ", ".join([f"{p:.0%}" for p in data["relative_positions"]])
            if data["relative_positions"]
            else "-"
        )

        print(f"{name:<20} {layers:<8} {has_tool:<8} {causal:<25} {relative:<25}")

    # Save results
    output_path = Path("qwen3_scale_ablation_results.json")
    with open(output_path, "w") as f:
        json_results = {}
        for name, data in all_results.items():
            if "layers" in data:
                data = dict(data)
                data["layers"] = {str(k): v for k, v in data["layers"].items()}
            json_results[name] = data
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    print("""
INTERPRETATION:

Compare with Gemma-3 results:
- FunctionGemma 270M (18L): Causal at 61-67% depth
- Gemma-3 1B (26L): Causal at 12-23% depth

Do Qwen3 models show consistent relative positions across scales?
""")


if __name__ == "__main__":
    main()
