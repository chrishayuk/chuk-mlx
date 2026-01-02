#!/usr/bin/env python3
"""
Gemma Scale Ablation Study

Compare tool-calling layer positions across Gemma model scales:
- Base Gemma 270M vs FunctionGemma 270M (is tool-calling learned?)
- Gemma-3 1B
- Gemma-3 4B

Questions:
1. Do tool-calling layers move relatively (stay at ~60-70% depth)?
2. Do they expand (more layers dedicated to tool decisions)?
3. Is tool-calling absent in base models?

Run: uv run python examples/introspection/gemma_scale_ablation.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
from huggingface_hub import hf_hub_download
from jinja2 import Template

from _loader import load_model


@dataclass
class ModelSpec:
    """Specification for a model to test."""
    name: str
    model_id: str
    num_layers: int
    has_tools: bool  # Whether it's tool-tuned


# Models to test - Gemma-3 family only
MODELS = [
    # FunctionGemma 270M (tool-tuned, 18 layers)
    ModelSpec("FunctionGemma 270M", "mlx-community/functiongemma-270m-it-bf16", 18, has_tools=True),
    # Gemma-3 270M base (18 layers) - for comparison
    ModelSpec("Gemma-3 270M", "mlx-community/gemma-3-270m-it-bf16", 18, has_tools=False),
    # Gemma-3 1B (26 layers)
    ModelSpec("Gemma-3 1B", "mlx-community/gemma-3-1b-it-bf16", 26, has_tools=False),
]


def load_gemma_model(model_id: str):
    """Load a Gemma model using the unified loader."""
    return load_model(model_id)


def load_chat_template(model_id: str) -> Template | None:
    """Load Jinja2 chat template if available."""
    try:
        template_path = hf_hub_download(model_id, "chat_template.jinja")
        with open(template_path) as f:
            return Template(f.read())
    except Exception:
        return None


def create_tool_prompt(template: Template | None, tokenizer, model_id: str) -> str:
    """Create a tool-calling prompt appropriate for the model."""

    # FunctionGemma uses specific tool format
    if "functiongemma" in model_id.lower():
        tools = [{
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
        }]

        messages = [
            {"role": "developer", "content": "You are a helpful assistant with access to tools. Use them when appropriate."},
            {"role": "user", "content": "What's the weather in Tokyo?"},
        ]

        if template:
            return template.render(
                messages=messages,
                tools=tools,
                add_generation_prompt=True,
                bos_token="<bos>",
                eos_token="<eos>"
            )

    # For Gemma-3 models, use a simpler approach - just ask about weather
    # and check if it tries to use function-calling syntax
    messages = [
        {"role": "user", "content": "You have access to a get_weather(location) function. Use it to get the weather in Tokyo. Respond with a function call."},
    ]

    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Manual fallback
        return "<bos><start_of_turn>user\nYou have access to a get_weather(location) function. Use it to get the weather in Tokyo. Respond with a function call.<end_of_turn>\n<start_of_turn>model\n"


def has_tool_call(text: str) -> bool:
    """Check if output contains tool-calling markers."""
    markers = [
        "<start_function_call>", "<function_call>", "get_weather(",
        '{"name":', '"function_call"', "```tool", "<tool_call>"
    ]
    return any(m.lower() in text.lower() for m in markers)


def generate_with_mlp_ablation(
    model: GemmaForCausalLM,
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

    output_ids = generated[0, input_ids.shape[1]:].tolist()
    return tokenizer.decode(output_ids, skip_special_tokens=False)


def run_ablation_sweep(
    model: GemmaForCausalLM,
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
    print("Gemma Scale Ablation Study")
    print("Comparing tool-calling layers across model scales")
    print("=" * 80)

    all_results = {}

    for spec in MODELS:
        print(f"\n{'─' * 60}")
        print(f"Model: {spec.name}")
        print(f"ID: {spec.model_id}")
        print(f"Layers: {spec.num_layers}")
        print(f"Tool-tuned: {spec.has_tools}")
        print(f"{'─' * 60}")

        try:
            model, tokenizer, config, model_path = load_gemma_model(spec.model_id)
            template = load_chat_template(spec.model_id)

            prompt = create_tool_prompt(template, tokenizer, spec.model_id)
            print(f"\nPrompt length: {len(prompt)} chars")

            results = run_ablation_sweep(model, tokenizer, prompt, spec.num_layers)

            # Find causal layers
            causal_layers = [l for l, data in results["layers"].items() if data["changed"] and data["coherent"]]
            causal_broken = [l for l, data in results["layers"].items() if data["changed"] and not data["coherent"]]

            print(f"\n  Causal layers (coherent): {causal_layers}")
            print(f"  Causal layers (broken): {causal_broken}")

            if causal_layers:
                relative_positions = [l / spec.num_layers for l in causal_layers]
                print(f"  Relative positions: {[f'{p:.1%}' for p in relative_positions]}")

            all_results[spec.name] = {
                "model_id": spec.model_id,
                "num_layers": spec.num_layers,
                "has_tools": spec.has_tools,
                "original_has_tool": results["original_has_tool"],
                "causal_layers": causal_layers,
                "causal_broken": causal_broken,
                "relative_positions": [l / spec.num_layers for l in causal_layers] if causal_layers else [],
                "layers": results["layers"],
            }

            # Clean up
            del model
            mx.metal.clear_cache()

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results[spec.name] = {"error": str(e)}

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Tool-Calling Layers Across Scales")
    print("=" * 80)

    print(f"\n{'Model':<25} {'Layers':<8} {'Tool?':<8} {'Causal Layers':<20} {'Relative Pos':<20}")
    print("-" * 85)

    for name, data in all_results.items():
        if "error" in data:
            print(f"{name:<25} ERROR: {data['error'][:40]}")
            continue

        layers = data["num_layers"]
        has_tool = "Yes" if data["original_has_tool"] else "No"
        causal = str(data["causal_layers"]) if data["causal_layers"] else "None"
        relative = ", ".join([f"{p:.0%}" for p in data["relative_positions"]]) if data["relative_positions"] else "-"

        print(f"{name:<25} {layers:<8} {has_tool:<8} {causal:<20} {relative:<20}")

    # Save results
    output_path = Path("gemma_scale_ablation_results.json")
    with open(output_path, "w") as f:
        # Convert layer keys to strings for JSON
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

If relative positions are CONSISTENT across scales (~60-70%):
    → The "decision zone" is an architectural constant
    → Tool-calling emerges at a fixed relative depth

If causal layers EXPAND with scale:
    → Larger models dedicate more capacity to tool reasoning
    → May decompose into sub-decisions (which tool, arguments, confidence)

If base model has NO causal layers but FunctionGemma does:
    → Tool-calling is learned, not emergent
    → Fine-tuning creates new circuits at specific layers
""")


if __name__ == "__main__":
    main()
