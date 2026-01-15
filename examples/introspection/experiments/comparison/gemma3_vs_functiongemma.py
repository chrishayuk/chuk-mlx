#!/usr/bin/env python3
"""
Compare Gemma 3 270M vs FunctionGemma 270M using introspection.

This example demonstrates:
1. How both models process the same prompts
2. When function-related tokens emerge in FunctionGemma vs regular Gemma
3. Differences in prediction patterns between base and function-tuned models

FunctionGemma is fine-tuned from Gemma for on-device function calling, so comparing
them reveals what the fine-tuning learned.

Run: uv run python examples/introspection/gemma3_vs_functiongemma.py
"""

import mlx.core as mx
from huggingface_hub import hf_hub_download, snapshot_download
from jinja2 import Template
from mlx.utils import tree_unflatten
from transformers import AutoTokenizer

from chuk_lazarus.introspection import (
    CaptureConfig,
    LogitLens,
    ModelHooks,
    PositionSelection,
)
from chuk_lazarus.models_v2.families.gemma import GemmaConfig, GemmaForCausalLM
from chuk_lazarus.models_v2.families.gemma.convert import load_hf_config, load_weights


def load_gemma_model(model_id: str):
    """Load a Gemma model from HuggingFace."""
    print(f"\nDownloading {model_id}...")
    model_path = snapshot_download(
        model_id,
        allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.jinja"],
    )
    print(f"Model path: {model_path}")

    # Load config
    hf_config = load_hf_config(model_path)
    config = GemmaConfig.from_hf_config(hf_config)
    print(f"Config: {config.num_hidden_layers} layers, {config.hidden_size} hidden dim")

    # Create model
    model = GemmaForCausalLM(config)

    # Load weights
    print("Loading weights...")
    raw_weights = load_weights(model_path)
    sanitized_weights = model.sanitize(raw_weights)
    nested_weights = tree_unflatten(list(sanitized_weights.items()))
    model.update(nested_weights)
    mx.eval(model.parameters())
    print(f"Loaded {len(sanitized_weights)} weight tensors")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return model, tokenizer, config, model_path


def load_chat_template(model_id: str) -> Template | None:
    """Load the Jinja2 chat template if available."""
    try:
        template_path = hf_hub_download(model_id, "chat_template.jinja")
        with open(template_path) as f:
            template_str = f.read()
        return Template(template_str)
    except Exception:
        return None


def analyze_prompt(model, tokenizer, config, prompt: str, model_name: str = "Model"):
    """Run introspection analysis on a prompt."""
    # Tokenize
    input_ids = tokenizer.encode(prompt, return_tensors="np")
    input_ids = mx.array(input_ids)
    tokens = [tokenizer.decode([tid]) for tid in input_ids[0].tolist()]

    # Setup hooks - capture every 3rd layer
    layers_to_capture = list(range(0, config.num_hidden_layers, 3))
    layers_to_capture.append(config.num_hidden_layers - 1)
    layers_to_capture = sorted(set(layers_to_capture))

    hooks = ModelHooks(model)
    hooks.configure(
        CaptureConfig(
            layers=layers_to_capture,
            capture_hidden_states=True,
            positions=PositionSelection.LAST,
        )
    )

    # Forward pass
    logits = hooks.forward(input_ids)

    # Get final prediction
    last_logits = logits[0, -1, :]
    probs = mx.softmax(last_logits)
    top_5_idx = mx.argsort(probs)[::-1][:5].tolist()
    top_5_probs = [float(probs[i]) for i in top_5_idx]
    top_5_tokens = [tokenizer.decode([i]) for i in top_5_idx]

    # Logit lens
    lens = LogitLens(hooks, tokenizer)
    predictions = lens.get_layer_predictions(position=-1, top_k=1)

    return {
        "model_name": model_name,
        "tokens": tokens,
        "top_5": list(zip(top_5_tokens, top_5_probs)),
        "layer_predictions": [(p.layer_idx, p.top_tokens[0], p.top_probs[0]) for p in predictions],
        "final_token": top_5_tokens[0],
        "final_prob": top_5_probs[0],
    }


def print_comparison(results: list[dict], prompt: str):
    """Print a side-by-side comparison."""
    print(f"\n{'=' * 80}")
    print(f"Prompt: {prompt}")
    print(f"{'=' * 80}")

    # Final predictions
    print("\n--- Final Predictions ---")
    for r in results:
        print(f"\n{r['model_name']}:")
        for tok, prob in r["top_5"][:3]:
            bar = "#" * int(prob * 40)
            clean_tok = repr(tok) if tok.startswith("<") or len(tok) > 10 else f"'{tok}'"
            print(f"  {prob:.4f} {bar} {clean_tok}")

    # Layer-by-layer comparison
    print("\n--- Layer-by-Layer (top prediction) ---")
    print(f"{'Layer':<8}", end="")
    for r in results:
        print(f"{r['model_name']:<35}", end="")
    print()
    print("-" * 78)

    # Get all layers from first result
    layers = [l for l, _, _ in results[0]["layer_predictions"]]
    for layer in layers:
        print(f"{layer:<8}", end="")
        for r in results:
            for l, tok, prob in r["layer_predictions"]:
                if l == layer:
                    clean_tok = repr(tok) if tok.startswith("<") else tok
                    print(f"{clean_tok[:15]:<15} ({prob:.3f})     ", end="")
                    break
        print()


def track_token_comparison(
    models_data: list[tuple], tokenizer, config, prompt: str, tokens_to_track: list[str]
):
    """Track specific tokens across layers for both models."""
    print(f"\n{'=' * 80}")
    print(f"Token Tracking: {tokens_to_track}")
    print(f"Prompt: {prompt}")
    print(f"{'=' * 80}")

    for model, model_name in models_data:
        # Tokenize
        input_ids = tokenizer.encode(prompt, return_tensors="np")
        input_ids = mx.array(input_ids)

        # Setup hooks
        layers_to_capture = list(range(0, config.num_hidden_layers, 2))
        layers_to_capture.append(config.num_hidden_layers - 1)
        layers_to_capture = sorted(set(layers_to_capture))

        hooks = ModelHooks(model)
        hooks.configure(
            CaptureConfig(
                layers=layers_to_capture,
                capture_hidden_states=True,
                positions=PositionSelection.LAST,
            )
        )
        hooks.forward(input_ids)

        lens = LogitLens(hooks, tokenizer)

        print(f"\n{model_name}:")
        for token in tokens_to_track:
            try:
                evolution = lens.track_token(token, position=-1)
                emergence = evolution.emergence_layer
                final_prob = evolution.probabilities[-1] if evolution.probabilities else 0

                if emergence is not None:
                    print(f"  '{token}': emerges at layer {emergence}, final prob {final_prob:.4f}")
                else:
                    print(f"  '{token}': never top-1, final prob {final_prob:.4f}")
            except Exception as e:
                print(f"  '{token}': couldn't track - {e}")


def main():
    print("=" * 80)
    print("Gemma 3 270M vs FunctionGemma 270M Comparison")
    print("=" * 80)
    print()
    print("NOTE: To see FunctionGemma's function-calling behavior, the prompt")
    print("must match its training format. Look for <start_function_call> in")
    print("the token tracking section - if it doesn't emerge, the template")
    print("may not match the model's expected format.")
    print()

    # Model IDs
    gemma3_id = "mlx-community/gemma-3-1b-it-bf16"  # Using 1B as 270M not always available
    functiongemma_id = "mlx-community/functiongemma-270m-it-bf16"

    # Try to use 270M Gemma if available, else fallback
    try:
        gemma3_model, gemma3_tokenizer, gemma3_config, _ = load_gemma_model(
            "mlx-community/gemma-3-270m-it-bf16"
        )
    except Exception:
        print("Gemma 3 270M not found, using 1B variant...")
        try:
            gemma3_model, gemma3_tokenizer, gemma3_config, _ = load_gemma_model(gemma3_id)
        except Exception as e:
            print(f"Could not load Gemma 3: {e}")
            print("Using tiny test model instead...")
            gemma3_config = GemmaConfig.tiny()
            gemma3_model = GemmaForCausalLM(gemma3_config)

            class MockTokenizer:
                def encode(self, text, return_tensors=None):
                    return [[1, 2, 3, 4, 5]]

                def decode(self, ids):
                    return f"[{ids[0] if isinstance(ids, list) else ids}]"

            gemma3_tokenizer = MockTokenizer()

    functiongemma_model, functiongemma_tokenizer, functiongemma_config, _ = load_gemma_model(
        functiongemma_id
    )

    # Load chat template for FunctionGemma
    functiongemma_template = load_chat_template(functiongemma_id)

    print("\n\n" + "#" * 80)
    print("# COMPARISON 1: Simple Completion (No Function Calling Context)")
    print("#" * 80)

    test_prompts = [
        "The capital of France is",
        "Hello",
        "2 + 2 equals",
    ]

    for prompt in test_prompts:
        gemma_result = analyze_prompt(
            gemma3_model, gemma3_tokenizer, gemma3_config, prompt, "Gemma3"
        )
        func_result = analyze_prompt(
            functiongemma_model,
            functiongemma_tokenizer,
            functiongemma_config,
            prompt,
            "FunctionGemma",
        )
        print_comparison([gemma_result, func_result], prompt)

    print("\n\n" + "#" * 80)
    print("# COMPARISON 2: Function-Related Prompts")
    print("#" * 80)

    # These prompts might trigger function-calling behavior in FunctionGemma
    function_prompts = [
        "What is the weather in",
        "Get the current temperature for",
        "Create a calendar event for",
        "Call the API to",
    ]

    for prompt in function_prompts:
        try:
            gemma_result = analyze_prompt(
                gemma3_model, gemma3_tokenizer, gemma3_config, prompt, "Gemma3"
            )
            func_result = analyze_prompt(
                functiongemma_model,
                functiongemma_tokenizer,
                functiongemma_config,
                prompt,
                "FunctionGemma",
            )
            print_comparison([gemma_result, func_result], prompt)
        except Exception as e:
            print(f"Error with prompt '{prompt}': {e}")

    print("\n\n" + "#" * 80)
    print("# COMPARISON 3: Token Evolution Tracking")
    print("#" * 80)

    # Track function-related tokens
    track_token_comparison(
        [(gemma3_model, "Gemma3"), (functiongemma_model, "FunctionGemma")],
        functiongemma_tokenizer,
        functiongemma_config,
        "What is the weather",
        ["get", " get", "Get", "!", " in"],
    )

    print("\n\n" + "#" * 80)
    print("# COMPARISON 4: FunctionGemma with Tool Context")
    print("#" * 80)

    if functiongemma_template:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_temperature",
                    "description": "Gets the current temperature for a location.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City name"},
                        },
                        "required": ["location"],
                    },
                },
            },
        ]

        # Test both soft and hard tool-use system messages
        test_variants = [
            {
                "name": "Soft Tool Context",
                "system": "You are a helpful assistant that can call functions when needed.",
            },
            {
                "name": "Hard Tool Context",
                "system": "When a tool can answer the user's question, you MUST call it using <start_function_call>. Do not respond in plain text if a tool applies.",
            },
        ]

        for variant in test_variants:
            messages = [
                {"role": "developer", "content": variant["system"]},
                {"role": "user", "content": "What is the weather in Tokyo?"},
            ]

            prompt_with_tools = functiongemma_template.render(
                messages=messages,
                tools=tools,
                add_generation_prompt=True,
                bos_token="<bos>",
                eos_token="<eos>",
            )

            print(f"\n--- {variant['name']} ---")

            # Print the rendered prompt for debugging
            print("\n[RENDERED PROMPT - last 500 chars]:")
            print(prompt_with_tools[-500:])
            print("[END PROMPT]\n")

            # Tokenize and show tokens for debugging
            input_ids = functiongemma_tokenizer.encode(prompt_with_tools, return_tensors="np")
            input_ids_list = input_ids[0].tolist()
            print(f"Total tokens: {len(input_ids_list)}")
            print("Last 20 tokens decoded:")
            for i, tid in enumerate(input_ids_list[-20:]):
                tok = functiongemma_tokenizer.decode([tid])
                print(f"  {i}: [{tid}] '{repr(tok)}'")

            func_result = analyze_prompt(
                functiongemma_model,
                functiongemma_tokenizer,
                functiongemma_config,
                prompt_with_tools,
                variant["name"],
            )

            print("\nFinal predictions:")
            for tok, prob in func_result["top_5"][:5]:
                bar = "#" * int(prob * 40)
                clean_tok = repr(tok) if tok.startswith("<") else f"'{tok}'"
                print(f"  {prob:.4f} {bar} {clean_tok}")

            print("\nLayer evolution:")
            for layer, tok, prob in func_result["layer_predictions"]:
                clean_tok = repr(tok) if tok.startswith("<") else tok
                print(f"  Layer {layer:2d}: {clean_tok} ({prob:.4f})")

        # Track key function-calling tokens
        print("\n\n--- Token Evolution for Function-Call Markers ---")
        track_tokens = [
            "<start_function_call>",
            "<function_call>",
            "{",
            '"',
            "get_current_temperature",
            "I",
        ]
        print(f"Tracking tokens: {track_tokens}")

        # Use hard tool context for tracking
        messages = [
            {"role": "developer", "content": test_variants[1]["system"]},
            {"role": "user", "content": "What is the weather in Tokyo?"},
        ]
        prompt_with_tools = functiongemma_template.render(
            messages=messages,
            tools=tools,
            add_generation_prompt=True,
            bos_token="<bos>",
            eos_token="<eos>",
        )

        track_token_comparison(
            [(functiongemma_model, "FunctionGemma")],
            functiongemma_tokenizer,
            functiongemma_config,
            prompt_with_tools,
            track_tokens,
        )

    print("\n" + "=" * 80)
    print("Comparison complete!")
    print("=" * 80)

    print("""
Key observations to look for:

1. TEMPLATE MATCH: Check the rendered prompt tokens - if function-calling special
   tokens like <start_function_call> aren't in the vocabulary or prompt, the model
   won't enter function-calling mode.

2. HARD vs SOFT: Compare final predictions between soft and hard system messages.
   If hard context makes <start_function_call> become top-1, the template is close
   but the model needs stronger policy. If neither works, template is mismatched.

3. TOKEN EVOLUTION: Watch which tokens emerge first:
   - If "I" dominates early and stays dominant → model is in assistant mode
   - If function tokens ({, <start_function_call>) emerge mid-layers → function mode

4. NEUTRAL PROMPTS: Both models should be nearly identical on "The capital of France"
   and simple completions. Differences only appear in function-calling contexts.

If <start_function_call> never emerges, inspect the rendered prompt to verify:
   - Tool definitions are formatted correctly
   - System message uses expected role name (developer vs system)
   - The generation prompt ends with the right assistant turn format
""")


if __name__ == "__main__":
    main()
