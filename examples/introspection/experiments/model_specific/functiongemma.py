#!/usr/bin/env python3
"""
Introspection demo with FunctionGemma.

This shows how logit lens and hooks can reveal how FunctionGemma
routes to different functions across its layers.

FunctionGemma is a 270M parameter model from Google, designed specifically
for on-device function calling. It needs its special chat template with
tool definitions to properly generate function call outputs.

Run: uv run python examples/introspection_functiongemma.py
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


def load_chat_template(model_id: str) -> Template:
    """Load the Jinja2 chat template from the model."""
    template_path = hf_hub_download(model_id, "chat_template.jinja")
    with open(template_path) as f:
        template_str = f.read()
    return Template(template_str)


def load_functiongemma():
    """Load FunctionGemma model, tokenizer, and chat template."""
    model_id = "mlx-community/functiongemma-270m-it-bf16"

    print(f"Downloading {model_id}...")
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

    # Sanitize weights (handles tied embeddings, etc.)
    sanitized_weights = model.sanitize(raw_weights)

    # Load into model
    nested_weights = tree_unflatten(list(sanitized_weights.items()))
    model.update(nested_weights)
    mx.eval(model.parameters())
    print(f"Loaded {len(sanitized_weights)} weight tensors")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load chat template
    print("Loading chat template...")
    chat_template = load_chat_template(model_id)

    return model, tokenizer, config, chat_template


def analyze_prompt(model, tokenizer, config, prompt: str, track_tokens: list[str] | None = None):
    """Run introspection on a prompt."""
    print(f"\n{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"{'='*60}")

    # Tokenize
    input_ids = tokenizer.encode(prompt, return_tensors="np")
    input_ids = mx.array(input_ids)
    tokens = [tokenizer.decode([tid]) for tid in input_ids[0].tolist()]
    print(f"\nTokens ({len(tokens)}): {tokens[:10]}..." if len(tokens) > 10 else f"\nTokens ({len(tokens)}): {tokens}")

    # Setup hooks - capture every 4th layer for efficiency
    layers_to_capture = list(range(0, config.num_hidden_layers, 4))
    layers_to_capture.append(config.num_hidden_layers - 1)  # Always include last
    layers_to_capture = sorted(set(layers_to_capture))

    print(f"\nCapturing layers: {layers_to_capture}")

    hooks = ModelHooks(model)
    hooks.configure(CaptureConfig(
        layers=layers_to_capture,
        capture_hidden_states=True,
        positions=PositionSelection.LAST,
    ))

    # Forward pass
    print("\nRunning forward pass...")
    logits = hooks.forward(input_ids)

    # Get final prediction
    last_logits = logits[0, -1, :]
    probs = mx.softmax(last_logits)
    top_5_idx = mx.argsort(probs)[::-1][:5].tolist()
    top_5_probs = [float(probs[i]) for i in top_5_idx]
    top_5_tokens = [tokenizer.decode([i]) for i in top_5_idx]

    print(f"\nFinal prediction (top 5):")
    for tok, prob in zip(top_5_tokens, top_5_probs):
        bar = "#" * int(prob * 50)
        print(f"  {prob:.4f} {bar} '{tok}'")

    # Logit lens analysis
    print(f"\n--- Logit Lens Analysis ---")
    lens = LogitLens(hooks, tokenizer)

    predictions = lens.get_layer_predictions(position=-1, top_k=3)
    print("\nTop prediction at each layer:")
    for pred in predictions:
        top_tok = pred.top_tokens[0]
        top_prob = pred.top_probs[0]
        print(f"  Layer {pred.layer_idx:2d}: '{top_tok}' ({top_prob:.4f})")

    # Track specific tokens if provided
    if track_tokens:
        print(f"\n--- Token Evolution ---")
        for token in track_tokens:
            try:
                evolution = lens.track_token(token, position=-1)
                print(f"\nToken '{token}':")
                for layer, prob, rank in zip(evolution.layers, evolution.probabilities, evolution.ranks):
                    rank_str = f"rank {rank}" if rank else "not in top-100"
                    bar = "#" * int(prob * 100)
                    print(f"  Layer {layer:2d}: {prob:.4f} {bar} ({rank_str})")

                if evolution.emergence_layer is not None:
                    print(f"  --> Becomes top-1 at layer {evolution.emergence_layer}")
            except Exception as e:
                print(f"\nCouldn't track '{token}': {e}")

    return hooks, lens


def analyze_with_tools(
    model, tokenizer, config, chat_template: Template, user_message: str, tools: list[dict]
):
    """Run introspection on a function calling prompt with proper chat format."""
    print(f"\n{'='*60}")
    print(f"User: {user_message}")
    print(f"Tools: {[t['function']['name'] for t in tools]}")
    print(f"{'='*60}")

    # Format with chat template
    messages = [
        {"role": "developer", "content": "You are a model that can do function calling with the following functions"},
        {"role": "user", "content": user_message},
    ]

    prompt = chat_template.render(
        messages=messages,
        tools=tools,
        add_generation_prompt=True,
        bos_token="<bos>",
        eos_token="<eos>",
    )

    # Tokenize
    input_ids = tokenizer.encode(prompt, return_tensors="np")
    input_ids = mx.array(input_ids)
    print(f"\nTokens: {input_ids.shape[1]} tokens")

    # Setup hooks - capture key layers
    layers_to_capture = list(range(0, config.num_hidden_layers, 3))
    layers_to_capture.append(config.num_hidden_layers - 1)
    layers_to_capture = sorted(set(layers_to_capture))

    hooks = ModelHooks(model)
    hooks.configure(CaptureConfig(
        layers=layers_to_capture,
        capture_hidden_states=True,
        positions=PositionSelection.LAST,
    ))

    # Forward pass
    print("Running forward pass...")
    logits = hooks.forward(input_ids)

    # Get final prediction
    last_logits = logits[0, -1, :]
    probs = mx.softmax(last_logits)
    top_5_idx = mx.argsort(probs)[::-1][:5].tolist()
    top_5_probs = [float(probs[i]) for i in top_5_idx]
    top_5_tokens = [tokenizer.decode([i]) for i in top_5_idx]

    print(f"\nFinal prediction (top 5):")
    for tok, prob in zip(top_5_tokens, top_5_probs):
        bar = "#" * int(prob * 50)
        # Clean up special characters for display
        clean_tok = repr(tok) if tok.startswith("<") else tok
        print(f"  {prob:.4f} {bar} {clean_tok}")

    # Logit lens
    lens = LogitLens(hooks, tokenizer)
    predictions = lens.get_layer_predictions(position=-1, top_k=1)
    print("\nTop prediction at each layer:")
    for pred in predictions:
        top_tok = pred.top_tokens[0]
        clean_tok = repr(top_tok) if top_tok.startswith("<") else top_tok
        print(f"  Layer {pred.layer_idx:2d}: {clean_tok} ({pred.top_probs[0]:.4f})")

    return hooks


def main():
    print("=" * 60)
    print("FunctionGemma Introspection Demo")
    print("=" * 60)

    # Load model
    model, tokenizer, config, chat_template = load_functiongemma()

    # Define tools for function calling tests
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_temperature",
                "description": "Gets the current temperature for a given location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The city name"},
                    },
                    "required": ["location"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "create_event",
                "description": "Creates a calendar event.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Event title"},
                        "date": {"type": "string", "description": "Event date"},
                    },
                    "required": ["title", "date"],
                },
            },
        },
    ]

    # Test 1: Simple completion (no chat template needed)
    print("\n\n" + "#" * 60)
    print("# SIMPLE PROMPTS (No Chat Template)")
    print("#" * 60)

    analyze_prompt(
        model, tokenizer, config,
        prompt="The capital of France is",
        track_tokens=["Paris", " Paris"],
    )

    analyze_prompt(
        model, tokenizer, config,
        prompt="Hello",
        track_tokens=["!", "there", "world"],
    )

    # Test 2: Function calling with chat template
    print("\n\n" + "#" * 60)
    print("# FUNCTION CALLING (With Chat Template + Tools)")
    print("#" * 60)

    analyze_with_tools(
        model, tokenizer, config, chat_template,
        user_message="What is the weather in Tokyo?",
        tools=tools,
    )

    analyze_with_tools(
        model, tokenizer, config, chat_template,
        user_message="Create an event called Team Meeting for tomorrow",
        tools=tools,
    )

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
