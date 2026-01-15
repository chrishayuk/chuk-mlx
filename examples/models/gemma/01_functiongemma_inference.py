#!/usr/bin/env python3
"""
FunctionGemma Inference Example

This example shows how to use FunctionGemma for function calling inference
using the mlx-community pretrained weights.

FunctionGemma is a 270M parameter model from Google, designed specifically
for on-device function calling. It's excellent for:
- Tool use / API calling
- MCP (Model Context Protocol) integration
- Lightweight RAG pipelines
- On-device agents

FunctionGemma uses special tokens for structured function calling:
- <start_function_declaration> / <end_function_declaration> - Define tools
- <start_function_call> / <end_function_call> - Model requests tool use
- <start_function_response> / <end_function_response> - Tool results
- <escape> - Wraps string values in structured data

Requirements:
    pip install mlx-lm huggingface_hub jinja2

Usage:
    python 01_functiongemma_inference.py
"""

import json
import re

try:
    from huggingface_hub import hf_hub_download
    from jinja2 import Template
    from mlx_lm import generate, load
except ImportError:
    print("Please install required packages: pip install mlx-lm huggingface_hub jinja2")
    raise


def load_chat_template(model_id: str) -> Template:
    """Load the Jinja2 chat template from the model."""
    template_path = hf_hub_download(model_id, "chat_template.jinja")
    with open(template_path) as f:
        template_str = f.read()
    return Template(template_str)


def apply_chat_template(
    template: Template,
    messages: list[dict],
    tools: list[dict] | None = None,
    add_generation_prompt: bool = True,
) -> str:
    """Apply the chat template to format messages with tools."""
    return template.render(
        messages=messages,
        tools=tools or [],
        add_generation_prompt=add_generation_prompt,
        bos_token="<bos>",
        eos_token="<eos>",
    )


def parse_function_call(response: str) -> dict | None:
    """
    Parse a function call from the model response.

    Returns dict with 'name' and 'arguments' if found, else None.
    """
    # Look for function call pattern: call:function_name{args}
    pattern = r"call:(\w+)\{(.+?)\}"
    match = re.search(pattern, response, re.DOTALL)

    if match:
        func_name = match.group(1)
        args_str = match.group(2)

        # Parse arguments (handle <escape> tokens)
        args = {}
        # Pattern: key:<escape>value<escape>
        arg_pattern = r"(\w+):<escape>([^<]+)<escape>"
        for arg_match in re.finditer(arg_pattern, args_str):
            args[arg_match.group(1)] = arg_match.group(2)

        return {"name": func_name, "arguments": args}

    return None


def main():
    """Run FunctionGemma inference example."""
    print("=" * 60)
    print("FunctionGemma Inference Example")
    print("=" * 60)

    # Load the model from mlx-community
    # bf16 gives better function calling accuracy than quantized versions
    model_name = "mlx-community/functiongemma-270m-it-bf16"

    print(f"\nLoading model: {model_name}")
    model, tokenizer = load(model_name)

    # Load the chat template
    print("Loading chat template...")
    chat_template = load_chat_template(model_name)
    print("Model loaded successfully!")

    # Define tools in OpenAI-compatible format
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_temperature",
                "description": "Gets the current temperature for a given location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city name, e.g. San Francisco",
                        },
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

    # Test prompts
    test_cases = [
        "What's the temperature in London?",
        "What is the weather in Tokyo?",
        "Create an event called Team Meeting for tomorrow",
    ]

    print("\n" + "=" * 60)
    print("Testing Function Calling")
    print("=" * 60)

    for query in test_cases:
        print(f"\nUser: {query}")

        # Format messages
        messages = [
            {
                "role": "developer",
                "content": "You are a model that can do function calling with the following functions",
            },
            {"role": "user", "content": query},
        ]

        # Apply chat template with tools
        prompt = apply_chat_template(
            chat_template,
            messages,
            tools=tools,
            add_generation_prompt=True,
        )

        # Generate
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=100,
            verbose=False,
        )

        # Clean up response - get the function call part
        clean_response = response.split("<end_of_turn>")[0].strip()

        print(f"Raw: {clean_response}")

        # Parse function call
        func_call = parse_function_call(response)
        if func_call:
            print(f"Parsed: {func_call['name']}({json.dumps(func_call['arguments'])})")
        else:
            print("(No structured function call detected)")

        print("-" * 40)


def demo_with_chuk_lazarus():
    """
    Alternative: Using chuk-lazarus Gemma implementation directly.

    This shows how you could use our native implementation
    (requires loading weights from mlx-community model).
    """
    import mlx.core as mx

    from chuk_lazarus.models_v2.families.gemma import GemmaConfig, GemmaForCausalLM

    print("\n" + "=" * 60)
    print("Using chuk-lazarus Gemma Implementation")
    print("=" * 60)

    # Create config for FunctionGemma 270M
    config = GemmaConfig.functiongemma_270m()
    print(f"Config: hidden_size={config.hidden_size}, layers={config.num_hidden_layers}")

    # Create model (without pretrained weights for demo)
    model = GemmaForCausalLM(config)
    print("Model architecture created!")

    # For actual inference, you would load weights:
    # weights = mx.load("path/to/model.safetensors")
    # model.update(weights)

    # Quick forward pass test with random input
    test_input = mx.array([[1, 2, 3, 4, 5]])
    output = model(test_input)
    print(f"Output shape: {output.logits.shape}")
    print(f"Expected: (1, 5, {config.vocab_size})")


if __name__ == "__main__":
    # Run the main inference example
    main()

    # Optionally show native implementation
    print("\n" + "=" * 60)
    demo_with_chuk_lazarus()
