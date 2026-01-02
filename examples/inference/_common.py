"""
Shared utilities for inference examples.

Provides common argument parsing, chat loops, and result display.
"""

from __future__ import annotations

import argparse
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chuk_lazarus.inference import UnifiedPipeline


def create_parser(
    description: str,
    model_aliases: dict[str, Enum],
    default_model: str = None,
    *,
    include_chat: bool = True,
    temperature: float = 0.7,
    max_tokens: int = 256,
    prompt: str = "What is the capital of France?",
) -> argparse.ArgumentParser:
    """Create a standard argument parser for inference examples.

    Args:
        description: Description for the example
        model_aliases: Dict of alias -> model enum
        default_model: Default model alias (first key if not specified)
        include_chat: Include --chat flag
        temperature: Default temperature
        max_tokens: Default max tokens
        prompt: Default prompt
    """
    if default_model is None:
        default_model = next(iter(model_aliases.keys()))

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", choices=list(model_aliases.keys()), default=default_model)
    parser.add_argument("--model-id", help="Custom HuggingFace model ID")
    parser.add_argument("--prompt", default=prompt)
    parser.add_argument("--max-tokens", type=int, default=max_tokens)
    parser.add_argument("--temperature", type=float, default=temperature)
    if include_chat:
        parser.add_argument("--chat", action="store_true", help="Interactive chat mode")
    parser.add_argument("--list", action="store_true", help="List models")
    return parser


def list_models(model_aliases: dict[str, Enum], width: int = 15) -> None:
    """Print available models."""
    print("Available models:")
    for alias, model in model_aliases.items():
        print(f"  {alias:{width}} -> {model.value}")


def chat_loop(pipeline: UnifiedPipeline, name: str = "Assistant") -> None:
    """Run an interactive chat loop."""
    from chuk_lazarus.inference import ChatHistory

    print(f"\n{name} Chat (type 'quit' to exit)")
    history = ChatHistory()
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not user_input or user_input.lower() == "quit":
            break
        history.add_user(user_input)
        result = pipeline.chat_with_history(history)
        print(f"\n{name}: {result.text}")
        history.add_assistant(result.text)


def print_result(pipeline: UnifiedPipeline, prompt: str) -> None:
    """Generate and print a chat result."""
    result = pipeline.chat(prompt)
    print(f"\nUser: {prompt}")
    print(f"Assistant: {result.text}")
    print(f"[{result.stats.tokens_per_second:.1f} tok/s]")
