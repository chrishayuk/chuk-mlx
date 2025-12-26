"""
Chat formatting and message handling utilities.

Provides typed structures for chat messages and formatting
with tokenizer chat templates.

Design principles:
- Use enums for roles, not strings
- Pydantic models for all data structures
- No dictionary goop
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


class Role(str, Enum):
    """Chat message roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    MODEL = "model"  # Used by some models (Gemma)

    def display_name(self) -> str:
        """Get display name for formatting."""
        return self.value.capitalize()


class ChatMessage(BaseModel):
    """A single chat message."""

    role: Role
    content: str

    def to_tokenizer_format(self) -> dict[str, str]:
        """Convert to dict format expected by tokenizers."""
        return {"role": self.role.value, "content": self.content}


class ChatHistory(BaseModel):
    """Container for chat conversation history."""

    messages: list[ChatMessage] = Field(default_factory=list)
    system_message: str | None = Field(None, description="Optional system prompt")

    def add_user(self, content: str) -> ChatHistory:
        """Add a user message."""
        self.messages.append(ChatMessage(role=Role.USER, content=content))
        return self

    def add_assistant(self, content: str) -> ChatHistory:
        """Add an assistant message."""
        self.messages.append(ChatMessage(role=Role.ASSISTANT, content=content))
        return self

    def add_system(self, content: str) -> ChatHistory:
        """Set the system message."""
        self.system_message = content
        return self

    def clear(self) -> ChatHistory:
        """Clear all messages but keep system prompt."""
        self.messages = []
        return self

    def to_tokenizer_format(self) -> list[dict[str, str]]:
        """Convert to list of message dicts for tokenizer."""
        result = []
        if self.system_message:
            result.append(
                ChatMessage(role=Role.SYSTEM, content=self.system_message).to_tokenizer_format()
            )
        for msg in self.messages:
            result.append(msg.to_tokenizer_format())
        return result


class FallbackTemplate(str, Enum):
    """Fallback templates when tokenizer has no chat template."""

    SIMPLE = "simple"
    CHATML = "chatml"


# Constants for formatting
ASSISTANT_SUFFIX = "Assistant:"
NEWLINE_DOUBLE = "\n\n"


def format_chat_prompt(
    tokenizer: PreTrainedTokenizer,
    user_message: str,
    system_message: str | None = None,
    add_generation_prompt: bool = True,
) -> str:
    """Format a single-turn chat prompt using tokenizer's template.

    Args:
        tokenizer: HuggingFace tokenizer with chat_template
        user_message: The user's message
        system_message: Optional system prompt
        add_generation_prompt: Whether to add generation prompt suffix

    Returns:
        Formatted prompt string
    """
    history = ChatHistory()
    if system_message:
        history.add_system(system_message)
    history.add_user(user_message)

    return format_history(tokenizer, history, add_generation_prompt=add_generation_prompt)


def format_history(
    tokenizer: PreTrainedTokenizer,
    history: ChatHistory,
    add_generation_prompt: bool = True,
) -> str:
    """Format chat history using tokenizer's template.

    Args:
        tokenizer: HuggingFace tokenizer with chat_template
        history: ChatHistory containing messages
        add_generation_prompt: Whether to add generation prompt suffix

    Returns:
        Formatted prompt string
    """
    messages = history.to_tokenizer_format()

    # Try tokenizer's chat template
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception:
            pass

    # Fallback to simple format
    return _format_simple(history)


def _format_simple(history: ChatHistory) -> str:
    """Simple fallback format for models without chat templates."""
    parts: list[str] = []

    if history.system_message:
        system_msg = ChatMessage(role=Role.SYSTEM, content=history.system_message)
        parts.append(f"{system_msg.role.display_name()}: {system_msg.content}")

    for msg in history.messages:
        parts.append(f"{msg.role.display_name()}: {msg.content}")

    prompt = NEWLINE_DOUBLE.join(parts)
    prompt += NEWLINE_DOUBLE + ASSISTANT_SUFFIX
    return prompt
