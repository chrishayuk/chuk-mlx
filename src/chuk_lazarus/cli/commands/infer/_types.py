"""Type definitions for inference CLI commands.

This module contains Pydantic models and enums for the infer command.
"""

from __future__ import annotations

from argparse import Namespace
from enum import Enum
from pathlib import Path

from pydantic import Field

from .._base import CommandConfig, CommandResult, OutputMixin


class InputMode(str, Enum):
    """Mode for providing input prompts."""

    SINGLE = "single"
    FILE = "file"
    INTERACTIVE = "interactive"


class InferenceConfig(CommandConfig):
    """Configuration for inference command.

    Attributes:
        model: Path or HuggingFace name of the model
        adapter: Optional path to LoRA adapter
        prompt: Single prompt text
        prompt_file: Path to file with prompts
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
    """

    model: str = Field(
        ...,
        description="Path or HuggingFace name of the model",
    )
    adapter: str | None = Field(
        default=None,
        description="Path to LoRA adapter weights",
    )
    prompt: str | None = Field(
        default=None,
        description="Single prompt text",
    )
    prompt_file: Path | None = Field(
        default=None,
        description="Path to file with prompts (one per line)",
    )
    max_tokens: int = Field(
        default=100,
        ge=1,
        le=8192,
        description="Maximum tokens to generate",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )

    @classmethod
    def from_args(cls, args: Namespace) -> InferenceConfig:
        """Create config from argparse Namespace."""
        return cls(
            model=args.model,
            adapter=getattr(args, "adapter", None),
            prompt=getattr(args, "prompt", None),
            prompt_file=getattr(args, "prompt_file", None),
            max_tokens=getattr(args, "max_tokens", 100),
            temperature=getattr(args, "temperature", 0.7),
        )

    @property
    def input_mode(self) -> InputMode:
        """Determine input mode based on config."""
        if self.prompt:
            return InputMode.SINGLE
        elif self.prompt_file:
            return InputMode.FILE
        else:
            return InputMode.INTERACTIVE


class GenerationResult(CommandResult, OutputMixin):
    """Result of a single generation.

    Attributes:
        prompt: The input prompt
        response: Generated response text
        tokens_generated: Number of tokens generated
    """

    prompt: str = Field(
        ...,
        description="The input prompt",
    )
    response: str = Field(
        ...,
        description="Generated response text",
    )
    tokens_generated: int = Field(
        default=0,
        ge=0,
        description="Number of tokens generated",
    )

    def to_display(self) -> str:
        """Format result for display."""
        return f"\nPrompt: {self.prompt}\nResponse: {self.response}"


class InferenceResult(CommandResult, OutputMixin):
    """Result of inference command with multiple generations.

    Attributes:
        generations: List of generation results
        model: Model used for inference
        adapter: Optional adapter used
    """

    generations: list[GenerationResult] = Field(
        default_factory=list,
        description="List of generation results",
    )
    model: str = Field(
        ...,
        description="Model used for inference",
    )
    adapter: str | None = Field(
        default=None,
        description="Adapter used (if any)",
    )

    def to_display(self) -> str:
        """Format result for display."""
        lines = [self.format_header("Inference Results")]
        lines.append(self.format_field("Model", self.model))
        if self.adapter:
            lines.append(self.format_field("Adapter", self.adapter))
        lines.append(self.format_field("Generations", len(self.generations)))
        lines.append("")

        for gen in self.generations:
            lines.append(gen.to_display())

        return "\n".join(lines)


__all__ = [
    "GenerationResult",
    "InferenceConfig",
    "InferenceResult",
    "InputMode",
]
