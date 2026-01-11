"""Base classes and utilities for CLI command handlers.

This module provides shared base classes for command configurations and results,
ensuring consistent patterns across all CLI modules.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from argparse import Namespace
from pathlib import Path
from typing import Any, ClassVar, TypeVar

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class CommandConfig(BaseModel, ABC):
    """Base configuration for CLI commands.

    All command configs should inherit from this class and implement
    the `from_args` classmethod to parse argparse.Namespace objects.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_default=True,
    )

    @classmethod
    @abstractmethod
    def from_args(cls, args: Namespace) -> CommandConfig:
        """Create config from argparse Namespace.

        Args:
            args: Parsed command-line arguments

        Returns:
            Validated configuration instance
        """
        ...


class CommandResult(BaseModel, ABC):
    """Base result for CLI commands.

    All command results should inherit from this class and implement
    the `to_display` method for consistent output formatting.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    @abstractmethod
    def to_display(self) -> str:
        """Format result for display.

        Returns:
            Human-readable string representation
        """
        ...


# Type variables for generic command patterns
ConfigT = TypeVar("ConfigT", bound=CommandConfig)
ResultT = TypeVar("ResultT", bound=CommandResult)


class OutputMixin:
    """Mixin providing common output formatting utilities."""

    SEPARATOR: ClassVar[str] = "=" * 60

    @staticmethod
    def format_header(title: str, width: int = 60) -> str:
        """Format a section header.

        Args:
            title: Header title
            width: Total width of separator

        Returns:
            Formatted header string
        """
        sep = "=" * width
        return f"\n{sep}\n{title}\n{sep}"

    @staticmethod
    def format_field(name: str, value: Any, indent: int = 2) -> str:
        """Format a field for display.

        Args:
            name: Field name
            value: Field value
            indent: Number of spaces for indentation

        Returns:
            Formatted field string
        """
        prefix = " " * indent
        return f"{prefix}{name}: {value}"

    @staticmethod
    def format_table_row(
        columns: list[tuple[str, Any]],
        widths: list[int] | None = None,
    ) -> str:
        """Format a table row.

        Args:
            columns: List of (name, value) tuples
            widths: Optional column widths

        Returns:
            Formatted row string
        """
        if widths is None:
            widths = [20] * len(columns)

        parts = []
        for (name, value), width in zip(columns, widths):
            formatted = f"{value}"
            parts.append(formatted.ljust(width))
        return "  ".join(parts)


class PathMixin:
    """Mixin providing common path handling utilities."""

    @staticmethod
    def ensure_parent_exists(path: Path) -> Path:
        """Ensure parent directory exists.

        Args:
            path: File path

        Returns:
            The same path (for chaining)
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def resolve_path(path: str | Path | None) -> Path | None:
        """Resolve a path to absolute form.

        Args:
            path: Path to resolve

        Returns:
            Resolved absolute path or None
        """
        if path is None:
            return None
        return Path(path).resolve()


# Common field definitions for reuse
class CommonFields:
    """Common Pydantic field definitions for command configs."""

    @staticmethod
    def tokenizer_field() -> Any:
        """Field for tokenizer path/name."""
        return Field(
            ...,
            description="Path or HuggingFace name of the tokenizer",
        )

    @staticmethod
    def model_field() -> Any:
        """Field for model path/name."""
        return Field(
            ...,
            description="Path or HuggingFace name of the model",
        )

    @staticmethod
    def output_field() -> Any:
        """Field for optional output path."""
        return Field(
            default=None,
            description="Output file path",
        )

    @staticmethod
    def verbose_field() -> Any:
        """Field for verbose flag."""
        return Field(
            default=False,
            description="Enable verbose output",
        )

    @staticmethod
    def seed_field() -> Any:
        """Field for random seed."""
        return Field(
            default=None,
            description="Random seed for reproducibility",
        )


__all__ = [
    "CommandConfig",
    "CommandResult",
    "CommonFields",
    "ConfigT",
    "OutputMixin",
    "PathMixin",
    "ResultT",
    "logger",
]
