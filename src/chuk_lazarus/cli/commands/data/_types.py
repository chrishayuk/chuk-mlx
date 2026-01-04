"""Shared types for data CLI commands."""

from enum import Enum


class SampleTextField(str, Enum):
    """Field names for sample text content."""

    TEXT = "text"
    CONTENT = "content"
    INPUT = "input"
    MESSAGES = "messages"


class SampleIdField(str, Enum):
    """Field names for sample IDs."""

    ID = "id"
    SAMPLE_ID = "sample_id"


class OutputFormat(str, Enum):
    """Output format options."""

    JSON = "json"
    JSONL = "jsonl"
    TEXT = "text"
