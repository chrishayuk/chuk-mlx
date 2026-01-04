"""Shared utilities for data CLI commands."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ._types import SampleIdField, SampleTextField

logger = logging.getLogger(__name__)


def load_dataset(path: Path | str) -> list[dict[str, Any]]:
    """Load a dataset from JSON or JSONL file.

    Args:
        path: Path to the dataset file.

    Returns:
        List of sample dictionaries.
    """
    path_str = str(path)
    with open(path_str) as f:
        if path_str.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        else:
            return json.load(f)


def get_sample_id(sample: dict[str, Any], index: int) -> str:
    """Extract sample ID from a sample dictionary.

    Args:
        sample: Sample dictionary.
        index: Index of the sample (used for auto-generation).

    Returns:
        Sample ID string.
    """
    return (
        sample.get(SampleIdField.ID.value)
        or sample.get(SampleIdField.SAMPLE_ID.value)
        or f"sample_{index:06d}"
    )


def get_sample_text(sample: dict[str, Any]) -> str | None:
    """Extract text content from a sample dictionary.

    Args:
        sample: Sample dictionary.

    Returns:
        Text content or None if not found.
    """
    text = (
        sample.get(SampleTextField.TEXT.value)
        or sample.get(SampleTextField.CONTENT.value)
        or sample.get(SampleTextField.INPUT.value)
    )
    if text is None and SampleTextField.MESSAGES.value in sample:
        messages = sample[SampleTextField.MESSAGES.value]
        text = " ".join(m.get("content", "") for m in messages)
    return text


def format_header(title: str, width: int = 60) -> str:
    """Format a section header.

    Args:
        title: Header title.
        width: Width of the header line.

    Returns:
        Formatted header string.
    """
    return f"\n{'=' * width}\n{title}\n{'=' * width}"
