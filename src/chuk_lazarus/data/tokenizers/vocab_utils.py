"""
Vocabulary I/O utilities with Pydantic models and async support.

This module provides async-native vocabulary loading and saving using
the VocabularyData Pydantic model - no raw dictionary operations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import aiofiles

from chuk_lazarus.data.tokenizers.types import VocabularyData

if TYPE_CHECKING:
    import os

# =============================================================================
# Constants
# =============================================================================

VOCAB_FILENAME = "tokenizer.json"
DEFAULT_VERSION = "1.0"


# =============================================================================
# Async I/O Functions (preferred)
# =============================================================================


async def load_vocabulary_async(vocab_file: str | Path) -> VocabularyData:
    """
    Load vocabulary from a JSON file asynchronously.

    Args:
        vocab_file: Path to the vocabulary JSON file.

    Returns:
        VocabularyData with vocab, special_tokens, and added_tokens.

    Raises:
        ValueError: If vocab_file is not provided or doesn't exist.
        ValidationError: If the JSON doesn't match the expected schema.
    """
    path = Path(vocab_file)

    if not path.exists():
        raise ValueError(f"Vocabulary file not found: {path}")

    async with aiofiles.open(path, encoding="utf-8") as f:
        content = await f.read()

    data = json.loads(content)

    return VocabularyData(
        vocab=data.get("vocab", {}),
        special_tokens=data.get("special_tokens", {}),
        added_tokens=data.get("added_tokens", []),
    )


async def save_vocabulary_async(
    vocab_data: VocabularyData,
    save_directory: str | Path,
    version: str = DEFAULT_VERSION,
) -> Path:
    """
    Save vocabulary to a JSON file asynchronously.

    Args:
        vocab_data: The VocabularyData model to save.
        save_directory: Directory to save the vocabulary file.
        version: Version string for the vocabulary format.

    Returns:
        Path to the saved vocabulary file.
    """
    save_dir = Path(save_directory)
    save_dir.mkdir(parents=True, exist_ok=True)

    vocab_file = save_dir / VOCAB_FILENAME

    output = {
        "version": version,
        "vocab": vocab_data.vocab,
        "special_tokens": vocab_data.special_tokens,
        "added_tokens": vocab_data.added_tokens,
    }

    async with aiofiles.open(vocab_file, mode="w", encoding="utf-8") as f:
        await f.write(json.dumps(output, indent=2))

    return vocab_file


# =============================================================================
# Sync I/O Functions (for compatibility with sync contexts)
# =============================================================================


def load_vocabulary(vocab_file: str | Path | os.PathLike[str]) -> VocabularyData:
    """
    Load vocabulary from a JSON file synchronously.

    Args:
        vocab_file: Path to the vocabulary JSON file.

    Returns:
        VocabularyData with vocab, special_tokens, and added_tokens.

    Raises:
        ValueError: If vocab_file is not provided or doesn't exist.
        ValidationError: If the JSON doesn't match the expected schema.
    """
    path = Path(vocab_file)

    if not path.exists():
        raise ValueError(f"Vocabulary file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    return VocabularyData(
        vocab=data.get("vocab", {}),
        special_tokens=data.get("special_tokens", {}),
        added_tokens=data.get("added_tokens", []),
    )


def save_vocabulary(
    vocab_data: VocabularyData,
    save_directory: str | Path,
    version: str = DEFAULT_VERSION,
) -> Path:
    """
    Save vocabulary to a JSON file synchronously.

    Args:
        vocab_data: The VocabularyData model to save.
        save_directory: Directory to save the vocabulary file.
        version: Version string for the vocabulary format.

    Returns:
        Path to the saved vocabulary file.
    """
    save_dir = Path(save_directory)
    save_dir.mkdir(parents=True, exist_ok=True)

    vocab_file = save_dir / VOCAB_FILENAME

    output = {
        "version": version,
        "vocab": vocab_data.vocab,
        "special_tokens": vocab_data.special_tokens,
        "added_tokens": vocab_data.added_tokens,
    }

    with open(vocab_file, mode="w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    return vocab_file
