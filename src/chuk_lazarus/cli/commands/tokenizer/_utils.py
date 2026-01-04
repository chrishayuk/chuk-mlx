"""Shared utilities for tokenizer CLI commands."""

from pathlib import Path


def load_texts(file: Path | None) -> list[str]:
    """Load texts from file or stdin.

    Args:
        file: Path to text file, or None for stdin.

    Returns:
        List of text strings.
    """
    if file:
        with open(file) as f:
            return [line.strip() for line in f if line.strip()]
    else:
        print("Enter texts (one per line, Ctrl+D to finish):")
        texts = []
        try:
            while True:
                line = input()
                if line.strip():
                    texts.append(line.strip())
        except EOFError:
            pass
        return texts
