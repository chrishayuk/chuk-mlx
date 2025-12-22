"""
Tokenizer backends for different performance profiles.

Provides a unified interface for tokenization with multiple backend implementations:
- huggingface: HuggingFace/SentencePiece compatible (default, portable)
- fast: MLX Data CharTrie-based (parallel, high-throughput)
"""

from .base import BackendType, TokenizerBackend
from .fast import FastBackend, is_fast_backend_available
from .huggingface import HuggingFaceBackend

# Backwards compatibility alias
CompatBackend = HuggingFaceBackend

__all__ = [
    "TokenizerBackend",
    "BackendType",
    "HuggingFaceBackend",
    "CompatBackend",  # Backwards compatibility
    "FastBackend",
    "is_fast_backend_available",
    "create_backend",
    "get_best_backend",
]


def create_backend(
    backend_type: BackendType | str,
    tokenizer_or_vocab: "TokenizerBackend | dict[str, int] | None" = None,
) -> TokenizerBackend:
    """
    Create a tokenizer backend of the specified type.

    Args:
        backend_type: Backend type to create
        tokenizer_or_vocab: Either a HuggingFace tokenizer (for huggingface backend)
                           or a vocabulary dict (for fast backend)

    Returns:
        TokenizerBackend instance
    """
    if isinstance(backend_type, str):
        backend_type = BackendType(backend_type)

    if backend_type in (BackendType.HUGGINGFACE, BackendType.COMPAT):
        return HuggingFaceBackend(tokenizer_or_vocab)
    elif backend_type == BackendType.FAST:
        if not is_fast_backend_available():
            raise ImportError(
                "Fast backend requires mlx-data. Install with: pip install chuk-lazarus[fast]"
            )
        return FastBackend(tokenizer_or_vocab)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")


def get_best_backend(
    tokenizer_or_vocab: "TokenizerBackend | dict[str, int] | None" = None,
    prefer_fast: bool = True,
) -> TokenizerBackend:
    """
    Get the best available backend.

    Args:
        tokenizer_or_vocab: Tokenizer or vocabulary to use
        prefer_fast: If True, use fast backend when available

    Returns:
        Best available TokenizerBackend
    """
    if prefer_fast and is_fast_backend_available():
        try:
            return FastBackend(tokenizer_or_vocab)
        except Exception:
            pass  # Fall back to HuggingFace

    return HuggingFaceBackend(tokenizer_or_vocab)
