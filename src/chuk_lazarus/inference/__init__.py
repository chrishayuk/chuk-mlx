"""
Inference and text generation utilities.

Provides:
- generate_sequence: Token-by-token generation
- generate_response: Full response generation with tokenization
"""

from .generator import generate_response, generate_sequence

__all__ = [
    "generate_response",
    "generate_sequence",
]
