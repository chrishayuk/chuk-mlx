"""Instrumentation commands for tokenizers."""

from .histogram import instrument_histogram
from .oov import instrument_oov
from .vocab_diff import instrument_vocab_diff
from .waste import instrument_waste

__all__ = [
    "instrument_histogram",
    "instrument_oov",
    "instrument_waste",
    "instrument_vocab_diff",
]
