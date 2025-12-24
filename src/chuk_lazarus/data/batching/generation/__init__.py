"""
Batch file generation and I/O.

This submodule provides:
- LengthCache: Async cache for sequence lengths
- BatchWriter: Write BatchPlan to NPZ files
- BatchReader: Read NPZ batch files

This is the unified batch generation module (Phase 5.6),
replacing the old separate batch_generation module.
"""

from .io import (
    BatchReader,
    BatchWriter,
    CollatedBatch,
    default_collate,
    pad_sequences,
)
from .length_cache import (
    LengthCache,
    LengthEntry,
)

__all__ = [
    # Length cache
    "LengthCache",
    "LengthEntry",
    # Batch I/O
    "BatchWriter",
    "BatchReader",
    "CollatedBatch",
    "default_collate",
    "pad_sequences",
]
