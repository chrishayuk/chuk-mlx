"""
Dataset protocols for type-safe training pipelines.

Protocols define the interfaces that datasets must implement.
This enables trainers to work with any dataset that conforms to the interface,
without requiring inheritance from a specific base class.

The protocol approach is more Pythonic and flexible than ABC inheritance:
- Structural subtyping (duck typing with type hints)
- No need to modify existing dataset classes
- Composition over inheritance
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol, TypeVar, runtime_checkable

import mlx.core as mx

# Generic sample type
T = TypeVar("T")


@runtime_checkable
class Dataset(Protocol[T]):
    """
    Protocol for all datasets.

    Defines the minimal interface that trainers expect:
    - __len__: Number of samples
    - __getitem__: Access by index
    - __iter__: Iteration support
    """

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        ...

    def __getitem__(self, idx: int) -> T:
        """Get a single sample by index."""
        ...

    def __iter__(self) -> Iterator[T]:
        """Iterate over samples."""
        ...


@runtime_checkable
class BatchableDataset(Protocol):
    """
    Protocol for datasets that support batching.

    Extends Dataset with batch iteration capabilities.
    """

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        ...

    def __getitem__(self, idx: int) -> Any:
        """Get a single sample by index."""
        ...

    def iter_batches(
        self,
        batch_size: int,
        shuffle: bool = True,
        pad_token_id: int = 0,
    ) -> Iterator[dict[str, mx.array]]:
        """
        Iterate over batches.

        Args:
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle samples
            pad_token_id: Token ID for padding

        Yields:
            Dictionary of batched tensors
        """
        ...


@runtime_checkable
class SFTDatasetProtocol(Protocol):
    """
    Protocol for SFT (Supervised Fine-Tuning) datasets.

    SFT datasets provide:
    - Prompt-response pairs
    - Tokenized sequences with loss masks
    - Batch iteration with proper padding
    """

    def __len__(self) -> int:
        """Return the number of samples."""
        ...

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Get a single tokenized sample.

        Returns:
            Dictionary with:
            - input_ids: Token IDs
            - labels: Target token IDs
            - loss_mask: Mask for loss computation
            - prompt_length: Length of prompt (for masking)
        """
        ...

    def iter_batches(
        self,
        batch_size: int,
        shuffle: bool = True,
        pad_token_id: int = 0,
    ) -> Iterator[dict[str, mx.array]]:
        """
        Iterate over batches.

        Yields:
            Dictionary with batched and padded tensors:
            - input_ids: (batch, seq_len)
            - labels: (batch, seq_len)
            - loss_mask: (batch, seq_len)
            - attention_mask: (batch, seq_len)
        """
        ...


@runtime_checkable
class PreferenceDatasetProtocol(Protocol):
    """
    Protocol for preference datasets (DPO, RLHF).

    Preference datasets provide:
    - Prompt with chosen and rejected responses
    - Tokenized pairs with proper masking
    """

    def __len__(self) -> int:
        """Return the number of preference pairs."""
        ...

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Get a single tokenized preference pair.

        Returns:
            Dictionary with:
            - prompt_length: Length of prompt
            - chosen_input_ids: Token IDs for chosen response
            - rejected_input_ids: Token IDs for rejected response
        """
        ...

    def iter_batches(
        self,
        batch_size: int,
        shuffle: bool = True,
        pad_token_id: int = 0,
    ) -> Iterator[dict[str, mx.array]]:
        """
        Iterate over batches of preference pairs.

        Yields:
            Dictionary with batched tensors:
            - chosen_input_ids: (batch, seq_len)
            - rejected_input_ids: (batch, seq_len)
            - chosen_attention_mask: (batch, seq_len)
            - rejected_attention_mask: (batch, seq_len)
            - prompt_lengths: (batch,)
        """
        ...


@runtime_checkable
class ClassificationDatasetProtocol(Protocol):
    """
    Protocol for classification datasets.

    Classification datasets provide:
    - Text samples with integer labels
    - Support for multi-class classification
    """

    def __len__(self) -> int:
        """Return the number of samples."""
        ...

    def __getitem__(self, idx: int) -> Any:
        """
        Get a single sample.

        Returns:
            ClassificationSample with text and label
        """
        ...

    @property
    def num_classes(self) -> int:
        """Number of unique classes."""
        ...

    @property
    def texts(self) -> list[str]:
        """List of all texts."""
        ...


@runtime_checkable
class TokenizerProtocol(Protocol):
    """
    Protocol for tokenizers.

    Tokenizers convert text to token IDs and back.
    """

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        ...

    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs to text."""
        ...

    @property
    def vocab_size(self) -> int:
        """Vocabulary size."""
        ...

    @property
    def eos_token_id(self) -> int | None:
        """End-of-sequence token ID."""
        ...

    @property
    def pad_token_id(self) -> int | None:
        """Padding token ID."""
        ...
