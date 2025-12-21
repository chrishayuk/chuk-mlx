"""
Base Dataset - Abstract base class for all datasets.

This module provides a unified interface for datasets, reducing code duplication
and ensuring consistent behavior across SFTDataset, PreferenceDataset, etc.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

import mlx.core as mx


class BaseDataset(ABC):
    """
    Abstract base class for all datasets.

    Provides common functionality:
    - Batch iteration with shuffling
    - Padding utilities
    - Length and indexing interface

    Subclasses must implement:
    - __len__(): Return dataset size
    - __getitem__(): Return a single sample
    - _collate_batch(): Collate samples into a batch
    """

    def __init__(self):
        self._samples: list[Any] = []

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Get a single sample by index.

        Args:
            idx: Sample index

        Returns:
            Dictionary with sample data
        """
        pass

    @abstractmethod
    def _collate_batch(self, samples: list[dict], pad_token_id: int) -> dict[str, mx.array]:
        """
        Collate a list of samples into a padded batch.

        Args:
            samples: List of sample dictionaries
            pad_token_id: Token ID to use for padding

        Returns:
            Dictionary of batched and padded tensors
        """
        pass

    def iter_batches(
        self, batch_size: int, shuffle: bool = True, pad_token_id: int = 0, drop_last: bool = False
    ) -> Iterator[dict[str, mx.array]]:
        """
        Iterate over batches.

        Args:
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle samples
            pad_token_id: Token ID for padding
            drop_last: Whether to drop the last incomplete batch

        Yields:
            Dictionary of batched tensors
        """
        import random

        indices = list(range(len(self)))

        if shuffle:
            random.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            end = start + batch_size

            if drop_last and end > len(indices):
                break

            batch_indices = indices[start:end]
            samples = [self[i] for i in batch_indices]

            yield self._collate_batch(samples, pad_token_id)

    def get_batches(
        self, batch_size: int, shuffle: bool = False, pad_token_id: int = 0
    ) -> list[dict[str, mx.array]]:
        """
        Get all batches as a list (for PPO-style updates).

        Args:
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle samples
            pad_token_id: Token ID for padding

        Returns:
            List of batch dictionaries
        """
        return list(
            self.iter_batches(batch_size=batch_size, shuffle=shuffle, pad_token_id=pad_token_id)
        )

    @staticmethod
    def pad_sequences(
        sequences: list[list[int]],
        pad_value: int = 0,
        max_length: int | None = None,
        pad_left: bool = False,
    ) -> list[list[int]]:
        """
        Pad sequences to the same length.

        Args:
            sequences: List of token sequences
            pad_value: Value to use for padding
            max_length: Maximum length (None = max of sequences)
            pad_left: Whether to pad on the left side

        Returns:
            List of padded sequences
        """
        if not sequences:
            return []

        if max_length is None:
            max_length = max(len(seq) for seq in sequences)

        padded = []
        for seq in sequences:
            if len(seq) >= max_length:
                padded.append(seq[:max_length])
            else:
                padding = [pad_value] * (max_length - len(seq))
                if pad_left:
                    padded.append(padding + seq)
                else:
                    padded.append(seq + padding)

        return padded

    @staticmethod
    def create_attention_mask(sequences: list[list[int]], pad_value: int = 0) -> list[list[float]]:
        """
        Create attention masks for padded sequences.

        Args:
            sequences: Padded sequences
            pad_value: The padding value to mask

        Returns:
            Attention masks (1.0 for real tokens, 0.0 for padding)
        """
        return [[1.0 if token != pad_value else 0.0 for token in seq] for seq in sequences]

    @staticmethod
    def create_labels_with_mask(
        input_ids: list[list[int]],
        response_starts: list[int],
        pad_token_id: int = 0,
        ignore_index: int = -100,
    ) -> tuple:
        """
        Create labels and loss mask for causal LM training.

        Labels are shifted input_ids. Loss mask is 1.0 only for response tokens.

        Args:
            input_ids: Input token sequences
            response_starts: Index where response starts for each sequence
            pad_token_id: Padding token ID
            ignore_index: Index to ignore in loss (-100 for CrossEntropy)

        Returns:
            Tuple of (labels, loss_mask)
        """
        labels = []
        loss_masks = []

        for seq, resp_start in zip(input_ids, response_starts):
            # Labels are shifted by 1 (predict next token)
            label = seq[1:] + [pad_token_id]

            # Loss mask: only compute loss on response tokens
            mask = [0.0] * resp_start + [1.0] * (len(seq) - resp_start)

            # Set masked positions in labels to ignore_index
            label = [lbl if m > 0 else ignore_index for lbl, m in zip(label, mask)]

            labels.append(label)
            loss_masks.append(mask)

        return labels, loss_masks
