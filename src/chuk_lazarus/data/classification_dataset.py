"""
Classification dataset for text classification tasks.

Provides a simple, Pydantic-native dataset for loading classification
data from JSONL files. Supports:
- Text classification (sentiment, topic, etc.)
- Multi-class and binary classification
- Train/test/validation splits
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Iterator


class ClassificationSample(BaseModel):
    """A single classification sample."""

    model_config = {"frozen": True}

    text: str = Field(description="Input text")
    label: int = Field(description="Class label (integer)")
    sample_id: str | None = Field(default=None, description="Optional sample ID")


class ClassificationDataset:
    """
    Dataset for text classification tasks.

    Loads data from JSONL files with format:
        {"text": "...", "label": 0}
        {"text": "...", "label": 1}

    Example:
        >>> dataset = ClassificationDataset.from_jsonl("train.jsonl")
        >>> len(dataset)
        100
        >>> dataset[0]
        ClassificationSample(text="great movie", label=1)

        >>> for sample in dataset:
        ...     print(sample.text, sample.label)
    """

    def __init__(self, samples: list[ClassificationSample]) -> None:
        """
        Initialize with list of samples.

        Args:
            samples: List of ClassificationSample objects.
        """
        self._samples = samples
        self._label_set = frozenset(s.label for s in samples)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> ClassificationDataset:
        """
        Load dataset from JSONL file.

        Expected format (one JSON object per line):
            {"text": "sample text", "label": 0}
            {"text": "another sample", "label": 1}

        Args:
            path: Path to JSONL file.

        Returns:
            ClassificationDataset instance.
        """
        path = Path(path)
        samples = []

        with open(path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                sample = ClassificationSample(
                    text=data["text"],
                    label=data["label"],
                    sample_id=data.get("id", f"{path.stem}_{i}"),
                )
                samples.append(sample)

        return cls(samples)

    @classmethod
    def from_list(
        cls,
        data: list[tuple[str, int]],
    ) -> ClassificationDataset:
        """
        Create dataset from list of (text, label) tuples.

        Args:
            data: List of (text, label) tuples.

        Returns:
            ClassificationDataset instance.
        """
        samples = [
            ClassificationSample(text=text, label=label, sample_id=f"sample_{i}")
            for i, (text, label) in enumerate(data)
        ]
        return cls(samples)

    @property
    def num_classes(self) -> int:
        """Number of unique classes in the dataset."""
        return len(self._label_set)

    @property
    def labels(self) -> frozenset[int]:
        """Set of unique labels."""
        return self._label_set

    @property
    def texts(self) -> list[str]:
        """List of all texts."""
        return [s.text for s in self._samples]

    def __len__(self) -> int:
        """Number of samples."""
        return len(self._samples)

    def __getitem__(self, idx: int) -> ClassificationSample:
        """Get sample by index."""
        return self._samples[idx]

    def __iter__(self) -> Iterator[ClassificationSample]:
        """Iterate over samples."""
        return iter(self._samples)

    def get_label_counts(self) -> dict[int, int]:
        """Get count of samples per label."""
        counts: dict[int, int] = {}
        for sample in self._samples:
            counts[sample.label] = counts.get(sample.label, 0) + 1
        return counts

    def filter_by_label(self, label: int) -> ClassificationDataset:
        """Return new dataset with only samples of given label."""
        filtered = [s for s in self._samples if s.label == label]
        return ClassificationDataset(filtered)

    def split(
        self,
        train_ratio: float = 0.8,
        seed: int | None = None,
    ) -> tuple[ClassificationDataset, ClassificationDataset]:
        """
        Split dataset into train and test sets.

        Args:
            train_ratio: Fraction of data for training (0-1).
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (train_dataset, test_dataset).
        """
        import random

        indices = list(range(len(self._samples)))
        if seed is not None:
            random.seed(seed)
        random.shuffle(indices)

        split_idx = int(len(indices) * train_ratio)
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]

        train_samples = [self._samples[i] for i in train_indices]
        test_samples = [self._samples[i] for i in test_indices]

        return ClassificationDataset(train_samples), ClassificationDataset(test_samples)

    def __repr__(self) -> str:
        """String representation."""
        return f"ClassificationDataset(samples={len(self)}, classes={self.num_classes})"


def load_classification_data(path: str | Path) -> ClassificationDataset:
    """
    Load classification dataset from JSONL file.

    Convenience function for ClassificationDataset.from_jsonl().

    Args:
        path: Path to JSONL file.

    Returns:
        ClassificationDataset instance.
    """
    return ClassificationDataset.from_jsonl(path)
