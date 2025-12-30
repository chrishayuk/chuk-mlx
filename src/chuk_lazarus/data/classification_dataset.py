"""
Classification dataset.

Supports:
- Text classification (sentiment, topic, etc.)
- Numeric feature classification (tabular data)
- Multi-class and binary classification
- Train/test/validation splits
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Iterator


class ClassificationSample(BaseModel):
    """A single classification sample."""

    model_config = {"frozen": True}

    label: int = Field(description="Class label (integer)")
    text: str | None = Field(default=None, description="Input text (for text classification)")
    features: list[float] | None = Field(
        default=None, description="Input features (for tabular data)"
    )


class ClassificationDataset:
    """
    Dataset for classification tasks.

    Supports both text and numeric features:

    Text classification:
        >>> dataset = ClassificationDataset.from_jsonl("train.jsonl")
        >>> dataset[0].text
        "great movie"

    Numeric features:
        >>> dataset = ClassificationDataset.from_features(X, y)
        >>> dataset[0].features
        [0.72, 0.85]
    """

    def __init__(self, samples: list[ClassificationSample]) -> None:
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
        """Create dataset from list of (text, label) tuples."""
        samples = [ClassificationSample(text=text, label=label) for text, label in data]
        return cls(samples)

    @classmethod
    def from_features(
        cls,
        X: list[list[float]],
        y: list[int],
    ) -> ClassificationDataset:
        """
        Create dataset from numeric features.

        Args:
            X: List of feature vectors, e.g., [[0.72, 0.85], [0.30, 0.25], ...]
            y: List of labels, e.g., [1, 0, ...]
        """
        samples = [
            ClassificationSample(features=features, label=label) for features, label in zip(X, y)
        ]
        return cls(samples)

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        feature_columns: list[str],
        label_column: str,
        normalize: float | None = None,
    ) -> ClassificationDataset:
        """
        Load dataset from CSV file.

        Args:
            path: Path to CSV file.
            feature_columns: Column names to use as features.
            label_column: Column name for labels.
            normalize: If set, divide features by this value (e.g., 100.0).

        Example:
            >>> dataset = ClassificationDataset.from_csv(
            ...     "exam.csv",
            ...     feature_columns=["coursework", "exam"],
            ...     label_column="label",
            ...     normalize=100.0,
            ... )
        """
        path = Path(path)
        samples = []

        with open(path) as f:
            for row in csv.DictReader(f):
                features = [float(row[col]) for col in feature_columns]
                if normalize:
                    features = [x / normalize for x in features]
                label = int(row[label_column])
                samples.append(ClassificationSample(features=features, label=label))

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
