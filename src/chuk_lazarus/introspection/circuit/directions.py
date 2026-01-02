"""
Direction extraction for activation steering.

Extracts interpretable directions from activation spaces.
Generic - works with any binary classification task.

Example use cases:
- Arithmetic: direction from "compute" to "suppress"
- Tool-calling: direction from "no tool" to "use tool"
- Safety: direction from "safe" to "unsafe"
- Factual: direction from "lie" to "truth"

Example:
    >>> from chuk_lazarus.introspection.circuit import (
    ...     CollectedActivations, DirectionExtractor, DirectionMethod
    ... )
    >>>
    >>> activations = CollectedActivations.load("arithmetic_activations")
    >>> extractor = DirectionExtractor(activations)
    >>>
    >>> # Extract direction separating positive from negative class
    >>> direction = extractor.extract_direction(layer=24)
    >>> print(f"Separation score: {direction.separation_score:.3f}")
    >>> print(f"Classification accuracy: {direction.accuracy:.1%}")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .collector import CollectedActivations


class DirectionMethod(str, Enum):
    """Method for extracting directions."""

    DIFFERENCE_OF_MEANS = "diff_means"
    LDA = "lda"
    PROBE_WEIGHTS = "probe_weights"
    CONTRASTIVE = "contrastive"
    PCA = "pca"  # First principal component of the difference


@dataclass
class ExtractedDirection:
    """A direction vector with metadata.

    Generic - the direction can represent any linear feature in activation space.
    """

    name: str
    layer: int
    direction: np.ndarray  # [hidden_size]
    method: DirectionMethod

    # Statistics
    mean_projection_positive: float = 0.0  # Mean projection for positive class (label=1)
    mean_projection_negative: float = 0.0  # Mean projection for negative class (label=0)
    separation_score: float = 0.0  # Cohen's d or similar

    # Validation
    accuracy: float = 0.0  # Classification accuracy using this direction
    correlation_with_output: float = 0.0  # Correlation with model behavior

    # Label info (for interpretation)
    positive_label: str = "positive"
    negative_label: str = "negative"

    metadata: dict = field(default_factory=dict)

    @property
    def normalized_direction(self) -> np.ndarray:
        """Get unit-normalized direction."""
        norm = np.linalg.norm(self.direction)
        if norm > 0:
            return self.direction / norm
        return self.direction

    def project(self, activations: np.ndarray) -> np.ndarray:
        """Project activations onto this direction."""
        direction = self.normalized_direction
        return activations @ direction

    def classify(self, activations: np.ndarray) -> np.ndarray:
        """Classify activations using the direction and midpoint threshold."""
        projections = self.project(activations)
        threshold = (self.mean_projection_positive + self.mean_projection_negative) / 2
        return (projections > threshold).astype(int)

    def summary(self) -> dict:
        return {
            "name": self.name,
            "layer": self.layer,
            "method": self.method.value,
            "norm": float(np.linalg.norm(self.direction)),
            "separation_score": self.separation_score,
            "accuracy": self.accuracy,
            "positive_label": self.positive_label,
            "negative_label": self.negative_label,
        }


@dataclass
class DirectionBundle:
    """Collection of related directions across layers."""

    name: str
    directions: dict[int, ExtractedDirection] = field(default_factory=dict)  # layer -> direction

    # Metadata
    model_id: str = ""
    positive_label: str = "positive"
    negative_label: str = "negative"

    def add(self, direction: ExtractedDirection) -> None:
        """Add a direction to the bundle."""
        self.directions[direction.layer] = direction

    def get(self, layer: int) -> ExtractedDirection | None:
        """Get direction for a specific layer."""
        return self.directions.get(layer)

    @property
    def layers(self) -> list[int]:
        """Get sorted list of layers."""
        return sorted(self.directions.keys())

    def get_separation_by_layer(self) -> dict[int, float]:
        """Get separation score for each layer."""
        return {layer: d.separation_score for layer, d in self.directions.items()}

    def get_accuracy_by_layer(self) -> dict[int, float]:
        """Get classification accuracy for each layer."""
        return {layer: d.accuracy for layer, d in self.directions.items()}

    def find_best_layer(self) -> int | None:
        """Find layer with highest separation score."""
        if not self.directions:
            return None
        return max(
            self.directions.keys(), key=lambda layer: self.directions[layer].separation_score
        )

    def save(self, path: str | Path) -> None:
        """Save directions to files."""
        path = Path(path)

        # Save directions as numpy
        arrays = {}
        for layer, direction in self.directions.items():
            arrays[f"layer_{layer}"] = direction.direction

        np.savez(path.with_suffix(".npz"), **arrays)

        # Save metadata as JSON
        metadata = {
            "name": self.name,
            "model_id": self.model_id,
            "positive_label": self.positive_label,
            "negative_label": self.negative_label,
            "directions": {str(layer): d.summary() for layer, d in self.directions.items()},
        }

        with open(path.with_suffix(".json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved directions to: {path}")

    @classmethod
    def load(cls, path: str | Path) -> DirectionBundle:
        """Load directions from files."""
        path = Path(path)

        # Load arrays
        data = np.load(path.with_suffix(".npz"))

        # Load metadata
        with open(path.with_suffix(".json")) as f:
            metadata = json.load(f)

        bundle = cls(
            name=metadata["name"],
            model_id=metadata.get("model_id", ""),
            positive_label=metadata.get("positive_label", "positive"),
            negative_label=metadata.get("negative_label", "negative"),
        )

        for key, arr in data.items():
            layer = int(key.split("_")[1])
            dir_meta = metadata["directions"].get(str(layer), {})
            bundle.add(
                ExtractedDirection(
                    name=dir_meta.get("name", f"direction_L{layer}"),
                    layer=layer,
                    direction=arr,
                    method=DirectionMethod(dir_meta.get("method", "diff_means")),
                    separation_score=dir_meta.get("separation_score", 0.0),
                    accuracy=dir_meta.get("accuracy", 0.0),
                    positive_label=metadata.get("positive_label", "positive"),
                    negative_label=metadata.get("negative_label", "negative"),
                )
            )

        return bundle


class DirectionExtractor:
    """
    Extracts interpretable directions from activations.

    Generic - works with any binary classification in the activations.

    Example:
        >>> extractor = DirectionExtractor(activations)
        >>> direction = extractor.extract_direction(layer=24)
        >>> bundle = extractor.extract_all_layers()
    """

    def __init__(self, activations: CollectedActivations):
        self.activations = activations

    def extract_direction(
        self,
        layer: int,
        method: DirectionMethod = DirectionMethod.DIFFERENCE_OF_MEANS,
        positive_label: int = 1,
        negative_label: int = 0,
    ) -> ExtractedDirection:
        """
        Extract direction separating two classes at a specific layer.

        Args:
            layer: Layer to extract from
            method: Extraction method
            positive_label: Label for positive class (default 1)
            negative_label: Label for negative class (default 0)

        Returns:
            ExtractedDirection
        """
        X = self.activations.get_activations_numpy(layer)
        labels = np.array(self.activations.labels)

        pos_mask = labels == positive_label
        neg_mask = labels == negative_label

        pos_acts = X[pos_mask]
        neg_acts = X[neg_mask]

        if method == DirectionMethod.DIFFERENCE_OF_MEANS:
            direction = self._diff_of_means(pos_acts, neg_acts)
        elif method == DirectionMethod.LDA:
            direction = self._lda_direction(X, labels)
        elif method == DirectionMethod.PROBE_WEIGHTS:
            direction = self._probe_weights(X, labels)
        elif method == DirectionMethod.PCA:
            direction = self._pca_direction(pos_acts, neg_acts)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Compute statistics
        direction_norm = direction / (np.linalg.norm(direction) + 1e-8)
        pos_projections = pos_acts @ direction_norm
        neg_projections = neg_acts @ direction_norm

        mean_pos = pos_projections.mean()
        mean_neg = neg_projections.mean()
        std_pooled = np.sqrt((pos_projections.std() ** 2 + neg_projections.std() ** 2) / 2)
        separation = abs(mean_pos - mean_neg) / (std_pooled + 1e-8)

        # Classification accuracy using threshold at midpoint
        threshold = (mean_pos + mean_neg) / 2
        projections = X @ direction_norm
        predictions = projections > threshold
        accuracy = (predictions == (labels == positive_label)).mean()

        # Get label names from dataset
        pos_name = self.activations.dataset_label_names.get(positive_label, "positive")
        neg_name = self.activations.dataset_label_names.get(negative_label, "negative")

        return ExtractedDirection(
            name=f"direction_L{layer}",
            layer=layer,
            direction=direction,
            method=method,
            mean_projection_positive=float(mean_pos),
            mean_projection_negative=float(mean_neg),
            separation_score=float(separation),
            accuracy=float(accuracy),
            positive_label=pos_name,
            negative_label=neg_name,
        )

    def extract_all_layers(
        self,
        method: DirectionMethod = DirectionMethod.DIFFERENCE_OF_MEANS,
        positive_label: int = 1,
        negative_label: int = 0,
    ) -> DirectionBundle:
        """
        Extract directions for all captured layers.

        Returns:
            DirectionBundle with directions for each layer
        """
        bundle = DirectionBundle(
            name=f"{self.activations.dataset_name}_directions",
            model_id=self.activations.model_id,
            positive_label=self.activations.dataset_label_names.get(positive_label, "positive"),
            negative_label=self.activations.dataset_label_names.get(negative_label, "negative"),
        )

        for layer in self.activations.captured_layers:
            direction = self.extract_direction(layer, method, positive_label, negative_label)
            bundle.add(direction)

        return bundle

    def extract_per_category(
        self,
        layer: int,
        method: DirectionMethod = DirectionMethod.DIFFERENCE_OF_MEANS,
    ) -> dict[str, ExtractedDirection]:
        """
        Extract directions for each category vs others.

        Useful for understanding what each category "means" in activation space.
        """
        X = self.activations.get_activations_numpy(layer)
        categories = np.array(self.activations.categories)

        unique_cats = np.unique(categories)
        results = {}

        for cat in unique_cats:
            cat_mask = categories == cat
            cat_acts = X[cat_mask]
            other_acts = X[~cat_mask]

            if len(cat_acts) < 2:
                continue

            direction = self._diff_of_means(cat_acts, other_acts)

            # Compute statistics
            direction_norm = direction / (np.linalg.norm(direction) + 1e-8)
            cat_projections = cat_acts @ direction_norm
            other_projections = other_acts @ direction_norm

            results[cat] = ExtractedDirection(
                name=f"{cat}_direction",
                layer=layer,
                direction=direction,
                method=method,
                mean_projection_positive=float(cat_projections.mean()),
                mean_projection_negative=float(other_projections.mean()),
                separation_score=float(abs(cat_projections.mean() - other_projections.mean())),
                positive_label=cat,
                negative_label="other",
            )

        return results

    def check_orthogonality(
        self,
        directions: list[ExtractedDirection],
    ) -> np.ndarray:
        """
        Check orthogonality between directions.

        Returns cosine similarity matrix.
        """
        n = len(directions)
        similarities = np.zeros((n, n))

        for i, dir_i in enumerate(directions):
            for j, dir_j in enumerate(directions):
                similarities[i, j] = np.dot(dir_i.normalized_direction, dir_j.normalized_direction)

        return similarities

    def print_summary(self, bundle: DirectionBundle) -> None:
        """Print summary of direction extraction results."""
        print("\n" + "=" * 60)
        print(f"DIRECTION SUMMARY: {bundle.name}")
        print(f"Model: {bundle.model_id}")
        print(f"Labels: {bundle.negative_label} → {bundle.positive_label}")
        print("=" * 60)

        print(f"\n{'Layer':<8} {'Separation':<12} {'Accuracy':<10} {'Norm':<10}")
        print("-" * 40)

        for layer in bundle.layers:
            d = bundle.directions[layer]
            print(
                f"L{layer:<6} {d.separation_score:<12.3f} {d.accuracy:<10.1%} {np.linalg.norm(d.direction):<10.2f}"
            )

        best = bundle.find_best_layer()
        if best is not None:
            print(
                f"\nBest layer: L{best} (separation={bundle.directions[best].separation_score:.3f})"
            )

        print("=" * 60)

    @staticmethod
    def _diff_of_means(positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
        """Compute difference of means direction."""
        return positive.mean(axis=0) - negative.mean(axis=0)

    @staticmethod
    def _lda_direction(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Compute LDA direction (maximizes class separation)."""
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        lda = LinearDiscriminantAnalysis(n_components=1)
        lda.fit(X, labels)
        return lda.coef_.flatten()

    @staticmethod
    def _probe_weights(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Get direction from logistic regression probe weights."""
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X, labels)
        return clf.coef_.flatten()

    @staticmethod
    def _pca_direction(positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
        """Compute first PC of the concatenated difference vectors."""
        # Center each class
        pos_centered = positive - positive.mean(axis=0)
        neg_centered = negative - negative.mean(axis=0)

        # Concatenate
        all_centered = np.vstack([pos_centered, neg_centered])

        # PCA
        from sklearn.decomposition import PCA

        pca = PCA(n_components=1)
        pca.fit(all_centered)

        return pca.components_[0]


# =============================================================================
# Convenience functions
# =============================================================================


def extract_direction(
    activations: CollectedActivations,
    layer: int,
    method: DirectionMethod = DirectionMethod.DIFFERENCE_OF_MEANS,
) -> ExtractedDirection:
    """Convenience function to extract direction at a single layer."""
    extractor = DirectionExtractor(activations)
    return extractor.extract_direction(layer, method)


def extract_all_directions(
    activations: CollectedActivations,
    method: DirectionMethod = DirectionMethod.DIFFERENCE_OF_MEANS,
) -> DirectionBundle:
    """Convenience function to extract directions for all layers."""
    extractor = DirectionExtractor(activations)
    return extractor.extract_all_layers(method)
