"""
Geometry analysis for activation spaces.

Provides tools for understanding the structure of tool-calling activations:
- PCA: Dimensionality reduction and variance analysis
- UMAP: Nonlinear visualization
- Linear probes: Binary and multi-class classification
- Clustering: K-means and hierarchical

Example:
    >>> from chuk_lazarus.introspection.circuit import (
    ...     CollectedActivations, GeometryAnalyzer, train_linear_probe
    ... )
    >>>
    >>> activations = CollectedActivations.load("tool_activations")
    >>> analyzer = GeometryAnalyzer(activations)
    >>>
    >>> # PCA analysis
    >>> pca_result = analyzer.compute_pca(layer=11, n_components=50)
    >>> print(f"Components for 90% variance: {pca_result.components_for_variance(0.9)}")
    >>>
    >>> # Linear probe
    >>> probe = analyzer.train_probe(layer=11)
    >>> print(f"Accuracy: {probe.accuracy:.2%}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .collector import CollectedActivations


class ProbeType(str, Enum):
    """Type of linear probe to train."""

    BINARY = "binary"  # Tool vs no-tool
    MULTICLASS = "multiclass"  # Which category
    TOOL_TYPE = "tool_type"  # Which specific tool


@dataclass
class PCAResult:
    """Result of PCA analysis."""

    layer: int
    n_components: int
    explained_variance_ratio: np.ndarray
    cumulative_variance: np.ndarray
    components: np.ndarray  # [n_components, hidden_size]
    mean: np.ndarray
    transformed: np.ndarray | None = None  # [n_samples, n_components]

    def components_for_variance(self, threshold: float = 0.9) -> int:
        """Number of components needed to explain threshold variance."""
        return int(np.searchsorted(self.cumulative_variance, threshold) + 1)

    @property
    def intrinsic_dimensionality_90(self) -> int:
        """Intrinsic dimensionality at 90% variance."""
        return self.components_for_variance(0.9)

    @property
    def intrinsic_dimensionality_95(self) -> int:
        """Intrinsic dimensionality at 95% variance."""
        return self.components_for_variance(0.95)

    def summary(self) -> dict:
        return {
            "layer": self.layer,
            "n_components": self.n_components,
            "variance_1": float(self.explained_variance_ratio[0]),
            "variance_10": float(
                self.cumulative_variance[min(9, len(self.cumulative_variance) - 1)]
            ),
            "dim_90": self.intrinsic_dimensionality_90,
            "dim_95": self.intrinsic_dimensionality_95,
        }


@dataclass
class UMAPResult:
    """Result of UMAP projection."""

    layer: int
    embedding: np.ndarray  # [n_samples, 2 or 3]
    labels: np.ndarray
    category_labels: list[str]
    n_neighbors: int
    min_dist: float

    def get_tool_mask(self) -> np.ndarray:
        """Boolean mask for tool-calling samples."""
        return self.labels == 1

    def get_coordinates_by_category(self, category: str) -> np.ndarray:
        """Get UMAP coordinates for a specific category."""
        mask = np.array(self.category_labels) == category
        return self.embedding[mask]


@dataclass
class ProbeResult:
    """Result of linear probe training."""

    layer: int
    probe_type: ProbeType
    accuracy: float
    train_accuracy: float
    weights: np.ndarray
    bias: np.ndarray
    classes: list[str | int]

    # Per-class metrics
    precision: dict[str, float] = field(default_factory=dict)
    recall: dict[str, float] = field(default_factory=dict)
    f1: dict[str, float] = field(default_factory=dict)

    # Cross-validation results
    cv_accuracies: list[float] = field(default_factory=list)
    cv_mean: float = 0.0
    cv_std: float = 0.0

    def get_direction(self) -> np.ndarray:
        """Get the probe direction (for binary classification)."""
        if self.probe_type != ProbeType.BINARY:
            raise ValueError("Direction only available for binary probes")
        # For logistic regression, weights point towards positive class
        return self.weights.flatten()

    def summary(self) -> dict:
        return {
            "layer": self.layer,
            "probe_type": self.probe_type.value,
            "accuracy": self.accuracy,
            "train_accuracy": self.train_accuracy,
            "cv_mean": self.cv_mean,
            "cv_std": self.cv_std,
            "n_classes": len(self.classes),
        }


@dataclass
class ClusterResult:
    """Result of clustering analysis."""

    layer: int
    n_clusters: int
    labels: np.ndarray  # Cluster assignment per sample
    centroids: np.ndarray  # [n_clusters, hidden_size]
    inertia: float
    silhouette_score: float

    def get_cluster_sizes(self) -> dict[int, int]:
        """Get number of samples per cluster."""
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))


@dataclass
class GeometryResult:
    """Combined geometry analysis results."""

    layer: int
    pca: PCAResult | None = None
    umap: UMAPResult | None = None
    binary_probe: ProbeResult | None = None
    category_probe: ProbeResult | None = None
    tool_probe: ProbeResult | None = None
    clusters: ClusterResult | None = None

    # Cosine similarity matrix between categories
    category_similarities: np.ndarray | None = None

    def summary(self) -> dict:
        result = {"layer": self.layer}
        if self.pca:
            result["pca"] = self.pca.summary()
        if self.binary_probe:
            result["binary_probe"] = self.binary_probe.summary()
        if self.category_probe:
            result["category_probe"] = self.category_probe.summary()
        return result


class GeometryAnalyzer:
    """
    Analyzes the geometry of activation spaces.

    Provides PCA, UMAP, probes, and clustering analysis.
    """

    def __init__(self, activations: CollectedActivations):
        self.activations = activations
        self._validate()

    def _validate(self):
        """Validate that activations are usable."""
        if len(self.activations) == 0:
            raise ValueError("No activations in dataset")
        if not self.activations.captured_layers:
            raise ValueError("No layers captured in activations")

    def compute_pca(
        self,
        layer: int,
        n_components: int = 50,
        transform: bool = True,
    ) -> PCAResult:
        """
        Compute PCA on activations at a specific layer.

        Args:
            layer: Layer index
            n_components: Number of components to compute
            transform: Whether to transform the data

        Returns:
            PCAResult with explained variance and components
        """
        from sklearn.decomposition import PCA

        X = self.activations.get_activations_numpy(layer)
        if X is None:
            raise ValueError(f"Layer {layer} not in activations")

        n_components = min(n_components, X.shape[0], X.shape[1])
        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(X) if transform else None

        return PCAResult(
            layer=layer,
            n_components=n_components,
            explained_variance_ratio=pca.explained_variance_ratio_,
            cumulative_variance=np.cumsum(pca.explained_variance_ratio_),
            components=pca.components_,
            mean=pca.mean_,
            transformed=transformed,
        )

    def compute_umap(
        self,
        layer: int,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
    ) -> UMAPResult:
        """
        Compute UMAP projection for visualization.

        Args:
            layer: Layer index
            n_components: Output dimensions (2 or 3)
            n_neighbors: UMAP n_neighbors parameter
            min_dist: UMAP min_dist parameter

        Returns:
            UMAPResult with 2D/3D coordinates
        """
        try:
            from umap import UMAP
        except ImportError:
            raise ImportError("Install umap-learn: pip install umap-learn") from None

        X = self.activations.get_activations_numpy(layer)
        if X is None:
            raise ValueError(f"Layer {layer} not in activations")

        reducer = UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=42,
        )
        embedding = reducer.fit_transform(X)

        return UMAPResult(
            layer=layer,
            embedding=embedding,
            labels=np.array(self.activations.labels),
            category_labels=self.activations.category_labels,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
        )

    def train_probe(
        self,
        layer: int,
        probe_type: ProbeType = ProbeType.BINARY,
        test_size: float = 0.2,
        cv_folds: int = 5,
    ) -> ProbeResult:
        """
        Train a linear probe on activations.

        Args:
            layer: Layer index
            probe_type: Type of probe (binary, multiclass, tool_type)
            test_size: Fraction of data for testing
            cv_folds: Number of cross-validation folds

        Returns:
            ProbeResult with accuracy and weights
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.model_selection import cross_val_score, train_test_split

        X = self.activations.get_activations_numpy(layer)
        if X is None:
            raise ValueError(f"Layer {layer} not in activations")

        # Get labels based on probe type
        if probe_type == ProbeType.BINARY:
            y = np.array(self.activations.labels)
            classes = [0, 1]
        elif probe_type == ProbeType.MULTICLASS:
            y = np.array(self.activations.category_labels)
            classes = list(np.unique(y))
        elif probe_type == ProbeType.TOOL_TYPE:
            y = np.array([t if t else "no_tool" for t in self.activations.tool_labels])
            # Filter out classes with too few samples for stratified split
            unique, counts = np.unique(y, return_counts=True)
            valid_classes = unique[counts >= 2]
            if len(valid_classes) < 2:
                raise ValueError("Not enough classes with sufficient samples")
            mask = np.isin(y, valid_classes)
            X = X[mask]
            y = y[mask]
            classes = list(valid_classes)
        else:
            raise ValueError(f"Unknown probe type: {probe_type}")

        # Check if stratified split is possible
        unique, counts = np.unique(y, return_counts=True)
        min_count = counts.min()
        use_stratify = min_count >= 2

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y if use_stratify else None
        )

        # Train probe
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)

        # Evaluate
        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)

        # Cross-validation
        cv_scores = cross_val_score(clf, X, y, cv=cv_folds)

        # Per-class metrics
        y_pred = clf.predict(X_test)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, labels=classes, zero_division=0
        )

        precision_dict = {str(c): float(p) for c, p in zip(classes, precision)}
        recall_dict = {str(c): float(r) for c, r in zip(classes, recall)}
        f1_dict = {str(c): float(f) for c, f in zip(classes, f1)}

        return ProbeResult(
            layer=layer,
            probe_type=probe_type,
            accuracy=test_acc,
            train_accuracy=train_acc,
            weights=clf.coef_,
            bias=clf.intercept_,
            classes=[str(c) for c in classes],
            precision=precision_dict,
            recall=recall_dict,
            f1=f1_dict,
            cv_accuracies=cv_scores.tolist(),
            cv_mean=cv_scores.mean(),
            cv_std=cv_scores.std(),
        )

    def compute_clusters(
        self,
        layer: int,
        n_clusters: int = 8,
    ) -> ClusterResult:
        """
        Cluster activations using K-means.

        Args:
            layer: Layer index
            n_clusters: Number of clusters

        Returns:
            ClusterResult with cluster assignments
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        X = self.activations.get_activations_numpy(layer)
        if X is None:
            raise ValueError(f"Layer {layer} not in activations")

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        sil_score = silhouette_score(X, labels)

        return ClusterResult(
            layer=layer,
            n_clusters=n_clusters,
            labels=labels,
            centroids=kmeans.cluster_centers_,
            inertia=kmeans.inertia_,
            silhouette_score=sil_score,
        )

    def compute_category_similarities(self, layer: int) -> np.ndarray:
        """
        Compute cosine similarity between category centroids.

        Returns matrix of shape [n_categories, n_categories].
        """
        from sklearn.metrics.pairwise import cosine_similarity

        X = self.activations.get_activations_numpy(layer)
        categories = np.array(self.activations.category_labels)
        unique_cats = np.unique(categories)

        # Compute centroids
        centroids = []
        for cat in unique_cats:
            mask = categories == cat
            centroid = X[mask].mean(axis=0)
            centroids.append(centroid)

        centroids = np.array(centroids)
        similarities = cosine_similarity(centroids)

        return similarities

    def analyze_layer(
        self,
        layer: int,
        include_umap: bool = True,
        include_clusters: bool = False,
    ) -> GeometryResult:
        """
        Run full geometry analysis on a layer.

        Args:
            layer: Layer index
            include_umap: Whether to compute UMAP (slower)
            include_clusters: Whether to compute clusters

        Returns:
            GeometryResult with all analyses
        """
        result = GeometryResult(layer=layer)

        # PCA
        result.pca = self.compute_pca(layer)

        # Probes
        result.binary_probe = self.train_probe(layer, ProbeType.BINARY)
        result.category_probe = self.train_probe(layer, ProbeType.MULTICLASS)

        # Only train tool probe if there are multiple tools
        unique_tools = {t for t in self.activations.tool_labels if t}
        if len(unique_tools) > 1:
            result.tool_probe = self.train_probe(layer, ProbeType.TOOL_TYPE)

        # Optional analyses
        if include_umap:
            try:
                result.umap = self.compute_umap(layer)
            except ImportError:
                pass

        if include_clusters:
            result.clusters = self.compute_clusters(layer)

        # Category similarities
        result.category_similarities = self.compute_category_similarities(layer)

        return result

    def compare_layers(
        self,
        layers: list[int] | None = None,
    ) -> dict[int, GeometryResult]:
        """
        Compare geometry across multiple layers.

        Args:
            layers: Layers to compare (default: all captured)

        Returns:
            Dict mapping layer index to GeometryResult
        """
        if layers is None:
            layers = self.activations.captured_layers

        results = {}
        for layer in layers:
            print(f"Analyzing layer {layer}...")
            results[layer] = self.analyze_layer(layer, include_umap=False)

        return results

    def print_layer_comparison(self, results: dict[int, GeometryResult]) -> None:
        """Print comparison table of layer geometry."""
        print("\n" + "=" * 80)
        print("LAYER GEOMETRY COMPARISON")
        print("=" * 80)

        print(
            f"\n{'Layer':<8} {'Dim90':<8} {'Dim95':<8} {'Probe Acc':<12} {'Cat Acc':<12} {'CV±std'}"
        )
        print("-" * 60)

        for layer in sorted(results.keys()):
            r = results[layer]
            dim90 = r.pca.intrinsic_dimensionality_90 if r.pca else "N/A"
            dim95 = r.pca.intrinsic_dimensionality_95 if r.pca else "N/A"
            probe_acc = f"{r.binary_probe.accuracy:.2%}" if r.binary_probe else "N/A"
            cat_acc = f"{r.category_probe.accuracy:.2%}" if r.category_probe else "N/A"
            cv = (
                f"{r.binary_probe.cv_mean:.2%}±{r.binary_probe.cv_std:.2%}"
                if r.binary_probe
                else "N/A"
            )

            print(f"{layer:<8} {dim90:<8} {dim95:<8} {probe_acc:<12} {cat_acc:<12} {cv}")


# =============================================================================
# Convenience functions
# =============================================================================


def compute_pca(
    activations: CollectedActivations,
    layer: int,
    n_components: int = 50,
) -> PCAResult:
    """Convenience function to compute PCA."""
    analyzer = GeometryAnalyzer(activations)
    return analyzer.compute_pca(layer, n_components)


def compute_umap(
    activations: CollectedActivations,
    layer: int,
    n_components: int = 2,
) -> UMAPResult:
    """Convenience function to compute UMAP."""
    analyzer = GeometryAnalyzer(activations)
    return analyzer.compute_umap(layer, n_components)


def train_linear_probe(
    activations: CollectedActivations,
    layer: int,
    probe_type: ProbeType = ProbeType.BINARY,
) -> ProbeResult:
    """Convenience function to train a linear probe."""
    analyzer = GeometryAnalyzer(activations)
    return analyzer.train_probe(layer, probe_type)
