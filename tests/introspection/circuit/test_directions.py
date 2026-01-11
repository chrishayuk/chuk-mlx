"""Tests for direction extraction module."""

import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from chuk_lazarus.introspection.circuit.collector import CollectedActivations
from chuk_lazarus.introspection.circuit.directions import (
    DirectionBundle,
    DirectionExtractor,
    DirectionMethod,
    ExtractedDirection,
    extract_all_directions,
    extract_direction,
)

# Check if sklearn is available and working
try:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: F401

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

sklearn_required = pytest.mark.skipif(
    not SKLEARN_AVAILABLE,
    reason="sklearn not available or incompatible with numpy version",
)


class TestDirectionMethod:
    """Tests for DirectionMethod enum."""

    def test_direction_method_values(self):
        """Test direction method enum values."""
        assert DirectionMethod.DIFFERENCE_OF_MEANS.value == "diff_means"
        assert DirectionMethod.LDA.value == "lda"
        assert DirectionMethod.PROBE_WEIGHTS.value == "probe_weights"
        assert DirectionMethod.CONTRASTIVE.value == "contrastive"
        assert DirectionMethod.PCA.value == "pca"


class TestExtractedDirection:
    """Tests for ExtractedDirection."""

    def test_create_direction(self):
        """Test creating an extracted direction."""
        direction = ExtractedDirection(
            name="test_dir",
            layer=10,
            direction=np.array([1.0, 0.0, 0.0]),
            method=DirectionMethod.DIFFERENCE_OF_MEANS,
        )
        assert direction.name == "test_dir"
        assert direction.layer == 10
        assert direction.method == DirectionMethod.DIFFERENCE_OF_MEANS

    def test_normalized_direction(self):
        """Test normalized direction property."""
        direction = ExtractedDirection(
            name="test",
            layer=0,
            direction=np.array([3.0, 4.0]),
            method=DirectionMethod.DIFFERENCE_OF_MEANS,
        )
        normalized = direction.normalized_direction
        norm = np.linalg.norm(normalized)
        assert np.isclose(norm, 1.0)

    def test_normalized_direction_zero(self):
        """Test normalized direction with zero vector."""
        direction = ExtractedDirection(
            name="test",
            layer=0,
            direction=np.array([0.0, 0.0]),
            method=DirectionMethod.DIFFERENCE_OF_MEANS,
        )
        normalized = direction.normalized_direction
        assert np.allclose(normalized, [0.0, 0.0])

    def test_project(self):
        """Test projecting activations onto direction."""
        direction = ExtractedDirection(
            name="test",
            layer=0,
            direction=np.array([1.0, 0.0]),
            method=DirectionMethod.DIFFERENCE_OF_MEANS,
        )
        acts = np.array([[2.0, 3.0], [4.0, 5.0]])
        projections = direction.project(acts)
        assert projections.shape == (2,)

    def test_classify(self):
        """Test classifying activations using direction."""
        direction = ExtractedDirection(
            name="test",
            layer=0,
            direction=np.array([1.0, 0.0]),
            method=DirectionMethod.DIFFERENCE_OF_MEANS,
            mean_projection_positive=3.0,
            mean_projection_negative=1.0,
        )
        acts = np.array([[4.0, 0.0], [1.0, 0.0]])
        predictions = direction.classify(acts)
        assert predictions[0] == 1  # Above threshold
        assert predictions[1] == 0  # Below threshold

    def test_summary(self):
        """Test direction summary."""
        direction = ExtractedDirection(
            name="test_dir",
            layer=5,
            direction=np.array([1.0, 2.0, 3.0]),
            method=DirectionMethod.LDA,
            separation_score=2.5,
            accuracy=0.85,
            positive_label="pos",
            negative_label="neg",
        )
        summary = direction.summary()
        assert summary["name"] == "test_dir"
        assert summary["layer"] == 5
        assert summary["method"] == "lda"
        assert summary["separation_score"] == 2.5
        assert summary["accuracy"] == 0.85
        assert summary["positive_label"] == "pos"
        assert summary["negative_label"] == "neg"
        assert "norm" in summary


class TestDirectionBundle:
    """Tests for DirectionBundle."""

    def test_create_empty_bundle(self):
        """Test creating empty direction bundle."""
        bundle = DirectionBundle(name="test_bundle")
        assert bundle.name == "test_bundle"
        assert len(bundle.directions) == 0

    def test_add_direction(self):
        """Test adding a direction to bundle."""
        bundle = DirectionBundle(name="test")
        direction = ExtractedDirection(
            name="dir1",
            layer=5,
            direction=np.array([1.0, 0.0]),
            method=DirectionMethod.DIFFERENCE_OF_MEANS,
        )
        bundle.add(direction)
        assert 5 in bundle.directions
        assert bundle.directions[5] == direction

    def test_get_direction(self):
        """Test getting a direction from bundle."""
        bundle = DirectionBundle(name="test")
        direction = ExtractedDirection(
            name="dir1",
            layer=5,
            direction=np.array([1.0, 0.0]),
            method=DirectionMethod.DIFFERENCE_OF_MEANS,
        )
        bundle.add(direction)
        retrieved = bundle.get(5)
        assert retrieved == direction

    def test_get_nonexistent_direction(self):
        """Test getting non-existent direction returns None."""
        bundle = DirectionBundle(name="test")
        assert bundle.get(99) is None

    def test_layers_property(self):
        """Test getting sorted list of layers."""
        bundle = DirectionBundle(name="test")
        for layer in [2, 0, 4]:
            bundle.add(
                ExtractedDirection(
                    name=f"dir{layer}",
                    layer=layer,
                    direction=np.array([1.0]),
                    method=DirectionMethod.DIFFERENCE_OF_MEANS,
                )
            )
        assert bundle.layers == [0, 2, 4]

    def test_get_separation_by_layer(self):
        """Test getting separation scores by layer."""
        bundle = DirectionBundle(name="test")
        bundle.add(
            ExtractedDirection(
                name="dir1",
                layer=0,
                direction=np.array([1.0]),
                method=DirectionMethod.DIFFERENCE_OF_MEANS,
                separation_score=1.5,
            )
        )
        bundle.add(
            ExtractedDirection(
                name="dir2",
                layer=2,
                direction=np.array([1.0]),
                method=DirectionMethod.DIFFERENCE_OF_MEANS,
                separation_score=2.5,
            )
        )
        seps = bundle.get_separation_by_layer()
        assert seps[0] == 1.5
        assert seps[2] == 2.5

    def test_get_accuracy_by_layer(self):
        """Test getting accuracy by layer."""
        bundle = DirectionBundle(name="test")
        bundle.add(
            ExtractedDirection(
                name="dir1",
                layer=0,
                direction=np.array([1.0]),
                method=DirectionMethod.DIFFERENCE_OF_MEANS,
                accuracy=0.75,
            )
        )
        bundle.add(
            ExtractedDirection(
                name="dir2",
                layer=2,
                direction=np.array([1.0]),
                method=DirectionMethod.DIFFERENCE_OF_MEANS,
                accuracy=0.85,
            )
        )
        accs = bundle.get_accuracy_by_layer()
        assert accs[0] == 0.75
        assert accs[2] == 0.85

    def test_find_best_layer(self):
        """Test finding layer with highest separation."""
        bundle = DirectionBundle(name="test")
        bundle.add(
            ExtractedDirection(
                name="dir1",
                layer=0,
                direction=np.array([1.0]),
                method=DirectionMethod.DIFFERENCE_OF_MEANS,
                separation_score=1.5,
            )
        )
        bundle.add(
            ExtractedDirection(
                name="dir2",
                layer=2,
                direction=np.array([1.0]),
                method=DirectionMethod.DIFFERENCE_OF_MEANS,
                separation_score=2.5,
            )
        )
        best = bundle.find_best_layer()
        assert best == 2

    def test_find_best_layer_empty(self):
        """Test finding best layer with empty bundle."""
        bundle = DirectionBundle(name="test")
        assert bundle.find_best_layer() is None

    def test_save_and_load(self):
        """Test saving and loading direction bundle."""
        bundle = DirectionBundle(
            name="test_bundle",
            model_id="test-model",
            positive_label="pos",
            negative_label="neg",
        )
        bundle.add(
            ExtractedDirection(
                name="dir1",
                layer=0,
                direction=np.array([1.0, 2.0, 3.0]),
                method=DirectionMethod.DIFFERENCE_OF_MEANS,
                separation_score=1.5,
                accuracy=0.8,
            )
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bundle"
            bundle.save(path)
            loaded = DirectionBundle.load(path)
            assert loaded.name == "test_bundle"
            assert loaded.model_id == "test-model"
            assert 0 in loaded.directions
            assert loaded.directions[0].layer == 0


class TestDirectionExtractor:
    """Tests for DirectionExtractor."""

    @pytest.fixture
    def sample_activations(self):
        """Create sample activations for testing."""
        acts = CollectedActivations()
        # Create activations: 5 samples, 64 features
        acts.hidden_states = {
            0: mx.array(np.random.randn(5, 64).astype(np.float32)),
            2: mx.array(np.random.randn(5, 64).astype(np.float32)),
        }
        acts.labels = [0, 0, 1, 1, 1]
        acts.categories = ["cat1", "cat1", "cat2", "cat2", "cat2"]
        acts.dataset_label_names = {0: "negative", 1: "positive"}
        acts.dataset_name = "test_dataset"
        acts.model_id = "test-model"
        return acts

    def test_init(self, sample_activations):
        """Test extractor initialization."""
        extractor = DirectionExtractor(sample_activations)
        assert extractor.activations is sample_activations

    def test_extract_direction_diff_means(self, sample_activations):
        """Test extracting direction using difference of means."""
        extractor = DirectionExtractor(sample_activations)
        direction = extractor.extract_direction(layer=0, method=DirectionMethod.DIFFERENCE_OF_MEANS)
        assert direction.layer == 0
        assert direction.method == DirectionMethod.DIFFERENCE_OF_MEANS
        assert direction.direction.shape == (64,)
        assert direction.positive_label == "positive"
        assert direction.negative_label == "negative"

    @pytest.mark.skip(reason="sklearn import issues with MLX numpy")
    def test_extract_direction_lda(self, sample_activations):
        """Test extracting direction using LDA."""
        extractor = DirectionExtractor(sample_activations)
        direction = extractor.extract_direction(layer=0, method=DirectionMethod.LDA)
        assert direction.method == DirectionMethod.LDA
        assert direction.direction.shape == (64,)

    @pytest.mark.skip(reason="sklearn import issues with MLX numpy")
    def test_extract_direction_probe_weights(self, sample_activations):
        """Test extracting direction using probe weights."""
        extractor = DirectionExtractor(sample_activations)
        direction = extractor.extract_direction(layer=0, method=DirectionMethod.PROBE_WEIGHTS)
        assert direction.method == DirectionMethod.PROBE_WEIGHTS
        assert direction.direction.shape == (64,)

    @pytest.mark.skip(reason="sklearn import issues with MLX numpy")
    def test_extract_direction_pca(self, sample_activations):
        """Test extracting direction using PCA."""
        extractor = DirectionExtractor(sample_activations)
        direction = extractor.extract_direction(layer=0, method=DirectionMethod.PCA)
        assert direction.method == DirectionMethod.PCA
        assert direction.direction.shape == (64,)

    def test_extract_direction_invalid_method(self, sample_activations):
        """Test extracting direction with invalid method raises error."""
        extractor = DirectionExtractor(sample_activations)
        with pytest.raises(ValueError, match="Unknown method"):
            extractor.extract_direction(layer=0, method="invalid")

    def test_extract_direction_custom_labels(self, sample_activations):
        """Test extracting direction with custom label values."""
        extractor = DirectionExtractor(sample_activations)
        direction = extractor.extract_direction(layer=0, positive_label=1, negative_label=0)
        assert direction.positive_label == "positive"
        assert direction.negative_label == "negative"

    def test_extract_all_layers(self, sample_activations):
        """Test extracting directions for all layers."""
        extractor = DirectionExtractor(sample_activations)
        bundle = extractor.extract_all_layers()
        assert bundle.name == "test_dataset_directions"
        assert bundle.model_id == "test-model"
        assert len(bundle.directions) == 2
        assert 0 in bundle.directions
        assert 2 in bundle.directions

    @sklearn_required
    def test_extract_all_layers_custom_method(self, sample_activations):
        """Test extracting all layers with custom method."""
        extractor = DirectionExtractor(sample_activations)
        bundle = extractor.extract_all_layers(method=DirectionMethod.LDA)
        for direction in bundle.directions.values():
            assert direction.method == DirectionMethod.LDA

    def test_extract_per_category(self, sample_activations):
        """Test extracting directions per category."""
        extractor = DirectionExtractor(sample_activations)
        directions = extractor.extract_per_category(layer=0)
        assert "cat1" in directions or "cat2" in directions
        for cat_name, direction in directions.items():
            assert direction.positive_label == cat_name
            assert direction.negative_label == "other"

    def test_extract_per_category_insufficient_samples(self):
        """Test per-category extraction with insufficient samples."""
        acts = CollectedActivations()
        acts.hidden_states = {0: mx.array(np.random.randn(2, 64).astype(np.float32))}
        acts.labels = [0, 1]
        acts.categories = ["cat1", "cat2"]
        extractor = DirectionExtractor(acts)
        directions = extractor.extract_per_category(layer=0)
        # Categories with < 2 samples should be skipped
        assert len(directions) == 0

    def test_check_orthogonality(self, sample_activations):
        """Test checking orthogonality between directions."""
        extractor = DirectionExtractor(sample_activations)
        dir1 = extractor.extract_direction(layer=0)
        dir2 = extractor.extract_direction(layer=2)
        similarities = extractor.check_orthogonality([dir1, dir2])
        assert similarities.shape == (2, 2)
        # Diagonal should be 1 (self-similarity)
        assert np.isclose(similarities[0, 0], 1.0)
        assert np.isclose(similarities[1, 1], 1.0)

    def test_print_summary(self, sample_activations, capsys):
        """Test printing direction summary."""
        extractor = DirectionExtractor(sample_activations)
        bundle = extractor.extract_all_layers()
        extractor.print_summary(bundle)
        captured = capsys.readouterr()
        assert "DIRECTION SUMMARY" in captured.out
        assert "test_dataset_directions" in captured.out

    def test_diff_of_means(self):
        """Test difference of means calculation."""
        positive = np.array([[1.0, 2.0], [3.0, 4.0]])
        negative = np.array([[0.0, 1.0], [1.0, 2.0]])
        direction = DirectionExtractor._diff_of_means(positive, negative)
        expected = positive.mean(axis=0) - negative.mean(axis=0)
        assert np.allclose(direction, expected)

    @sklearn_required
    def test_lda_direction(self):
        """Test LDA direction calculation."""
        X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
        labels = np.array([0, 0, 1, 1])
        direction = DirectionExtractor._lda_direction(X, labels)
        assert direction.shape == (2,)

    @sklearn_required
    def test_probe_weights(self):
        """Test probe weights direction calculation."""
        X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
        labels = np.array([0, 0, 1, 1])
        direction = DirectionExtractor._probe_weights(X, labels)
        assert direction.shape == (2,)

    @sklearn_required
    def test_pca_direction(self):
        """Test PCA direction calculation."""
        positive = np.array([[1.0, 2.0], [2.0, 3.0]])
        negative = np.array([[3.0, 4.0], [4.0, 5.0]])
        direction = DirectionExtractor._pca_direction(positive, negative)
        assert direction.shape == (2,)


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.fixture
    def sample_activations(self):
        """Create sample activations."""
        acts = CollectedActivations()
        acts.hidden_states = {0: mx.array(np.random.randn(5, 64).astype(np.float32))}
        acts.labels = [0, 0, 1, 1, 1]
        acts.dataset_label_names = {0: "negative", 1: "positive"}
        acts.dataset_name = "test"
        acts.model_id = "test-model"
        return acts

    def test_extract_direction(self, sample_activations):
        """Test extract_direction convenience function."""
        direction = extract_direction(sample_activations, layer=0)
        assert isinstance(direction, ExtractedDirection)
        assert direction.layer == 0

    @sklearn_required
    def test_extract_direction_with_method(self, sample_activations):
        """Test extract_direction with custom method."""
        direction = extract_direction(sample_activations, layer=0, method=DirectionMethod.LDA)
        assert direction.method == DirectionMethod.LDA

    def test_extract_all_directions(self, sample_activations):
        """Test extract_all_directions convenience function."""
        bundle = extract_all_directions(sample_activations)
        assert isinstance(bundle, DirectionBundle)
        assert len(bundle.directions) > 0

    @sklearn_required
    def test_extract_all_directions_with_method(self, sample_activations):
        """Test extract_all_directions with custom method."""
        bundle = extract_all_directions(sample_activations, method=DirectionMethod.PROBE_WEIGHTS)
        for direction in bundle.directions.values():
            assert direction.method == DirectionMethod.PROBE_WEIGHTS
