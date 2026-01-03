"""Tests for probing Pydantic models."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from chuk_lazarus.introspection.enums import DirectionMethod
from chuk_lazarus.introspection.models.probing import (
    ProbeLayerResult,
    ProbeResult,
    ProbeTopNeuron,
)


class TestProbeLayerResult:
    """Tests for ProbeLayerResult model."""

    def test_instantiation_minimal(self):
        """Test creating layer result with minimal fields."""
        result = ProbeLayerResult(layer=5, accuracy=0.95)
        assert result.layer == 5
        assert result.accuracy == 0.95
        assert result.std == 0.0

    def test_instantiation_with_std(self):
        """Test creating layer result with standard deviation."""
        result = ProbeLayerResult(layer=5, accuracy=0.95, std=0.02)
        assert result.std == 0.02

    def test_default_std_value(self):
        """Test default std value is 0.0."""
        result = ProbeLayerResult(layer=0, accuracy=0.5)
        assert result.std == 0.0

    def test_missing_required_field_raises_error(self):
        """Test that missing required field raises ValidationError."""
        with pytest.raises(ValidationError):
            ProbeLayerResult(layer=5)  # Missing accuracy


class TestProbeTopNeuron:
    """Tests for ProbeTopNeuron model."""

    def test_instantiation(self):
        """Test creating top neuron with all fields."""
        neuron = ProbeTopNeuron(index=42, weight=0.85)
        assert neuron.index == 42
        assert neuron.weight == 0.85

    def test_negative_weight(self):
        """Test creating top neuron with negative weight."""
        neuron = ProbeTopNeuron(index=10, weight=-0.75)
        assert neuron.weight == -0.75

    def test_missing_required_field_raises_error(self):
        """Test that missing required field raises ValidationError."""
        with pytest.raises(ValidationError):
            ProbeTopNeuron(index=5)  # Missing weight


class TestProbeResult:
    """Tests for ProbeResult model."""

    def test_instantiation_minimal(self):
        """Test creating probe result with minimal fields."""
        result = ProbeResult(
            model_id="test-model",
            class_a_label="positive",
            class_b_label="negative",
            num_class_a=50,
            num_class_b=50,
            best_layer=5,
            best_accuracy=0.92,
            method=DirectionMethod.MEAN_DIFFERENCE,
        )
        assert result.model_id == "test-model"
        assert result.class_a_label == "positive"
        assert result.class_b_label == "negative"
        assert result.num_class_a == 50
        assert result.num_class_b == 50
        assert result.best_layer == 5
        assert result.best_accuracy == 0.92
        assert result.method == DirectionMethod.MEAN_DIFFERENCE
        assert result.layer_results == []
        assert result.direction is None
        assert result.direction_norm == 0.0
        assert result.top_neurons == []
        assert result.separation == 0.0
        assert result.class_a_mean_projection == 0.0
        assert result.class_b_mean_projection == 0.0

    def test_instantiation_with_all_fields(self):
        """Test creating probe result with all fields."""
        direction = np.random.randn(768)
        layer_results = [
            ProbeLayerResult(layer=3, accuracy=0.85, std=0.03),
            ProbeLayerResult(layer=4, accuracy=0.90, std=0.02),
            ProbeLayerResult(layer=5, accuracy=0.92, std=0.015),
        ]
        top_neurons = [
            ProbeTopNeuron(index=42, weight=0.85),
            ProbeTopNeuron(index=100, weight=0.75),
            ProbeTopNeuron(index=250, weight=-0.70),
        ]

        result = ProbeResult(
            model_id="test-model",
            class_a_label="positive",
            class_b_label="negative",
            num_class_a=50,
            num_class_b=50,
            best_layer=5,
            best_accuracy=0.92,
            method=DirectionMethod.LOGISTIC,
            layer_results=layer_results,
            direction=direction,
            direction_norm=float(np.linalg.norm(direction)),
            top_neurons=top_neurons,
            separation=2.5,
            class_a_mean_projection=1.2,
            class_b_mean_projection=-1.3,
        )
        assert len(result.layer_results) == 3
        assert result.direction is not None
        assert result.direction_norm > 0
        assert len(result.top_neurons) == 3
        assert result.separation == 2.5
        assert result.class_a_mean_projection == 1.2
        assert result.class_b_mean_projection == -1.3

    def test_all_direction_methods(self):
        """Test creating probe result with all direction methods."""
        for method in DirectionMethod:
            result = ProbeResult(
                model_id="test",
                class_a_label="a",
                class_b_label="b",
                num_class_a=10,
                num_class_b=10,
                best_layer=0,
                best_accuracy=0.8,
                method=method,
            )
            assert result.method == method

    def test_default_values(self):
        """Test default values for optional fields."""
        result = ProbeResult(
            model_id="test",
            class_a_label="a",
            class_b_label="b",
            num_class_a=10,
            num_class_b=10,
            best_layer=0,
            best_accuracy=0.8,
            method=DirectionMethod.MEAN_DIFFERENCE,
        )
        assert result.layer_results == []
        assert result.direction is None
        assert result.direction_norm == 0.0
        assert result.top_neurons == []
        assert result.separation == 0.0
        assert result.class_a_mean_projection == 0.0
        assert result.class_b_mean_projection == 0.0

    def test_save_direction_success(self):
        """Test saving direction to npz file."""
        direction = np.random.randn(768)
        result = ProbeResult(
            model_id="test-model",
            class_a_label="positive",
            class_b_label="negative",
            num_class_a=50,
            num_class_b=50,
            best_layer=5,
            best_accuracy=0.92,
            method=DirectionMethod.LOGISTIC,
            direction=direction,
            separation=2.5,
            class_a_mean_projection=1.2,
            class_b_mean_projection=-1.3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "direction.npz"
            result.save_direction(path)
            assert path.exists()

            # Verify saved data
            data = np.load(path, allow_pickle=True)
            assert "direction" in data
            assert "layer" in data
            assert "label_positive" in data
            assert "label_negative" in data
            assert "model_id" in data
            assert "method" in data
            assert "accuracy" in data
            assert "separation" in data
            assert np.allclose(data["direction"], direction)
            assert int(data["layer"]) == 5
            assert str(data["label_positive"]) == "positive"
            assert str(data["label_negative"]) == "negative"
            assert str(data["model_id"]) == "test-model"
            assert str(data["method"]) == "logistic"
            assert float(data["accuracy"]) == 0.92
            assert float(data["separation"]) == 2.5

    def test_save_direction_raises_error_when_no_direction(self):
        """Test save_direction raises error when direction is None."""
        result = ProbeResult(
            model_id="test",
            class_a_label="a",
            class_b_label="b",
            num_class_a=10,
            num_class_b=10,
            best_layer=0,
            best_accuracy=0.8,
            method=DirectionMethod.MEAN_DIFFERENCE,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "direction.npz"
            with pytest.raises(ValueError, match="No direction to save"):
                result.save_direction(path)

    def test_save_direction_with_string_path(self):
        """Test save_direction accepts string path."""
        direction = np.random.randn(768)
        result = ProbeResult(
            model_id="test",
            class_a_label="a",
            class_b_label="b",
            num_class_a=10,
            num_class_b=10,
            best_layer=0,
            best_accuracy=0.8,
            method=DirectionMethod.MEAN_DIFFERENCE,
            direction=direction,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/direction.npz"
            result.save_direction(path)
            assert Path(path).exists()

    def test_load_direction_success(self):
        """Test loading direction from npz file."""
        direction = np.random.randn(768)
        original = ProbeResult(
            model_id="test-model",
            class_a_label="positive",
            class_b_label="negative",
            num_class_a=50,
            num_class_b=50,
            best_layer=5,
            best_accuracy=0.92,
            method=DirectionMethod.LOGISTIC,
            direction=direction,
            separation=2.5,
            class_a_mean_projection=1.2,
            class_b_mean_projection=-1.3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "direction.npz"
            original.save_direction(path)
            loaded = ProbeResult.load_direction(path)

            assert loaded.model_id == original.model_id
            assert loaded.class_a_label == original.class_a_label
            assert loaded.class_b_label == original.class_b_label
            assert loaded.best_layer == original.best_layer
            assert loaded.best_accuracy == original.best_accuracy
            assert loaded.method == original.method
            assert np.allclose(loaded.direction, original.direction)
            assert loaded.separation == original.separation
            assert loaded.class_a_mean_projection == original.class_a_mean_projection
            assert loaded.class_b_mean_projection == original.class_b_mean_projection

    def test_load_direction_sets_num_classes_to_zero(self):
        """Test load_direction sets num_class_a and num_class_b to 0."""
        direction = np.random.randn(768)
        original = ProbeResult(
            model_id="test",
            class_a_label="a",
            class_b_label="b",
            num_class_a=50,
            num_class_b=40,
            best_layer=3,
            best_accuracy=0.85,
            method=DirectionMethod.MEAN_DIFFERENCE,
            direction=direction,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "direction.npz"
            original.save_direction(path)
            loaded = ProbeResult.load_direction(path)

            # These fields are not saved, so should be set to 0
            assert loaded.num_class_a == 0
            assert loaded.num_class_b == 0

    def test_load_direction_with_string_path(self):
        """Test load_direction accepts string path."""
        direction = np.random.randn(768)
        original = ProbeResult(
            model_id="test",
            class_a_label="a",
            class_b_label="b",
            num_class_a=10,
            num_class_b=10,
            best_layer=0,
            best_accuracy=0.8,
            method=DirectionMethod.MEAN_DIFFERENCE,
            direction=direction,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/direction.npz"
            original.save_direction(path)
            loaded = ProbeResult.load_direction(path)
            assert loaded.model_id == "test"

    def test_numpy_array_allowed_in_direction(self):
        """Test that numpy arrays are allowed via ConfigDict."""
        direction = np.random.randn(768)
        result = ProbeResult(
            model_id="test",
            class_a_label="a",
            class_b_label="b",
            num_class_a=10,
            num_class_b=10,
            best_layer=0,
            best_accuracy=0.8,
            method=DirectionMethod.MEAN_DIFFERENCE,
            direction=direction,
        )
        assert isinstance(result.direction, np.ndarray)
        assert result.direction.shape == (768,)

    def test_multiple_layer_results(self):
        """Test probe result with multiple layer results."""
        layer_results = [
            ProbeLayerResult(layer=i, accuracy=0.5 + i * 0.05, std=0.02)
            for i in range(10)
        ]
        result = ProbeResult(
            model_id="test",
            class_a_label="a",
            class_b_label="b",
            num_class_a=10,
            num_class_b=10,
            best_layer=9,
            best_accuracy=0.95,
            method=DirectionMethod.MEAN_DIFFERENCE,
            layer_results=layer_results,
        )
        assert len(result.layer_results) == 10
        assert result.layer_results[9].layer == 9
        assert result.layer_results[9].accuracy == 0.95

    def test_top_neurons_with_positive_and_negative_weights(self):
        """Test top neurons can have both positive and negative weights."""
        top_neurons = [
            ProbeTopNeuron(index=10, weight=0.9),
            ProbeTopNeuron(index=20, weight=-0.85),
            ProbeTopNeuron(index=30, weight=0.8),
            ProbeTopNeuron(index=40, weight=-0.75),
        ]
        result = ProbeResult(
            model_id="test",
            class_a_label="a",
            class_b_label="b",
            num_class_a=10,
            num_class_b=10,
            best_layer=0,
            best_accuracy=0.9,
            method=DirectionMethod.MEAN_DIFFERENCE,
            top_neurons=top_neurons,
        )
        assert len(result.top_neurons) == 4
        positive_weights = [n for n in result.top_neurons if n.weight > 0]
        negative_weights = [n for n in result.top_neurons if n.weight < 0]
        assert len(positive_weights) == 2
        assert len(negative_weights) == 2
