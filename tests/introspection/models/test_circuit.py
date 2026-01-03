"""Tests for circuit Pydantic models."""

import tempfile
from pathlib import Path

import numpy as np

from chuk_lazarus.introspection.enums import InvocationMethod, TestStatus
from chuk_lazarus.introspection.models.circuit import (
    CapturedCircuit,
    CircuitComparisonResult,
    CircuitDirection,
    CircuitEntry,
    CircuitInvocationResult,
    CircuitTestResult,
)


class TestCircuitEntry:
    """Tests for CircuitEntry model."""

    def test_instantiation_minimal(self):
        """Test creating entry with minimal required fields."""
        entry = CircuitEntry(prompt="2 + 3 = ")
        assert entry.prompt == "2 + 3 = "
        assert entry.operand_a is None
        assert entry.operand_b is None
        assert entry.operator is None
        assert entry.result is None
        assert entry.activation is None

    def test_instantiation_with_all_fields(self):
        """Test creating entry with all fields."""
        activation = np.array([1.0, 2.0, 3.0])
        entry = CircuitEntry(
            prompt="2 + 3 = 5",
            operand_a=2,
            operand_b=3,
            operator="+",
            result=5,
            activation=activation,
        )
        assert entry.prompt == "2 + 3 = 5"
        assert entry.operand_a == 2
        assert entry.operand_b == 3
        assert entry.operator == "+"
        assert entry.result == 5
        assert np.array_equal(entry.activation, activation)

    def test_numpy_array_allowed(self):
        """Test that numpy arrays are allowed via ConfigDict."""
        activation = np.random.randn(768)
        entry = CircuitEntry(prompt="test", activation=activation)
        assert isinstance(entry.activation, np.ndarray)
        assert entry.activation.shape == (768,)


class TestCircuitDirection:
    """Tests for CircuitDirection model."""

    def test_instantiation_minimal(self):
        """Test creating direction with minimal fields."""
        direction = np.array([1.0, 0.0, 0.0])
        circuit_dir = CircuitDirection(direction=direction, norm=1.0)
        assert np.array_equal(circuit_dir.direction, direction)
        assert circuit_dir.norm == 1.0
        assert circuit_dir.r2_score == 0.0
        assert circuit_dir.mae == 0.0
        assert circuit_dir.scale == 1.0
        assert circuit_dir.intercept == 0.0

    def test_instantiation_with_all_fields(self):
        """Test creating direction with all fields."""
        direction = np.array([0.5, 0.5, 0.707])
        circuit_dir = CircuitDirection(
            direction=direction,
            norm=1.0,
            r2_score=0.95,
            mae=0.5,
            scale=2.0,
            intercept=1.5,
        )
        assert circuit_dir.r2_score == 0.95
        assert circuit_dir.mae == 0.5
        assert circuit_dir.scale == 2.0
        assert circuit_dir.intercept == 1.5

    def test_default_values(self):
        """Test default values for optional fields."""
        direction = np.array([1.0, 0.0])
        circuit_dir = CircuitDirection(direction=direction, norm=1.0)
        assert circuit_dir.r2_score == 0.0
        assert circuit_dir.mae == 0.0
        assert circuit_dir.scale == 1.0
        assert circuit_dir.intercept == 0.0


class TestCapturedCircuit:
    """Tests for CapturedCircuit model."""

    def test_instantiation_minimal(self):
        """Test creating circuit with minimal fields."""
        circuit = CapturedCircuit(model_id="test-model", layer=5)
        assert circuit.model_id == "test-model"
        assert circuit.layer == 5
        assert circuit.entries == []
        assert circuit.direction is None
        assert circuit.activations is None

    def test_instantiation_with_entries(self):
        """Test creating circuit with entries."""
        entries = [
            CircuitEntry(prompt="2 + 3 = ", operand_a=2, operand_b=3, operator="+"),
            CircuitEntry(prompt="4 + 5 = ", operand_a=4, operand_b=5, operator="+"),
        ]
        circuit = CapturedCircuit(
            model_id="test-model",
            layer=5,
            entries=entries,
        )
        assert len(circuit.entries) == 2
        assert circuit.entries[0].prompt == "2 + 3 = "

    def test_num_entries_property(self):
        """Test num_entries property."""
        entries = [CircuitEntry(prompt=f"prompt_{i}") for i in range(5)]
        circuit = CapturedCircuit(model_id="test", layer=0, entries=entries)
        assert circuit.num_entries == 5

    def test_num_entries_property_empty(self):
        """Test num_entries property with no entries."""
        circuit = CapturedCircuit(model_id="test", layer=0)
        assert circuit.num_entries == 0

    def test_has_direction_property_false(self):
        """Test has_direction property returns False when no direction."""
        circuit = CapturedCircuit(model_id="test", layer=0)
        assert circuit.has_direction is False

    def test_has_direction_property_true(self):
        """Test has_direction property returns True when direction exists."""
        direction = CircuitDirection(
            direction=np.array([1.0, 0.0]),
            norm=1.0,
        )
        circuit = CapturedCircuit(model_id="test", layer=0, direction=direction)
        assert circuit.has_direction is True

    def test_save_and_load_minimal(self):
        """Test saving and loading circuit with minimal data."""
        entries = [
            CircuitEntry(prompt="2 + 3 = ", operand_a=2, operand_b=3, operator="+", result=5),
        ]
        circuit = CapturedCircuit(model_id="test-model", layer=5, entries=entries)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "circuit.npz"
            circuit.save(path)
            loaded = CapturedCircuit.load(path)

            assert loaded.model_id == circuit.model_id
            assert loaded.layer == circuit.layer
            assert len(loaded.entries) == len(circuit.entries)
            assert loaded.entries[0].prompt == circuit.entries[0].prompt
            assert loaded.entries[0].operand_a == circuit.entries[0].operand_a

    def test_save_and_load_with_activations(self):
        """Test saving and loading circuit with activations."""
        activations = np.random.randn(3, 768)
        entries = [
            CircuitEntry(prompt="p1", activation=activations[0]),
            CircuitEntry(prompt="p2", activation=activations[1]),
            CircuitEntry(prompt="p3", activation=activations[2]),
        ]
        circuit = CapturedCircuit(
            model_id="test-model",
            layer=5,
            entries=entries,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "circuit.npz"
            circuit.save(path)
            loaded = CapturedCircuit.load(path)

            assert loaded.entries[0].activation is not None
            assert np.allclose(loaded.entries[0].activation, activations[0])

    def test_save_and_load_with_stacked_activations(self):
        """Test saving and loading circuit with stacked activation matrix."""
        activations = np.random.randn(3, 768)
        entries = [
            CircuitEntry(prompt="p1", operand_a=1, operand_b=2, operator="+", result=3),
            CircuitEntry(prompt="p2", operand_a=2, operand_b=3, operator="+", result=5),
            CircuitEntry(prompt="p3", operand_a=3, operand_b=4, operator="+", result=7),
        ]
        circuit = CapturedCircuit(
            model_id="test-model",
            layer=5,
            entries=entries,
            activations=activations,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "circuit.npz"
            circuit.save(path)
            loaded = CapturedCircuit.load(path)

            assert loaded.activations is not None
            assert np.allclose(loaded.activations, activations)

    def test_save_and_load_with_direction(self):
        """Test saving and loading circuit with direction."""
        direction_vec = np.random.randn(768)
        direction = CircuitDirection(
            direction=direction_vec,
            norm=float(np.linalg.norm(direction_vec)),
            r2_score=0.92,
            mae=1.5,
            scale=2.0,
            intercept=0.5,
        )
        entries = [CircuitEntry(prompt="test", operand_a=1, operand_b=2, operator="+", result=3)]
        circuit = CapturedCircuit(
            model_id="test-model",
            layer=5,
            entries=entries,
            direction=direction,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "circuit.npz"
            circuit.save(path)
            loaded = CapturedCircuit.load(path)

            assert loaded.direction is not None
            assert np.allclose(loaded.direction.direction, direction_vec)
            assert loaded.direction.r2_score == 0.92
            assert loaded.direction.mae == 1.5
            assert loaded.direction.scale == 2.0
            assert loaded.direction.intercept == 0.5

    def test_save_with_string_path(self):
        """Test save accepts string path."""
        entries = [CircuitEntry(prompt="test", operand_a=1, operand_b=2, operator="+", result=3)]
        circuit = CapturedCircuit(model_id="test", layer=0, entries=entries)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/circuit.npz"
            circuit.save(path)
            assert Path(path).exists()


class TestCircuitInvocationResult:
    """Tests for CircuitInvocationResult model."""

    def test_instantiation_minimal(self):
        """Test creating invocation result with minimal fields."""
        result = CircuitInvocationResult(
            operand_a=2,
            operand_b=3,
            predicted=5.0,
            method=InvocationMethod.STEER,
        )
        assert result.operand_a == 2
        assert result.operand_b == 3
        assert result.predicted == 5.0
        assert result.method == InvocationMethod.STEER
        assert result.true_result is None
        assert result.error is None

    def test_instantiation_with_all_fields(self):
        """Test creating invocation result with all fields."""
        result = CircuitInvocationResult(
            operand_a=2,
            operand_b=3,
            predicted=4.8,
            true_result=5,
            error=0.2,
            method=InvocationMethod.LINEAR,
        )
        assert result.true_result == 5
        assert result.error == 0.2
        assert result.method == InvocationMethod.LINEAR

    def test_default_values(self):
        """Test default values for optional fields."""
        result = CircuitInvocationResult(
            operand_a=10,
            operand_b=20,
            predicted=30.0,
            method=InvocationMethod.EXTRAPOLATE,
        )
        assert result.true_result is None
        assert result.error is None


class TestCircuitTestResult:
    """Tests for CircuitTestResult model."""

    def test_instantiation_minimal(self):
        """Test creating test result with minimal fields."""
        result = CircuitTestResult(
            prompt="2 + 3 = ",
            true_result=5.0,
            predicted=4.8,
            error=0.2,
        )
        assert result.prompt == "2 + 3 = "
        assert result.true_result == 5.0
        assert result.predicted == 4.8
        assert result.error == 0.2
        assert result.in_training is False
        assert result.status == TestStatus.NOVEL

    def test_instantiation_with_all_fields(self):
        """Test creating test result with all fields."""
        result = CircuitTestResult(
            prompt="10 + 20 = ",
            true_result=30.0,
            predicted=30.0,
            error=0.0,
            in_training=True,
            status=TestStatus.IN_TRAINING,
        )
        assert result.in_training is True
        assert result.status == TestStatus.IN_TRAINING

    def test_default_status(self):
        """Test default status is NOVEL."""
        result = CircuitTestResult(
            prompt="test",
            true_result=1.0,
            predicted=1.0,
            error=0.0,
        )
        assert result.status == TestStatus.NOVEL


class TestCircuitComparisonResult:
    """Tests for CircuitComparisonResult model."""

    def test_instantiation_minimal(self):
        """Test creating comparison result with minimal fields."""
        similarity = np.array([[1.0, 0.9], [0.9, 1.0]])
        result = CircuitComparisonResult(
            circuit_names=["circuit1", "circuit2"],
            similarity_matrix=similarity,
        )
        assert result.circuit_names == ["circuit1", "circuit2"]
        assert np.array_equal(result.similarity_matrix, similarity)
        assert result.angles == {}
        assert result.shared_neurons == []

    def test_instantiation_with_all_fields(self):
        """Test creating comparison result with all fields."""
        similarity = np.array([[1.0, 0.8, 0.6], [0.8, 1.0, 0.7], [0.6, 0.7, 1.0]])
        angles = {
            ("circuit1", "circuit2"): 36.87,
            ("circuit1", "circuit3"): 53.13,
            ("circuit2", "circuit3"): 45.57,
        }
        shared_neurons = [
            (0, [("circuit1", 0.5), ("circuit2", 0.6)]),
            (10, [("circuit1", 0.3), ("circuit3", 0.4)]),
        ]
        result = CircuitComparisonResult(
            circuit_names=["circuit1", "circuit2", "circuit3"],
            similarity_matrix=similarity,
            angles=angles,
            shared_neurons=shared_neurons,
        )
        assert len(result.circuit_names) == 3
        assert len(result.angles) == 3
        assert len(result.shared_neurons) == 2
        assert result.shared_neurons[0][0] == 0
        assert result.shared_neurons[0][1] == [("circuit1", 0.5), ("circuit2", 0.6)]

    def test_numpy_array_in_similarity_matrix(self):
        """Test that numpy arrays work in similarity matrix."""
        similarity = np.eye(4)
        result = CircuitComparisonResult(
            circuit_names=["c1", "c2", "c3", "c4"],
            similarity_matrix=similarity,
        )
        assert isinstance(result.similarity_matrix, np.ndarray)
        assert result.similarity_matrix.shape == (4, 4)
