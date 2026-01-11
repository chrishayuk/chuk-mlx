"""Tests for probe battery module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import mlx.core as mx
import numpy as np
import pytest

from chuk_lazarus.introspection.circuit.probes import (
    ProbeBattery,
    ProbeDataset,
    ProbeResult,
    StratigraphyResult,
    create_arithmetic_probe,
    create_code_trace_probe,
    create_factual_consistency_probe,
    create_suppression_probe,
    create_tool_decision_probe,
    get_default_probe_datasets,
)

# Check if sklearn is available and working (not just importable)
try:
    # Actually test if sklearn works with current numpy version
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    _test_lr = LogisticRegression()
    _test_lr.fit(np.random.randn(10, 5), [0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    SKLEARN_AVAILABLE = True
except (ImportError, Exception):
    # sklearn is either not installed or incompatible with numpy version
    SKLEARN_AVAILABLE = False

sklearn_required = pytest.mark.skipif(
    not SKLEARN_AVAILABLE,
    reason="sklearn not available or incompatible with numpy version",
)


class TestProbeDataset:
    """Tests for ProbeDataset."""

    def test_create_probe_dataset(self):
        """Test creating a probe dataset."""
        dataset = ProbeDataset(
            name="test_probe",
            description="Test probe",
            prompts=["p1", "p2", "p3"],
            labels=[0, 1, 0],
            label_names=["neg", "pos"],
            category="test",
        )
        assert dataset.name == "test_probe"
        assert len(dataset) == 3

    def test_len(self):
        """Test dataset length."""
        dataset = ProbeDataset(name="test", description="", prompts=["p1", "p2"], labels=[0, 1])
        assert len(dataset) == 2

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "description": "Test probe",
            "prompts": ["p1", "p2"],
            "labels": [0, 1],
            "label_names": ["neg", "pos"],
            "category": "test",
        }
        dataset = ProbeDataset.from_dict("test_probe", data)
        assert dataset.name == "test_probe"
        assert dataset.description == "Test probe"
        assert len(dataset) == 2

    def test_from_dict_minimal(self):
        """Test creating from minimal dictionary."""
        data = {"prompts": ["p1"], "labels": [0]}
        dataset = ProbeDataset.from_dict("test", data)
        assert dataset.name == "test"
        assert dataset.category == "custom"

    def test_to_dict(self):
        """Test converting to dictionary."""
        dataset = ProbeDataset(
            name="test",
            description="desc",
            prompts=["p1"],
            labels=[0],
            label_names=["label"],
            category="cat",
        )
        data = dataset.to_dict()
        assert data["description"] == "desc"
        assert data["prompts"] == ["p1"]
        assert data["labels"] == [0]

    def test_baseline_accuracy(self):
        """Test baseline accuracy calculation."""
        dataset = ProbeDataset(
            name="test", description="", prompts=["p1", "p2", "p3"], labels=[0, 0, 1]
        )
        baseline = dataset.baseline_accuracy
        assert baseline == 2 / 3  # Majority class is 0

    def test_baseline_accuracy_empty(self):
        """Test baseline accuracy with empty dataset."""
        dataset = ProbeDataset(name="test", description="", prompts=[], labels=[])
        assert dataset.baseline_accuracy == 0.5

    def test_num_classes(self):
        """Test number of classes."""
        dataset = ProbeDataset(
            name="test", description="", prompts=["p1", "p2", "p3"], labels=[0, 1, 2]
        )
        assert dataset.num_classes == 3


class TestProbeResult:
    """Tests for ProbeResult."""

    def test_create_probe_result(self):
        """Test creating a probe result."""
        result = ProbeResult(
            probe_name="test",
            layer=5,
            accuracy=0.85,
            cv_std=0.02,
            baseline=0.5,
            above_chance=0.35,
            n_samples=100,
        )
        assert result.probe_name == "test"
        assert result.layer == 5
        assert result.accuracy == 0.85

    def test_is_significant(self):
        """Test significance check."""
        result = ProbeResult(
            probe_name="test",
            layer=5,
            accuracy=0.75,
            cv_std=0.02,
            baseline=0.5,
            above_chance=0.25,
            n_samples=100,
        )
        assert result.is_significant is True

    def test_is_not_significant_low_accuracy(self):
        """Test not significant with low accuracy."""
        result = ProbeResult(
            probe_name="test",
            layer=5,
            accuracy=0.55,
            cv_std=0.02,
            baseline=0.5,
            above_chance=0.05,
            n_samples=100,
        )
        assert result.is_significant is False

    def test_is_not_significant_low_above_chance(self):
        """Test not significant with low above-chance score."""
        result = ProbeResult(
            probe_name="test",
            layer=5,
            accuracy=0.75,
            cv_std=0.02,
            baseline=0.7,
            above_chance=0.05,
            n_samples=100,
        )
        assert result.is_significant is False


class TestStratigraphyResult:
    """Tests for StratigraphyResult."""

    def test_create_stratigraphy_result(self):
        """Test creating a stratigraphy result."""
        result = StratigraphyResult(model_id="test-model", num_layers=10)
        assert result.model_id == "test-model"
        assert result.num_layers == 10
        assert len(result.probes) == 0

    def test_get_accuracy_matrix(self):
        """Test getting accuracy matrix."""
        result = StratigraphyResult(model_id="test", num_layers=10)
        result.probes["probe1"] = {
            0: ProbeResult(
                probe_name="probe1",
                layer=0,
                accuracy=0.5,
                cv_std=0.02,
                baseline=0.5,
                above_chance=0.0,
                n_samples=100,
            ),
            2: ProbeResult(
                probe_name="probe1",
                layer=2,
                accuracy=0.8,
                cv_std=0.02,
                baseline=0.5,
                above_chance=0.3,
                n_samples=100,
            ),
        }
        result.probes["probe2"] = {
            0: ProbeResult(
                probe_name="probe2",
                layer=0,
                accuracy=0.6,
                cv_std=0.02,
                baseline=0.5,
                above_chance=0.1,
                n_samples=100,
            ),
            2: ProbeResult(
                probe_name="probe2",
                layer=2,
                accuracy=0.7,
                cv_std=0.02,
                baseline=0.5,
                above_chance=0.2,
                n_samples=100,
            ),
        }
        matrix = result.get_accuracy_matrix(layers=[0, 2])
        assert matrix["probe1"] == [0.5, 0.8]
        assert matrix["probe2"] == [0.6, 0.7]

    def test_find_emergence_layer(self):
        """Test finding emergence layer."""
        result = StratigraphyResult(model_id="test", num_layers=10)
        result.probes["test_probe"] = {
            0: ProbeResult(
                probe_name="test",
                layer=0,
                accuracy=0.5,
                cv_std=0.02,
                baseline=0.5,
                above_chance=0.0,
                n_samples=100,
            ),
            2: ProbeResult(
                probe_name="test",
                layer=2,
                accuracy=0.6,
                cv_std=0.02,
                baseline=0.5,
                above_chance=0.1,
                n_samples=100,
            ),
            4: ProbeResult(
                probe_name="test",
                layer=4,
                accuracy=0.8,
                cv_std=0.02,
                baseline=0.5,
                above_chance=0.3,
                n_samples=100,
            ),
        }
        emergence = result.find_emergence_layer("test_probe", threshold=0.75)
        assert emergence == 4

    def test_find_emergence_layer_none(self):
        """Test finding emergence layer when threshold never met."""
        result = StratigraphyResult(model_id="test", num_layers=10)
        result.probes["test_probe"] = {
            0: ProbeResult(
                probe_name="test",
                layer=0,
                accuracy=0.5,
                cv_std=0.02,
                baseline=0.5,
                above_chance=0.0,
                n_samples=100,
            ),
            2: ProbeResult(
                probe_name="test",
                layer=2,
                accuracy=0.6,
                cv_std=0.02,
                baseline=0.5,
                above_chance=0.1,
                n_samples=100,
            ),
        }
        emergence = result.find_emergence_layer("test_probe", threshold=0.9)
        assert emergence is None

    def test_find_emergence_layer_nonexistent_probe(self):
        """Test finding emergence layer for non-existent probe."""
        result = StratigraphyResult(model_id="test", num_layers=10)
        assert result.find_emergence_layer("nonexistent") is None

    def test_find_destruction_layer(self):
        """Test finding destruction layer."""
        result = StratigraphyResult(model_id="test", num_layers=10)
        result.probes["test_probe"] = {
            0: ProbeResult(
                probe_name="test",
                layer=0,
                accuracy=0.8,
                cv_std=0.02,
                baseline=0.5,
                above_chance=0.3,
                n_samples=100,
            ),
            2: ProbeResult(
                probe_name="test",
                layer=2,
                accuracy=0.9,
                cv_std=0.02,
                baseline=0.5,
                above_chance=0.4,
                n_samples=100,
            ),
            4: ProbeResult(
                probe_name="test",
                layer=4,
                accuracy=0.4,
                cv_std=0.02,
                baseline=0.5,
                above_chance=-0.1,
                n_samples=100,
            ),
        }
        destruction = result.find_destruction_layer("test_probe", threshold=0.5)
        assert destruction == 4

    def test_find_destruction_layer_none(self):
        """Test finding destruction layer when no drop occurs."""
        result = StratigraphyResult(model_id="test", num_layers=10)
        result.probes["test_probe"] = {
            0: ProbeResult(
                probe_name="test",
                layer=0,
                accuracy=0.8,
                cv_std=0.02,
                baseline=0.5,
                above_chance=0.3,
                n_samples=100,
            ),
            2: ProbeResult(
                probe_name="test",
                layer=2,
                accuracy=0.9,
                cv_std=0.02,
                baseline=0.5,
                above_chance=0.4,
                n_samples=100,
            ),
        }
        destruction = result.find_destruction_layer("test_probe")
        assert destruction is None

    def test_get_all_emergence_layers(self):
        """Test getting emergence layers for all probes."""
        result = StratigraphyResult(model_id="test", num_layers=10)
        result.probes["probe1"] = {
            0: ProbeResult(
                probe_name="p1",
                layer=0,
                accuracy=0.5,
                cv_std=0.02,
                baseline=0.5,
                above_chance=0.0,
                n_samples=100,
            ),
            2: ProbeResult(
                probe_name="p1",
                layer=2,
                accuracy=0.8,
                cv_std=0.02,
                baseline=0.5,
                above_chance=0.3,
                n_samples=100,
            ),
        }
        result.probes["probe2"] = {
            0: ProbeResult(
                probe_name="p2",
                layer=0,
                accuracy=0.8,
                cv_std=0.02,
                baseline=0.5,
                above_chance=0.3,
                n_samples=100,
            ),
        }
        emergence = result.get_all_emergence_layers(threshold=0.75)
        assert emergence["probe1"] == 2
        assert emergence["probe2"] == 0

    def test_save_and_load(self):
        """Test saving and loading stratigraphy results."""
        result = StratigraphyResult(model_id="test-model", num_layers=10)
        result.probes["probe1"] = {
            0: ProbeResult(
                probe_name="probe1",
                layer=0,
                accuracy=0.8,
                cv_std=0.02,
                baseline=0.5,
                above_chance=0.3,
                n_samples=100,
            )
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.json"
            result.save(path)
            loaded = StratigraphyResult.load(path)
            assert loaded.model_id == "test-model"
            assert loaded.num_layers == 10
            assert "probe1" in loaded.probes
            assert 0 in loaded.probes["probe1"]


@sklearn_required
class TestProbeBattery:
    """Tests for ProbeBattery."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = Mock()
        model.model = Mock()
        model.model.layers = [Mock() for _ in range(4)]
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=np.array([[1, 2, 3]]))
        return tokenizer

    def test_init(self, mock_model, mock_tokenizer):
        """Test initialization."""
        battery = ProbeBattery(mock_model, mock_tokenizer, model_id="test-model")
        assert battery.model is mock_model
        assert battery.tokenizer is mock_tokenizer
        assert battery.model_id == "test-model"
        assert battery.num_layers == 4

    def test_detect_structure_missing_layers_raises(self, mock_tokenizer):
        """Test structure detection raises on missing layers."""
        bad_model = Mock(spec=[])  # Empty spec = no attributes
        # This model has no .model or .layers attributes at all
        with pytest.raises(ValueError, match="Cannot detect"):
            ProbeBattery(bad_model, mock_tokenizer)

    @patch("chuk_lazarus.introspection.ablation.AblationStudy")
    def test_from_pretrained(self, mock_ablation_study):
        """Test loading from pretrained."""
        mock_study = Mock()
        mock_model = Mock()
        mock_model.model = Mock()
        mock_model.model.layers = [Mock() for _ in range(4)]
        mock_study.adapter.model = mock_model
        mock_study.adapter.tokenizer = Mock()
        mock_ablation_study.from_pretrained.return_value = mock_study
        battery = ProbeBattery.from_pretrained("test-model")
        assert battery.model_id == "test-model"

    def test_add_dataset(self, mock_model, mock_tokenizer):
        """Test adding a probe dataset."""
        battery = ProbeBattery(mock_model, mock_tokenizer)
        dataset = ProbeDataset(name="test", description="", prompts=["p1"], labels=[0])
        battery.add_dataset(dataset)
        assert "test" in battery.datasets

    def test_load_datasets_from_file(self, mock_model, mock_tokenizer):
        """Test loading datasets from JSON file."""
        battery = ProbeBattery(mock_model, mock_tokenizer)
        data = {
            "probe1": {
                "description": "Test probe",
                "prompts": ["p1", "p2"],
                "labels": [0, 1],
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "probes.json"
            with open(path, "w") as f:
                json.dump(data, f)
            battery.load_datasets(path)
            assert "probe1" in battery.datasets

    def test_load_datasets_single_dataset_file(self, mock_model, mock_tokenizer):
        """Test loading single dataset from file."""
        battery = ProbeBattery(mock_model, mock_tokenizer)
        data = {"description": "Test", "prompts": ["p1"], "labels": [0]}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "probe.json"
            with open(path, "w") as f:
                json.dump(data, f)
            battery.load_datasets(path)
            assert "probe" in battery.datasets

    def test_load_datasets_from_directory(self, mock_model, mock_tokenizer):
        """Test loading datasets from directory."""
        battery = ProbeBattery(mock_model, mock_tokenizer)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            data = {"description": "Test", "prompts": ["p1"], "labels": [0]}
            with open(path / "probe1.json", "w") as f:
                json.dump(data, f)
            battery.load_datasets(path)
            assert "probe1" in battery.datasets

    def test_load_datasets_invalid_path_raises(self, mock_model, mock_tokenizer):
        """Test loading from invalid path raises error."""
        battery = ProbeBattery(mock_model, mock_tokenizer)
        with pytest.raises(ValueError, match="not found"):
            battery.load_datasets("/nonexistent/path")

    @patch("chuk_lazarus.introspection.hooks.ModelHooks")
    def test_get_activations(self, mock_hooks_cls, mock_model, mock_tokenizer):
        """Test getting activations for a prompt."""
        mock_hooks = Mock()
        mock_state = Mock()
        mock_state.hidden_states = {0: mx.ones((1, 5, 64), dtype=mx.float32)}
        mock_hooks.state = mock_state
        mock_hooks_cls.return_value = mock_hooks
        battery = ProbeBattery(mock_model, mock_tokenizer)
        acts = battery.get_activations("Test prompt", layer=0)
        assert isinstance(acts, np.ndarray)
        assert acts.shape == (64,)

    @patch("chuk_lazarus.introspection.hooks.ModelHooks")
    def test_collect_dataset_activations(self, mock_hooks_cls, mock_model, mock_tokenizer):
        """Test collecting activations for a dataset."""
        mock_hooks = Mock()
        mock_state = Mock()
        mock_state.hidden_states = {0: mx.ones((1, 5, 64), dtype=mx.float32)}
        mock_hooks.state = mock_state
        mock_hooks_cls.return_value = mock_hooks
        battery = ProbeBattery(mock_model, mock_tokenizer)
        dataset = ProbeDataset(name="test", description="", prompts=["p1", "p2"], labels=[0, 1])
        X, y = battery.collect_dataset_activations(dataset, layer=0)
        assert X.shape == (2, 64)
        assert y.shape == (2,)

    def test_train_probe(self):
        """Test training a probe."""
        X = np.random.randn(20, 64)
        y = np.array([0] * 10 + [1] * 10)
        # Create mock battery (we don't need actual model for this test)
        battery = Mock(spec=ProbeBattery)
        battery.train_probe = ProbeBattery.train_probe.__get__(battery, ProbeBattery)
        accuracy, std = battery.train_probe(X, y, cv_folds=3)
        assert 0.0 <= accuracy <= 1.0
        assert std >= 0.0

    def test_train_probe_insufficient_samples(self):
        """Test training probe with insufficient samples."""
        X = np.random.randn(1, 64)
        y = np.array([0])
        battery = Mock(spec=ProbeBattery)
        battery.train_probe = ProbeBattery.train_probe.__get__(battery, ProbeBattery)
        accuracy, std = battery.train_probe(X, y)
        assert accuracy == 0.5
        assert std == 0.0

    @patch("chuk_lazarus.introspection.hooks.ModelHooks")
    def test_run_probe(self, mock_hooks_cls, mock_model, mock_tokenizer):
        """Test running a single probe."""
        mock_hooks = Mock()
        mock_state = Mock()
        mock_state.hidden_states = {0: mx.ones((1, 5, 64), dtype=mx.float32)}
        mock_hooks.state = mock_state
        mock_hooks_cls.return_value = mock_hooks
        battery = ProbeBattery(mock_model, mock_tokenizer)
        # Need at least 5 samples per class for cv_folds=5 (default)
        dataset = ProbeDataset(
            name="test",
            description="",
            prompts=[f"p{i}" for i in range(10)],
            labels=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        )
        battery.add_dataset(dataset)
        result = battery.run_probe("test", layer=0)
        assert isinstance(result, ProbeResult)
        assert result.probe_name == "test"
        assert result.layer == 0

    @patch("chuk_lazarus.introspection.hooks.ModelHooks")
    def test_run_all_probes(self, mock_hooks_cls, mock_model, mock_tokenizer):
        """Test running all probes."""
        mock_hooks = Mock()
        mock_state = Mock()
        mock_state.hidden_states = {0: mx.ones((1, 5, 64), dtype=mx.float32)}
        mock_hooks.state = mock_state
        mock_hooks_cls.return_value = mock_hooks
        battery = ProbeBattery(mock_model, mock_tokenizer)
        # Need at least 5 samples per class for cv_folds=5 (default)
        dataset = ProbeDataset(
            name="test",
            description="",
            prompts=[f"p{i}" for i in range(10)],
            labels=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        )
        battery.add_dataset(dataset)
        results = battery.run_all_probes(layers=[0], progress=False)
        assert isinstance(results, StratigraphyResult)
        assert "test" in results.probes

    @patch("chuk_lazarus.introspection.hooks.ModelHooks")
    def test_run_all_probes_with_category_filter(self, mock_hooks_cls, mock_model, mock_tokenizer):
        """Test running probes with category filter."""
        mock_hooks = Mock()
        mock_state = Mock()
        mock_state.hidden_states = {0: mx.ones((1, 5, 64), dtype=mx.float32)}
        mock_hooks.state = mock_state
        mock_hooks_cls.return_value = mock_hooks
        battery = ProbeBattery(mock_model, mock_tokenizer)
        dataset1 = ProbeDataset(
            name="test1",
            description="",
            prompts=["p1"],
            labels=[0],
            category="cat1",
        )
        dataset2 = ProbeDataset(
            name="test2",
            description="",
            prompts=["p2"],
            labels=[1],
            category="cat2",
        )
        battery.add_dataset(dataset1)
        battery.add_dataset(dataset2)
        results = battery.run_all_probes(layers=[0], categories=["cat1"], progress=False)
        assert "test1" in results.probes
        assert "test2" not in results.probes

    def test_print_results_table(self, mock_model, mock_tokenizer, capsys):
        """Test printing results table."""
        battery = ProbeBattery(mock_model, mock_tokenizer)
        results = StratigraphyResult(model_id="test", num_layers=4)
        results.probes["probe1"] = {
            0: ProbeResult(
                probe_name="probe1",
                layer=0,
                accuracy=0.8,
                cv_std=0.02,
                baseline=0.5,
                above_chance=0.3,
                n_samples=100,
            )
        }
        battery.print_results_table(results)
        captured = capsys.readouterr()
        assert "PROBE ACCURACY BY LAYER" in captured.out

    def test_print_stratigraphy(self, mock_model, mock_tokenizer, capsys):
        """Test printing stratigraphy."""
        battery = ProbeBattery(mock_model, mock_tokenizer)
        battery.add_dataset(
            ProbeDataset(
                name="probe1",
                description="Test probe",
                prompts=["p1"],
                labels=[0],
                category="test",
            )
        )
        results = StratigraphyResult(model_id="test-model", num_layers=4)
        results.probes["probe1"] = {
            0: ProbeResult(
                probe_name="probe1",
                layer=0,
                accuracy=0.8,
                cv_std=0.02,
                baseline=0.5,
                above_chance=0.3,
                n_samples=100,
            )
        }
        battery.print_stratigraphy(results, threshold=0.75)
        captured = capsys.readouterr()
        assert "COMPUTATIONAL STRATIGRAPHY" in captured.out


class TestPrebuiltProbes:
    """Tests for pre-built probe datasets."""

    def test_create_arithmetic_probe(self):
        """Test creating arithmetic probe."""
        probe = create_arithmetic_probe()
        assert probe.name == "arithmetic_mode"
        assert len(probe) > 0
        assert probe.category == "computation"
        assert len(probe.prompts) == len(probe.labels)
        assert probe.label_names == ["retrieval", "arithmetic"]
        # Should have balanced classes
        assert probe.labels.count(0) == probe.labels.count(1)

    def test_create_code_trace_probe(self):
        """Test creating code trace probe."""
        probe = create_code_trace_probe()
        assert probe.name == "code_trace"
        assert len(probe) > 0
        assert probe.category == "computation"
        assert len(probe.prompts) == len(probe.labels)
        assert probe.label_names == ["discussion", "trace"]

    def test_create_factual_consistency_probe(self):
        """Test creating factual consistency probe."""
        probe = create_factual_consistency_probe()
        assert probe.name == "factual_consistency"
        assert len(probe) > 0
        assert probe.category == "factual"
        assert len(probe.prompts) == len(probe.labels)
        assert probe.label_names == ["contradiction", "consistent"]

    def test_create_tool_decision_probe(self):
        """Test creating tool decision probe."""
        probe = create_tool_decision_probe()
        assert probe.name == "tool_decision"
        assert len(probe) > 0
        assert probe.category == "decision"
        assert len(probe.prompts) == len(probe.labels)
        assert probe.label_names == ["no_tool", "tool"]

    def test_create_suppression_probe(self):
        """Test creating suppression probe."""
        probe = create_suppression_probe()
        assert probe.name == "suppression_mode"
        assert len(probe) > 0
        assert probe.category == "alignment"
        assert len(probe.prompts) == len(probe.labels)
        assert probe.label_names == ["compute", "suppress"]
        assert probe.description is not None

    def test_get_default_probe_datasets(self):
        """Test getting all default probe datasets."""
        datasets = get_default_probe_datasets()
        assert "arithmetic_mode" in datasets
        assert "code_trace" in datasets
        assert "factual_consistency" in datasets
        assert "tool_decision" in datasets
        assert "suppression_mode" in datasets
        assert len(datasets) == 5
        # All should be ProbeDataset instances
        for name, dataset in datasets.items():
            assert isinstance(dataset, ProbeDataset)
            assert len(dataset.prompts) == len(dataset.labels)


class TestProbeDatasetEdgeCases:
    """Additional edge case tests for ProbeDataset."""

    def test_baseline_accuracy_balanced(self):
        """Test baseline accuracy with balanced classes."""
        dataset = ProbeDataset(name="test", description="", prompts=["p1", "p2"], labels=[0, 1])
        assert dataset.baseline_accuracy == 0.5

    def test_baseline_accuracy_all_same(self):
        """Test baseline accuracy with all same labels."""
        dataset = ProbeDataset(
            name="test", description="", prompts=["p1", "p2", "p3"], labels=[1, 1, 1]
        )
        assert dataset.baseline_accuracy == 1.0

    def test_num_classes_binary(self):
        """Test num_classes with binary classification."""
        dataset = ProbeDataset(name="test", description="", prompts=["p1", "p2"], labels=[0, 1])
        assert dataset.num_classes == 2

    def test_num_classes_multiclass(self):
        """Test num_classes with multiple classes."""
        dataset = ProbeDataset(
            name="test", description="", prompts=["p1", "p2", "p3"], labels=[0, 1, 2]
        )
        assert dataset.num_classes == 3

    def test_default_label_names(self):
        """Test default label names are set correctly."""
        dataset = ProbeDataset(name="test", description="", prompts=["p1"], labels=[0])
        assert dataset.label_names == ["class_0", "class_1"]

    def test_from_dict_with_defaults(self):
        """Test from_dict uses defaults correctly."""
        data = {"prompts": ["p1", "p2"], "labels": [0, 1]}
        dataset = ProbeDataset.from_dict("test", data)
        assert dataset.description == ""
        assert dataset.label_names == ["class_0", "class_1"]
        assert dataset.category == "custom"


class TestProbeResultEdgeCases:
    """Additional edge case tests for ProbeResult."""

    def test_is_significant_boundary_accuracy(self):
        """Test significance at accuracy boundary."""
        result = ProbeResult(
            probe_name="test",
            layer=5,
            accuracy=0.6,
            cv_std=0.02,
            baseline=0.5,
            above_chance=0.1,
            n_samples=100,
        )
        # accuracy == 0.6 (meets threshold) and above_chance == 0.1 (meets threshold)
        assert result.is_significant is False  # above_chance must be > 0.1, not >=

    def test_is_significant_boundary_above_chance(self):
        """Test significance at above_chance boundary."""
        result = ProbeResult(
            probe_name="test",
            layer=5,
            accuracy=0.61,
            cv_std=0.02,
            baseline=0.5,
            above_chance=0.11,
            n_samples=100,
        )
        assert result.is_significant is True

    def test_is_significant_both_conditions_met(self):
        """Test significance when both conditions are met."""
        result = ProbeResult(
            probe_name="test",
            layer=5,
            accuracy=0.85,
            cv_std=0.02,
            baseline=0.5,
            above_chance=0.35,
            n_samples=100,
        )
        assert result.is_significant is True

    def test_above_chance_calculation(self):
        """Test that above_chance is correctly stored."""
        result = ProbeResult(
            probe_name="test",
            layer=5,
            accuracy=0.8,
            cv_std=0.02,
            baseline=0.6,
            above_chance=0.2,
            n_samples=50,
        )
        assert result.above_chance == 0.2


class TestStratigraphyResultEdgeCases:
    """Additional edge case tests for StratigraphyResult."""

    def test_get_accuracy_matrix_with_missing_layers(self):
        """Test accuracy matrix with missing layer data."""
        result = StratigraphyResult(model_id="test", num_layers=10)
        result.probes["probe1"] = {
            0: ProbeResult(
                probe_name="probe1",
                layer=0,
                accuracy=0.5,
                cv_std=0.02,
                baseline=0.5,
                above_chance=0.0,
                n_samples=100,
            ),
            2: ProbeResult(
                probe_name="probe1",
                layer=2,
                accuracy=0.8,
                cv_std=0.02,
                baseline=0.5,
                above_chance=0.3,
                n_samples=100,
            ),
        }
        matrix = result.get_accuracy_matrix(layers=[0, 1, 2])
        assert matrix["probe1"][0] == 0.5
        assert matrix["probe1"][1] == 0.0  # Missing layer
        assert matrix["probe1"][2] == 0.8

    def test_get_accuracy_matrix_auto_detect_layers(self):
        """Test accuracy matrix with automatic layer detection."""
        result = StratigraphyResult(model_id="test", num_layers=10)
        result.probes["probe1"] = {
            0: ProbeResult(
                probe_name="probe1",
                layer=0,
                accuracy=0.5,
                cv_std=0.02,
                baseline=0.5,
                above_chance=0.0,
                n_samples=100,
            ),
            5: ProbeResult(
                probe_name="probe1",
                layer=5,
                accuracy=0.8,
                cv_std=0.02,
                baseline=0.5,
                above_chance=0.3,
                n_samples=100,
            ),
        }
        result.probes["probe2"] = {
            0: ProbeResult(
                probe_name="probe2",
                layer=0,
                accuracy=0.6,
                cv_std=0.02,
                baseline=0.5,
                above_chance=0.1,
                n_samples=100,
            ),
            3: ProbeResult(
                probe_name="probe2",
                layer=3,
                accuracy=0.7,
                cv_std=0.02,
                baseline=0.5,
                above_chance=0.2,
                n_samples=100,
            ),
        }
        matrix = result.get_accuracy_matrix()
        assert len(matrix["probe1"]) == 3  # Layers 0, 3, 5
        assert len(matrix["probe2"]) == 3

    def test_find_emergence_layer_low_above_chance(self):
        """Test emergence not found when above_chance is too low."""
        result = StratigraphyResult(model_id="test", num_layers=10)
        result.probes["test_probe"] = {
            0: ProbeResult(
                probe_name="test",
                layer=0,
                accuracy=0.8,
                cv_std=0.02,
                baseline=0.75,
                above_chance=0.05,
                n_samples=100,
            ),  # Low above_chance
        }
        emergence = result.find_emergence_layer("test_probe", threshold=0.75)
        assert emergence is None

    def test_find_destruction_layer_nonexistent_probe(self):
        """Test destruction layer for non-existent probe."""
        result = StratigraphyResult(model_id="test", num_layers=10)
        assert result.find_destruction_layer("nonexistent") is None

    def test_find_destruction_layer_never_high(self):
        """Test destruction layer when accuracy never gets high."""
        result = StratigraphyResult(model_id="test", num_layers=10)
        result.probes["test_probe"] = {
            0: ProbeResult(
                probe_name="test",
                layer=0,
                accuracy=0.5,
                cv_std=0.02,
                baseline=0.5,
                above_chance=0.0,
                n_samples=100,
            ),
            2: ProbeResult(
                probe_name="test",
                layer=2,
                accuracy=0.6,
                cv_std=0.02,
                baseline=0.5,
                above_chance=0.1,
                n_samples=100,
            ),
        }
        destruction = result.find_destruction_layer("test_probe")
        assert destruction is None

    def test_save_and_load_multiple_probes(self):
        """Test saving and loading with multiple probes and layers."""
        result = StratigraphyResult(model_id="test-model", num_layers=10)
        result.probes["probe1"] = {
            0: ProbeResult(
                probe_name="probe1",
                layer=0,
                accuracy=0.8,
                cv_std=0.02,
                baseline=0.5,
                above_chance=0.3,
                n_samples=100,
            ),
            2: ProbeResult(
                probe_name="probe1",
                layer=2,
                accuracy=0.9,
                cv_std=0.03,
                baseline=0.5,
                above_chance=0.4,
                n_samples=100,
            ),
        }
        result.probes["probe2"] = {
            0: ProbeResult(
                probe_name="probe2",
                layer=0,
                accuracy=0.7,
                cv_std=0.01,
                baseline=0.5,
                above_chance=0.2,
                n_samples=100,
            ),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.json"
            result.save(path)

            # Verify JSON structure
            with open(path) as f:
                data = json.load(f)
            assert data["model_id"] == "test-model"
            assert data["num_layers"] == 10
            assert "probe1" in data["probes"]
            assert "probe2" in data["probes"]
            assert "0" in data["probes"]["probe1"]
            assert "2" in data["probes"]["probe1"]

            # Load and verify
            loaded = StratigraphyResult.load(path)
            assert loaded.model_id == "test-model"
            assert loaded.num_layers == 10
            assert len(loaded.probes) == 2
            assert 0 in loaded.probes["probe1"]
            assert 2 in loaded.probes["probe1"]
            assert loaded.probes["probe1"][0].accuracy == 0.8
            assert loaded.probes["probe1"][2].cv_std == 0.03

    def test_save_load_preserves_all_fields(self):
        """Test that save/load preserves all ProbeResult fields."""
        result = StratigraphyResult(model_id="test", num_layers=5)
        result.probes["test"] = {
            0: ProbeResult(
                probe_name="test",
                layer=0,
                accuracy=0.85,
                cv_std=0.025,
                baseline=0.55,
                above_chance=0.30,
                n_samples=150,
            )
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            result.save(path)
            loaded = StratigraphyResult.load(path)

            r = loaded.probes["test"][0]
            assert r.probe_name == "test"
            assert r.layer == 0
            assert r.accuracy == 0.85
            assert r.cv_std == 0.025
            assert r.baseline == 0.55
            assert r.above_chance == 0.30
            assert r.n_samples == 150


@sklearn_required
class TestProbeBatteryEdgeCases:
    """Additional edge case tests for ProbeBattery."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = Mock()
        model.model = Mock()
        model.model.layers = [Mock() for _ in range(4)]
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=np.array([[1, 2, 3]]))
        return tokenizer

    def test_detect_structure_direct_layers(self, mock_tokenizer):
        """Test structure detection with direct layers attribute."""
        model = Mock()
        model.layers = [Mock() for _ in range(6)]
        del model.model  # No model.model, just layers
        battery = ProbeBattery(model, mock_tokenizer)
        assert battery.num_layers == 6

    @patch("chuk_lazarus.introspection.ablation.AblationStudy")
    def test_from_pretrained_with_dataset_dir(self, mock_ablation_study):
        """Test loading from pretrained with custom dataset directory."""
        mock_study = Mock()
        mock_model = Mock()
        mock_model.model = Mock()
        mock_model.model.layers = [Mock() for _ in range(4)]
        mock_study.adapter.model = mock_model
        mock_study.adapter.tokenizer = Mock()
        mock_ablation_study.from_pretrained.return_value = mock_study

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            data = {"description": "Test", "prompts": ["p1"], "labels": [0]}
            with open(path / "test.json", "w") as f:
                json.dump(data, f)

            battery = ProbeBattery.from_pretrained("test-model", dataset_dir=path)
            assert "test" in battery.datasets

    @patch("chuk_lazarus.introspection.ablation.AblationStudy")
    def test_from_pretrained_without_dataset_dir(self, mock_ablation_study):
        """Test loading from pretrained without dataset directory."""
        mock_study = Mock()
        mock_model = Mock()
        mock_model.model = Mock()
        mock_model.model.layers = [Mock() for _ in range(4)]
        mock_study.adapter.model = mock_model
        mock_study.adapter.tokenizer = Mock()
        mock_ablation_study.from_pretrained.return_value = mock_study

        battery = ProbeBattery.from_pretrained("test-model")
        assert battery.model_id == "test-model"

    def test_load_datasets_yaml_without_pyyaml(self, mock_model, mock_tokenizer):
        """Test loading YAML files when PyYAML is not installed."""
        battery = ProbeBattery(mock_model, mock_tokenizer)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.yaml"
            with open(path, "w") as f:
                f.write("prompts: [p1]\nlabels: [0]")

            with patch("builtins.__import__", side_effect=ImportError):
                # Should print message and skip
                battery.load_datasets(path)
                # No datasets should be loaded
                assert len(battery.datasets) == 0

    def test_load_datasets_yaml_with_pyyaml(self, mock_model, mock_tokenizer):
        """Test loading YAML files with PyYAML installed."""
        battery = ProbeBattery(mock_model, mock_tokenizer)

        try:
            import yaml

            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "test.yaml"
                data = {"description": "Test", "prompts": ["p1"], "labels": [0]}
                with open(path, "w") as f:
                    yaml.dump(data, f)

                battery.load_datasets(path)
                assert "test" in battery.datasets
        except ImportError:
            pytest.skip("PyYAML not available")

    def test_load_datasets_from_directory_yaml_and_json(self, mock_model, mock_tokenizer):
        """Test loading both YAML and JSON files from directory."""
        battery = ProbeBattery(mock_model, mock_tokenizer)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            # Create JSON file
            data_json = {"description": "JSON probe", "prompts": ["p1"], "labels": [0]}
            with open(path / "probe1.json", "w") as f:
                json.dump(data_json, f)

            # Try YAML if available
            try:
                import yaml

                data_yaml = {
                    "description": "YAML probe",
                    "prompts": ["p2"],
                    "labels": [1],
                }
                with open(path / "probe2.yaml", "w") as f:
                    yaml.dump(data_yaml, f)
            except ImportError:
                pass

            battery.load_datasets(path)
            assert "probe1" in battery.datasets

    @patch("chuk_lazarus.introspection.hooks.ModelHooks")
    def test_get_activations_bfloat16(self, mock_hooks_cls, mock_model, mock_tokenizer):
        """Test getting activations with bfloat16 dtype."""
        mock_hooks = Mock()
        mock_state = Mock()
        # Simulate bfloat16 activations
        mock_state.hidden_states = {0: mx.ones((1, 5, 64), dtype=mx.bfloat16)}
        mock_hooks.state = mock_state
        mock_hooks_cls.return_value = mock_hooks

        battery = ProbeBattery(mock_model, mock_tokenizer)
        acts = battery.get_activations("Test prompt", layer=0)
        assert isinstance(acts, np.ndarray)
        assert acts.dtype == np.float32  # Should be converted to float32

    @patch("chuk_lazarus.introspection.hooks.ModelHooks")
    def test_get_activations_2d_array(self, mock_hooks_cls, mock_model, mock_tokenizer):
        """Test getting activations with 2D array (no batch dimension)."""
        mock_hooks = Mock()
        mock_state = Mock()
        # 2D array instead of 3D
        mock_state.hidden_states = {0: mx.ones((5, 64), dtype=mx.float32)}
        mock_hooks.state = mock_state
        mock_hooks_cls.return_value = mock_hooks

        battery = ProbeBattery(mock_model, mock_tokenizer)
        acts = battery.get_activations("Test prompt", layer=0)
        assert isinstance(acts, np.ndarray)
        assert acts.shape == (64,)

    @patch("chuk_lazarus.introspection.hooks.ModelHooks")
    def test_get_activations_custom_position(self, mock_hooks_cls, mock_model, mock_tokenizer):
        """Test getting activations at a custom position."""
        mock_hooks = Mock()
        mock_state = Mock()
        mock_state.hidden_states = {0: mx.ones((1, 5, 64), dtype=mx.float32)}
        mock_hooks.state = mock_state
        mock_hooks_cls.return_value = mock_hooks

        battery = ProbeBattery(mock_model, mock_tokenizer)
        acts = battery.get_activations("Test prompt", layer=0, position=2)
        assert isinstance(acts, np.ndarray)
        assert acts.shape == (64,)

    def test_train_probe_with_cv_folds(self):
        """Test training probe with custom cv_folds."""
        X = np.random.randn(30, 64)
        y = np.array([0] * 15 + [1] * 15)
        battery = Mock(spec=ProbeBattery)
        battery.train_probe = ProbeBattery.train_probe.__get__(battery, ProbeBattery)
        accuracy, std = battery.train_probe(X, y, cv_folds=3)
        assert 0.0 <= accuracy <= 1.0
        assert std >= 0.0

    def test_train_probe_few_samples(self):
        """Test training probe with very few samples (less than cv_folds)."""
        # Need at least 2 samples per class for stratified k-fold with k=3
        X = np.random.randn(6, 64)
        y = np.array([0, 0, 0, 1, 1, 1])
        battery = Mock(spec=ProbeBattery)
        battery.train_probe = ProbeBattery.train_probe.__get__(battery, ProbeBattery)
        accuracy, std = battery.train_probe(X, y, cv_folds=3)
        assert 0.0 <= accuracy <= 1.0

    @patch("chuk_lazarus.introspection.hooks.ModelHooks")
    def test_run_all_probes_default_layers(self, mock_hooks_cls, mock_model, mock_tokenizer):
        """Test running all probes with default layer selection."""
        mock_hooks = Mock()
        mock_state = Mock()
        mock_state.hidden_states = {i: mx.ones((1, 5, 64), dtype=mx.float32) for i in range(4)}
        mock_hooks.state = mock_state
        mock_hooks_cls.return_value = mock_hooks

        battery = ProbeBattery(mock_model, mock_tokenizer)
        # Need at least 5 samples per class for cv_folds=5 (default)
        dataset = ProbeDataset(
            name="test",
            description="",
            prompts=[f"p{i}" for i in range(10)],
            labels=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        )
        battery.add_dataset(dataset)

        # Don't specify layers - should use default evenly spaced
        results = battery.run_all_probes(progress=False)
        assert isinstance(results, StratigraphyResult)
        # Should include last layer
        assert 3 in results.probes["test"]

    @patch("chuk_lazarus.introspection.hooks.ModelHooks")
    def test_run_all_probes_with_progress(self, mock_hooks_cls, mock_model, mock_tokenizer, capsys):
        """Test running all probes with progress output."""
        mock_hooks = Mock()
        mock_state = Mock()
        mock_state.hidden_states = {0: mx.ones((1, 5, 64), dtype=mx.float32)}
        mock_hooks.state = mock_state
        mock_hooks_cls.return_value = mock_hooks

        battery = ProbeBattery(mock_model, mock_tokenizer)
        # Need at least 5 samples per class for cv_folds=5 (default)
        dataset = ProbeDataset(
            name="test",
            description="",
            prompts=[f"p{i}" for i in range(10)],
            labels=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        )
        battery.add_dataset(dataset)

        battery.run_all_probes(layers=[0], progress=True)
        captured = capsys.readouterr()
        assert "Probing:" in captured.out
        assert "test" in captured.out

    def test_print_results_table_multiple_probes(self, mock_model, mock_tokenizer, capsys):
        """Test printing results table with multiple probes."""
        battery = ProbeBattery(mock_model, mock_tokenizer)
        results = StratigraphyResult(model_id="test", num_layers=4)
        results.probes["probe1"] = {
            0: ProbeResult(
                probe_name="probe1",
                layer=0,
                accuracy=0.8,
                cv_std=0.02,
                baseline=0.5,
                above_chance=0.3,
                n_samples=100,
            ),
            2: ProbeResult(
                probe_name="probe1",
                layer=2,
                accuracy=0.9,
                cv_std=0.02,
                baseline=0.5,
                above_chance=0.4,
                n_samples=100,
            ),
        }
        results.probes["probe2"] = {
            0: ProbeResult(
                probe_name="probe2",
                layer=0,
                accuracy=0.7,
                cv_std=0.02,
                baseline=0.5,
                above_chance=0.2,
                n_samples=100,
            ),
            2: ProbeResult(
                probe_name="probe2",
                layer=2,
                accuracy=0.6,
                cv_std=0.02,
                baseline=0.5,
                above_chance=0.1,
                n_samples=100,
            ),
        }

        battery.print_results_table(results)
        captured = capsys.readouterr()
        assert "probe1" in captured.out
        assert "probe2" in captured.out
        assert "0.80" in captured.out or "0.8" in captured.out
        assert "0.90*" in captured.out or "0.9*" in captured.out  # > 0.85

    def test_print_results_table_missing_layer(self, mock_model, mock_tokenizer, capsys):
        """Test printing results table with missing layer data."""
        battery = ProbeBattery(mock_model, mock_tokenizer)
        results = StratigraphyResult(model_id="test", num_layers=4)
        results.probes["probe1"] = {
            0: ProbeResult(
                probe_name="probe1",
                layer=0,
                accuracy=0.8,
                cv_std=0.02,
                baseline=0.5,
                above_chance=0.3,
                n_samples=100,
            ),
            # Layer 1 missing
            2: ProbeResult(
                probe_name="probe1",
                layer=2,
                accuracy=0.9,
                cv_std=0.02,
                baseline=0.5,
                above_chance=0.4,
                n_samples=100,
            ),
        }

        battery.print_results_table(results)
        captured = capsys.readouterr()
        assert "-" in captured.out  # Should show - for missing layer

    def test_print_stratigraphy_multiple_categories(self, mock_model, mock_tokenizer, capsys):
        """Test printing stratigraphy with multiple categories."""
        battery = ProbeBattery(mock_model, mock_tokenizer)
        battery.add_dataset(
            ProbeDataset(
                name="probe1",
                description="Test probe 1",
                prompts=["p1"],
                labels=[0],
                category="cat1",
            )
        )
        battery.add_dataset(
            ProbeDataset(
                name="probe2",
                description="Test probe 2",
                prompts=["p2"],
                labels=[1],
                category="cat2",
            )
        )

        results = StratigraphyResult(model_id="test-model", num_layers=4)
        results.probes["probe1"] = {
            0: ProbeResult(
                probe_name="probe1",
                layer=0,
                accuracy=0.8,
                cv_std=0.02,
                baseline=0.5,
                above_chance=0.3,
                n_samples=100,
            )
        }
        results.probes["probe2"] = {
            2: ProbeResult(
                probe_name="probe2",
                layer=2,
                accuracy=0.85,
                cv_std=0.02,
                baseline=0.5,
                above_chance=0.35,
                n_samples=100,
            )
        }

        battery.print_stratigraphy(results, threshold=0.75)
        captured = capsys.readouterr()
        assert "CAT1:" in captured.out or "cat1:" in captured.out
        assert "CAT2:" in captured.out or "cat2:" in captured.out
        assert "probe1" in captured.out
        assert "probe2" in captured.out

    def test_print_stratigraphy_never_emerges(self, mock_model, mock_tokenizer, capsys):
        """Test printing stratigraphy when probe never emerges."""
        battery = ProbeBattery(mock_model, mock_tokenizer)
        battery.add_dataset(
            ProbeDataset(
                name="probe1",
                description="Test probe",
                prompts=["p1"],
                labels=[0],
                category="test",
            )
        )

        results = StratigraphyResult(model_id="test-model", num_layers=4)
        results.probes["probe1"] = {
            0: ProbeResult(
                probe_name="probe1",
                layer=0,
                accuracy=0.5,
                cv_std=0.02,
                baseline=0.5,
                above_chance=0.0,
                n_samples=100,
            )
        }

        battery.print_stratigraphy(results, threshold=0.75)
        captured = capsys.readouterr()
        assert "Never" in captured.out

    def test_print_stratigraphy_sorted_by_emergence(self, mock_model, mock_tokenizer, capsys):
        """Test that stratigraphy prints probes sorted by emergence layer."""
        battery = ProbeBattery(mock_model, mock_tokenizer)
        battery.add_dataset(
            ProbeDataset(
                name="late",
                description="Late probe",
                prompts=["p1"],
                labels=[0],
                category="test",
            )
        )
        battery.add_dataset(
            ProbeDataset(
                name="early",
                description="Early probe",
                prompts=["p2"],
                labels=[1],
                category="test",
            )
        )

        results = StratigraphyResult(model_id="test-model", num_layers=10)
        results.probes["late"] = {
            5: ProbeResult(
                probe_name="late",
                layer=5,
                accuracy=0.8,
                cv_std=0.02,
                baseline=0.5,
                above_chance=0.3,
                n_samples=100,
            )
        }
        results.probes["early"] = {
            1: ProbeResult(
                probe_name="early",
                layer=1,
                accuracy=0.8,
                cv_std=0.02,
                baseline=0.5,
                above_chance=0.3,
                n_samples=100,
            )
        }

        battery.print_stratigraphy(results, threshold=0.75)
        captured = capsys.readouterr()
        # Early should appear before late
        early_pos = captured.out.find("early")
        late_pos = captured.out.find("late")
        assert early_pos < late_pos


class TestProbeBatteryWithMockedSklearn:
    """Tests for ProbeBattery with mocked sklearn to avoid NumPy compatibility issues."""

    def _mock_sklearn_modules(self, **mocks):
        """Mock sklearn modules in sys.modules to avoid NumPy 2.x compatibility issues.

        This method creates a context manager that temporarily replaces sklearn modules
        in sys.modules with mocks, avoiding the NumPy 2.3.0 compatibility errors.
        """
        import sys
        from contextlib import contextmanager

        @contextmanager
        def mock_context():
            from unittest.mock import MagicMock

            # Build the mock sklearn hierarchy
            mock_sklearn = MagicMock()
            mock_linear_model = MagicMock()
            mock_model_selection = MagicMock()
            mock_preprocessing = MagicMock()

            # Assign provided mocks
            if "LogisticRegression" in mocks:
                mock_linear_model.LogisticRegression = mocks["LogisticRegression"]
            if "cross_val_score" in mocks:
                mock_model_selection.cross_val_score = mocks["cross_val_score"]
            if "StandardScaler" in mocks:
                mock_preprocessing.StandardScaler = mocks["StandardScaler"]

            # Link submodules
            mock_sklearn.linear_model = mock_linear_model
            mock_sklearn.model_selection = mock_model_selection
            mock_sklearn.preprocessing = mock_preprocessing

            # Save original modules
            original_modules = {}
            modules_to_mock = [
                "sklearn",
                "sklearn.linear_model",
                "sklearn.model_selection",
                "sklearn.preprocessing",
            ]
            for mod in modules_to_mock:
                original_modules[mod] = sys.modules.get(mod)

            # Install mocks
            sys.modules["sklearn"] = mock_sklearn
            sys.modules["sklearn.linear_model"] = mock_linear_model
            sys.modules["sklearn.model_selection"] = mock_model_selection
            sys.modules["sklearn.preprocessing"] = mock_preprocessing

            try:
                yield
            finally:
                # Restore original modules
                for mod, orig in original_modules.items():
                    if orig is None:
                        sys.modules.pop(mod, None)
                    else:
                        sys.modules[mod] = orig

        return mock_context()

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = Mock()
        model.model = Mock()
        model.model.layers = [Mock() for _ in range(4)]
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=np.array([[1, 2, 3]]))
        return tokenizer

    def test_detect_structure_model_layers(self, mock_tokenizer):
        """Test _detect_structure with model.model.layers pattern (lines 248-249)."""
        model = Mock()
        model.model = Mock()
        model.model.layers = [Mock() for _ in range(6)]
        battery = ProbeBattery(model, mock_tokenizer)
        assert battery.num_layers == 6
        assert battery._layers is model.model.layers

    def test_detect_structure_direct_layers(self, mock_tokenizer):
        """Test _detect_structure with direct layers attribute (lines 250-251)."""
        model = Mock()
        model.layers = [Mock() for _ in range(8)]
        # Ensure model.model doesn't exist
        delattr(model, "model") if hasattr(model, "model") else None
        battery = ProbeBattery(model, mock_tokenizer)
        assert battery.num_layers == 8
        assert battery._layers is model.layers

    def test_detect_structure_raises_on_missing(self, mock_tokenizer):
        """Test _detect_structure raises ValueError when layers not found (lines 252-253)."""
        model = Mock()
        # Remove both layers attributes
        if hasattr(model, "model"):
            delattr(model, "model")
        if hasattr(model, "layers"):
            delattr(model, "layers")
        with pytest.raises(ValueError, match="Cannot detect model layer structure"):
            ProbeBattery(model, mock_tokenizer)

    @patch("chuk_lazarus.introspection.ablation.AblationStudy")
    def test_from_pretrained_with_dataset_dir(self, mock_ablation_study, mock_tokenizer):
        """Test from_pretrained with custom dataset_dir (lines 264-284)."""
        # Setup mock
        mock_study = Mock()
        mock_model = Mock()
        mock_model.model = Mock()
        mock_model.model.layers = [Mock() for _ in range(4)]
        mock_study.adapter.model = mock_model
        mock_study.adapter.tokenizer = mock_tokenizer
        mock_ablation_study.from_pretrained.return_value = mock_study

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            # Create a test dataset file
            data = {"description": "Test", "prompts": ["p1"], "labels": [0]}
            with open(path / "test.json", "w") as f:
                json.dump(data, f)

            battery = ProbeBattery.from_pretrained("test-model", dataset_dir=path)
            assert battery.model_id == "test-model"
            assert "test" in battery.datasets

    @patch("chuk_lazarus.introspection.ablation.AblationStudy")
    def test_from_pretrained_with_default_dir(self, mock_ablation_study, mock_tokenizer):
        """Test from_pretrained tries default directory (lines 276-282)."""
        mock_study = Mock()
        mock_model = Mock()
        mock_model.model = Mock()
        mock_model.model.layers = [Mock() for _ in range(4)]
        mock_study.adapter.model = mock_model
        mock_study.adapter.tokenizer = mock_tokenizer
        mock_ablation_study.from_pretrained.return_value = mock_study

        # Without dataset_dir, should try default location
        battery = ProbeBattery.from_pretrained("test-model")
        assert battery.model_id == "test-model"

    def test_load_datasets_from_directory(self, mock_model, mock_tokenizer):
        """Test load_datasets with directory path (lines 292-296)."""
        battery = ProbeBattery(mock_model, mock_tokenizer)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            # Create JSON file
            json_data = {"description": "JSON probe", "prompts": ["p1"], "labels": [0]}
            with open(path / "probe1.json", "w") as f:
                json.dump(json_data, f)

            # Create YAML file if PyYAML is available
            try:
                import yaml

                yaml_data = {
                    "description": "YAML probe",
                    "prompts": ["p2"],
                    "labels": [1],
                }
                with open(path / "probe2.yaml", "w") as f:
                    yaml.dump(yaml_data, f)
            except ImportError:
                pass

            battery.load_datasets(path)
            assert "probe1" in battery.datasets

    def test_load_datasets_invalid_path(self, mock_model, mock_tokenizer):
        """Test load_datasets raises on invalid path (lines 297-298)."""
        battery = ProbeBattery(mock_model, mock_tokenizer)
        with pytest.raises(ValueError, match="Path not found"):
            battery.load_datasets("/nonexistent/invalid/path")

    def test_load_dataset_file_yaml_with_import_error(self, mock_model, mock_tokenizer):
        """Test _load_dataset_file handles YAML import error (lines 303-310)."""
        battery = ProbeBattery(mock_model, mock_tokenizer)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.yaml"
            with open(path, "w") as f:
                f.write("prompts: [p1]\nlabels: [0]")

            # Mock yaml import to raise ImportError
            with patch("builtins.__import__", side_effect=ImportError):
                # Should print message and return without error
                battery._load_dataset_file(path)
                # No datasets should be loaded
                assert len(battery.datasets) == 0

    def test_add_dataset(self, mock_model, mock_tokenizer):
        """Test add_dataset method (line 327)."""
        battery = ProbeBattery(mock_model, mock_tokenizer)
        dataset = ProbeDataset(
            name="custom_probe",
            description="Custom test probe",
            prompts=["test prompt"],
            labels=[1],
        )
        battery.add_dataset(dataset)
        assert "custom_probe" in battery.datasets
        assert battery.datasets["custom_probe"] is dataset

    @patch("chuk_lazarus.introspection.hooks.ModelHooks")
    def test_get_activations_detailed(self, mock_hooks_cls, mock_model, mock_tokenizer):
        """Test get_activations method in detail (lines 336-360)."""
        import mlx.core as mx

        # Setup hooks mock
        mock_hooks = Mock()
        mock_state = Mock()
        # Create 3D tensor: (batch=1, seq_len=5, hidden_size=64)
        mock_state.hidden_states = {2: mx.ones((1, 5, 64), dtype=mx.float32)}
        mock_hooks.state = mock_state
        mock_hooks_cls.return_value = mock_hooks

        battery = ProbeBattery(mock_model, mock_tokenizer)
        acts = battery.get_activations("Test prompt", layer=2, position=-1)

        # Verify it's a numpy array of correct shape
        assert isinstance(acts, np.ndarray)
        assert acts.shape == (64,)

        # Verify hooks were configured correctly
        mock_hooks.configure.assert_called_once()
        mock_hooks.forward.assert_called_once()

    @patch("chuk_lazarus.introspection.hooks.ModelHooks")
    def test_get_activations_bfloat16_conversion(self, mock_hooks_cls, mock_model, mock_tokenizer):
        """Test get_activations converts bfloat16 to float32 (lines 356-357)."""
        import mlx.core as mx

        mock_hooks = Mock()
        mock_state = Mock()
        # Use bfloat16 dtype
        mock_state.hidden_states = {0: mx.ones((1, 5, 64), dtype=mx.bfloat16)}
        mock_hooks.state = mock_state
        mock_hooks_cls.return_value = mock_hooks

        battery = ProbeBattery(mock_model, mock_tokenizer)
        acts = battery.get_activations("Test", layer=0)

        # Should be converted to float32 numpy array
        assert acts.dtype == np.float32

    @patch("chuk_lazarus.introspection.hooks.ModelHooks")
    def test_get_activations_2d_tensor(self, mock_hooks_cls, mock_model, mock_tokenizer):
        """Test get_activations with 2D tensor (lines 358-360)."""
        import mlx.core as mx

        mock_hooks = Mock()
        mock_state = Mock()
        # 2D tensor (no batch dimension)
        mock_state.hidden_states = {0: mx.ones((5, 64), dtype=mx.float32)}
        mock_hooks.state = mock_state
        mock_hooks_cls.return_value = mock_hooks

        battery = ProbeBattery(mock_model, mock_tokenizer)
        acts = battery.get_activations("Test", layer=0)

        assert acts.shape == (64,)

    @patch("chuk_lazarus.introspection.hooks.ModelHooks")
    def test_collect_dataset_activations(self, mock_hooks_cls, mock_model, mock_tokenizer):
        """Test collect_dataset_activations (lines 368-375)."""
        import mlx.core as mx

        mock_hooks = Mock()
        mock_state = Mock()
        mock_state.hidden_states = {0: mx.ones((1, 5, 64), dtype=mx.float32)}
        mock_hooks.state = mock_state
        mock_hooks_cls.return_value = mock_hooks

        battery = ProbeBattery(mock_model, mock_tokenizer)
        dataset = ProbeDataset(
            name="test", description="", prompts=["p1", "p2", "p3"], labels=[0, 1, 0]
        )

        X, y = battery.collect_dataset_activations(dataset, layer=0)

        assert X.shape == (3, 64)
        assert y.shape == (3,)
        assert np.array_equal(y, [0, 1, 0])

    def test_train_probe_with_mocked_sklearn(self, mock_model, mock_tokenizer):
        """Test train_probe with mocked sklearn (lines 384-401)."""
        from unittest.mock import MagicMock

        # Create test data
        X = np.random.randn(20, 64)
        y = np.array([0] * 10 + [1] * 10)

        # Setup mocks
        mock_scaler_instance = MagicMock()
        mock_scaler_instance.fit_transform.return_value = X  # Return scaled data
        mock_scaler_class = MagicMock(return_value=mock_scaler_instance)

        mock_probe_instance = MagicMock()
        mock_cross_val_score = MagicMock(return_value=np.array([0.85, 0.80, 0.82, 0.88, 0.85]))

        battery = ProbeBattery(mock_model, mock_tokenizer)

        with self._mock_sklearn_modules(
            LogisticRegression=MagicMock(return_value=mock_probe_instance),
            cross_val_score=mock_cross_val_score,
            StandardScaler=mock_scaler_class,
        ):
            accuracy, std = battery.train_probe(X, y, cv_folds=5)

        assert 0.0 <= accuracy <= 1.0
        assert std >= 0.0

    def test_train_probe_insufficient_samples(self, mock_model, mock_tokenizer):
        """Test train_probe with too few samples (lines 388-393)."""
        from unittest.mock import MagicMock

        X = np.random.randn(1, 64)
        y = np.array([0])

        battery = ProbeBattery(mock_model, mock_tokenizer)

        # Even though we won't use sklearn when n_samples < 2,
        # we still need to mock it because the import happens at function call
        with self._mock_sklearn_modules(
            LogisticRegression=MagicMock(),
            cross_val_score=MagicMock(),
            StandardScaler=MagicMock(),
        ):
            accuracy, std = battery.train_probe(X, y, cv_folds=5)

        # Should return default values when n_samples < 2
        assert accuracy == 0.5
        assert std == 0.0

    @patch("chuk_lazarus.introspection.hooks.ModelHooks")
    def test_run_probe(self, mock_hooks_cls, mock_model, mock_tokenizer):
        """Test run_probe method (lines 409-415)."""
        from unittest.mock import MagicMock

        import mlx.core as mx

        mock_hooks = Mock()
        mock_state = Mock()
        mock_state.hidden_states = {0: mx.ones((1, 5, 64), dtype=mx.float32)}
        mock_hooks.state = mock_state
        mock_hooks_cls.return_value = mock_hooks

        battery = ProbeBattery(mock_model, mock_tokenizer)
        dataset = ProbeDataset(
            name="test_probe",
            description="Test",
            prompts=["p1", "p2", "p3", "p4", "p5"],
            labels=[0, 1, 0, 1, 0],
        )
        battery.add_dataset(dataset)

        # Mock sklearn for train_probe
        mock_scaler_instance = MagicMock()
        mock_scaler_instance.fit_transform.return_value = np.random.randn(5, 64)
        mock_scaler_class = MagicMock(return_value=mock_scaler_instance)

        mock_cross_val_score = MagicMock(return_value=np.array([0.8, 0.85]))

        with self._mock_sklearn_modules(
            LogisticRegression=MagicMock(),
            cross_val_score=mock_cross_val_score,
            StandardScaler=mock_scaler_class,
        ):
            result = battery.run_probe("test_probe", layer=0)

        assert isinstance(result, ProbeResult)
        assert result.probe_name == "test_probe"
        assert result.layer == 0

    @patch("chuk_lazarus.introspection.hooks.ModelHooks")
    def test_run_all_probes_default_layers(self, mock_hooks_cls, mock_model, mock_tokenizer):
        """Test run_all_probes with default layer selection (lines 442-472)."""
        from unittest.mock import MagicMock

        import mlx.core as mx

        mock_hooks = Mock()
        mock_state = Mock()
        # Create hidden states for all layers
        mock_state.hidden_states = {i: mx.ones((1, 5, 64), dtype=mx.float32) for i in range(4)}
        mock_hooks.state = mock_state
        mock_hooks_cls.return_value = mock_hooks

        battery = ProbeBattery(mock_model, mock_tokenizer)
        dataset = ProbeDataset(
            name="test",
            description="",
            prompts=["p1", "p2", "p3", "p4", "p5"],
            labels=[0, 1, 0, 1, 0],
            category="test_cat",
        )
        battery.add_dataset(dataset)

        # Mock sklearn
        mock_scaler_instance = MagicMock()
        mock_scaler_instance.fit_transform.return_value = np.random.randn(5, 64)
        mock_scaler_class = MagicMock(return_value=mock_scaler_instance)

        mock_cross_val_score = MagicMock(return_value=np.array([0.8, 0.85]))

        with self._mock_sklearn_modules(
            LogisticRegression=MagicMock(),
            cross_val_score=mock_cross_val_score,
            StandardScaler=mock_scaler_class,
        ):
            # Don't specify layers - should use default (evenly spaced)
            result = battery.run_all_probes(layers=None, progress=False)

        assert isinstance(result, StratigraphyResult)
        assert "test" in result.probes
        # Should include last layer
        assert 3 in result.probes["test"]

    @patch("chuk_lazarus.introspection.hooks.ModelHooks")
    def test_run_all_probes_with_categories(self, mock_hooks_cls, mock_model, mock_tokenizer):
        """Test run_all_probes with category filtering (lines 454-457)."""
        from unittest.mock import MagicMock

        import mlx.core as mx

        mock_hooks = Mock()
        mock_state = Mock()
        mock_state.hidden_states = {0: mx.ones((1, 5, 64), dtype=mx.float32)}
        mock_hooks.state = mock_state
        mock_hooks_cls.return_value = mock_hooks

        battery = ProbeBattery(mock_model, mock_tokenizer)

        dataset1 = ProbeDataset(
            name="cat1_probe",
            description="",
            prompts=["p1", "p2", "p3"],
            labels=[0, 1, 0],
            category="cat1",
        )
        dataset2 = ProbeDataset(
            name="cat2_probe",
            description="",
            prompts=["p4", "p5", "p6"],
            labels=[1, 0, 1],
            category="cat2",
        )
        battery.add_dataset(dataset1)
        battery.add_dataset(dataset2)

        # Mock sklearn
        mock_scaler_instance = MagicMock()
        mock_scaler_instance.fit_transform.return_value = np.random.randn(3, 64)
        mock_scaler_class = MagicMock(return_value=mock_scaler_instance)

        mock_cross_val_score = MagicMock(return_value=np.array([0.8, 0.85]))

        with self._mock_sklearn_modules(
            LogisticRegression=MagicMock(),
            cross_val_score=mock_cross_val_score,
            StandardScaler=mock_scaler_class,
        ):
            # Filter by category
            result = battery.run_all_probes(layers=[0], categories=["cat1"], progress=False)

        assert "cat1_probe" in result.probes
        assert "cat2_probe" not in result.probes

    @patch("chuk_lazarus.introspection.hooks.ModelHooks")
    def test_run_all_probes_with_progress(self, mock_hooks_cls, mock_model, mock_tokenizer, capsys):
        """Test run_all_probes with progress output (lines 460-470)."""
        from unittest.mock import MagicMock

        import mlx.core as mx

        mock_hooks = Mock()
        mock_state = Mock()
        mock_state.hidden_states = {0: mx.ones((1, 5, 64), dtype=mx.float32)}
        mock_hooks.state = mock_state
        mock_hooks_cls.return_value = mock_hooks

        battery = ProbeBattery(mock_model, mock_tokenizer)
        dataset = ProbeDataset(
            name="test",
            description="",
            prompts=["p1", "p2", "p3"],
            labels=[0, 1, 0],
            category="test_cat",
        )
        battery.add_dataset(dataset)

        # Mock sklearn
        mock_scaler_instance = MagicMock()
        mock_scaler_instance.fit_transform.return_value = np.random.randn(3, 64)
        mock_scaler_class = MagicMock(return_value=mock_scaler_instance)

        mock_cross_val_score = MagicMock(return_value=np.array([0.8, 0.85]))

        with self._mock_sklearn_modules(
            LogisticRegression=MagicMock(),
            cross_val_score=mock_cross_val_score,
            StandardScaler=mock_scaler_class,
        ):
            battery.run_all_probes(layers=[0], progress=True)

        captured = capsys.readouterr()
        assert "Probing:" in captured.out
        assert "test" in captured.out
        assert "L 0:" in captured.out or "L0:" in captured.out

    def test_print_results_table_with_star(self, mock_model, mock_tokenizer, capsys):
        """Test print_results_table shows star for high accuracy (lines 491-499)."""
        battery = ProbeBattery(mock_model, mock_tokenizer)
        results = StratigraphyResult(model_id="test", num_layers=4)
        results.probes["probe1"] = {
            0: ProbeResult(
                probe_name="probe1",
                layer=0,
                accuracy=0.90,
                cv_std=0.02,
                baseline=0.5,
                above_chance=0.4,
                n_samples=100,
            ),  # > 0.85
            2: ProbeResult(
                probe_name="probe1",
                layer=2,
                accuracy=0.75,
                cv_std=0.02,
                baseline=0.5,
                above_chance=0.25,
                n_samples=100,
            ),  # < 0.85
        }

        battery.print_results_table(results)
        captured = capsys.readouterr()

        # Should show star for 0.90
        assert "*" in captured.out
        assert "0.90*" in captured.out or "0.9*" in captured.out

    def test_print_stratigraphy_detailed(self, mock_model, mock_tokenizer, capsys):
        """Test print_stratigraphy with detailed output (lines 536)."""
        battery = ProbeBattery(mock_model, mock_tokenizer)

        # Add datasets with descriptions
        dataset = ProbeDataset(
            name="probe1",
            description="This is a detailed description",
            prompts=["p1"],
            labels=[0],
            category="test",
        )
        battery.add_dataset(dataset)

        results = StratigraphyResult(model_id="test-model", num_layers=4)
        results.probes["probe1"] = {
            0: ProbeResult(
                probe_name="probe1",
                layer=0,
                accuracy=0.8,
                cv_std=0.02,
                baseline=0.5,
                above_chance=0.3,
                n_samples=100,
            )
        }

        battery.print_stratigraphy(results, threshold=0.75)
        captured = capsys.readouterr()

        # Should print description (line 536)
        assert "This is a detailed description" in captured.out

    @patch("chuk_lazarus.introspection.hooks.ModelHooks")
    def test_run_all_probes_adds_last_layer(self, mock_hooks_cls, mock_tokenizer):
        """Test run_all_probes appends last layer when needed (line 446)."""
        from unittest.mock import MagicMock

        import mlx.core as mx

        # Create a model with 25 layers
        # num_layers // 10 = 2, so range(0, 25, 2) = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
        # This DOES include 24 (last layer), but let's try 26 layers
        # With 26 layers: range(0, 26, 2) = [0, 2, 4, ..., 24], which doesn't include 25
        model = Mock()
        model.model = Mock()
        model.model.layers = [Mock() for _ in range(26)]

        mock_hooks = Mock()
        mock_state = Mock()
        # Create hidden states for all layers
        mock_state.hidden_states = {i: mx.ones((1, 5, 64), dtype=mx.float32) for i in range(26)}
        mock_hooks.state = mock_state
        mock_hooks_cls.return_value = mock_hooks

        battery = ProbeBattery(model, mock_tokenizer)
        dataset = ProbeDataset(
            name="test",
            description="",
            prompts=["p1", "p2", "p3"],
            labels=[0, 1, 0],
        )
        battery.add_dataset(dataset)

        # Mock sklearn
        mock_scaler_instance = MagicMock()
        mock_scaler_instance.fit_transform.return_value = np.random.randn(3, 64)
        mock_scaler_class = MagicMock(return_value=mock_scaler_instance)

        mock_cross_val_score = MagicMock(return_value=np.array([0.8, 0.85]))

        with self._mock_sklearn_modules(
            LogisticRegression=MagicMock(),
            cross_val_score=mock_cross_val_score,
            StandardScaler=mock_scaler_class,
        ):
            # Don't specify layers - should use default and add last layer
            result = battery.run_all_probes(layers=None, progress=False)

        # Should include last layer (25)
        assert 25 in result.probes["test"]

    def test_print_results_table_with_missing_layer(self, mock_model, mock_tokenizer, capsys):
        """Test print_results_table displays dash for missing layer (line 498)."""
        battery = ProbeBattery(mock_model, mock_tokenizer)
        results = StratigraphyResult(model_id="test", num_layers=4)
        # probe1 has layers 0 and 2
        results.probes["probe1"] = {
            0: ProbeResult(
                probe_name="probe1",
                layer=0,
                accuracy=0.80,
                cv_std=0.02,
                baseline=0.5,
                above_chance=0.3,
                n_samples=100,
            ),
            2: ProbeResult(
                probe_name="probe1",
                layer=2,
                accuracy=0.75,
                cv_std=0.02,
                baseline=0.5,
                above_chance=0.25,
                n_samples=100,
            ),
        }
        # probe2 has layers 0, 1, and 2
        results.probes["probe2"] = {
            0: ProbeResult(
                probe_name="probe2",
                layer=0,
                accuracy=0.85,
                cv_std=0.02,
                baseline=0.5,
                above_chance=0.35,
                n_samples=100,
            ),
            1: ProbeResult(
                probe_name="probe2",
                layer=1,
                accuracy=0.70,
                cv_std=0.02,
                baseline=0.5,
                above_chance=0.20,
                n_samples=100,
            ),
            2: ProbeResult(
                probe_name="probe2",
                layer=2,
                accuracy=0.65,
                cv_std=0.02,
                baseline=0.5,
                above_chance=0.15,
                n_samples=100,
            ),
        }

        battery.print_results_table(results)
        captured = capsys.readouterr()

        # The layers will be [0, 1, 2] (union of all probe layers)
        # probe1 is missing layer 1, so it should show a dash
        lines = captured.out.split("\n")
        # Find the probe1 row
        probe1_row = None
        for line in lines:
            if line.startswith("probe1"):
                probe1_row = line
                break

        assert probe1_row is not None
        # Should contain a dash for the missing layer 1
        assert "-" in probe1_row


class TestProbeBatteryWithoutSklearn:
    """Tests for ProbeBattery that don't require sklearn."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = Mock()
        model.model = Mock()
        model.model.layers = [Mock() for _ in range(4)]
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=np.array([[1, 2, 3]]))
        return tokenizer

    def test_probe_battery_init_with_model_id(self, mock_model, mock_tokenizer):
        """Test initialization with custom model ID."""
        battery = ProbeBattery(mock_model, mock_tokenizer, model_id="custom-model")
        assert battery.model_id == "custom-model"
        assert battery.num_layers == 4
        assert len(battery.datasets) == 0

    def test_load_datasets_single_file_json_in_dict(self, mock_model, mock_tokenizer):
        """Test loading a JSON file with single dataset structure."""
        battery = ProbeBattery(mock_model, mock_tokenizer)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "single.json"
            data = {
                "description": "Single dataset",
                "prompts": ["p1", "p2"],
                "labels": [0, 1],
                "label_names": ["a", "b"],
                "category": "test",
            }
            with open(path, "w") as f:
                json.dump(data, f)

            battery.load_datasets(path)
            assert "single" in battery.datasets
            assert battery.datasets["single"].description == "Single dataset"
            assert battery.datasets["single"].category == "test"

    def test_load_datasets_multiple_in_file(self, mock_model, mock_tokenizer):
        """Test loading multiple datasets from one JSON file."""
        battery = ProbeBattery(mock_model, mock_tokenizer)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "multiple.json"
            data = {
                "dataset1": {
                    "description": "First",
                    "prompts": ["p1"],
                    "labels": [0],
                },
                "dataset2": {
                    "description": "Second",
                    "prompts": ["p2"],
                    "labels": [1],
                },
            }
            with open(path, "w") as f:
                json.dump(data, f)

            battery.load_datasets(path)
            assert "dataset1" in battery.datasets
            assert "dataset2" in battery.datasets
            assert battery.datasets["dataset1"].description == "First"
            assert battery.datasets["dataset2"].description == "Second"

    def test_print_results_table_empty_results(self, mock_model, mock_tokenizer, capsys):
        """Test printing results table with no probes."""
        battery = ProbeBattery(mock_model, mock_tokenizer)
        results = StratigraphyResult(model_id="test", num_layers=4)

        battery.print_results_table(results)
        captured = capsys.readouterr()
        assert "PROBE ACCURACY BY LAYER" in captured.out

    def test_print_stratigraphy_probe_not_in_datasets(self, mock_model, mock_tokenizer, capsys):
        """Test printing stratigraphy when probe is in results but not in battery datasets."""
        battery = ProbeBattery(mock_model, mock_tokenizer)
        results = StratigraphyResult(model_id="test-model", num_layers=4)
        # Add a probe to results that isn't in battery.datasets
        results.probes["unknown_probe"] = {
            0: ProbeResult(
                probe_name="unknown_probe",
                layer=0,
                accuracy=0.8,
                cv_std=0.02,
                baseline=0.5,
                above_chance=0.3,
                n_samples=100,
            )
        }

        battery.print_stratigraphy(results, threshold=0.75)
        captured = capsys.readouterr()
        # Should handle missing dataset gracefully
        assert "unknown_probe" in captured.out or "OTHER" in captured.out.upper()

    def test_stratigraphy_result_empty_probes(self):
        """Test StratigraphyResult with no probes."""
        result = StratigraphyResult(model_id="test", num_layers=5)
        matrix = result.get_accuracy_matrix()
        assert matrix == {}

        all_emergence = result.get_all_emergence_layers()
        assert all_emergence == {}

    def test_stratigraphy_result_get_accuracy_matrix_empty_layers(self):
        """Test getting accuracy matrix with empty layer list."""
        result = StratigraphyResult(model_id="test", num_layers=10)
        result.probes["probe1"] = {
            0: ProbeResult(
                probe_name="probe1",
                layer=0,
                accuracy=0.5,
                cv_std=0.02,
                baseline=0.5,
                above_chance=0.0,
                n_samples=100,
            ),
        }
        matrix = result.get_accuracy_matrix(layers=[])
        assert matrix["probe1"] == []

    def test_probe_dataset_to_dict_complete(self):
        """Test to_dict includes all fields."""
        dataset = ProbeDataset(
            name="test",
            description="Test description",
            prompts=["p1", "p2"],
            labels=[0, 1],
            label_names=["negative", "positive"],
            category="test_category",
        )
        data = dataset.to_dict()
        assert "description" in data
        assert "category" in data
        assert "label_names" in data
        assert "prompts" in data
        assert "labels" in data
        assert data["category"] == "test_category"
        assert data["label_names"] == ["negative", "positive"]

    def test_stratigraphy_result_save_with_path_object(self):
        """Test saving StratigraphyResult using Path object."""
        result = StratigraphyResult(model_id="test", num_layers=5)
        result.probes["test"] = {
            0: ProbeResult(
                probe_name="test",
                layer=0,
                accuracy=0.8,
                cv_std=0.02,
                baseline=0.5,
                above_chance=0.3,
                n_samples=100,
            )
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "result.json"
            result.save(path)
            assert path.exists()

            # Verify content
            with open(path) as f:
                data = json.load(f)
            assert data["model_id"] == "test"

    def test_stratigraphy_result_load_with_path_object(self):
        """Test loading StratigraphyResult using Path object."""
        result = StratigraphyResult(model_id="test", num_layers=5)
        result.probes["test"] = {
            0: ProbeResult(
                probe_name="test",
                layer=0,
                accuracy=0.8,
                cv_std=0.02,
                baseline=0.5,
                above_chance=0.3,
                n_samples=100,
            )
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "result.json"
            result.save(path)

            loaded = StratigraphyResult.load(path)
            assert loaded.model_id == "test"
            assert loaded.num_layers == 5

    def test_probe_dataset_num_classes_single_class(self):
        """Test num_classes with only one class."""
        dataset = ProbeDataset(name="test", description="", prompts=["p1", "p2"], labels=[0, 0])
        assert dataset.num_classes == 1

    def test_find_emergence_layer_exact_threshold(self):
        """Test finding emergence layer when accuracy exactly matches threshold."""
        result = StratigraphyResult(model_id="test", num_layers=10)
        result.probes["test_probe"] = {
            0: ProbeResult(
                probe_name="test",
                layer=0,
                accuracy=0.75,
                cv_std=0.02,
                baseline=0.5,
                above_chance=0.25,
                n_samples=100,
            ),
        }
        # Threshold is 0.75, accuracy is 0.75, above_chance is 0.25 (> 0.1)
        emergence = result.find_emergence_layer("test_probe", threshold=0.75)
        assert emergence == 0

    def test_find_destruction_layer_exact_threshold(self):
        """Test finding destruction layer when accuracy exactly matches threshold."""
        result = StratigraphyResult(model_id="test", num_layers=10)
        result.probes["test_probe"] = {
            0: ProbeResult(
                probe_name="test",
                layer=0,
                accuracy=0.8,
                cv_std=0.02,
                baseline=0.5,
                above_chance=0.3,
                n_samples=100,
            ),
            2: ProbeResult(
                probe_name="test",
                layer=2,
                accuracy=0.5,
                cv_std=0.02,
                baseline=0.5,
                above_chance=0.0,
                n_samples=100,
            ),  # Exactly at threshold
        }
        # Should not count as destruction since it's not < threshold
        destruction = result.find_destruction_layer("test_probe", threshold=0.5)
        assert destruction is None

    def test_find_destruction_layer_below_threshold(self):
        """Test finding destruction layer when it goes below threshold."""
        result = StratigraphyResult(model_id="test", num_layers=10)
        result.probes["test_probe"] = {
            0: ProbeResult(
                probe_name="test",
                layer=0,
                accuracy=0.8,
                cv_std=0.02,
                baseline=0.5,
                above_chance=0.3,
                n_samples=100,
            ),
            2: ProbeResult(
                probe_name="test",
                layer=2,
                accuracy=0.49,
                cv_std=0.02,
                baseline=0.5,
                above_chance=-0.01,
                n_samples=100,
            ),  # Below threshold
        }
        destruction = result.find_destruction_layer("test_probe", threshold=0.5)
        assert destruction == 2

    def test_probe_result_repr_or_str(self):
        """Test that ProbeResult can be represented as string."""
        result = ProbeResult(
            probe_name="test",
            layer=5,
            accuracy=0.85,
            cv_std=0.02,
            baseline=0.5,
            above_chance=0.35,
            n_samples=100,
        )
        # Should not raise an error
        str_repr = str(result)
        assert "test" in str_repr or "ProbeResult" in str_repr or str_repr is not None
