"""Tests for experiments base module."""

import tempfile
from pathlib import Path

import pytest

from chuk_lazarus.experiments.base import (
    ExperimentBase,
    ExperimentConfig,
    ExperimentResult,
)


class TestExperimentConfig:
    """Tests for ExperimentConfig."""

    def test_config_minimal(self):
        """Test minimal config."""
        config = ExperimentConfig(name="test_exp", description="Test")
        assert config.name == "test_exp"
        assert config.description == "Test"
        assert config.parameters == {}

    def test_config_full(self):
        """Test full config."""
        config = ExperimentConfig(
            name="full_exp",
            description="A test experiment",
            parameters={"lr": 1e-4, "epochs": 5},
            model="test-model",
            training={"batch_size": 8},
        )
        assert config.name == "full_exp"
        assert config.description == "A test experiment"
        assert config.parameters["lr"] == 1e-4
        assert config.parameters["epochs"] == 5
        assert config.model == "test-model"
        assert config.training["batch_size"] == 8

    def test_config_to_dict(self):
        """Test converting config to dict."""
        config = ExperimentConfig(
            name="test",
            description="Test",
            parameters={"key": "value"},
        )
        d = config.to_dict()
        assert d["name"] == "test"
        assert d["description"] == "Test"
        assert d["parameters"] == {"key": "value"}

    def test_config_from_yaml(self):
        """Test loading config from YAML."""
        import yaml

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(
                {
                    "name": "yaml_test",
                    "description": "From YAML",
                    "model": "test-model",
                    "custom_param": 42,  # Extra field goes to parameters
                },
                f,
            )
            f.flush()

            config = ExperimentConfig.from_yaml(Path(f.name))

        assert config.name == "yaml_test"
        assert config.description == "From YAML"
        assert config.model == "test-model"
        assert config.parameters["custom_param"] == 42

        Path(f.name).unlink()


class TestExperimentResult:
    """Tests for ExperimentResult."""

    def test_result_basic(self):
        """Test basic result."""
        result = ExperimentResult(
            experiment_name="test",
            status="success",
            started_at="2024-01-01T00:00:00",
            finished_at="2024-01-01T01:00:00",
            duration_seconds=3600.0,
            run_results={"output": "value"},
            eval_results={"accuracy": 0.95},
            config={"name": "test"},
        )
        assert result.experiment_name == "test"
        assert result.status == "success"
        assert result.error is None

    def test_result_with_error(self):
        """Test result with error."""
        result = ExperimentResult(
            experiment_name="test",
            status="failed",
            started_at="2024-01-01T00:00:00",
            finished_at="2024-01-01T00:01:00",
            duration_seconds=60.0,
            run_results={},
            eval_results={},
            config={},
            error="Something went wrong",
        )
        assert result.status == "failed"
        assert result.error == "Something went wrong"

    def test_result_to_dict(self):
        """Test converting result to dict."""
        result = ExperimentResult(
            experiment_name="test",
            status="success",
            started_at="2024-01-01T00:00:00",
            finished_at="2024-01-01T01:00:00",
            duration_seconds=3600.0,
            run_results={"key": "value"},
            eval_results={"accuracy": 0.9},
            config={"name": "test"},
        )
        d = result.to_dict()
        assert d["experiment_name"] == "test"
        assert d["run_results"]["key"] == "value"

    def test_result_from_dict(self):
        """Test creating result from dict."""
        data = {
            "experiment_name": "test",
            "status": "success",
            "started_at": "2024-01-01T00:00:00",
            "finished_at": "2024-01-01T01:00:00",
            "duration_seconds": 3600.0,
            "run_results": {},
            "eval_results": {},
            "config": {},
        }
        result = ExperimentResult.from_dict(data)
        assert result.experiment_name == "test"
        assert result.status == "success"


class ConcreteExperiment(ExperimentBase):
    """Concrete implementation for testing."""

    def setup(self):
        """Set up the experiment."""
        pass

    def run(self):
        """Run the experiment."""
        return {"output": "test_output"}

    def evaluate(self):
        """Evaluate the experiment."""
        return {"accuracy": 0.95}


class TestExperimentBase:
    """Tests for ExperimentBase."""

    def test_experiment_init(self):
        """Test experiment initialization."""
        config = ExperimentConfig(
            name="test",
            description="Test experiment",
            parameters={"test_param": 1},
        )
        exp = ConcreteExperiment(config)
        assert exp.config == config

    def test_experiment_init_with_dir(self):
        """Test experiment initialization with directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExperimentConfig(
                name="test",
                description="Test experiment",
                experiment_dir=Path(tmpdir),
            )
            exp = ConcreteExperiment(config)

            # Directories should be set up
            assert exp.config.data_dir is not None
            assert exp.config.checkpoint_dir is not None
            assert exp.config.results_dir is not None

    def test_experiment_run(self):
        """Test running an experiment."""
        config = ExperimentConfig(name="test", description="Test")
        exp = ConcreteExperiment(config)
        result = exp.run()
        assert result == {"output": "test_output"}

    def test_experiment_evaluate(self):
        """Test evaluating an experiment."""
        config = ExperimentConfig(name="test", description="Test")
        exp = ConcreteExperiment(config)
        eval_result = exp.evaluate()
        assert eval_result == {"accuracy": 0.95}

    def test_experiment_cleanup(self):
        """Test cleanup (should be no-op by default)."""
        config = ExperimentConfig(name="test", description="Test")
        exp = ConcreteExperiment(config)
        # Should not raise
        exp.cleanup()

    def test_experiment_log(self):
        """Test logging."""
        config = ExperimentConfig(name="test", description="Test")
        exp = ConcreteExperiment(config)
        # Should not raise
        exp.log("Test message")
        exp.log("Debug message", level="debug")
        exp.log("Warning message", level="warning")

    def test_get_parameter(self):
        """Test getting parameters."""
        config = ExperimentConfig(
            name="test",
            description="Test",
            parameters={
                "simple": 1,
                "nested": {"key": "value", "deep": {"level": 3}},
            },
        )
        exp = ConcreteExperiment(config)

        assert exp.get_parameter("simple") == 1
        assert exp.get_parameter("nested.key") == "value"
        assert exp.get_parameter("nested.deep.level") == 3
        assert exp.get_parameter("nonexistent") is None
        assert exp.get_parameter("nonexistent", "default") == "default"

    def test_save_results(self):
        """Test saving results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExperimentConfig(
                name="test",
                description="Test",
                experiment_dir=Path(tmpdir),
            )
            exp = ConcreteExperiment(config)

            path = exp.save_results({"accuracy": 0.95})
            assert path.exists()

            import json

            with open(path) as f:
                data = json.load(f)
            assert data["accuracy"] == 0.95

    def test_save_results_no_dir(self):
        """Test saving results without results_dir."""
        config = ExperimentConfig(name="test", description="Test")
        exp = ConcreteExperiment(config)

        with pytest.raises(ValueError, match="results_dir not set"):
            exp.save_results({"key": "value"})

    def test_load_latest_results(self):
        """Test loading latest results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExperimentConfig(
                name="test",
                description="Test",
                experiment_dir=Path(tmpdir),
            )
            exp = ConcreteExperiment(config)

            # Save some results
            exp.save_results({"version": 1}, name="results")
            import time

            time.sleep(0.1)
            exp.save_results({"version": 2}, name="results")

            # Load latest
            latest = exp.load_latest_results()
            assert latest["version"] == 2

    def test_load_latest_results_none(self):
        """Test loading latest results when none exist."""
        config = ExperimentConfig(name="test", description="Test")
        exp = ConcreteExperiment(config)
        result = exp.load_latest_results()
        assert result is None
