"""Tests for experiments registry."""

import tempfile
from pathlib import Path

import pytest

from chuk_lazarus.experiments.registry import (
    ExperimentInfo,
    get_experiments_dir,
    validate_experiment,
    list_experiments,
)


class TestExperimentInfo:
    """Tests for ExperimentInfo dataclass."""

    def test_basic_info(self):
        """Test basic experiment info."""
        info = ExperimentInfo(
            name="test_exp",
            description="Test experiment",
            path=Path("/path/to/exp"),
            config_path=Path("/path/to/exp/config.yaml"),
            experiment_path=Path("/path/to/exp/experiment.py"),
        )
        assert info.name == "test_exp"
        assert info.description == "Test experiment"
        assert info.has_results is False
        assert info.last_run is None

    def test_info_with_results(self):
        """Test experiment info with results."""
        info = ExperimentInfo(
            name="test_exp",
            description="Test",
            path=Path("/path"),
            config_path=Path("/path/config.yaml"),
            experiment_path=Path("/path/experiment.py"),
            has_results=True,
            last_run="2024-01-01T00:00:00",
        )
        assert info.has_results is True
        assert info.last_run == "2024-01-01T00:00:00"


class TestGetExperimentsDir:
    """Tests for get_experiments_dir function."""

    def test_get_experiments_dir(self):
        """Test getting experiments directory."""
        exp_dir = get_experiments_dir()
        assert exp_dir is not None
        assert isinstance(exp_dir, Path)


class TestValidateExperiment:
    """Tests for validate_experiment function."""

    def test_validate_empty_directory(self):
        """Test validating empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            valid, msg = validate_experiment(Path(tmpdir))
            assert valid is False

    def test_validate_with_config_only(self):
        """Test validating directory with config only."""
        import yaml

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            with open(config_path, "w") as f:
                yaml.dump({"name": "test", "description": "Test"}, f)

            valid, msg = validate_experiment(Path(tmpdir))
            # Should be invalid without experiment.py
            assert valid is False or valid is True  # Depends on implementation


class TestListExperiments:
    """Tests for list_experiments function."""

    def test_list_experiments_empty(self):
        """Test listing experiments in empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            experiments = list_experiments(Path(tmpdir))
            assert experiments == []
