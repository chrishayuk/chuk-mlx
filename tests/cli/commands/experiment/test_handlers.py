"""Tests for experiment CLI handlers."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

from chuk_lazarus.cli.commands.experiment.handlers import (
    experiment_list,
    experiment_info,
    experiment_run,
    experiment_status,
)
from chuk_lazarus.experiments.registry import ExperimentInfo
from chuk_lazarus.experiments.base import ExperimentResult


@pytest.fixture
def mock_experiment_info():
    """Create a mock ExperimentInfo."""
    return ExperimentInfo(
        name="test_experiment",
        description="A test experiment for unit testing",
        path=Path("/fake/experiments/test_experiment"),
        config_path=Path("/fake/experiments/test_experiment/config.yaml"),
        experiment_path=Path("/fake/experiments/test_experiment/experiment.py"),
        has_results=False,
    )


@pytest.fixture
def mock_experiment_info_with_results():
    """Create a mock ExperimentInfo with results."""
    return ExperimentInfo(
        name="test_experiment",
        description="A test experiment with results",
        path=Path("/fake/experiments/test_experiment"),
        config_path=Path("/fake/experiments/test_experiment/config.yaml"),
        experiment_path=Path("/fake/experiments/test_experiment/experiment.py"),
        has_results=True,
    )


@pytest.fixture
def mock_experiment_result():
    """Create a mock ExperimentResult."""
    return ExperimentResult(
        experiment_name="test_experiment",
        status="success",
        started_at="2024-01-01T10:00:00",
        finished_at="2024-01-01T10:05:00",
        duration_seconds=300.0,
        run_results={"training_loss": 0.5},
        eval_results={"accuracy": 0.95, "f1_score": 0.92},
        config={"name": "test_experiment", "model": "test-model"},
        error=None,
    )


@pytest.fixture
def mock_experiment_result_with_error():
    """Create a mock ExperimentResult with an error."""
    return ExperimentResult(
        experiment_name="test_experiment",
        status="failed",
        started_at="2024-01-01T10:00:00",
        finished_at="2024-01-01T10:01:00",
        duration_seconds=60.0,
        run_results={},
        eval_results={},
        config={"name": "test_experiment"},
        error="Model loading failed: OOM",
    )


class TestExperimentList:
    """Tests for experiment_list handler."""

    def test_list_no_experiments(self, capsys):
        """Test listing when no experiments found."""
        with patch(
            "chuk_lazarus.cli.commands.experiment.handlers._list_experiments",
            return_value=[],
        ):
            experiment_list(experiments_dir="/fake/experiments")

        captured = capsys.readouterr()
        assert "No experiments found" in captured.out

    def test_list_experiments_table_output(self, capsys, mock_experiment_info):
        """Test listing experiments in table format."""
        experiments = [mock_experiment_info]
        with patch(
            "chuk_lazarus.cli.commands.experiment.handlers._list_experiments",
            return_value=experiments,
        ):
            experiment_list(experiments_dir="/fake/experiments")

        captured = capsys.readouterr()
        assert "Available Experiments" in captured.out
        assert "test_experiment" in captured.out
        assert "no runs" in captured.out
        assert "Total: 1 experiments" in captured.out

    def test_list_experiments_with_results(self, capsys, mock_experiment_info_with_results):
        """Test listing experiments with results shows status."""
        experiments = [mock_experiment_info_with_results]
        with patch(
            "chuk_lazarus.cli.commands.experiment.handlers._list_experiments",
            return_value=experiments,
        ):
            experiment_list(experiments_dir="/fake/experiments")

        captured = capsys.readouterr()
        assert "has results" in captured.out

    def test_list_experiments_json_output(self, capsys, mock_experiment_info):
        """Test listing experiments in JSON format."""
        experiments = [mock_experiment_info]
        with patch(
            "chuk_lazarus.cli.commands.experiment.handlers._list_experiments",
            return_value=experiments,
        ):
            experiment_list(experiments_dir="/fake/experiments", json_output=True)

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert len(data) == 1
        assert data[0]["name"] == "test_experiment"
        assert data[0]["description"] == "A test experiment for unit testing"
        assert data[0]["has_results"] is False

    def test_list_experiments_long_description_truncated(self, capsys):
        """Test that long descriptions are truncated in table output."""
        info = ExperimentInfo(
            name="test_experiment",
            description="A" * 50,  # Very long description
            path=Path("/fake/experiments/test_experiment"),
            config_path=Path("/fake/experiments/test_experiment/config.yaml"),
            experiment_path=Path("/fake/experiments/test_experiment/experiment.py"),
            has_results=False,
        )
        with patch(
            "chuk_lazarus.cli.commands.experiment.handlers._list_experiments",
            return_value=[info],
        ):
            experiment_list(experiments_dir="/fake/experiments")

        captured = capsys.readouterr()
        # Description should be truncated to 35 chars + "..."
        assert "..." in captured.out

    def test_list_experiments_default_dir(self, capsys, mock_experiment_info):
        """Test listing with default experiments directory."""
        experiments = [mock_experiment_info]
        with patch(
            "chuk_lazarus.cli.commands.experiment.handlers._list_experiments",
            return_value=experiments,
        ):
            experiment_list()  # No experiments_dir

        captured = capsys.readouterr()
        assert "test_experiment" in captured.out


class TestExperimentInfo:
    """Tests for experiment_info handler."""

    def test_info_not_found(self, capsys):
        """Test info for non-existent experiment."""
        with patch(
            "chuk_lazarus.cli.commands.experiment.handlers.get_experiment_info",
            side_effect=ValueError("Experiment not found: missing"),
        ):
            experiment_info("missing", experiments_dir="/fake/experiments")

        captured = capsys.readouterr()
        assert "Error:" in captured.out
        assert "Experiment not found" in captured.out

    def test_info_table_output(self, capsys, mock_experiment_info):
        """Test info in table format."""
        config = {
            "name": "test_experiment",
            "description": "A test experiment",
            "model": "test-model",
            "training": {"epochs": 10, "batch_size": 32},
            "parameters": {"learning_rate": 0.001},
        }
        yaml_content = "name: test_experiment\ndescription: A test experiment"

        with patch(
            "chuk_lazarus.cli.commands.experiment.handlers.get_experiment_info",
            return_value=mock_experiment_info,
        ):
            with patch("builtins.open", mock_open(read_data=yaml_content)):
                with patch("yaml.safe_load", return_value=config):
                    experiment_info("test_experiment")

        captured = capsys.readouterr()
        assert "Experiment: test_experiment" in captured.out
        assert "Configuration:" in captured.out
        assert "model: test-model" in captured.out

    def test_info_with_dict_config(self, capsys, mock_experiment_info):
        """Test info with nested dict in config."""
        config = {
            "name": "test_experiment",
            "description": "A test experiment",
            "model": "test-model",
            "training": {"epochs": 10, "batch_size": 32},
        }

        with patch(
            "chuk_lazarus.cli.commands.experiment.handlers.get_experiment_info",
            return_value=mock_experiment_info,
        ):
            with patch("builtins.open", mock_open()):
                with patch("yaml.safe_load", return_value=config):
                    experiment_info("test_experiment")

        captured = capsys.readouterr()
        assert "training:" in captured.out
        assert "epochs: 10" in captured.out
        assert "batch_size: 32" in captured.out

    def test_info_with_list_config(self, capsys, mock_experiment_info):
        """Test info with list values in config."""
        config = {
            "name": "test_experiment",
            "description": "A test experiment",
            "layers": [0, 4, 8, 12],
        }

        with patch(
            "chuk_lazarus.cli.commands.experiment.handlers.get_experiment_info",
            return_value=mock_experiment_info,
        ):
            with patch("builtins.open", mock_open()):
                with patch("yaml.safe_load", return_value=config):
                    experiment_info("test_experiment")

        captured = capsys.readouterr()
        assert "layers: 0, 4, 8, 12" in captured.out

    def test_info_with_scalar_config(self, capsys, mock_experiment_info):
        """Test info with scalar values in config."""
        config = {
            "name": "test_experiment",
            "description": "A test experiment",
            "max_tokens": 512,
        }

        with patch(
            "chuk_lazarus.cli.commands.experiment.handlers.get_experiment_info",
            return_value=mock_experiment_info,
        ):
            with patch("builtins.open", mock_open()):
                with patch("yaml.safe_load", return_value=config):
                    experiment_info("test_experiment")

        captured = capsys.readouterr()
        assert "max_tokens: 512" in captured.out

    def test_info_json_output(self, capsys, mock_experiment_info):
        """Test info in JSON format."""
        config = {
            "name": "test_experiment",
            "description": "A test experiment",
            "model": "test-model",
        }

        with patch(
            "chuk_lazarus.cli.commands.experiment.handlers.get_experiment_info",
            return_value=mock_experiment_info,
        ):
            with patch("builtins.open", mock_open()):
                with patch("yaml.safe_load", return_value=config):
                    experiment_info("test_experiment", json_output=True)

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["name"] == "test_experiment"
        assert data["config"]["model"] == "test-model"
        assert data["has_results"] is False

    def test_info_with_recent_runs(self, capsys, mock_experiment_info_with_results):
        """Test info shows recent runs when results exist."""
        config = {
            "name": "test_experiment",
            "description": "A test experiment",
        }
        runs = [
            {
                "started_at": "2024-01-01T10:00:00.000",
                "duration_seconds": 300.0,
                "status": "success",
            }
        ]

        with patch(
            "chuk_lazarus.cli.commands.experiment.handlers.get_experiment_info",
            return_value=mock_experiment_info_with_results,
        ):
            with patch("builtins.open", mock_open()):
                with patch("yaml.safe_load", return_value=config):
                    with patch(
                        "chuk_lazarus.cli.commands.experiment.handlers.list_experiment_runs",
                        return_value=runs,
                    ):
                        experiment_info("test_experiment")

        captured = capsys.readouterr()
        assert "Recent Runs:" in captured.out
        assert "success" in captured.out

    def test_info_default_dir(self, capsys, mock_experiment_info):
        """Test info with default experiments directory."""
        config = {"name": "test_experiment", "description": "A test"}

        with patch(
            "chuk_lazarus.cli.commands.experiment.handlers.get_experiment_info",
            return_value=mock_experiment_info,
        ):
            with patch("builtins.open", mock_open()):
                with patch("yaml.safe_load", return_value=config):
                    experiment_info("test_experiment")  # No experiments_dir

        captured = capsys.readouterr()
        assert "test_experiment" in captured.out


class TestExperimentRun:
    """Tests for experiment_run handler."""

    def test_run_invalid_param_format(self, capsys):
        """Test run with invalid parameter format."""
        experiment_run(
            "test_experiment",
            params=["invalid_no_equals"],
        )

        captured = capsys.readouterr()
        assert "Invalid parameter format" in captured.out
        assert "expected key=value" in captured.out

    def test_run_success(self, capsys, mock_experiment_result):
        """Test successful experiment run."""
        with patch(
            "chuk_lazarus.cli.commands.experiment.handlers._run_experiment",
            return_value=mock_experiment_result,
        ):
            experiment_run("test_experiment")

        captured = capsys.readouterr()
        assert "Running experiment: test_experiment" in captured.out
        assert "Results:" in captured.out
        assert "Status: success" in captured.out
        assert "Duration:" in captured.out

    def test_run_with_eval_results_float(self, capsys, mock_experiment_result):
        """Test run with float eval results."""
        with patch(
            "chuk_lazarus.cli.commands.experiment.handlers._run_experiment",
            return_value=mock_experiment_result,
        ):
            experiment_run("test_experiment")

        captured = capsys.readouterr()
        assert "Evaluation Metrics:" in captured.out
        assert "accuracy: 0.9500" in captured.out
        assert "f1_score: 0.9200" in captured.out

    def test_run_with_eval_results_non_float(self, capsys):
        """Test run with non-float eval results."""
        result = ExperimentResult(
            experiment_name="test_experiment",
            status="success",
            started_at="2024-01-01T10:00:00",
            finished_at="2024-01-01T10:05:00",
            duration_seconds=300.0,
            run_results={},
            eval_results={"model_name": "test-model", "epoch": 10},
            config={},
            error=None,
        )
        with patch(
            "chuk_lazarus.cli.commands.experiment.handlers._run_experiment",
            return_value=result,
        ):
            experiment_run("test_experiment")

        captured = capsys.readouterr()
        assert "model_name: test-model" in captured.out
        assert "epoch: 10" in captured.out

    def test_run_with_error(self, capsys, mock_experiment_result_with_error):
        """Test run that produces an error in result."""
        with patch(
            "chuk_lazarus.cli.commands.experiment.handlers._run_experiment",
            return_value=mock_experiment_result_with_error,
        ):
            experiment_run("test_experiment")

        captured = capsys.readouterr()
        assert "Error: Model loading failed" in captured.out

    def test_run_dry_run(self, capsys, mock_experiment_result):
        """Test dry run mode."""
        with patch(
            "chuk_lazarus.cli.commands.experiment.handlers._run_experiment",
            return_value=mock_experiment_result,
        ):
            experiment_run("test_experiment", dry_run=True)

        captured = capsys.readouterr()
        assert "(dry run mode)" in captured.out

    def test_run_with_param_overrides(self, capsys, mock_experiment_result):
        """Test run with parameter overrides."""
        with patch(
            "chuk_lazarus.cli.commands.experiment.handlers._run_experiment",
            return_value=mock_experiment_result,
        ) as mock_run:
            experiment_run(
                "test_experiment",
                params=["learning_rate=0.001", "epochs=10"],
            )

            # Check the overrides were passed
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["config_overrides"]["learning_rate"] == 0.001
            assert call_kwargs["config_overrides"]["epochs"] == 10

    def test_run_with_json_param_value(self, capsys, mock_experiment_result):
        """Test run with JSON parameter value."""
        with patch(
            "chuk_lazarus.cli.commands.experiment.handlers._run_experiment",
            return_value=mock_experiment_result,
        ) as mock_run:
            experiment_run(
                "test_experiment",
                params=['layers=[0,4,8]'],
            )

            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["config_overrides"]["layers"] == [0, 4, 8]

    def test_run_with_string_param_value(self, capsys, mock_experiment_result):
        """Test run with plain string parameter value (non-JSON)."""
        with patch(
            "chuk_lazarus.cli.commands.experiment.handlers._run_experiment",
            return_value=mock_experiment_result,
        ) as mock_run:
            experiment_run(
                "test_experiment",
                params=['model_name=my-custom-model', 'description=A test run'],
            )

            call_kwargs = mock_run.call_args[1]
            # String values should be kept as strings (not parsed as JSON)
            assert call_kwargs["config_overrides"]["model_name"] == "my-custom-model"
            assert call_kwargs["config_overrides"]["description"] == "A test run"

    def test_run_with_config_file(self, capsys, mock_experiment_result):
        """Test run with config file override."""
        custom_config = {"learning_rate": 0.01, "batch_size": 64}
        yaml_content = "learning_rate: 0.01\nbatch_size: 64"

        with patch(
            "chuk_lazarus.cli.commands.experiment.handlers._run_experiment",
            return_value=mock_experiment_result,
        ) as mock_run:
            with patch("builtins.open", mock_open(read_data=yaml_content)):
                with patch("yaml.safe_load", return_value=custom_config):
                    experiment_run(
                        "test_experiment",
                        config_file="/path/to/override.yaml",
                    )

            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["config_overrides"]["learning_rate"] == 0.01
            assert call_kwargs["config_overrides"]["batch_size"] == 64

    def test_run_exception(self, capsys):
        """Test run that raises exception."""
        with patch(
            "chuk_lazarus.cli.commands.experiment.handlers._run_experiment",
            side_effect=RuntimeError("Experiment crashed"),
        ):
            experiment_run("test_experiment")

        captured = capsys.readouterr()
        assert "Error running experiment" in captured.out

    def test_run_with_experiments_dir(self, capsys, mock_experiment_result):
        """Test run with custom experiments directory."""
        with patch(
            "chuk_lazarus.cli.commands.experiment.handlers._run_experiment",
            return_value=mock_experiment_result,
        ) as mock_run:
            experiment_run("test_experiment", experiments_dir="/custom/experiments")

            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["experiments_dir"] == Path("/custom/experiments")

    def test_run_no_eval_results(self, capsys):
        """Test run with empty eval results."""
        result = ExperimentResult(
            experiment_name="test_experiment",
            status="success",
            started_at="2024-01-01T10:00:00",
            finished_at="2024-01-01T10:05:00",
            duration_seconds=300.0,
            run_results={},
            eval_results={},  # Empty
            config={},
            error=None,
        )
        with patch(
            "chuk_lazarus.cli.commands.experiment.handlers._run_experiment",
            return_value=result,
        ):
            experiment_run("test_experiment")

        captured = capsys.readouterr()
        assert "Evaluation Metrics:" not in captured.out


class TestExperimentStatus:
    """Tests for experiment_status handler."""

    def test_status_not_found(self, capsys):
        """Test status for non-existent experiment."""
        with patch(
            "chuk_lazarus.cli.commands.experiment.handlers.get_experiment_status",
            side_effect=ValueError("Experiment not found: missing"),
        ):
            experiment_status("missing")

        captured = capsys.readouterr()
        assert "Error:" in captured.out
        assert "Experiment not found" in captured.out

    def test_status_no_results(self, capsys):
        """Test status for experiment without results."""
        status = {
            "name": "test_experiment",
            "description": "A test experiment",
            "path": "/fake/experiments/test_experiment",
            "has_results": False,
            "latest_result": None,
        }

        with patch(
            "chuk_lazarus.cli.commands.experiment.handlers.get_experiment_status",
            return_value=status,
        ):
            experiment_status("test_experiment")

        captured = capsys.readouterr()
        assert "Experiment Status: test_experiment" in captured.out
        assert "Has Results: No" in captured.out

    def test_status_with_results(self, capsys):
        """Test status with latest results."""
        status = {
            "name": "test_experiment",
            "description": "A test experiment",
            "path": "/fake/experiments/test_experiment",
            "has_results": True,
            "latest_result": {
                "status": "success",
                "started_at": "2024-01-01T10:00:00.000",
                "duration_seconds": 300.0,
                "eval_results": {"accuracy": 0.95},
            },
        }

        with patch(
            "chuk_lazarus.cli.commands.experiment.handlers.get_experiment_status",
            return_value=status,
        ):
            experiment_status("test_experiment")

        captured = capsys.readouterr()
        assert "Has Results: Yes" in captured.out
        assert "Latest Run:" in captured.out
        assert "Status: success" in captured.out
        assert "Duration: 300.00s" in captured.out
        assert "Metrics:" in captured.out
        assert "accuracy: 0.9500" in captured.out

    def test_status_with_non_float_metrics(self, capsys):
        """Test status with non-float metric values."""
        status = {
            "name": "test_experiment",
            "description": "A test experiment",
            "path": "/fake/experiments/test_experiment",
            "has_results": True,
            "latest_result": {
                "status": "success",
                "started_at": "2024-01-01T10:00:00.000",
                "duration_seconds": 300.0,
                "eval_results": {"model": "test-model", "epochs": 10},
            },
        }

        with patch(
            "chuk_lazarus.cli.commands.experiment.handlers.get_experiment_status",
            return_value=status,
        ):
            experiment_status("test_experiment")

        captured = capsys.readouterr()
        assert "model: test-model" in captured.out
        assert "epochs: 10" in captured.out

    def test_status_with_error(self, capsys):
        """Test status showing error from latest run."""
        status = {
            "name": "test_experiment",
            "description": "A test experiment",
            "path": "/fake/experiments/test_experiment",
            "has_results": True,
            "latest_result": {
                "status": "failed",
                "started_at": "2024-01-01T10:00:00.000",
                "duration_seconds": 60.0,
                "eval_results": {},
                "error": "Out of memory",
            },
        }

        with patch(
            "chuk_lazarus.cli.commands.experiment.handlers.get_experiment_status",
            return_value=status,
        ):
            experiment_status("test_experiment")

        captured = capsys.readouterr()
        assert "Error: Out of memory" in captured.out

    def test_status_json_output(self, capsys):
        """Test status in JSON format."""
        status = {
            "name": "test_experiment",
            "description": "A test experiment",
            "path": "/fake/experiments/test_experiment",
            "has_results": True,
            "latest_result": {"status": "success"},
        }

        with patch(
            "chuk_lazarus.cli.commands.experiment.handlers.get_experiment_status",
            return_value=status,
        ):
            experiment_status("test_experiment", json_output=True)

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["name"] == "test_experiment"
        assert data["has_results"] is True

    def test_status_show_all_runs(self, capsys):
        """Test status with show_all flag."""
        status = {
            "name": "test_experiment",
            "description": "A test experiment",
            "path": "/fake/experiments/test_experiment",
            "has_results": True,
            "latest_result": {
                "status": "success",
                "started_at": "2024-01-01T12:00:00.000",
                "duration_seconds": 300.0,
                "eval_results": {},
            },
        }
        runs = [
            {
                "started_at": "2024-01-01T12:00:00.000",
                "duration_seconds": 300.0,
                "status": "success",
            },
            {
                "started_at": "2024-01-01T10:00:00.000",
                "duration_seconds": 250.0,
                "status": "failed",
            },
        ]

        with patch(
            "chuk_lazarus.cli.commands.experiment.handlers.get_experiment_status",
            return_value=status,
        ):
            with patch(
                "chuk_lazarus.cli.commands.experiment.handlers.list_experiment_runs",
                return_value=runs,
            ):
                experiment_status("test_experiment", show_all=True)

        captured = capsys.readouterr()
        assert "All Runs:" in captured.out
        assert "success" in captured.out
        assert "failed" in captured.out

    def test_status_show_all_single_run(self, capsys):
        """Test status with show_all but only one run exists."""
        status = {
            "name": "test_experiment",
            "description": "A test experiment",
            "path": "/fake/experiments/test_experiment",
            "has_results": True,
            "latest_result": {
                "status": "success",
                "started_at": "2024-01-01T12:00:00.000",
                "duration_seconds": 300.0,
                "eval_results": {},
            },
        }
        runs = [
            {
                "started_at": "2024-01-01T12:00:00.000",
                "duration_seconds": 300.0,
                "status": "success",
            },
        ]

        with patch(
            "chuk_lazarus.cli.commands.experiment.handlers.get_experiment_status",
            return_value=status,
        ):
            with patch(
                "chuk_lazarus.cli.commands.experiment.handlers.list_experiment_runs",
                return_value=runs,
            ):
                experiment_status("test_experiment", show_all=True)

        captured = capsys.readouterr()
        # Should not show "All Runs" section if only 1 run
        assert "All Runs:" not in captured.out

    def test_status_with_experiments_dir(self, capsys):
        """Test status with custom experiments directory."""
        status = {
            "name": "test_experiment",
            "description": "A test experiment",
            "path": "/custom/experiments/test_experiment",
            "has_results": False,
            "latest_result": None,
        }

        with patch(
            "chuk_lazarus.cli.commands.experiment.handlers.get_experiment_status",
            return_value=status,
        ) as mock_status:
            experiment_status("test_experiment", experiments_dir="/custom/experiments")

            mock_status.assert_called_once_with("test_experiment", Path("/custom/experiments"))

    def test_status_empty_eval_results(self, capsys):
        """Test status with empty eval_results dict."""
        status = {
            "name": "test_experiment",
            "description": "A test experiment",
            "path": "/fake/experiments/test_experiment",
            "has_results": True,
            "latest_result": {
                "status": "success",
                "started_at": "2024-01-01T12:00:00.000",
                "duration_seconds": 300.0,
                "eval_results": {},
            },
        }

        with patch(
            "chuk_lazarus.cli.commands.experiment.handlers.get_experiment_status",
            return_value=status,
        ):
            experiment_status("test_experiment")

        captured = capsys.readouterr()
        # Should not show "Metrics:" section if eval_results is empty
        assert "Metrics:" not in captured.out
