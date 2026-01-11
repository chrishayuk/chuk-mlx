"""
Experiment runner for executing experiments.

Handles:
- Loading experiment class and config
- Running setup/run/evaluate/cleanup lifecycle
- Saving results with timestamps
- Error handling and reporting
"""

import logging
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from .base import ExperimentConfig, ExperimentResult
from .registry import get_experiment, get_experiment_info, get_experiments_dir

logger = logging.getLogger(__name__)


def load_config(experiment_path: Path, overrides: dict | None = None) -> ExperimentConfig:
    """Load and merge experiment configuration.

    Args:
        experiment_path: Path to experiment directory
        overrides: Optional config overrides

    Returns:
        Merged ExperimentConfig
    """
    config_path = experiment_path / "config.yaml"

    if not config_path.exists():
        raise ValueError(f"Config not found: {config_path}")

    # Load base config
    config = ExperimentConfig.from_yaml(config_path)

    # Set paths
    config.experiment_dir = experiment_path
    config.data_dir = experiment_path / "data"
    config.checkpoint_dir = experiment_path / "checkpoints"
    config.results_dir = experiment_path / "results"

    # Apply overrides
    if overrides:
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            elif "." in key:
                # Handle nested keys like "parameters.learning_rate"
                parts = key.split(".", 1)
                if parts[0] == "parameters":
                    config.parameters[parts[1]] = value
                elif parts[0] == "training":
                    config.training[parts[1]] = value
            else:
                # Add to parameters
                config.parameters[key] = value

    return config


def run_experiment(
    name: str,
    experiments_dir: Path | None = None,
    config_overrides: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> ExperimentResult:
    """Run an experiment by name.

    Args:
        name: Experiment name (directory name)
        experiments_dir: Path to experiments directory
        config_overrides: Optional config parameter overrides
        dry_run: If True, only validate without running

    Returns:
        ExperimentResult with run details and metrics
    """
    exp_dir = experiments_dir or get_experiments_dir()
    exp_path = exp_dir / name

    logger.info(f"Running experiment: {name}")
    logger.info(f"Experiment path: {exp_path}")

    # Load config
    config = load_config(exp_path, config_overrides)
    logger.info(f"Loaded config: {config.name}")

    if dry_run:
        logger.info("Dry run - skipping execution")
        return ExperimentResult(
            experiment_name=name,
            status="dry_run",
            started_at=datetime.now().isoformat(),
            finished_at=datetime.now().isoformat(),
            duration_seconds=0,
            run_results={},
            eval_results={},
            config=config.to_dict(),
        )

    # Get experiment class
    experiment_class = get_experiment(name, exp_dir)
    logger.info(f"Loaded experiment class: {experiment_class.__name__}")

    # Create instance
    experiment = experiment_class(config)

    # Run lifecycle
    started_at = datetime.now()
    start_time = time.time()

    run_results = {}
    eval_results = {}
    error = None
    status = "success"

    try:
        # Setup
        logger.info("Running setup...")
        experiment.setup()

        # Run
        logger.info("Running experiment...")
        run_results = experiment.run() or {}

        # Evaluate
        logger.info("Running evaluation...")
        eval_results = experiment.evaluate() or {}

    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        logger.error(traceback.format_exc())
        error = str(e)
        status = "failed"

    finally:
        # Cleanup
        try:
            logger.info("Running cleanup...")
            experiment.cleanup()
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")

    finished_at = datetime.now()
    duration = time.time() - start_time

    # Create result
    result = ExperimentResult(
        experiment_name=name,
        status=status,
        started_at=started_at.isoformat(),
        finished_at=finished_at.isoformat(),
        duration_seconds=duration,
        run_results=run_results,
        eval_results=eval_results,
        config=config.to_dict(),
        error=error,
    )

    # Save result
    save_experiment_result(result, exp_path)

    logger.info(f"Experiment completed: {status}")
    logger.info(f"Duration: {duration:.2f}s")

    return result


def save_experiment_result(result: ExperimentResult, experiment_path: Path) -> Path:
    """Save experiment result to results directory.

    Args:
        result: ExperimentResult to save
        experiment_path: Path to experiment directory

    Returns:
        Path to saved result file
    """
    import json

    results_dir = experiment_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"run_{timestamp}.json"
    path = results_dir / filename

    with open(path, "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)

    logger.info(f"Saved result to: {path}")
    return path


def get_experiment_status(name: str, experiments_dir: Path | None = None) -> dict:
    """Get status of an experiment including latest results.

    Args:
        name: Experiment name
        experiments_dir: Path to experiments directory

    Returns:
        Dictionary with experiment status and results
    """
    import json

    info = get_experiment_info(name, experiments_dir)

    status = {
        "name": info.name,
        "description": info.description,
        "path": str(info.path),
        "has_results": info.has_results,
        "latest_result": None,
    }

    if info.has_results:
        results_dir = info.path / "results"
        results_files = sorted(results_dir.glob("run_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

        if results_files:
            with open(results_files[0]) as f:
                status["latest_result"] = json.load(f)

    return status


def list_experiment_runs(name: str, experiments_dir: Path | None = None, limit: int = 10) -> list[dict]:
    """List recent runs of an experiment.

    Args:
        name: Experiment name
        experiments_dir: Path to experiments directory
        limit: Maximum number of runs to return

    Returns:
        List of run summaries (most recent first)
    """
    import json

    info = get_experiment_info(name, experiments_dir)
    results_dir = info.path / "results"

    if not results_dir.exists():
        return []

    runs = []
    results_files = sorted(results_dir.glob("run_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

    for path in results_files[:limit]:
        try:
            with open(path) as f:
                data = json.load(f)
            runs.append(
                {
                    "file": path.name,
                    "status": data.get("status"),
                    "started_at": data.get("started_at"),
                    "duration_seconds": data.get("duration_seconds"),
                    "eval_results": data.get("eval_results", {}),
                }
            )
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")

    return runs
