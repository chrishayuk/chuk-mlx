"""
Experiment registry for discovering and loading experiments.

Experiments are discovered by looking for directories containing:
- experiment.py: Module defining a class inheriting from ExperimentBase
- config.yaml: Experiment configuration
"""

import importlib.util
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import ExperimentBase

logger = logging.getLogger(__name__)

# Default experiments directory (relative to project root)
DEFAULT_EXPERIMENTS_DIR = "experiments"


@dataclass
class ExperimentInfo:
    """Information about a discovered experiment."""

    name: str
    description: str
    path: Path
    config_path: Path
    experiment_path: Path
    has_results: bool = False
    last_run: str | None = None


def get_experiments_dir() -> Path:
    """Get the experiments directory.

    Looks for 'experiments' directory starting from:
    1. Current working directory
    2. Parent directories up to 5 levels
    """
    cwd = Path.cwd()

    # Check current and parent directories
    for _ in range(6):
        exp_dir = cwd / DEFAULT_EXPERIMENTS_DIR
        if exp_dir.is_dir():
            return exp_dir
        cwd = cwd.parent

    # Fallback to cwd/experiments
    return Path.cwd() / DEFAULT_EXPERIMENTS_DIR


def validate_experiment(path: Path) -> tuple[bool, str]:
    """Validate that a directory is a valid experiment.

    Args:
        path: Path to experiment directory

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not path.is_dir():
        return False, f"Not a directory: {path}"

    experiment_py = path / "experiment.py"
    config_yaml = path / "config.yaml"

    if not experiment_py.exists():
        return False, f"Missing experiment.py in {path}"

    if not config_yaml.exists():
        return False, f"Missing config.yaml in {path}"

    return True, ""


def discover_experiments(
    experiments_dir: Path | None = None,
) -> dict[str, ExperimentInfo]:
    """Discover all valid experiments in the experiments directory.

    Args:
        experiments_dir: Path to experiments directory.
                        Defaults to auto-detected location.

    Returns:
        Dictionary mapping experiment name to ExperimentInfo.
    """
    exp_dir = experiments_dir or get_experiments_dir()

    if not exp_dir.exists():
        logger.warning(f"Experiments directory not found: {exp_dir}")
        return {}

    experiments = {}

    for item in exp_dir.iterdir():
        if not item.is_dir():
            continue

        # Skip special directories
        if item.name.startswith(("_", ".")):
            continue

        is_valid, error = validate_experiment(item)
        if not is_valid:
            logger.debug(f"Skipping {item.name}: {error}")
            continue

        # Load config to get description
        config_path = item / "config.yaml"
        try:
            import yaml

            with open(config_path) as f:
                config = yaml.safe_load(f)
            description = config.get("description", "No description")
        except Exception as e:
            logger.warning(f"Failed to load config for {item.name}: {e}")
            description = "Failed to load description"

        # Check for results
        results_dir = item / "results"
        has_results = results_dir.exists() and any(results_dir.glob("*.json"))
        last_run = None
        if has_results:
            results_files = sorted(
                results_dir.glob("*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if results_files:
                last_run = results_files[0].stat().st_mtime

        experiments[item.name] = ExperimentInfo(
            name=item.name,
            description=description,
            path=item,
            config_path=config_path,
            experiment_path=item / "experiment.py",
            has_results=has_results,
            last_run=last_run,
        )

    return experiments


def list_experiments(experiments_dir: Path | None = None) -> list[ExperimentInfo]:
    """List all discovered experiments.

    Args:
        experiments_dir: Path to experiments directory.

    Returns:
        List of ExperimentInfo sorted by name.
    """
    experiments = discover_experiments(experiments_dir)
    return sorted(experiments.values(), key=lambda e: e.name)


def get_experiment(name: str, experiments_dir: Path | None = None) -> type["ExperimentBase"]:
    """Get an experiment class by name.

    Args:
        name: Experiment name (directory name)
        experiments_dir: Path to experiments directory.

    Returns:
        Experiment class (subclass of ExperimentBase)

    Raises:
        ValueError: If experiment not found or invalid
    """
    exp_dir = experiments_dir or get_experiments_dir()
    exp_path = exp_dir / name

    is_valid, error = validate_experiment(exp_path)
    if not is_valid:
        raise ValueError(f"Invalid experiment '{name}': {error}")

    # Load the experiment module
    experiment_py = exp_path / "experiment.py"

    spec = importlib.util.spec_from_file_location(f"experiments.{name}.experiment", experiment_py)
    if spec is None or spec.loader is None:
        raise ValueError(f"Failed to load experiment module: {experiment_py}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    # Find the experiment class
    from .base import ExperimentBase

    experiment_class = None
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if (
            isinstance(attr, type)
            and issubclass(attr, ExperimentBase)
            and attr is not ExperimentBase
        ):
            experiment_class = attr
            break

    if experiment_class is None:
        raise ValueError(f"No ExperimentBase subclass found in {experiment_py}")

    return experiment_class


def get_experiment_info(name: str, experiments_dir: Path | None = None) -> ExperimentInfo:
    """Get information about a specific experiment.

    Args:
        name: Experiment name
        experiments_dir: Path to experiments directory.

    Returns:
        ExperimentInfo for the experiment

    Raises:
        ValueError: If experiment not found
    """
    experiments = discover_experiments(experiments_dir)
    if name not in experiments:
        raise ValueError(f"Experiment not found: {name}")
    return experiments[name]
