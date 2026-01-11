"""
CLI handlers for experiment commands.
"""

import json
import logging
from pathlib import Path

from chuk_lazarus.experiments import list_experiments as _list_experiments
from chuk_lazarus.experiments import run_experiment as _run_experiment
from chuk_lazarus.experiments.registry import get_experiment_info
from chuk_lazarus.experiments.runner import get_experiment_status, list_experiment_runs

logger = logging.getLogger(__name__)


def experiment_list(experiments_dir: str | None = None, json_output: bool = False) -> None:
    """List all discovered experiments.

    Args:
        experiments_dir: Optional path to experiments directory
        json_output: Output as JSON instead of table
    """
    exp_dir = Path(experiments_dir) if experiments_dir else None
    experiments = _list_experiments(exp_dir)

    if not experiments:
        print("No experiments found.")
        print(f"Looking in: {exp_dir or 'auto-detected experiments/'}")
        return

    if json_output:
        data = [
            {
                "name": exp.name,
                "description": exp.description,
                "path": str(exp.path),
                "has_results": exp.has_results,
            }
            for exp in experiments
        ]
        print(json.dumps(data, indent=2))
        return

    # Table output
    print("\nAvailable Experiments:")
    print("=" * 70)
    print(f"{'Name':<25} {'Status':<10} {'Description'}")
    print("-" * 70)

    for exp in experiments:
        status = "has results" if exp.has_results else "no runs"
        desc = exp.description[:35] + "..." if len(exp.description) > 35 else exp.description
        print(f"{exp.name:<25} {status:<10} {desc}")

    print("-" * 70)
    print(f"Total: {len(experiments)} experiments")
    print()


def experiment_info(
    name: str, experiments_dir: str | None = None, json_output: bool = False
) -> None:
    """Show detailed information about an experiment.

    Args:
        name: Experiment name
        experiments_dir: Optional path to experiments directory
        json_output: Output as JSON
    """
    import yaml

    exp_dir = Path(experiments_dir) if experiments_dir else None

    try:
        info = get_experiment_info(name, exp_dir)
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Load full config
    with open(info.config_path) as f:
        config = yaml.safe_load(f)

    if json_output:
        data = {
            "name": info.name,
            "description": info.description,
            "path": str(info.path),
            "config": config,
            "has_results": info.has_results,
        }
        print(json.dumps(data, indent=2))
        return

    print(f"\nExperiment: {info.name}")
    print("=" * 60)
    print(f"Description: {info.description}")
    print(f"Path: {info.path}")
    print()
    print("Configuration:")
    print("-" * 40)

    for key, value in config.items():
        if key not in ("name", "description"):
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            elif isinstance(value, list):
                print(f"  {key}: {', '.join(str(v) for v in value)}")
            else:
                print(f"  {key}: {value}")

    print()

    # Show recent runs if available
    if info.has_results:
        runs = list_experiment_runs(name, exp_dir, limit=3)
        if runs:
            print("Recent Runs:")
            print("-" * 40)
            for run in runs:
                started = run.get("started_at", "unknown")[:19]
                duration = run.get("duration_seconds", 0)
                status = run.get("status", "unknown")
                print(f"  {started} - {status} ({duration:.1f}s)")
            print()


def experiment_run(
    name: str,
    experiments_dir: str | None = None,
    config_file: str | None = None,
    params: list[str] | None = None,
    dry_run: bool = False,
) -> None:
    """Run an experiment.

    Args:
        name: Experiment name
        experiments_dir: Optional path to experiments directory
        config_file: Optional path to override config file
        params: List of parameter overrides (key=value format)
        dry_run: If True, validate without running
    """
    import yaml

    exp_dir = Path(experiments_dir) if experiments_dir else None

    # Parse parameter overrides
    overrides = {}
    if params:
        for param in params:
            if "=" not in param:
                print(f"Invalid parameter format: {param} (expected key=value)")
                return
            key, value = param.split("=", 1)
            # Try to parse as JSON for complex values
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass  # Keep as string
            overrides[key] = value

    # Load custom config if provided
    if config_file:
        with open(config_file) as f:
            custom_config = yaml.safe_load(f)
        overrides.update(custom_config)

    print(f"\nRunning experiment: {name}")
    if dry_run:
        print("(dry run mode)")
    print("=" * 60)

    try:
        result = _run_experiment(
            name=name,
            experiments_dir=exp_dir,
            config_overrides=overrides if overrides else None,
            dry_run=dry_run,
        )

        print()
        print("Results:")
        print("-" * 40)
        print(f"Status: {result.status}")
        print(f"Duration: {result.duration_seconds:.2f}s")

        if result.eval_results:
            print()
            print("Evaluation Metrics:")
            for key, value in result.eval_results.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")

        if result.error:
            print()
            print(f"Error: {result.error}")

        print()

    except Exception as e:
        print(f"Error running experiment: {e}")
        logger.exception("Experiment run failed")


def experiment_status(
    name: str,
    experiments_dir: str | None = None,
    show_all: bool = False,
    json_output: bool = False,
) -> None:
    """Show experiment status and latest results.

    Args:
        name: Experiment name
        experiments_dir: Optional path to experiments directory
        show_all: Show all runs, not just latest
        json_output: Output as JSON
    """
    exp_dir = Path(experiments_dir) if experiments_dir else None

    try:
        status = get_experiment_status(name, exp_dir)
    except ValueError as e:
        print(f"Error: {e}")
        return

    if json_output:
        print(json.dumps(status, indent=2))
        return

    print(f"\nExperiment Status: {name}")
    print("=" * 60)
    print(f"Description: {status['description']}")
    print(f"Has Results: {'Yes' if status['has_results'] else 'No'}")

    if status["latest_result"]:
        result = status["latest_result"]
        print()
        print("Latest Run:")
        print("-" * 40)
        print(f"  Status: {result.get('status')}")
        print(f"  Started: {result.get('started_at', '')[:19]}")
        print(f"  Duration: {result.get('duration_seconds', 0):.2f}s")

        eval_results = result.get("eval_results", {})
        if eval_results:
            print()
            print("  Metrics:")
            for key, value in eval_results.items():
                if isinstance(value, float):
                    print(f"    {key}: {value:.4f}")
                else:
                    print(f"    {key}: {value}")

        if result.get("error"):
            print()
            print(f"  Error: {result.get('error')}")

    if show_all:
        runs = list_experiment_runs(name, exp_dir, limit=20)
        if len(runs) > 1:
            print()
            print("All Runs:")
            print("-" * 40)
            for run in runs:
                started = run.get("started_at", "unknown")[:19]
                duration = run.get("duration_seconds", 0)
                run_status = run.get("status", "unknown")
                print(f"  {started} - {run_status} ({duration:.1f}s)")

    print()
