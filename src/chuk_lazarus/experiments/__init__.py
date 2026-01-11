"""
Experiments Framework for chuk-lazarus.

Provides a structured way to define, discover, and run experiments
that leverage the lazarus training infrastructure.

Usage:
    from chuk_lazarus.experiments import ExperimentBase, ExperimentConfig

    class MyExperiment(ExperimentBase):
        def setup(self) -> None:
            self.model = self.load_model()

        def run(self) -> dict:
            # Experiment logic
            return {"accuracy": 0.95}

        def evaluate(self) -> dict:
            return {"final_score": 0.95}
"""

from .base import ExperimentBase, ExperimentConfig, ExperimentResult
from .registry import (
    ExperimentInfo,
    discover_experiments,
    get_experiment,
    list_experiments,
    validate_experiment,
)
from .runner import run_experiment

__all__ = [
    # Base classes
    "ExperimentBase",
    "ExperimentConfig",
    "ExperimentResult",
    # Registry
    "ExperimentInfo",
    "discover_experiments",
    "list_experiments",
    "get_experiment",
    "validate_experiment",
    # Runner
    "run_experiment",
]
