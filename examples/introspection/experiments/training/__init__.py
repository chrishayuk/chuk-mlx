"""Training experiments for introspection research."""

from .classifier_emergence import (
    ClassifierSignal,
    ExperimentSnapshot,
    TaskResult,
    analyze_model,
    generate_arithmetic_data,
    generate_test_prompts,
    run_baseline_experiment,
    run_full_experiment,
)

__all__ = [
    "ClassifierSignal",
    "ExperimentSnapshot",
    "TaskResult",
    "generate_arithmetic_data",
    "generate_test_prompts",
    "analyze_model",
    "run_baseline_experiment",
    "run_full_experiment",
]
