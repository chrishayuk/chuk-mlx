"""
Neural Compiler Experiment.

Demonstrates that LLMs serve as semantic frontends for deterministic
computation via WASM. Achieves 100% accuracy on arithmetic, multi-step
operations, and loops.
"""

from .experiment import NeuralCompilerExperiment

__all__ = ["NeuralCompilerExperiment"]
