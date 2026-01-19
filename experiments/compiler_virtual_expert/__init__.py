"""
Compiler Virtual Expert Experiment.

Implements a compiler as a virtual expert that verifies and executes
model-generated code, providing feedback for self-correction.
"""

from .compiler_plugin import CompilerExpertPlugin, ExecutionResult

__all__ = ["CompilerExpertPlugin", "ExecutionResult"]
