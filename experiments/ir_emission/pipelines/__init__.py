"""
IR Emission Pipelines.

Each pipeline tests a specific capability of the neural compiler:
- single_op: Single arithmetic operations (100% accuracy)
- multi_op: Multi-operation chains (100% accuracy)
- loop: Loop constructs demonstrating Turing completeness (100% accuracy)
- comparison: Comparison operations with deterministic evaluation (100% accuracy)
"""

from .base import NeuralCompilerBase, PipelineResult
from .comparison import ComparisonFullIRPipeline, ComparisonPipeline
from .loop import LoopPipeline
from .multi_op import MultiOpPipeline
from .single_op import SingleOpPipeline

__all__ = [
    "NeuralCompilerBase",
    "PipelineResult",
    "SingleOpPipeline",
    "MultiOpPipeline",
    "LoopPipeline",
    "ComparisonPipeline",
    "ComparisonFullIRPipeline",
]
