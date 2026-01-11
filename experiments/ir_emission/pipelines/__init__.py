"""
IR Emission Pipelines.

Each pipeline tests a specific capability of the neural compiler:
- single_op: Single arithmetic operations (100% accuracy)
- multi_op: Multi-operation chains (75% accuracy)
- loop: Loop constructs demonstrating Turing completeness (100% accuracy)
"""

from .base import NeuralCompilerBase, PipelineResult
from .single_op import SingleOpPipeline
from .multi_op import MultiOpPipeline
from .loop import LoopPipeline

__all__ = [
    "NeuralCompilerBase",
    "PipelineResult",
    "SingleOpPipeline",
    "MultiOpPipeline",
    "LoopPipeline",
]
