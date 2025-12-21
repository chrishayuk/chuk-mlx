"""Data generators for synthetic training data."""

# Generators
from .math_generator import (
    MathProblemGenerator,
    generate_lazarus_dataset,
)

# Types
from .types import (
    MathProblem,
    ProblemType,
    ToolCallTrace,
    TrainingSample,
)

__all__ = [
    "MathProblemGenerator",
    "generate_lazarus_dataset",
    "MathProblem",
    "ProblemType",
    "ToolCallTrace",
    "TrainingSample",
]
