"""Data generators for synthetic training data."""

from .math_generator import (
    MathProblemGenerator,
    MathProblem,
    ProblemType,
    ToolCallTrace,
    TrainingSample,
    generate_lazarus_dataset,
)
