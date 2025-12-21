"""Data generators for synthetic training data."""

# Types
from .types import (
    ProblemType,
    MathProblem,
    ToolCallTrace,
    TrainingSample,
)

# Generators
from .math_generator import (
    MathProblemGenerator,
    generate_lazarus_dataset,
)
