"""
Type definitions for data generators.

This module contains the data structures used by generators:
- ProblemType: Enum of problem categories
- MathProblem: A generated math problem
- ToolCallTrace: Trace showing tool calls to solve a problem
- TrainingSample: Complete training sample with correct and incorrect responses
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ProblemType(str, Enum):
    """Types of math problems."""

    ARITHMETIC = "arithmetic"
    FRACTIONS = "fractions"
    PERCENTAGES = "percentages"
    WORD_PROBLEM = "word_problem"
    MULTI_STEP = "multi_step"
    COMPARISON = "comparison"


class MathProblem(BaseModel):
    """A generated math problem."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(description="Unique problem ID")
    problem_type: ProblemType = Field(description="Type of math problem")
    problem_text: str = Field(description="Problem description")
    expression: str = Field(description="The mathematical expression")
    answer: float = Field(description="Numeric answer")
    answer_exact: str | None = Field(default=None, description="Exact answer for fractions, etc.")
    unit: str | None = Field(default=None, description="Unit of measurement")
    difficulty: int = Field(default=1, ge=1, le=5, description="Difficulty level (1-5)")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ToolCallTrace(BaseModel):
    """A trace showing how to solve the problem with tools."""

    model_config = ConfigDict(frozen=True)

    tool_name: str = Field(description="Name of the tool called")
    tool_args: dict[str, Any] = Field(description="Arguments passed to the tool")
    tool_result: Any = Field(description="Result returned by the tool")
    thought: str | None = Field(default=None, description="Chain of thought reasoning")


class TrainingSample(BaseModel):
    """A complete training sample."""

    model_config = ConfigDict(frozen=True)

    problem: MathProblem = Field(description="The math problem")
    correct_trace: list[ToolCallTrace] = Field(description="Correct tool call trace")
    correct_response: str = Field(description="Correct response text")
    incorrect_responses: list[str] = Field(default_factory=list, description="Incorrect responses")
