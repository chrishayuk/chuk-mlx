"""
Type definitions for data generators.

This module contains the data structures used by generators:
- ProblemType: Enum of problem categories
- MathProblem: A generated math problem
- ToolCallTrace: Trace showing tool calls to solve a problem
- TrainingSample: Complete training sample with correct and incorrect responses
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ProblemType(Enum):
    """Types of math problems."""
    ARITHMETIC = "arithmetic"
    FRACTIONS = "fractions"
    PERCENTAGES = "percentages"
    WORD_PROBLEM = "word_problem"
    MULTI_STEP = "multi_step"
    COMPARISON = "comparison"


@dataclass
class MathProblem:
    """A generated math problem."""
    id: str
    problem_type: ProblemType
    problem_text: str
    expression: str  # The mathematical expression
    answer: float
    answer_exact: Optional[str] = None  # For fractions, etc.
    unit: Optional[str] = None
    difficulty: int = 1  # 1-5
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCallTrace:
    """A trace showing how to solve the problem with tools."""
    tool_name: str
    tool_args: Dict[str, Any]
    tool_result: Any
    thought: Optional[str] = None


@dataclass
class TrainingSample:
    """A complete training sample."""
    problem: MathProblem
    correct_trace: List[ToolCallTrace]
    correct_response: str
    incorrect_responses: List[str] = field(default_factory=list)
