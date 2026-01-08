"""Calibration prompt datasets for uncertainty detection.

This module provides calibration prompts used to train and evaluate
uncertainty detection models. Prompts are categorized into "working"
(should compute correctly) and "broken" (may refuse or fail).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CalibrationPrompts(BaseModel):
    """Calibration prompts for uncertainty detection."""

    working: list[str] = Field(
        default_factory=list,
        description="Prompts that should compute correctly",
    )
    broken: list[str] = Field(
        default_factory=list,
        description="Prompts that may refuse or fail",
    )


# Default calibration prompts - arithmetic with format variations
_DEFAULT_WORKING_PROMPTS: list[str] = [
    # Standard format with trailing space
    "100 - 37 = ",
    "50 + 25 = ",
    "12 * 4 = ",
    "144 / 12 = ",
    "7 * 8 = ",
    "99 - 11 = ",
    "25 + 75 = ",
    "81 / 9 = ",
    "15 * 3 = ",
    "200 - 50 = ",
    # Larger numbers
    "1234 + 5678 = ",
    "9999 - 1111 = ",
    "123 * 45 = ",
    # Simple single digit
    "2 + 2 = ",
    "5 * 5 = ",
    "9 - 3 = ",
    "8 / 2 = ",
]

_DEFAULT_BROKEN_PROMPTS: list[str] = [
    # Missing trailing space (common format issue)
    "100 - 37 =",
    "50 + 25 =",
    "12 * 4 =",
    # No spaces at all
    "100-37=",
    "50+25=",
    # Different format
    "What is 100 minus 37?",
    "Calculate 50 plus 25",
    # Ambiguous
    "100 37",
    "fifty plus twenty-five",
]


def load_calibration_prompts() -> CalibrationPrompts:
    """Load default calibration prompts.

    Returns:
        CalibrationPrompts with working and broken prompt sets.

    Example:
        >>> prompts = load_calibration_prompts()
        >>> print(len(prompts.working))  # Working prompts
        >>> print(len(prompts.broken))   # Broken prompts
    """
    return CalibrationPrompts(
        working=_DEFAULT_WORKING_PROMPTS.copy(),
        broken=_DEFAULT_BROKEN_PROMPTS.copy(),
    )


__all__ = [
    "CalibrationPrompts",
    "load_calibration_prompts",
]
