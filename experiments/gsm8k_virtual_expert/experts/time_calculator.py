"""
Time Calculator Expert.

Handles time calculations, durations, schedules.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from ..extraction.gsm8k_extractor import ProblemSpec


@dataclass
class TimeResult:
    """Result from time calculation."""
    final_time: str
    total_duration: timedelta | None
    answer: str
    steps: list[str]


class TimeCalculatorExpert:
    """
    Expert for time/scheduling arithmetic.

    Pattern: "Leave at 9am, travel 3 hours, stop 30min. What time arrive?"
    """

    def solve(self, spec: ProblemSpec) -> TimeResult:
        """
        Solve time calculation problem.
        """
        steps = []

        # Get start time
        start_hour = spec.metadata.get("start_hour", 9)
        start_minute = spec.metadata.get("start_minute", 0)

        current = datetime(2000, 1, 1, start_hour, start_minute)
        steps.append(f"Start: {current.strftime('%I:%M %p')}")

        total_duration = timedelta()

        # Apply operations
        for op in spec.operations:
            if op.type == "add_time":
                if op.unit == "hours":
                    delta = timedelta(hours=op.amount)
                elif op.unit == "minutes":
                    delta = timedelta(minutes=op.amount)
                else:
                    delta = timedelta()

                current += delta
                total_duration += delta
                steps.append(f"+ {op.amount} {op.unit} → {current.strftime('%I:%M %p')}")

        final_time = current.strftime("%I:%M %p")
        steps.append(f"Final: {final_time}")

        # Also handle duration-based questions
        hours = total_duration.total_seconds() / 3600
        if hours == int(hours):
            answer = final_time
        else:
            answer = final_time

        return TimeResult(
            final_time=final_time,
            total_duration=total_duration,
            answer=answer,
            steps=steps,
        )

    def solve_from_spec(self, spec: ProblemSpec) -> str | None:
        """Solve and return time answer."""
        result = self.solve(spec)
        return result.answer
