"""
Rate/Ratio Expert.

Handles work rates, speed, proportions, inverse relationships.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..extraction.gsm8k_extractor import ProblemSpec


@dataclass
class RateResult:
    """Result from rate/ratio calculation."""
    answer: float | None
    unit: str | None
    formula: str
    steps: list[str]


class RateRatioExpert:
    """
    Expert for rate/ratio problems.

    Patterns:
    - Work rate: workers × time = total_work
    - Speed: distance = speed × time
    - Proportions: a/b = c/d
    """

    def solve(self, spec: ProblemSpec) -> RateResult:
        """
        Solve rate/ratio problem based on extracted metadata.
        """
        rate_type = spec.metadata.get("rate_type")

        if rate_type == "work_rate":
            return self._solve_work_rate(spec)
        elif rate_type == "speed":
            return self._solve_speed(spec)
        else:
            return self._solve_proportion(spec)

    def _solve_work_rate(self, spec: ProblemSpec) -> RateResult:
        """
        Solve work rate problem.

        Formula: total_work = workers × time
        If N workers do job in T hours:
            total_work = N × T worker-hours
        For M workers:
            new_time = total_work / M
        """
        workers = spec.metadata.get("workers", 1)
        time = spec.metadata.get("time", 1)
        target_workers = spec.metadata.get("target_workers")

        steps = []

        # Calculate total work
        total_work = workers * time
        steps.append(f"Total work = {workers} workers × {time} hours = {total_work} worker-hours")

        if target_workers:
            new_time = total_work / target_workers
            steps.append(f"For {target_workers} workers: {total_work} / {target_workers} = {new_time} hours")

            return RateResult(
                answer=new_time,
                unit="hours",
                formula="time = total_work / workers",
                steps=steps,
            )

        return RateResult(
            answer=total_work,
            unit="worker-hours",
            formula="work = workers × time",
            steps=steps,
        )

    def _solve_speed(self, spec: ProblemSpec) -> RateResult:
        """
        Solve speed/distance problem.

        Formula: distance = speed × time
        """
        speed = spec.metadata.get("speed", 0)
        time = spec.metadata.get("time", 0)
        time_unit = spec.metadata.get("time_unit", "hours")

        steps = []

        # Convert time to hours if needed
        if "minute" in time_unit.lower():
            time = time / 60
            steps.append(f"Convert time: {spec.metadata.get('time')} minutes = {time} hours")

        distance = speed * time
        steps.append(f"Distance = {speed} mph × {time} hours = {distance} miles")

        return RateResult(
            answer=distance,
            unit="miles",
            formula="distance = speed × time",
            steps=steps,
        )

    def _solve_proportion(self, spec: ProblemSpec) -> RateResult:
        """
        Solve general proportion problem.

        Formula: a/b = c/d → solve for unknown
        """
        # Try to extract proportion from constraints
        steps = ["Setting up proportion..."]

        # Default fallback
        return RateResult(
            answer=None,
            unit=None,
            formula="a/b = c/d",
            steps=steps,
        )

    def solve_from_spec(self, spec: ProblemSpec) -> float | None:
        """Solve and return numeric answer."""
        result = self.solve(spec)
        return result.answer
