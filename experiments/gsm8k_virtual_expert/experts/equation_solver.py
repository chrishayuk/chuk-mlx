"""
Equation Solver Expert.

Handles multi-constraint problems via symbolic solving.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..extraction.gsm8k_extractor import ProblemSpec


@dataclass
class EquationResult:
    """Result from equation solving."""
    solutions: dict[str, float]
    answer: float | None
    steps: list[str]


class EquationSolverExpert:
    """
    Expert for multi-constraint equation problems.

    Uses sympy for symbolic solving when available,
    falls back to algebraic methods.

    Example:
        "Sum of two numbers is 20. Difference is 4. Find both."
        x + y = 20
        x - y = 4
        Solution: x = 12, y = 8
    """

    def __init__(self):
        self._sympy_available = self._check_sympy()

    def _check_sympy(self) -> bool:
        try:
            import sympy
            return True
        except Exception:
            return False

    def solve(self, spec: ProblemSpec) -> EquationResult:
        """
        Solve system of equations.
        """
        if self._sympy_available:
            return self._solve_sympy(spec)
        return self._solve_algebraic(spec)

    def _solve_sympy(self, spec: ProblemSpec) -> EquationResult:
        """
        Solve using sympy.
        """
        from sympy import symbols, Eq, solve

        steps = []

        # Check for sum/difference constraints (common pattern)
        sum_val = None
        diff_val = None

        for c in spec.constraints:
            if c["type"] == "sum":
                sum_val = c["value"]
                steps.append(f"x + y = {sum_val}")
            elif c["type"] == "difference":
                diff_val = c["value"]
                steps.append(f"x - y = {diff_val}")

        if sum_val is not None and diff_val is not None:
            x, y = symbols("x y")
            eq1 = Eq(x + y, sum_val)
            eq2 = Eq(x - y, diff_val)

            solution = solve((eq1, eq2), (x, y))
            steps.append(f"Solution: x = {solution[x]}, y = {solution[y]}")

            return EquationResult(
                solutions={"x": float(solution[x]), "y": float(solution[y])},
                answer=float(solution[x]),  # Usually asking for the larger
                steps=steps,
            )

        # Fallback for other equation types
        steps.append("Could not parse equations")
        return EquationResult(solutions={}, answer=None, steps=steps)

    def _solve_algebraic(self, spec: ProblemSpec) -> EquationResult:
        """
        Fallback algebraic solution.
        """
        steps = ["Using algebraic method (sympy not available)"]

        # Sum/difference system: x + y = S, x - y = D
        # Solution: x = (S + D) / 2, y = (S - D) / 2
        sum_val = None
        diff_val = None

        for c in spec.constraints:
            if c["type"] == "sum":
                sum_val = c["value"]
            elif c["type"] == "difference":
                diff_val = c["value"]

        if sum_val is not None and diff_val is not None:
            x = (sum_val + diff_val) / 2
            y = (sum_val - diff_val) / 2

            steps.append(f"x = ({sum_val} + {diff_val}) / 2 = {x}")
            steps.append(f"y = ({sum_val} - {diff_val}) / 2 = {y}")

            return EquationResult(
                solutions={"x": x, "y": y},
                answer=x,
                steps=steps,
            )

        return EquationResult(solutions={}, answer=None, steps=steps)

    def solve_from_spec(self, spec: ProblemSpec) -> float | None:
        """Solve and return numeric answer."""
        result = self.solve(spec)
        return result.answer
