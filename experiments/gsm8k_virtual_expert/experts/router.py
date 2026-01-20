"""
Expert Router.

Routes GSM-8K problems to appropriate expert based on classification.
"""

from __future__ import annotations

from typing import Any

from ..extraction.gsm8k_extractor import (
    GSM8KExtractor,
    ProblemSpec,
    ProblemType,
    extract_problem,
)
from .calculator_chain import CalculatorChainExpert
from .rate_ratio import RateRatioExpert
from .allocation import AllocationExpert
from .comparison import ComparisonExpert
from .time_calculator import TimeCalculatorExpert
from .geometry import GeometryExpert
from .percentage import PercentageExpert
from .equation_solver import EquationSolverExpert


class ExpertRouter:
    """
    Routes GSM-8K problems to specialized experts.

    Two modes:
    1. Pattern-based: Use regex patterns to classify
    2. Probe-based: Use L4 linear probe for classification (when available)

    Example:
        >>> router = ExpertRouter()
        >>> answer = router.solve("Jenny has 5 apples. She buys 3. How many?")
        8.0
    """

    def __init__(self, use_probe: bool = False, probe_model=None):
        """
        Initialize router with experts.

        Args:
            use_probe: If True, use L4 probe for classification (requires model)
            probe_model: Trained probe model for classification
        """
        self.extractor = GSM8KExtractor()
        self.use_probe = use_probe
        self.probe = probe_model

        # Initialize all experts
        self.experts = {
            ProblemType.ARITHMETIC_CHAIN: CalculatorChainExpert(),
            ProblemType.RATE_RATIO: RateRatioExpert(),
            ProblemType.ALLOCATION: AllocationExpert(),
            ProblemType.COMPARISON: ComparisonExpert(),
            ProblemType.SCHEDULING_TIME: TimeCalculatorExpert(),
            ProblemType.GEOMETRY: GeometryExpert(),
            ProblemType.PERCENTAGE: PercentageExpert(),
            ProblemType.MULTI_CONSTRAINT: EquationSolverExpert(),
        }

        # Default expert for unknown types
        self.default_expert = CalculatorChainExpert()

    def classify(self, problem: str) -> ProblemType:
        """
        Classify problem type.

        Uses probe if available, otherwise pattern matching.
        """
        if self.use_probe and self.probe is not None:
            # TODO: Extract hidden states and run probe
            # predicted_type = self.probe.predict(hidden_state)
            # return ProblemType(predicted_type)
            pass

        # Fallback to pattern matching
        return self.extractor.classify(problem)

    def extract(self, problem: str) -> ProblemSpec:
        """
        Extract problem specification.
        """
        return self.extractor.extract(problem)

    def solve(self, problem: str) -> dict:
        """
        Classify, extract, and solve a GSM-8K problem.

        Returns:
            Dict with problem_type, answer, and explanation
        """
        # Extract specification (includes classification)
        spec = self.extract(problem)

        # Get appropriate expert
        expert = self.experts.get(spec.problem_type, self.default_expert)

        # Solve
        answer = expert.solve_from_spec(spec)

        return {
            "problem": problem,
            "problem_type": spec.problem_type.value,
            "answer": answer,
            "spec": spec.to_dict(),
            "expert": type(expert).__name__,
        }

    def solve_batch(self, problems: list[str]) -> list[dict]:
        """
        Solve a batch of problems.
        """
        return [self.solve(p) for p in problems]


def solve_gsm8k(problem: str) -> float | str | None:
    """
    Convenience function to solve a GSM-8K problem.

    Returns numeric answer or string (for time problems).
    """
    router = ExpertRouter()
    result = router.solve(problem)
    return result["answer"]


if __name__ == "__main__":
    print("Expert Router Tests")
    print("=" * 60)

    router = ExpertRouter()

    tests = [
        "Jenny has 5 apples. She buys 3 more. How many does she have?",
        "If 3 workers paint a house in 6 hours, how long for 2 workers?",
        "Split $100 between Alice and Bob where Alice gets twice what Bob gets. How much does Alice get?",
        "Tom has 3 times as many marbles as Jane. Jane has 5. How many more does Tom have?",
        "Train leaves at 9:00am, travels 3 hours. What time does it arrive?",
        "A rectangle is 5m by 3m. What is its area?",
        "A shirt costs $40. It's 25% off. What's the final price?",
        "The sum of two numbers is 20. Their difference is 4. Find the larger number.",
    ]

    for problem in tests:
        result = router.solve(problem)
        print(f"\nProblem: {problem[:60]}...")
        print(f"  Type: {result['problem_type']}")
        print(f"  Expert: {result['expert']}")
        print(f"  Answer: {result['answer']}")
