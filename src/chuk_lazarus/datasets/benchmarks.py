"""Benchmark datasets for model evaluation.

This module provides benchmark problems and test categories
for evaluating model capabilities, particularly for MoE
expert analysis and virtual expert systems.
"""

from __future__ import annotations

from typing import Any

# Expert test categories - organized by domain
_EXPERT_TEST_CATEGORIES: dict[str, list[str]] = {
    "MATH": [
        "127 * 89 = ",
        "456 + 789 = ",
        "1000 - 237 = ",
        "144 / 12 = ",
        "25 * 25 = ",
        "999 + 1 = ",
        "500 - 123 = ",
        "64 / 8 = ",
    ],
    "CODE": [
        "def fibonacci(n):",
        "for i in range(10):",
        "import numpy as np",
        "class MyClass:",
        "if __name__ == '__main__':",
        "try:\n    x = 1",
        "lambda x: x * 2",
        "list(map(str, [1,2,3]))",
    ],
    "LOGIC": [
        "If A implies B, and B implies C, then A implies",
        "All dogs are animals. Fido is a dog. Therefore Fido is",
        "If it rains, the ground is wet. The ground is wet. Can we conclude it rained?",
        "NOT (A AND B) is equivalent to",
        "If P then Q. Not Q. Therefore",
    ],
    "LANGUAGE": [
        "The capital of France is",
        "Translate to French: Hello",
        "The opposite of 'hot' is",
        "Complete the analogy: King is to Queen as Prince is to",
        "The past tense of 'run' is",
    ],
    "SCIENCE": [
        "Water boils at",
        "The chemical formula for water is",
        "Newton's first law states that",
        "The speed of light is approximately",
        "DNA stands for",
    ],
}


def load_expert_test_categories() -> dict[str, list[str]]:
    """Load test categories for expert analysis.

    Returns:
        Dictionary mapping category names to lists of test prompts.

    Example:
        >>> categories = load_expert_test_categories()
        >>> for cat, prompts in categories.items():
        ...     print(f"{cat}: {len(prompts)} prompts")
    """
    # Return a copy to prevent mutation
    return {k: v.copy() for k, v in _EXPERT_TEST_CATEGORIES.items()}


# Benchmark problems for evaluation
_EXPERT_BENCHMARK_PROBLEMS: list[dict[str, Any]] = [
    # Simple arithmetic
    {"prompt": "2 + 2 = ", "answer": "4", "difficulty": "easy"},
    {"prompt": "5 * 5 = ", "answer": "25", "difficulty": "easy"},
    {"prompt": "10 - 3 = ", "answer": "7", "difficulty": "easy"},
    {"prompt": "20 / 4 = ", "answer": "5", "difficulty": "easy"},
    # Medium arithmetic
    {"prompt": "23 + 45 = ", "answer": "68", "difficulty": "medium"},
    {"prompt": "12 * 11 = ", "answer": "132", "difficulty": "medium"},
    {"prompt": "100 - 37 = ", "answer": "63", "difficulty": "medium"},
    {"prompt": "144 / 12 = ", "answer": "12", "difficulty": "medium"},
    # Hard arithmetic
    {"prompt": "127 * 89 = ", "answer": "11303", "difficulty": "hard"},
    {"prompt": "12345 + 67890 = ", "answer": "80235", "difficulty": "hard"},
    {"prompt": "9999 - 1234 = ", "answer": "8765", "difficulty": "hard"},
    {"prompt": "1024 / 32 = ", "answer": "32", "difficulty": "hard"},
    # Multi-step
    {"prompt": "2 + 3 * 4 = ", "answer": "14", "difficulty": "medium"},
    {"prompt": "(10 + 5) * 2 = ", "answer": "30", "difficulty": "medium"},
]


def load_expert_benchmark() -> list[dict[str, Any]]:
    """Load benchmark problems for expert evaluation.

    Returns:
        List of benchmark problems with schema:
        {
            "prompt": str,
            "answer": str,
            "difficulty": "easy" | "medium" | "hard",
        }

    Example:
        >>> problems = load_expert_benchmark()
        >>> hard = [p for p in problems if p["difficulty"] == "hard"]
        >>> print(f"Hard problems: {len(hard)}")
    """
    # Return a deep copy to prevent mutation
    return [p.copy() for p in _EXPERT_BENCHMARK_PROBLEMS]


__all__ = [
    "load_expert_benchmark",
    "load_expert_test_categories",
]
