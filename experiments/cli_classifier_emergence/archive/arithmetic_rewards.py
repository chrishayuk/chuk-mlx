"""Arithmetic reward functions for GRPO training.

This script defines the reward function and prompt generator for testing
whether pure RL with verifiable rewards induces classifiers.

Usage:
    lazarus train grpo --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --reward-script experiments/cli_classifier_emergence/arithmetic_rewards.py \
        --iterations 100 --prompts-per-iteration 16
"""

import random
import re

# Arithmetic operations
OPS = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
}


def generate_arithmetic_prompt() -> tuple[str, int]:
    """Generate an arithmetic prompt and its correct answer.

    Returns:
        Tuple of (prompt, correct_answer)
    """
    op = random.choice(list(OPS.keys()))

    if op == "*":
        a = random.randint(2, 12)
        b = random.randint(2, 12)
    else:
        a = random.randint(10, 99)
        b = random.randint(10, 99)

    correct = OPS[op](a, b)
    prompt = f"Calculate: {a} {op} {b} = "

    return prompt, correct


def extract_number(response: str) -> int | None:
    """Extract the first number from a response."""
    # Try to find a number at the start or after common patterns
    patterns = [
        r"^[-]?\d+",  # Number at start
        r"=\s*([-]?\d+)",  # After equals
        r"is\s+([-]?\d+)",  # After "is"
        r":\s*([-]?\d+)",  # After colon
        r"([-]?\d+)\s*$",  # Number at end
    ]

    for pattern in patterns:
        match = re.search(pattern, response.strip())
        if match:
            try:
                return int(match.group(1) if match.lastindex else match.group())
            except ValueError:
                continue

    return None


def reward_fn(prompt: str, response: str) -> float:
    """Compute reward for an arithmetic response.

    Args:
        prompt: The arithmetic prompt (e.g., "Calculate: 5 + 3 = ")
        response: The model's response

    Returns:
        1.0 if correct, 0.0 if incorrect
    """
    # Parse the prompt to get the correct answer
    match = re.search(r"Calculate:\s*(\d+)\s*([\+\-\*])\s*(\d+)", prompt)
    if not match:
        return 0.0

    a, op, b = int(match.group(1)), match.group(2), int(match.group(3))
    correct = OPS[op](a, b)

    # Extract answer from response
    answer = extract_number(response)

    if answer is None:
        return 0.0

    return 1.0 if answer == correct else 0.0


def get_prompts() -> list[str]:
    """Generate a batch of arithmetic prompts.

    Returns:
        List of prompt strings
    """
    prompts = []
    for _ in range(32):  # Generate 32 prompts per call
        prompt, _ = generate_arithmetic_prompt()
        prompts.append(prompt)
    return prompts


# For testing
if __name__ == "__main__":
    # Test prompt generation
    print("Sample prompts:")
    for prompt in get_prompts()[:5]:
        print(f"  {prompt}")

    # Test reward function
    print("\nReward tests:")
    test_cases = [
        ("Calculate: 5 + 3 = ", "8", 1.0),
        ("Calculate: 5 + 3 = ", "The answer is 8.", 1.0),
        ("Calculate: 5 + 3 = ", "9", 0.0),
        ("Calculate: 12 * 5 = ", "60", 1.0),
        ("Calculate: 50 - 25 = ", "25", 1.0),
    ]

    for prompt, response, expected in test_cases:
        actual = reward_fn(prompt, response)
        status = "✓" if actual == expected else "✗"
        print(f"  {status} {prompt!r} + {response!r} -> {actual} (expected {expected})")
