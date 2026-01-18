"""
Training Data Generator for IR Program Synthesis

Generates (NL description, IR sequence) pairs for training the IR head.

Each program has multiple NL templates that get filled with operands,
creating diverse training examples that all map to the same IR structure.
"""

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

from programs import (
    IRProgram,
    SUM_1_TO_N,
    SUM_A_TO_B,
    FACTORIAL,
    POWER,
    COLLATZ_LENGTH,
    MAX_OF_TWO,
    ABS_DIFF,
    SUM_EVEN,
    compile_program,
    WASMRuntime,
)
from ir_emission.shared import IROpcode


# =============================================================================
# NL TEMPLATES
# =============================================================================

# Each program has multiple natural language templates
# {0}, {1}, etc. are replaced with operands

NL_TEMPLATES = {
    "sum_1_to_n": [
        "Sum 1 to {0}",
        "Sum from 1 to {0}",
        "Add numbers 1 to {0}",
        "What is 1 + 2 + ... + {0}?",
        "Calculate the sum of integers from 1 to {0}",
        "Compute sum(1, {0})",
        "Total of numbers 1 through {0}",
        "1 + 2 + 3 + ... + {0} = ?",
        "Find the sum 1 to {0}",
        "Add all integers from 1 to {0}",
    ],
    "sum_a_to_b": [
        "Sum {0} to {1}",
        "Sum from {0} to {1}",
        "Add numbers {0} to {1}",
        "What is {0} + ... + {1}?",
        "Calculate sum from {0} to {1}",
        "Compute sum({0}, {1})",
        "Total of {0} through {1}",
        "Add integers from {0} to {1}",
    ],
    "factorial": [
        "{0} factorial",
        "{0}!",
        "Factorial of {0}",
        "What is {0}!?",
        "Calculate {0} factorial",
        "Compute factorial({0})",
        "{0} * {0_minus_1} * ... * 1",
        "Product 1 to {0}",
    ],
    "power": [
        "{0} to the power of {1}",
        "{0}^{1}",
        "{0} raised to {1}",
        "What is {0}^{1}?",
        "Calculate {0} to the {1}th power",
        "Compute power({0}, {1})",
        "{0} ** {1}",
        "Exponentiate {0} by {1}",
    ],
    "collatz_length": [
        "Collatz length of {0}",
        "Collatz steps for {0}",
        "How many Collatz steps for {0}?",
        "Collatz sequence length starting at {0}",
        "Count Collatz iterations for {0}",
        "Steps to reach 1 from {0} via Collatz",
        "Collatz({0}) length",
        "How long is the Collatz sequence for {0}?",
    ],
    "max_of_two": [
        "Max of {0} and {1}",
        "Maximum of {0} and {1}",
        "What is max({0}, {1})?",
        "The larger of {0} and {1}",
        "Which is bigger, {0} or {1}?",
        "max({0}, {1})",
        "Greater of {0} and {1}",
    ],
    "abs_diff": [
        "Absolute difference of {0} and {1}",
        "|{0} - {1}|",
        "abs({0} - {1})",
        "Distance between {0} and {1}",
        "How far apart are {0} and {1}?",
        "Absolute value of {0} minus {1}",
    ],
    "sum_even": [
        "Sum of even numbers from 1 to {0}",
        "Add all even numbers up to {0}",
        "2 + 4 + 6 + ... up to {0}",
        "Sum evens to {0}",
        "Total of even integers from 1 to {0}",
        "What is 2 + 4 + ... + evens up to {0}?",
    ],
}


# =============================================================================
# OPERAND GENERATORS
# =============================================================================

def generate_operands_sum_1_to_n() -> tuple[list[int], int]:
    """Generate operands for sum_1_to_n."""
    n = random.randint(5, 100)
    expected = n * (n + 1) // 2
    return [n], expected


def generate_operands_sum_a_to_b() -> tuple[list[int], int]:
    """Generate operands for sum_a_to_b."""
    a = random.randint(1, 50)
    b = random.randint(a + 5, a + 100)
    expected = (b - a + 1) * (a + b) // 2
    return [a, b], expected


def generate_operands_factorial() -> tuple[list[int], int]:
    """Generate operands for factorial."""
    n = random.randint(3, 12)  # Keep small to avoid overflow
    expected = 1
    for i in range(2, n + 1):
        expected *= i
    return [n], expected


def generate_operands_power() -> tuple[list[int], int]:
    """Generate operands for power."""
    base = random.randint(2, 5)
    exp = random.randint(2, 15)
    expected = base ** exp
    return [base, exp], expected


def generate_operands_collatz() -> tuple[list[int], int]:
    """Generate operands for collatz_length."""
    n = random.randint(10, 1000)
    # Compute expected
    count = 0
    x = n
    while x > 1:
        if x % 2 == 0:
            x = x // 2
        else:
            x = 3 * x + 1
        count += 1
    return [n], count


def generate_operands_max_of_two() -> tuple[list[int], int]:
    """Generate operands for max_of_two."""
    a = random.randint(1, 100)
    b = random.randint(1, 100)
    return [a, b], max(a, b)


def generate_operands_abs_diff() -> tuple[list[int], int]:
    """Generate operands for abs_diff."""
    a = random.randint(1, 100)
    b = random.randint(1, 100)
    return [a, b], abs(a - b)


def generate_operands_sum_even() -> tuple[list[int], int]:
    """Generate operands for sum_even."""
    n = random.randint(5, 50)
    # Sum of even numbers from 1 to n: 2 + 4 + ... = 2*(1 + 2 + ... + n//2)
    k = n // 2
    expected = k * (k + 1)  # 2 * (1+2+...+k) = 2 * k*(k+1)/2 = k*(k+1)
    return [n], expected


OPERAND_GENERATORS: dict[str, Callable[[], tuple[list[int], int]]] = {
    "sum_1_to_n": generate_operands_sum_1_to_n,
    "sum_a_to_b": generate_operands_sum_a_to_b,
    "factorial": generate_operands_factorial,
    "power": generate_operands_power,
    "collatz_length": generate_operands_collatz,
    "max_of_two": generate_operands_max_of_two,
    "abs_diff": generate_operands_abs_diff,
    "sum_even": generate_operands_sum_even,
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TrainingExample:
    """A single training example."""
    nl_input: str           # Natural language input
    program_name: str       # Which program this maps to
    ir_opcodes: list[int]   # The IR opcode sequence
    operands: list[int]     # Extracted operands
    expected_result: int    # Expected computation result

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainingDataset:
    """Collection of training examples."""
    examples: list[TrainingExample]
    programs: list[str]

    def to_dict(self) -> dict:
        return {
            "num_examples": len(self.examples),
            "programs": self.programs,
            "examples": [e.to_dict() for e in self.examples],
        }

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "TrainingDataset":
        with open(path) as f:
            data = json.load(f)
        examples = [TrainingExample(**e) for e in data["examples"]]
        return cls(examples=examples, programs=data["programs"])


# =============================================================================
# GENERATOR
# =============================================================================

PROGRAMS = {
    "sum_1_to_n": SUM_1_TO_N,
    "sum_a_to_b": SUM_A_TO_B,
    "factorial": FACTORIAL,
    "power": POWER,
    "collatz_length": COLLATZ_LENGTH,
    "max_of_two": MAX_OF_TWO,
    "abs_diff": ABS_DIFF,
    "sum_even": SUM_EVEN,
}


def generate_example(program_name: str) -> TrainingExample:
    """Generate a single training example for a program."""
    program = PROGRAMS[program_name]
    templates = NL_TEMPLATES[program_name]
    generator = OPERAND_GENERATORS[program_name]

    # Generate operands
    operands, expected = generator()

    # Pick a random template
    template = random.choice(templates)

    # Fill in operands
    format_dict = {str(i): operands[i] for i in range(len(operands))}
    # Special case for factorial template
    if "{0_minus_1}" in template and len(operands) > 0:
        format_dict["0_minus_1"] = operands[0] - 1

    try:
        nl_input = template.format(*operands, **format_dict)
    except (IndexError, KeyError):
        nl_input = template.format(*operands)

    return TrainingExample(
        nl_input=nl_input,
        program_name=program_name,
        ir_opcodes=program.opcodes,
        operands=operands,
        expected_result=expected,
    )


def generate_dataset(
    programs: list[str],
    examples_per_program: int = 100,
    seed: int = 42,
) -> TrainingDataset:
    """Generate a training dataset."""
    random.seed(seed)

    examples = []
    for program_name in programs:
        for _ in range(examples_per_program):
            example = generate_example(program_name)
            examples.append(example)

    # Shuffle
    random.shuffle(examples)

    return TrainingDataset(examples=examples, programs=programs)


def validate_dataset(dataset: TrainingDataset) -> dict:
    """Validate that all examples execute correctly."""
    runtime = WASMRuntime(use_native=True)

    results = {"total": 0, "passed": 0, "failed": 0, "errors": []}

    for example in dataset.examples:
        results["total"] += 1

        try:
            program = PROGRAMS[example.program_name]
            wasm = compile_program(program, example.operands)
            result = runtime.execute(wasm)

            if result.success and result.result == example.expected_result:
                results["passed"] += 1
            else:
                results["failed"] += 1
                results["errors"].append({
                    "nl": example.nl_input,
                    "expected": example.expected_result,
                    "actual": result.result,
                    "error": result.error,
                })
        except Exception as e:
            results["failed"] += 1
            results["errors"].append({
                "nl": example.nl_input,
                "exception": str(e),
            })

    results["accuracy"] = results["passed"] / results["total"] if results["total"] > 0 else 0
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("IR Program Synthesis - Training Data Generator")
    print("=" * 60)

    # Define train/test split
    # Include IF/ELSE programs: max_of_two, abs_diff, sum_even
    # sum_even has loop + if/else like Collatz
    train_programs = [
        "sum_1_to_n", "sum_a_to_b", "factorial", "power",  # simple loops
        "max_of_two", "abs_diff",  # simple if/else
        "sum_even",  # loop + if/else (structural match to Collatz!)
    ]
    test_programs = ["collatz_length"]  # Held out for generalization

    print(f"\nTrain programs: {train_programs}")
    print(f"Test programs: {test_programs}")

    # Generate training data
    print("\n1. Generating training dataset...")
    train_dataset = generate_dataset(
        programs=train_programs,
        examples_per_program=200,
        seed=42,
    )
    print(f"   Generated {len(train_dataset.examples)} training examples")

    # Generate test data
    print("\n2. Generating test dataset...")
    test_dataset = generate_dataset(
        programs=test_programs,
        examples_per_program=50,
        seed=123,
    )
    print(f"   Generated {len(test_dataset.examples)} test examples")

    # Validate
    print("\n3. Validating training dataset...")
    train_results = validate_dataset(train_dataset)
    print(f"   Accuracy: {train_results['passed']}/{train_results['total']} = {train_results['accuracy']:.1%}")

    print("\n4. Validating test dataset...")
    test_results = validate_dataset(test_dataset)
    print(f"   Accuracy: {test_results['passed']}/{test_results['total']} = {test_results['accuracy']:.1%}")

    # Show examples
    print("\n5. Sample training examples:")
    for example in train_dataset.examples[:5]:
        print(f"   '{example.nl_input}' → {example.program_name}")
        print(f"      operands: {example.operands}, expected: {example.expected_result}")

    print("\n6. Sample test examples (Collatz - held out):")
    for example in test_dataset.examples[:5]:
        print(f"   '{example.nl_input}' → {example.program_name}")
        print(f"      operands: {example.operands}, expected: {example.expected_result}")

    # Save datasets
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    train_path = results_dir / "train_dataset.json"
    test_path = results_dir / "test_dataset.json"

    train_dataset.save(train_path)
    test_dataset.save(test_path)

    print(f"\n7. Saved datasets:")
    print(f"   Train: {train_path}")
    print(f"   Test: {test_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"""
Training data:
  Programs: {train_programs}
  Examples: {len(train_dataset.examples)}
  Accuracy: {train_results['accuracy']:.1%}

Test data (held out for generalization):
  Programs: {test_programs}
  Examples: {len(test_dataset.examples)}
  Accuracy: {test_results['accuracy']:.1%}

Next step: Train IR head on train_dataset.json
Goal: Model learns to emit IR sequences from NL
Test: Can it generalize to Collatz (unseen)?
""")
    print("=" * 60)


if __name__ == "__main__":
    main()
