"""
Training Data Generator for AST-Based IR Synthesis

Generates (NL description, template_id) pairs for training the template classifier.

KEY DIFFERENCE from original:
- Original: (NL, program_id) - fails on held-out programs
- This: (NL, template_id) - generalizes across programs with same structure

The hypothesis: sum_even and collatz share the same template.
If trained on sum_even → LOOP_CONDITIONAL_ACCUMULATE,
the model should generalize to collatz → LOOP_CONDITIONAL_ACCUMULATE.
"""

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, List, Dict, Tuple

from templates import TemplateID, PROGRAM_TO_TEMPLATE, template_name


# =============================================================================
# NL TEMPLATES
# =============================================================================

NL_TEMPLATES: Dict[str, List[str]] = {
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

def generate_operands_sum_1_to_n() -> Tuple[List[int], int]:
    n = random.randint(5, 100)
    expected = n * (n + 1) // 2
    return [n], expected


def generate_operands_sum_a_to_b() -> Tuple[List[int], int]:
    a = random.randint(1, 50)
    b = random.randint(a + 5, a + 100)
    expected = (b - a + 1) * (a + b) // 2
    return [a, b], expected


def generate_operands_factorial() -> Tuple[List[int], int]:
    n = random.randint(3, 12)
    expected = 1
    for i in range(2, n + 1):
        expected *= i
    return [n], expected


def generate_operands_power() -> Tuple[List[int], int]:
    base = random.randint(2, 5)
    exp = random.randint(2, 15)
    expected = base ** exp
    return [base, exp], expected


def generate_operands_collatz() -> Tuple[List[int], int]:
    n = random.randint(10, 1000)
    count = 0
    x = n
    while x > 1:
        if x % 2 == 0:
            x = x // 2
        else:
            x = 3 * x + 1
        count += 1
    return [n], count


def generate_operands_max_of_two() -> Tuple[List[int], int]:
    a = random.randint(1, 100)
    b = random.randint(1, 100)
    return [a, b], max(a, b)


def generate_operands_abs_diff() -> Tuple[List[int], int]:
    a = random.randint(1, 100)
    b = random.randint(1, 100)
    return [a, b], abs(a - b)


def generate_operands_sum_even() -> Tuple[List[int], int]:
    n = random.randint(5, 50)
    k = n // 2
    expected = k * (k + 1)
    return [n], expected


OPERAND_GENERATORS: Dict[str, Callable[[], Tuple[List[int], int]]] = {
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
    nl_input: str
    program_name: str
    template_id: int       # This is the KEY change - label with template
    template_name: str
    operands: List[int]
    expected_result: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainingDataset:
    """Collection of training examples."""
    examples: List[TrainingExample]
    programs: List[str]
    split: str

    def to_dict(self) -> dict:
        return {
            "split": self.split,
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
        return cls(
            examples=examples,
            programs=data["programs"],
            split=data.get("split", "unknown")
        )


# =============================================================================
# GENERATOR
# =============================================================================

def generate_example(program_name: str) -> TrainingExample:
    """Generate a single training example."""
    templates = NL_TEMPLATES[program_name]
    generator = OPERAND_GENERATORS[program_name]

    operands, expected = generator()
    template = random.choice(templates)

    try:
        nl_input = template.format(*operands)
    except (IndexError, KeyError):
        nl_input = template

    template_id = PROGRAM_TO_TEMPLATE[program_name]

    return TrainingExample(
        nl_input=nl_input,
        program_name=program_name,
        template_id=int(template_id),
        template_name=template_name(template_id),
        operands=operands,
        expected_result=expected,
    )


def generate_dataset(
    programs: List[str],
    examples_per_program: int = 100,
    seed: int = 42,
    split: str = "train",
) -> TrainingDataset:
    """Generate a training dataset."""
    random.seed(seed)

    examples = []
    for program_name in programs:
        for _ in range(examples_per_program):
            example = generate_example(program_name)
            examples.append(example)

    random.shuffle(examples)

    return TrainingDataset(
        examples=examples,
        programs=programs,
        split=split,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("AST-Based IR Synthesis - Training Data Generator")
    print("=" * 60)

    # KEY: Define train/test split based on TEMPLATES, not programs
    # Train: programs that give us examples of all templates EXCEPT collatz
    # Test: collatz (same template as sum_even)
    train_programs = [
        "sum_1_to_n", "sum_a_to_b", "factorial", "power",  # LOOP_ACCUMULATE
        "max_of_two", "abs_diff",  # IF_BRANCH
        "sum_even",  # LOOP_CONDITIONAL_ACCUMULATE (structural match to Collatz!)
    ]
    test_programs = ["collatz_length"]  # LOOP_CONDITIONAL_ACCUMULATE

    print(f"\nTrain programs: {train_programs}")
    print(f"Test programs: {test_programs}")

    # Show template mapping
    print("\nTemplate mapping:")
    for prog in train_programs + test_programs:
        tid = PROGRAM_TO_TEMPLATE[prog]
        print(f"  {prog} → {template_name(tid)} ({tid})")

    # Generate training data
    print("\n1. Generating training dataset...")
    train_dataset = generate_dataset(
        programs=train_programs,
        examples_per_program=200,
        seed=42,
        split="train",
    )
    print(f"   Generated {len(train_dataset.examples)} training examples")

    # Generate test data
    print("\n2. Generating test dataset...")
    test_dataset = generate_dataset(
        programs=test_programs,
        examples_per_program=50,
        seed=123,
        split="test",
    )
    print(f"   Generated {len(test_dataset.examples)} test examples")

    # Show template distribution
    print("\n3. Template distribution:")
    for dataset, name in [(train_dataset, "Train"), (test_dataset, "Test")]:
        counts = {}
        for ex in dataset.examples:
            tid = ex.template_id
            counts[tid] = counts.get(tid, 0) + 1
        print(f"   {name}:")
        for tid, count in sorted(counts.items()):
            print(f"     {template_name(TemplateID(tid))}: {count}")

    # Show examples
    print("\n4. Sample training examples:")
    for example in train_dataset.examples[:5]:
        print(f"   '{example.nl_input}'")
        print(f"      → {example.template_name} (from {example.program_name})")

    print("\n5. Sample test examples (Collatz - held out):")
    for example in test_dataset.examples[:5]:
        print(f"   '{example.nl_input}'")
        print(f"      → {example.template_name} (from {example.program_name})")

    # Save datasets
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    train_path = results_dir / "train_dataset.json"
    test_path = results_dir / "test_dataset.json"

    train_dataset.save(train_path)
    test_dataset.save(test_path)

    print(f"\n6. Saved datasets:")
    print(f"   Train: {train_path}")
    print(f"   Test: {test_path}")

    # Summary
    print("\n" + "=" * 60)
    print("KEY INSIGHT")
    print("=" * 60)
    print("""
The hypothesis:
  - sum_even uses LOOP_CONDITIONAL_ACCUMULATE template
  - collatz_length uses LOOP_CONDITIONAL_ACCUMULATE template
  - If model learns sum_even → LOOP_CONDITIONAL_ACCUMULATE
  - Then it SHOULD generalize to collatz → LOOP_CONDITIONAL_ACCUMULATE

This is compositional generalization via structural abstraction!
""")
    print("=" * 60)


if __name__ == "__main__":
    main()
