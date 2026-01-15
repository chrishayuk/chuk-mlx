#!/usr/bin/env python3
"""
Generate training data for NL → Canonical normalizer.

The insight: CoT is format normalization, not reasoning.
We train the model to rewrite varied NL into canonical "a op b = " form.
"""

import json
import random
from pathlib import Path

# Templates for varied NL expressions
NL_TEMPLATES = {
    "add": [
        "Add {a} and {b}",
        "The sum of {a} and {b} is",
        "What is {a} plus {b}?",
        "{a} added to {b} equals",
        "Calculate {a} + {b}",
        "Find the total of {a} and {b}",
        "Combine {a} with {b}",
        "If you have {a} and get {b} more, you have",
        "{a} increased by {b} is",
        "The result of adding {a} to {b} is",
    ],
    "sub": [
        "Subtract {b} from {a}",
        "The difference of {a} and {b} is",
        "What is {a} minus {b}?",
        "{a} take away {b} equals",
        "Calculate {a} - {b}",
        "Find {a} decreased by {b}",
        "{a} reduced by {b} is",
        "If you have {a} and lose {b}, you have",
        "Remove {b} from {a}",
        "The result of subtracting {b} from {a} is",
    ],
    "mul": [
        "Multiply {a} by {b}",
        "The product of {a} and {b} is",
        "What is {a} times {b}?",
        "{a} multiplied by {b} equals",
        "Calculate {a} * {b}",
        "Find {a} groups of {b}",
        "{a} times {b} gives",
        "If you have {a} sets of {b}, you have",
        "The result of multiplying {a} by {b} is",
        "{a} by {b} equals",
    ],
    "div": [
        "Divide {a} by {b}",
        "The quotient of {a} and {b} is",
        "What is {a} divided by {b}?",
        "{a} split into {b} parts gives",
        "Calculate {a} / {b}",
        "Find {a} shared among {b}",
        "{a} over {b} is",
        "If you split {a} into {b} equal parts, each is",
        "The result of dividing {a} by {b} is",
        "How many times does {b} go into {a}?",
    ],
}

# Canonical output format
CANONICAL_FORMAT = {
    "add": "{a} + {b} = ",
    "sub": "{a} - {b} = ",
    "mul": "{a} * {b} = ",
    "div": "{a} / {b} = ",
}

# Word problem templates (more complex NL)
# These MUST also map to clean canonical forms like "a op b = "
WORD_PROBLEMS = {
    "add": [
        "Janet has {a} apples. She buys {b} more. How many does she have?",
        "A store sold {a} items in the morning and {b} in the afternoon. Total sales?",
        "Tom walked {a} miles yesterday and {b} miles today. How far did he walk?",
        "There are {a} students in one class and {b} in another. How many total?",
        "Sarah has {a} dollars. She earns {b} more. How much does she have now?",
    ],
    "sub": [
        "Janet has {a} apples. She gives away {b}. How many remain?",
        "A tank holds {a} gallons. {b} gallons leak out. How much is left?",
        "Tom had {a} dollars. He spent {b}. How much remains?",
        "There were {a} birds. {b} flew away. How many are left?",
        "The temperature was {a} degrees. It dropped {b} degrees. What is it now?",
    ],
    "mul": [
        "Janet's ducks lay {a} eggs daily. How many eggs in {b} days?",
        "Each box holds {a} items. How many in {b} boxes?",
        "A car travels {a} miles per hour. How far in {b} hours?",
        "Each student needs {a} pencils. How many for {b} students?",
        "Tickets cost {a} dollars each. Cost for {b} tickets?",
    ],
    "div": [
        "Janet has {a} cookies to share among {b} friends. How many each?",
        "{a} students split into {b} equal groups. How many per group?",
        "A {a} mile journey in {b} hours. What speed?",
        "{a} items packed in boxes of {b}. How many boxes?",
        "Divide {a} dollars among {b} people. How much each?",
    ],
}

# Additional question forms that should also normalize
QUESTION_TEMPLATES = {
    "add": [
        "What is {a} plus {b}?",
        "What do you get when you add {a} to {b}?",
    ],
    "sub": [
        "What is {a} minus {b}?",
        "What do you get when you subtract {b} from {a}?",
    ],
    "mul": [
        "What is {a} times {b}?",
        "What is {a} multiplied by {b}?",
    ],
    "div": [
        "What is {a} divided by {b}?",
        "What do you get when you divide {a} by {b}?",
    ],
}


def generate_sample(op: str, template_type: str = "simple") -> dict:
    """Generate a single NL → Canonical training sample.

    template_type: "simple", "word_problem", or "question"
    """
    # Generate operands
    if op == "div":
        # Ensure clean division
        b = random.randint(1, 12)
        result = random.randint(1, 20)
        a = b * result
    else:
        a = random.randint(1, 99)
        b = random.randint(1, 99)

    # Pick template based on type
    if template_type == "word_problem":
        templates = WORD_PROBLEMS[op]
    elif template_type == "question":
        templates = QUESTION_TEMPLATES[op]
    else:
        templates = NL_TEMPLATES[op]

    template = random.choice(templates)
    nl_input = template.format(a=a, b=b)
    canonical_output = CANONICAL_FORMAT[op].format(a=a, b=b)

    # Calculate expected result
    if op == "add":
        expected = a + b
    elif op == "sub":
        expected = a - b
    elif op == "mul":
        expected = a * b
    else:
        expected = a // b

    return {
        "nl_input": nl_input,
        "canonical_output": canonical_output,
        "operation": op,
        "operands": [a, b],
        "expected_result": expected,
        "template_type": template_type,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--word-problem-ratio", type=float, default=0.3)
    parser.add_argument("--output-dir", default="experiments/ir_emission/data")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ops = ["add", "sub", "mul", "div"]
    samples_per_op = args.num_samples // len(ops)

    all_samples = []
    for op in ops:
        for i in range(samples_per_op):
            # Mix of template types: 40% simple, 40% word_problem, 20% question
            r = random.random()
            if r < 0.4:
                template_type = "simple"
            elif r < 0.8:
                template_type = "word_problem"
            else:
                template_type = "question"
            sample = generate_sample(op, template_type)
            all_samples.append(sample)

    random.shuffle(all_samples)

    # Split into train/val
    split_idx = int(len(all_samples) * 0.9)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    # Write files
    train_path = output_dir / "normalizer_train.jsonl"
    val_path = output_dir / "normalizer_val.jsonl"

    with open(train_path, "w") as f:
        for sample in train_samples:
            f.write(json.dumps(sample) + "\n")

    with open(val_path, "w") as f:
        for sample in val_samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Generated {len(train_samples)} training samples → {train_path}")
    print(f"Generated {len(val_samples)} validation samples → {val_path}")

    # Show examples
    print("\nExamples:")
    for op in ops:
        samples = [s for s in train_samples if s["operation"] == op][:2]
        for s in samples:
            print(f"  {s['nl_input'][:50]:50} → {s['canonical_output']}")


if __name__ == "__main__":
    main()
