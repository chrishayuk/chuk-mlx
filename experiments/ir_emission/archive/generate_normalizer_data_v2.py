#!/usr/bin/env python3
"""
Generate diverse training data for NL → Canonical normalizer (v2).

Much more varied templates to improve generalization.
"""

import json
import random
from pathlib import Path

# Canonical output format - ALWAYS this exact form
CANONICAL_FORMAT = {
    "add": "{a} + {b} = ",
    "sub": "{a} - {b} = ",
    "mul": "{a} * {b} = ",
    "div": "{a} / {b} = ",
}

# ============================================================================
# SIMPLE TEMPLATES - Direct expressions
# ============================================================================
SIMPLE_TEMPLATES = {
    "add": [
        "Add {a} and {b}",
        "Add {a} to {b}",
        "{a} plus {b}",
        "{a} and {b} added together",
        "The sum of {a} and {b}",
        "The sum of {a} and {b} is",
        "What is {a} plus {b}",
        "What is {a} plus {b}?",
        "What's {a} plus {b}?",
        "Calculate {a} + {b}",
        "Compute {a} + {b}",
        "Find {a} + {b}",
        "{a} added to {b}",
        "{a} increased by {b}",
        "The total of {a} and {b}",
        "Combine {a} and {b}",
        "{a} + {b}",
        "{a}+{b}",
        "sum of {a} {b}",
        "add together {a} and {b}",
    ],
    "sub": [
        "Subtract {b} from {a}",
        "{a} minus {b}",
        "{a} take away {b}",
        "The difference of {a} and {b}",
        "The difference of {a} and {b} is",
        "The difference between {a} and {b}",
        "What is {a} minus {b}",
        "What is {a} minus {b}?",
        "What's {a} minus {b}?",
        "Calculate {a} - {b}",
        "Compute {a} - {b}",
        "Find {a} - {b}",
        "{a} decreased by {b}",
        "{a} reduced by {b}",
        "{a} less {b}",
        "From {a} subtract {b}",
        "Remove {b} from {a}",
        "{a} - {b}",
        "{a}-{b}",
        "difference of {a} {b}",
    ],
    "mul": [
        "Multiply {a} by {b}",
        "{a} times {b}",
        "{a} multiplied by {b}",
        "The product of {a} and {b}",
        "The product of {a} and {b} is",
        "What is {a} times {b}",
        "What is {a} times {b}?",
        "What's {a} times {b}?",
        "Calculate {a} * {b}",
        "Calculate {a} x {b}",
        "Compute {a} * {b}",
        "Find {a} * {b}",
        "{a} by {b}",
        "{a} × {b}",
        "{a} x {b}",
        "{a} * {b}",
        "{a}*{b}",
        "product of {a} {b}",
        "{a} groups of {b}",
        "{b} groups of {a}",
    ],
    "div": [
        "Divide {a} by {b}",
        "{a} divided by {b}",
        "{a} over {b}",
        "The quotient of {a} and {b}",
        "The quotient of {a} and {b} is",
        "What is {a} divided by {b}",
        "What is {a} divided by {b}?",
        "What's {a} divided by {b}?",
        "Calculate {a} / {b}",
        "Calculate {a} ÷ {b}",
        "Compute {a} / {b}",
        "Find {a} / {b}",
        "{a} split by {b}",
        "{a} / {b}",
        "{a}/{b}",
        "{a} ÷ {b}",
        "quotient of {a} {b}",
        "How many times does {b} go into {a}",
        "How many times does {b} go into {a}?",
        "{a} into {b} parts",
    ],
}

# ============================================================================
# WORD PROBLEM TEMPLATES - Story-based
# ============================================================================
WORD_PROBLEMS = {
    "add": [
        # Possession
        "I have {a} apples. I get {b} more. How many do I have?",
        "Janet has {a} apples. She buys {b} more. How many does she have?",
        "Tom has {a} dollars. He earns {b} more. How much does he have?",
        "Sarah has {a} coins. She finds {b} more. How many coins does she have?",
        # Combining
        "There are {a} boys and {b} girls. How many children total?",
        "A store sold {a} items in the morning and {b} in the afternoon. Total?",
        "Team A scored {a} points. Team B scored {b}. Total points?",
        "{a} red balls and {b} blue balls. How many balls?",
        # Movement
        "Tom walked {a} miles yesterday and {b} miles today. Total distance?",
        "A car traveled {a} km then {b} km more. How far did it go?",
        # Time
        "I worked {a} hours Monday and {b} hours Tuesday. Total hours?",
        "She slept {a} hours at night and {b} hours napping. Total sleep?",
        # Money
        "I spent {a} dollars on food and {b} on drinks. Total spent?",
        "The shirt costs {a} dollars and pants cost {b}. Total cost?",
        # Misc
        "There are {a} cats and {b} dogs. How many pets?",
        "{a} students in class A and {b} in class B. How many students?",
    ],
    "sub": [
        # Possession loss
        "I have {a} apples. I give away {b}. How many remain?",
        "Janet has {a} apples. She eats {b}. How many are left?",
        "Tom has {a} dollars. He spends {b}. How much remains?",
        "Sarah has {a} coins. She loses {b}. How many does she have?",
        # Removal
        "There are {a} birds. {b} fly away. How many are left?",
        "A tank has {a} gallons. {b} leak out. How much remains?",
        "{a} cookies on the plate. {b} are eaten. How many left?",
        "{a} people in line. {b} leave. How many remain?",
        # Comparison
        "Tom is {a} years old. Jane is {b}. How much older is Tom?",
        "Building A is {a} meters tall. Building B is {b}. Difference?",
        "I have {a} dollars. You have {b}. How much more do I have?",
        # Temperature
        "The temperature was {a} degrees. It dropped {b}. What is it now?",
        "It was {a} degrees. It cooled by {b}. New temperature?",
        # Distance
        "I need to walk {a} miles. I've walked {b}. How far to go?",
        "The journey is {a} km. We've traveled {b}. How much left?",
        # Misc
        "{a} pages in the book. I read {b}. Pages remaining?",
    ],
    "mul": [
        # Repeated groups
        "Each box has {a} items. How many in {b} boxes?",
        "Each bag contains {a} apples. How many in {b} bags?",
        "Each row has {a} seats. How many seats in {b} rows?",
        "{a} cookies per plate. How many on {b} plates?",
        # Rate × time
        "A car goes {a} mph. How far in {b} hours?",
        "She types {a} words per minute. How many in {b} minutes?",
        "He runs {a} laps per hour. How many in {b} hours?",
        "The machine makes {a} parts per hour. How many in {b} hours?",
        # Price × quantity
        "Tickets cost {a} dollars each. Cost for {b} tickets?",
        "Apples are {a} cents each. Cost of {b} apples?",
        "Each book costs {a} dollars. Price of {b} books?",
        "Pens cost {a} dollars each. How much for {b} pens?",
        # Daily/weekly
        "Janet's ducks lay {a} eggs daily. How many in {b} days?",
        "He earns {a} dollars per day. Earnings in {b} days?",
        "She saves {a} dollars weekly. Savings in {b} weeks?",
        # Misc
        "{a} students per class. How many in {b} classes?",
    ],
    "div": [
        # Equal sharing
        "{a} cookies shared among {b} friends. How many each?",
        "{a} dollars split between {b} people. How much each?",
        "{a} candies divided among {b} children. How many each?",
        "Share {a} apples equally among {b} people. How many each?",
        # Grouping
        "{a} students in groups of {b}. How many groups?",
        "{a} eggs in cartons of {b}. How many cartons?",
        "{a} items packed in boxes of {b}. How many boxes?",
        "Pack {a} books into boxes of {b}. How many boxes?",
        # Rate
        "Drive {a} miles in {b} hours. Speed?",
        "Travel {a} km in {b} hours. Speed in km/h?",
        "Complete {a} tasks in {b} hours. Tasks per hour?",
        "Read {a} pages in {b} hours. Pages per hour?",
        # Price per unit
        "{a} dollars for {b} items. Price per item?",
        "Paid {a} dollars for {b} kg. Price per kg?",
        "{a} cents for {b} candies. Cost per candy?",
        # Misc
        "A {a} page book in {b} days. Pages per day?",
    ],
}

# ============================================================================
# QUESTION FORMS - Interrogative variations
# ============================================================================
QUESTION_FORMS = {
    "add": [
        "What is {a} plus {b}?",
        "What do you get when you add {a} and {b}?",
        "What's the sum of {a} and {b}?",
        "How much is {a} plus {b}?",
        "What does {a} plus {b} equal?",
        "If you add {a} and {b}, what do you get?",
        "What's {a} and {b} together?",
        "What is the total of {a} and {b}?",
    ],
    "sub": [
        "What is {a} minus {b}?",
        "What do you get when you subtract {b} from {a}?",
        "What's the difference between {a} and {b}?",
        "How much is {a} minus {b}?",
        "What does {a} minus {b} equal?",
        "If you take {b} from {a}, what remains?",
        "What's {a} take away {b}?",
        "What is {a} less {b}?",
    ],
    "mul": [
        "What is {a} times {b}?",
        "What do you get when you multiply {a} by {b}?",
        "What's the product of {a} and {b}?",
        "How much is {a} times {b}?",
        "What does {a} times {b} equal?",
        "If you multiply {a} and {b}, what do you get?",
        "What's {a} multiplied by {b}?",
        "What is {a} by {b}?",
    ],
    "div": [
        "What is {a} divided by {b}?",
        "What do you get when you divide {a} by {b}?",
        "What's the quotient of {a} and {b}?",
        "How much is {a} divided by {b}?",
        "What does {a} divided by {b} equal?",
        "If you divide {a} by {b}, what do you get?",
        "What's {a} over {b}?",
        "What is {a} split into {b}?",
    ],
}

# ============================================================================
# IMPERATIVE FORMS - Commands
# ============================================================================
IMPERATIVE_FORMS = {
    "add": [
        "Find {a} plus {b}.",
        "Calculate {a} + {b}.",
        "Compute the sum of {a} and {b}.",
        "Add {a} and {b} together.",
        "Work out {a} plus {b}.",
        "Determine {a} + {b}.",
        "Figure out {a} plus {b}.",
        "Solve {a} + {b}.",
    ],
    "sub": [
        "Find {a} minus {b}.",
        "Calculate {a} - {b}.",
        "Compute the difference of {a} and {b}.",
        "Subtract {b} from {a}.",
        "Work out {a} minus {b}.",
        "Determine {a} - {b}.",
        "Figure out {a} minus {b}.",
        "Solve {a} - {b}.",
    ],
    "mul": [
        "Find {a} times {b}.",
        "Calculate {a} * {b}.",
        "Compute the product of {a} and {b}.",
        "Multiply {a} by {b}.",
        "Work out {a} times {b}.",
        "Determine {a} * {b}.",
        "Figure out {a} times {b}.",
        "Solve {a} * {b}.",
    ],
    "div": [
        "Find {a} divided by {b}.",
        "Calculate {a} / {b}.",
        "Compute the quotient of {a} and {b}.",
        "Divide {a} by {b}.",
        "Work out {a} divided by {b}.",
        "Determine {a} / {b}.",
        "Figure out {a} over {b}.",
        "Solve {a} / {b}.",
    ],
}


def generate_sample(op: str) -> dict:
    """Generate a single NL → Canonical training sample."""
    # Generate operands
    if op == "div":
        b = random.randint(2, 12)
        result = random.randint(1, 20)
        a = b * result
    else:
        a = random.randint(1, 99)
        b = random.randint(1, 99)

    # Pick template type randomly
    # 30% simple, 40% word problem, 15% question, 15% imperative
    r = random.random()
    if r < 0.30:
        templates = SIMPLE_TEMPLATES[op]
        template_type = "simple"
    elif r < 0.70:
        templates = WORD_PROBLEMS[op]
        template_type = "word_problem"
    elif r < 0.85:
        templates = QUESTION_FORMS[op]
        template_type = "question"
    else:
        templates = IMPERATIVE_FORMS[op]
        template_type = "imperative"

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
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--output-dir", default="experiments/ir_emission/data")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ops = ["add", "sub", "mul", "div"]
    samples_per_op = args.num_samples // len(ops)

    all_samples = []
    for op in ops:
        for _ in range(samples_per_op):
            sample = generate_sample(op)
            all_samples.append(sample)

    random.shuffle(all_samples)

    # Split into train/val
    split_idx = int(len(all_samples) * 0.9)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    # Write files
    train_path = output_dir / "normalizer_train_v2.jsonl"
    val_path = output_dir / "normalizer_val_v2.jsonl"

    with open(train_path, "w") as f:
        for sample in train_samples:
            f.write(json.dumps(sample) + "\n")

    with open(val_path, "w") as f:
        for sample in val_samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Generated {len(train_samples)} training samples → {train_path}")
    print(f"Generated {len(val_samples)} validation samples → {val_path}")

    # Stats
    type_counts = {}
    for s in train_samples:
        t = s["template_type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    print("\nTemplate distribution:")
    for t, c in sorted(type_counts.items()):
        print(f"  {t}: {c} ({100 * c / len(train_samples):.1f}%)")

    # Show examples
    print("\nExamples:")
    for op in ops:
        samples = [s for s in train_samples if s["operation"] == op][:3]
        for s in samples:
            print(f"  {s['nl_input'][:50]:50} → {s['canonical_output']}")


if __name__ == "__main__":
    main()
