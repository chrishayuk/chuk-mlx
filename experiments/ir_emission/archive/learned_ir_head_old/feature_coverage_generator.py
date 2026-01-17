"""
Feature Coverage Generator

Generate synthetic data that covers feature→role mappings.
The hypothesis: features (not semantics) determine roles.
"""

import random
import json
from pathlib import Path
from collections import Counter

import functools
print = functools.partial(print, flush=True)


# =============================================================================
# VOCABULARY
# =============================================================================

NAMES = ["Tom", "Sara", "Mike", "Emma", "John", "Lisa", "David", "Amy",
         "Chris", "Rachel", "James", "Maria", "Alex", "Sophie", "Ben"]

ITEMS = ["apples", "oranges", "books", "pencils", "cookies", "eggs",
         "stickers", "marbles", "candies", "toys", "cards", "flowers",
         "bottles", "boxes", "bags", "pens", "shirts", "coins"]

CONTAINERS = ["boxes", "bags", "baskets", "crates", "packs", "cartons", "jars"]

GROUPS = ["kids", "friends", "people", "students", "teams", "groups", "classes"]


# =============================================================================
# TEMPLATES BY FEATURE PATTERN
# =============================================================================

TEMPLATES = {
    # =========================================================================
    # MULTIPLY patterns - "each/per" signals MUL_RIGHT
    # =========================================================================

    "mul_each": {
        "templates": [
            "{person} buys {n1} items at ${n2} each.",
            "{person} has {n1} {containers} with {n2} {items} each.",
            "There are {n1} rows with {n2} chairs each.",
            "{person} packs {n1} boxes with {n2} {items} each.",
            "{n1} {containers} contain {n2} {items} each.",
            "{person} gives {n2} {items} to each of {n1} friends.",
        ],
        "roles": {"n1": "MUL_LEFT", "n2": "MUL_RIGHT"},
        "operation": "MUL",
    },

    "mul_per": {
        "templates": [
            "{person} earns ${n2} per hour for {n1} hours.",
            "{person} drives {n1} miles at {n2} miles per hour.",
            "{person} reads {n2} pages per day for {n1} days.",
            "The machine produces {n2} items per minute for {n1} minutes.",
            "{person} saves ${n2} per week for {n1} weeks.",
            "{person} walks {n2} miles per day for {n1} days.",
        ],
        "roles": {"n1": "MUL_LEFT", "n2": "MUL_RIGHT"},
        "operation": "MUL",
    },

    "mul_times": {
        "templates": [
            "{person} does {n1} pushups {n2} times a day.",
            "{person} visits the store {n2} times, buying {n1} items each time.",
            "The bell rings {n2} times every {n1} hours.",
            "{person} waters plants {n2} times using {n1} liters each time.",
        ],
        "roles": {"n1": "MUL_LEFT", "n2": "MUL_RIGHT"},
        "operation": "MUL",
    },

    # =========================================================================
    # SUBTRACT patterns - consumption/loss verbs signal SUB_RIGHT
    # =========================================================================

    "sub_eats": {
        "templates": [
            "{person} has {n1} {items}. {person} eats {n2}.",
            "There are {n1} {items} on the plate. {person} eats {n2} of them.",
            "{person} baked {n1} cookies and ate {n2}.",
            "The jar has {n1} candies. {person} eats {n2} candies.",
        ],
        "roles": {"n1": "SUB_LEFT", "n2": "SUB_RIGHT"},
        "operation": "SUB",
    },

    "sub_spends": {
        "templates": [
            "{person} has ${n1}. {person} spends ${n2}.",
            "{person} saved ${n1} and spent ${n2} on a gift.",
            "The budget is ${n1}. {person} spends ${n2}.",
            "{person} had ${n1} but spent ${n2} at the store.",
        ],
        "roles": {"n1": "SUB_LEFT", "n2": "SUB_RIGHT"},
        "operation": "SUB",
    },

    "sub_gives": {
        "templates": [
            "{person} has {n1} {items}. {person} gives {n2} to a friend.",
            "There are {n1} {items}. {person} gives away {n2}.",
            "{person} collected {n1} stickers and gave {n2} to {other}.",
            "{person} had {n1} {items} and donated {n2}.",
        ],
        "roles": {"n1": "SUB_LEFT", "n2": "SUB_RIGHT"},
        "operation": "SUB",
    },

    "sub_uses": {
        "templates": [
            "{person} has {n1} {items}. {person} uses {n2} for baking.",
            "There are {n1} eggs. {person} uses {n2} for breakfast.",
            "{person} bought {n1} supplies and used {n2}.",
            "The box has {n1} {items}. {n2} are used.",
        ],
        "roles": {"n1": "SUB_LEFT", "n2": "SUB_RIGHT"},
        "operation": "SUB",
    },

    "sub_loses": {
        "templates": [
            "{person} has {n1} {items}. {person} loses {n2}.",
            "There were {n1} marbles. {person} lost {n2} of them.",
            "{person} had {n1} balloons but {n2} flew away.",
            "{person} started with {n1} points and lost {n2}.",
        ],
        "roles": {"n1": "SUB_LEFT", "n2": "SUB_RIGHT"},
        "operation": "SUB",
    },

    "sub_takes": {
        "templates": [
            "{person} has {n1} {items}. {other} takes {n2}.",
            "There are {n1} {items}. {person} takes {n2} away.",
            "The pile has {n1} {items}. {n2} are taken.",
            "{person} picked {n1} flowers. {other} took {n2}.",
        ],
        "roles": {"n1": "SUB_LEFT", "n2": "SUB_RIGHT"},
        "operation": "SUB",
    },

    # =========================================================================
    # DIVIDE patterns - "among/between/split/share" signals DIV_RIGHT
    # =========================================================================

    "div_among": {
        "templates": [
            "{n1} {items} are shared among {n2} {groups}.",
            "{person} divides {n1} candies among {n2} kids.",
            "There are {n1} {items} split among {n2} people.",
            "{person} distributes {n1} prizes among {n2} winners.",
        ],
        "roles": {"n1": "DIV_LEFT", "n2": "DIV_RIGHT"},
        "operation": "DIV",
    },

    "div_between": {
        "templates": [
            "{person} splits ${n1} between {n2} friends.",
            "{n1} tasks are divided between {n2} teams.",
            "The {n1} {items} are shared between {n2} people.",
            "{person} divides {n1} hours between {n2} projects.",
        ],
        "roles": {"n1": "DIV_LEFT", "n2": "DIV_RIGHT"},
        "operation": "DIV",
    },

    "div_equal": {
        "templates": [
            "{n1} {items} are split into {n2} equal groups.",
            "{person} makes {n2} equal piles from {n1} {items}.",
            "There are {n1} students in {n2} equal rows.",
            "{person} cuts the {n1}-inch rope into {n2} equal pieces.",
        ],
        "roles": {"n1": "DIV_LEFT", "n2": "DIV_RIGHT"},
        "operation": "DIV",
    },

    # =========================================================================
    # ADD patterns - "more/additional/finds/gets" signals ADD_RIGHT
    # =========================================================================

    "add_more": {
        "templates": [
            "{person} has {n1} {items}. {person} gets {n2} more.",
            "There are {n1} {items}. {n2} more are added.",
            "{person} found {n1} coins and then {n2} more.",
            "{person} collected {n1} stickers and got {n2} more.",
        ],
        "roles": {"n1": "ADD_LEFT", "n2": "ADD_RIGHT"},
        "operation": "ADD",
    },

    "add_finds": {
        "templates": [
            "{person} has {n1} {items}. {person} finds {n2}.",
            "There are {n1} {items}. {person} finds {n2} additional ones.",
            "{person} owned {n1} {items} and found {n2} more.",
            "The box has {n1} {items}. {person} adds {n2}.",
        ],
        "roles": {"n1": "ADD_LEFT", "n2": "ADD_RIGHT"},
        "operation": "ADD",
    },

    "add_receives": {
        "templates": [
            "{person} has {n1} {items}. {person} receives {n2} from {other}.",
            "There are {n1} {items}. {n2} more arrive.",
            "{person} had {n1} and was given {n2}.",
            "{person} started with {n1} points and earned {n2} more.",
        ],
        "roles": {"n1": "ADD_LEFT", "n2": "ADD_RIGHT"},
        "operation": "ADD",
    },
}

# Questions by operation result
QUESTIONS = {
    "MUL": [
        "How many in total?",
        "What is the total?",
        "How many altogether?",
        "How much does {person} have?",
    ],
    "SUB": [
        "How many are left?",
        "How many remain?",
        "How many does {person} have now?",
        "What is left?",
    ],
    "DIV": [
        "How many does each get?",
        "How many in each group?",
        "How many per person?",
        "What is each share?",
    ],
    "ADD": [
        "How many in total?",
        "How many does {person} have now?",
        "What is the total?",
        "How many altogether?",
    ],
}


# =============================================================================
# GENERATOR
# =============================================================================

def generate_example(template_key: str) -> dict:
    """Generate a single training example."""
    config = TEMPLATES[template_key]
    template = random.choice(config["templates"])
    roles = config["roles"]
    operation = config["operation"]

    # Generate numbers
    if operation == "DIV":
        # Ensure divisibility
        n2 = random.randint(2, 10)
        n1 = n2 * random.randint(2, 15)
    elif operation == "SUB":
        # Ensure n1 > n2
        n1 = random.randint(10, 100)
        n2 = random.randint(1, n1 - 1)
    else:
        n1 = random.randint(2, 50)
        n2 = random.randint(2, 20)

    # Fill template
    person = random.choice(NAMES)
    other = random.choice([n for n in NAMES if n != person])
    items = random.choice(ITEMS)
    containers = random.choice(CONTAINERS)
    groups = random.choice(GROUPS)

    problem = template.format(
        n1=n1, n2=n2,
        person=person, other=other,
        items=items, containers=containers, groups=groups
    )

    # Add question
    question = random.choice(QUESTIONS[operation]).format(person=person)
    full_problem = f"{problem} {question}"

    # Compute answer
    if operation == "MUL":
        answer = n1 * n2
    elif operation == "SUB":
        answer = n1 - n2
    elif operation == "DIV":
        answer = n1 // n2
    elif operation == "ADD":
        answer = n1 + n2

    # Return with labeled roles
    return {
        "problem": full_problem,
        "answer": answer,
        "numbers": [
            {"value": str(n1), "role": roles["n1"]},
            {"value": str(n2), "role": roles["n2"]},
        ],
        "operation": operation,
        "template_key": template_key,
    }


def generate_dataset(n: int) -> list[dict]:
    """Generate balanced dataset across template types."""
    examples = []
    template_keys = list(TEMPLATES.keys())

    for _ in range(n):
        key = random.choice(template_keys)
        examples.append(generate_example(key))

    return examples


# =============================================================================
# MAIN
# =============================================================================

def main():
    random.seed(42)

    print("=" * 70)
    print("  FEATURE COVERAGE GENERATOR")
    print("  Generating data based on feature→role mappings")
    print("=" * 70)

    # Generate datasets
    train_data = generate_dataset(10000)
    test_data = generate_dataset(2000)

    print(f"\nGenerated: {len(train_data)} train, {len(test_data)} test")

    # Statistics
    op_counts = Counter(ex["operation"] for ex in train_data)
    print("\nOperation distribution:")
    for op, count in op_counts.most_common():
        print(f"  {op}: {count} ({count/len(train_data):.1%})")

    template_counts = Counter(ex["template_key"] for ex in train_data)
    print("\nTemplate distribution:")
    for key, count in template_counts.most_common():
        print(f"  {key}: {count}")

    # Samples
    print("\n" + "=" * 70)
    print("SAMPLE PROBLEMS")
    print("=" * 70)

    for ex in train_data[:10]:
        print(f"\n  Problem: {ex['problem']}")
        print(f"  Answer: {ex['answer']}")
        print(f"  Numbers: {ex['numbers']}")
        print(f"  Operation: {ex['operation']}")

    # Save
    output_dir = Path(__file__).parent / "feature_coverage_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, data in [("train", train_data), ("test", test_data)]:
        path = output_dir / f"{name}.json"
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\nSaved {len(data)} examples to {path}")


if __name__ == "__main__":
    main()
