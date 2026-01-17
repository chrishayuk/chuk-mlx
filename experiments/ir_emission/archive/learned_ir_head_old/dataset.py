"""
IR Dataset Generation.

Creates training data for the learned IR head:
- Varied natural language inputs
- Target IR structure (operation, operand_a, operand_b)
"""

import random
from dataclasses import dataclass

import mlx.core as mx


@dataclass
class IRExample:
    """A single training example."""
    text: str
    op: str
    a: int
    b: int

    @property
    def op_idx(self) -> int:
        """Convert operation to index."""
        return {"add": 0, "subtract": 1, "multiply": 2, "divide": 3}[self.op]


# Templates for generating varied NL
NL_TEMPLATES = {
    "add": [
        "Add {a} and {b}",
        "{a} plus {b}",
        "What is {a} + {b}?",
        "The sum of {a} and {b}",
        "Calculate {a} + {b}",
        "{a} added to {b}",
        "Find the sum of {a} and {b}",
        "Compute {a} plus {b}",
    ],
    "subtract": [
        "Subtract {b} from {a}",
        "{a} minus {b}",
        "What is {a} - {b}?",
        "The difference of {a} and {b}",
        "Calculate {a} - {b}",
        "{b} subtracted from {a}",
        "Find {a} minus {b}",
        "Take {b} away from {a}",
    ],
    "multiply": [
        "Multiply {a} by {b}",
        "{a} times {b}",
        "What is {a} * {b}?",
        "The product of {a} and {b}",
        "Calculate {a} × {b}",
        "{a} multiplied by {b}",
        "Find {a} times {b}",
        "Compute {a} * {b}",
    ],
    "divide": [
        "Divide {a} by {b}",
        "{a} divided by {b}",
        "What is {a} / {b}?",
        "The quotient of {a} and {b}",
        "Calculate {a} ÷ {b}",
        "{a} over {b}",
        "Find {a} divided by {b}",
        "Compute {a} / {b}",
    ],
}

# Word problem templates for more variety
WORD_PROBLEM_TEMPLATES = {
    "add": [
        "Janet has {a} apples. She buys {b} more. How many does she have?",
        "A box contains {a} items. {b} more are added. Total items?",
        "There are {a} birds. {b} more arrive. How many birds now?",
    ],
    "subtract": [
        "Janet has {a} apples. She gives away {b}. How many remain?",
        "A tank has {a} gallons. {b} leak out. How much is left?",
        "There are {a} cookies. {b} are eaten. How many remain?",
    ],
    "multiply": [
        "Each box holds {a} items. How many in {b} boxes?",
        "Tickets cost {a} dollars each. Cost for {b} tickets?",
        "A car travels {a} miles per hour for {b} hours. Distance?",
    ],
    "divide": [
        "{a} cookies are shared equally among {b} people. How many each?",
        "A {a} mile trip takes {b} hours. What's the speed?",
        "{a} items are packed into boxes of {b}. How many boxes?",
    ],
}


def create_training_data(
    num_examples: int = 1000,
    max_value: int = 200,
    include_word_problems: bool = True,
    seed: int = 42,
) -> list[IRExample]:
    """
    Generate training data for IR head.

    Args:
        num_examples: Number of examples to generate
        max_value: Maximum operand value
        include_word_problems: Whether to include word problem templates
        seed: Random seed for reproducibility

    Returns:
        List of IRExample instances
    """
    random.seed(seed)
    examples = []

    ops = list(NL_TEMPLATES.keys())

    for _ in range(num_examples):
        op = random.choice(ops)

        # Generate operands (ensure valid division)
        if op == "divide":
            b = random.randint(1, min(20, max_value))  # Divisor 1-20
            a = b * random.randint(1, max_value // b)  # Ensure clean division
        else:
            a = random.randint(0, max_value)
            b = random.randint(0, max_value)

        # Select template
        templates = NL_TEMPLATES[op].copy()
        if include_word_problems:
            templates.extend(WORD_PROBLEM_TEMPLATES[op])

        template = random.choice(templates)
        text = template.format(a=a, b=b)

        examples.append(IRExample(text=text, op=op, a=a, b=b))

    return examples


class IRDataset:
    """Dataset wrapper for IR training."""

    def __init__(self, examples: list[IRExample], tokenizer, max_length: int = 64):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        tokens = self.tokenizer.encode(ex.text)

        # Pad or truncate
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [self.tokenizer.pad_token_id or 0] * (self.max_length - len(tokens))

        return {
            "input_ids": mx.array(tokens),
            "op": ex.op_idx,
            "a": ex.a,
            "b": ex.b,
            "text": ex.text,
        }

    def get_batch(self, indices: list[int]) -> dict:
        """Get a batch of examples."""
        batch = [self[i] for i in indices]

        return {
            "input_ids": mx.stack([b["input_ids"] for b in batch]),
            "op": mx.array([b["op"] for b in batch]),
            "a": mx.array([b["a"] for b in batch]),
            "b": mx.array([b["b"] for b in batch]),
            "texts": [b["text"] for b in batch],
        }

    def iter_batches(self, batch_size: int, shuffle: bool = True):
        """Iterate over batches."""
        indices = list(range(len(self)))
        if shuffle:
            random.shuffle(indices)

        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            yield self.get_batch(batch_indices)


def create_eval_data() -> list[IRExample]:
    """
    Create a fixed evaluation set for consistent benchmarking.

    These are the same test cases used in the regex pipeline.
    """
    return [
        # Basic operations
        IRExample("Add 11 and 94", "add", 11, 94),
        IRExample("Subtract 49 from 69", "subtract", 69, 49),
        IRExample("Multiply 7 by 8", "multiply", 7, 8),
        IRExample("Divide 48 by 6", "divide", 48, 6),

        # Alternate phrasings
        IRExample("The sum of 25 and 17", "add", 25, 17),
        IRExample("The difference of 100 and 37", "subtract", 100, 37),
        IRExample("What is 12 times 9?", "multiply", 12, 9),
        IRExample("What is 144 divided by 12?", "divide", 144, 12),

        # Word problems
        IRExample("Janet has 50 apples. She gives away 15.", "subtract", 50, 15),
        IRExample("Each box holds 8 items. How many in 7 boxes?", "multiply", 8, 7),
        IRExample("A tank has 200 gallons. 75 leak out.", "subtract", 200, 75),
        IRExample("Tickets cost 15 dollars each. Cost for 4?", "multiply", 15, 4),

        # Edge cases
        IRExample("0 plus 5", "add", 0, 5),
        IRExample("100 minus 100", "subtract", 100, 100),
        IRExample("1 times 255", "multiply", 1, 255),
        IRExample("0 divided by 7", "divide", 0, 7),
    ]
