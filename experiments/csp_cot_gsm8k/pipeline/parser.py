"""
Problem Parser.

Uses LLM to extract structured ProblemSpec from natural language.
NO REGEX - pure semantic parsing by the model.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any
from decimal import Decimal

from ..schema.problem import (
    ProblemSpec,
    ProblemType,
    Entity,
    Operation,
    OperationType,
    Query,
    Constraint,
)


class ProblemParser(ABC):
    """Abstract base class for problem parsers."""

    @abstractmethod
    def parse(self, problem: str) -> ProblemSpec | None:
        """Parse natural language problem into structured spec."""
        pass


# =============================================================================
# FEW-SHOT EXAMPLES
# =============================================================================

FEW_SHOT_EXAMPLES = [
    # Entity tracking - transfer
    {
        "problem": "Jenny has 5 apples. She gives 2 to Bob. How many does Jenny have?",
        "spec": {
            "problem_type": "entity_tracking",
            "entities": [
                {"name": "jenny", "attribute": "apples", "initial_value": 5},
                {"name": "bob", "attribute": "apples", "initial_value": 0},
            ],
            "operations": [
                {"type": "transfer", "source": "jenny", "target": "bob", "amount": 2}
            ],
            "query": {"target": "jenny", "question": "how_many"},
        },
    },
    # Entity tracking - add/subtract
    {
        "problem": "Sam had 10 marbles. He lost 3, then found 5. How many does he have now?",
        "spec": {
            "problem_type": "entity_tracking",
            "entities": [
                {"name": "sam", "attribute": "marbles", "initial_value": 10}
            ],
            "operations": [
                {"type": "subtract", "target": "sam", "amount": 3},
                {"type": "add", "target": "sam", "amount": 5},
            ],
            "query": {"target": "sam", "question": "how_many"},
        },
    },
    # Arithmetic chain - multiplication
    {
        "problem": "There are 6 bags with 4 oranges in each bag. How many oranges total?",
        "spec": {
            "problem_type": "arithmetic_chain",
            "entities": [
                {"name": "total", "attribute": "oranges", "initial_value": 6}
            ],
            "operations": [
                {"type": "multiply", "target": "total", "factor": 4}
            ],
            "query": {"target": "total", "question": "value"},
        },
    },
    # Comparison
    {
        "problem": "Tom has 15 marbles. Jane has 5 marbles. How many more does Tom have than Jane?",
        "spec": {
            "problem_type": "comparison",
            "entities": [
                {"name": "tom", "attribute": "marbles", "initial_value": 15},
                {"name": "jane", "attribute": "marbles", "initial_value": 5},
            ],
            "operations": [],
            "query": {
                "target": "difference",
                "question": "compare",
                "compare_a": "tom",
                "compare_b": "jane",
            },
        },
    },
    # Comparison with multiplication
    {
        "problem": "Alice has 3 times as many stickers as Bob. Bob has 5 stickers. How many more does Alice have than Bob?",
        "spec": {
            "problem_type": "comparison",
            "entities": [
                {"name": "bob", "attribute": "stickers", "initial_value": 5},
                {"name": "alice", "attribute": "stickers", "initial_value": 5},
            ],
            "operations": [
                {"type": "multiply", "target": "alice", "factor": 3}
            ],
            "query": {
                "target": "difference",
                "question": "compare",
                "compare_a": "alice",
                "compare_b": "bob",
            },
        },
    },
    # Allocation
    {
        "problem": "Split $100 between Alice and Bob. Alice gets twice what Bob gets. How much does Alice get?",
        "spec": {
            "problem_type": "allocation",
            "entities": [
                {"name": "alice", "attribute": "dollars"},
                {"name": "bob", "attribute": "dollars"},
            ],
            "operations": [],
            "constraints": [
                {"type": "sum", "entities": ["alice", "bob"], "value": 100},
                {"type": "ratio", "entities": ["alice", "bob"], "factor": 2},
            ],
            "query": {"target": "alice", "question": "value"},
        },
    },
    # Percentage
    {
        "problem": "A shirt costs $40. It's 25% off. What's the final price?",
        "spec": {
            "problem_type": "percentage",
            "entities": [
                {"name": "price", "attribute": "dollars", "initial_value": 40}
            ],
            "operations": [
                {"type": "multiply", "target": "price", "factor": 0.75}
            ],
            "query": {"target": "price", "question": "value"},
        },
    },
    # Division
    {
        "problem": "12 cookies are split equally between 4 kids. How many does each kid get?",
        "spec": {
            "problem_type": "arithmetic_chain",
            "entities": [
                {"name": "cookies_per_kid", "attribute": "cookies", "initial_value": 12}
            ],
            "operations": [
                {"type": "divide", "target": "cookies_per_kid", "factor": 4}
            ],
            "query": {"target": "cookies_per_kid", "question": "value"},
        },
    },
]


def build_few_shot_prompt(problem: str) -> str:
    """
    Build few-shot prompt for problem extraction.

    Returns a prompt that instructs the model to output JSON.
    """
    examples_text = ""
    for ex in FEW_SHOT_EXAMPLES:
        examples_text += f'Problem: "{ex["problem"]}"\n'
        examples_text += f"Spec: {json.dumps(ex['spec'], indent=2)}\n\n"

    prompt = f"""You are a math problem parser. Extract the structured specification from the problem.

Output ONLY valid JSON matching this schema:
- problem_type: one of "entity_tracking", "arithmetic_chain", "comparison", "allocation", "percentage"
- entities: list of {{name, attribute, initial_value (if known)}}
- operations: list of {{type, target, source (for transfer), amount, factor (for multiply/divide)}}
- constraints: list of {{type, entities, value, factor}} (for allocation problems)
- query: {{target, question, compare_a, compare_b (for comparisons)}}

{examples_text}
Problem: "{problem}"
Spec:"""

    return prompt


class FewShotParser(ProblemParser):
    """
    Parser that uses few-shot prompting with an LLM.

    The model extracts structured information through semantic understanding,
    not regex pattern matching.
    """

    def __init__(self, model=None, tokenizer=None, max_tokens: int = 500):
        """
        Initialize with model and tokenizer.

        If not provided, uses a simple JSON extraction fallback.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    def parse(self, problem: str) -> ProblemSpec | None:
        """
        Parse problem using few-shot LLM extraction.

        If no model is available, returns None (must use model-based parsing).
        """
        if self.model is None or self.tokenizer is None:
            # No model - can't parse
            return None

        prompt = build_few_shot_prompt(problem)
        response = self._generate(prompt)

        if response:
            return self._parse_response(response, problem)
        return None

    def _generate(self, prompt: str) -> str | None:
        """Generate response from model."""
        try:
            import mlx.core as mx

            tokens = self.tokenizer.encode(prompt)
            input_ids = mx.array([tokens])

            generated = []
            for _ in range(self.max_tokens):
                output = self.model(input_ids)
                logits = output.logits if hasattr(output, "logits") else output
                next_token = int(mx.argmax(logits[:, -1, :], axis=-1).item())

                # Stop on end tokens or JSON completion
                decoded = self.tokenizer.decode(generated + [next_token])
                if "}" in decoded and decoded.count("{") <= decoded.count("}"):
                    generated.append(next_token)
                    break

                generated.append(next_token)
                input_ids = mx.concatenate(
                    [input_ids, mx.array([[next_token]])], axis=1
                )

            return self.tokenizer.decode(generated).strip()
        except Exception as e:
            print(f"Generation error: {e}")
            return None

    def _parse_response(self, response: str, raw_problem: str) -> ProblemSpec | None:
        """Parse JSON response into ProblemSpec."""
        try:
            # Find JSON in response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start == -1 or end == 0:
                return None

            json_str = response[start:end]
            data = json.loads(json_str)

            return self._dict_to_spec(data, raw_problem)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Parse error: {e}")
            return None

    def _dict_to_spec(self, data: dict, raw_problem: str) -> ProblemSpec:
        """Convert parsed dict to ProblemSpec."""
        # Problem type
        type_str = data.get("problem_type", "unknown")
        try:
            problem_type = ProblemType(type_str)
        except ValueError:
            problem_type = ProblemType.UNKNOWN

        # Entities
        entities = []
        for e in data.get("entities", []):
            entities.append(
                Entity(
                    name=e.get("name", "unknown"),
                    attribute=e.get("attribute"),
                    initial_value=(
                        Decimal(str(e["initial_value"]))
                        if e.get("initial_value") is not None
                        else None
                    ),
                    unit=e.get("unit"),
                )
            )

        # Operations
        operations = []
        for o in data.get("operations", []):
            op_type_str = o.get("type", "add")
            try:
                op_type = OperationType(op_type_str)
            except ValueError:
                op_type = OperationType.ADD

            operations.append(
                Operation(
                    type=op_type,
                    target=o.get("target", ""),
                    source=o.get("source"),
                    amount=(
                        Decimal(str(o["amount"])) if o.get("amount") is not None else None
                    ),
                    factor=(
                        Decimal(str(o["factor"])) if o.get("factor") is not None else None
                    ),
                )
            )

        # Constraints
        constraints = []
        for c in data.get("constraints", []):
            constraints.append(
                Constraint(
                    type=c.get("type", ""),
                    entities=c.get("entities", []),
                    factor=(
                        Decimal(str(c["factor"])) if c.get("factor") is not None else None
                    ),
                    value=(
                        Decimal(str(c["value"])) if c.get("value") is not None else None
                    ),
                )
            )

        # Query
        query = None
        if "query" in data and data["query"]:
            q = data["query"]
            query = Query(
                target=q.get("target", ""),
                question=q.get("question", "value"),
                compare_a=q.get("compare_a"),
                compare_b=q.get("compare_b"),
            )

        return ProblemSpec(
            problem_type=problem_type,
            entities=entities,
            operations=operations,
            constraints=constraints,
            query=query,
            raw_text=raw_problem,
        )


# =============================================================================
# MANUAL SPEC BUILDER (for testing without model)
# =============================================================================


class ManualSpecBuilder:
    """
    Helper for manually building ProblemSpec from example format.

    Used for testing when no LLM is available.
    """

    @staticmethod
    def from_dict(data: dict, raw_problem: str = "") -> ProblemSpec:
        """Build spec from dict (like few-shot examples)."""
        parser = FewShotParser()
        return parser._dict_to_spec(data, raw_problem)

    @staticmethod
    def get_example_specs() -> dict[str, ProblemSpec]:
        """Get all few-shot examples as ProblemSpecs."""
        specs = {}
        for i, ex in enumerate(FEW_SHOT_EXAMPLES):
            spec = ManualSpecBuilder.from_dict(ex["spec"], ex["problem"])
            specs[f"example_{i}_{spec.problem_type.value}"] = spec
        return specs


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Problem Parser Tests")
    print("=" * 60)

    # Test 1: Manual spec building
    print("\nTest 1: Manual spec building from examples")
    specs = ManualSpecBuilder.get_example_specs()
    for name, spec in list(specs.items())[:3]:
        print(f"\n{name}:")
        print(f"  Type: {spec.problem_type.value}")
        print(f"  Entities: {[e.name for e in spec.entities]}")
        print(f"  Operations: {len(spec.operations)}")
        print(f"  Query: {spec.query.target if spec.query else None}")
        print(f"  Valid: {spec.is_valid()}")

    # Test 2: Few-shot prompt generation
    print("\nTest 2: Few-shot prompt")
    problem = "Lisa has 20 stickers. She gives 5 to Ann and 3 to Bob. How many does Lisa have left?"
    prompt = build_few_shot_prompt(problem)
    print(f"Prompt length: {len(prompt)} chars")
    print(f"Problem: {problem}")
    print(f"\nExpected output format:")
    print(
        json.dumps(
            {
                "problem_type": "entity_tracking",
                "entities": [
                    {"name": "lisa", "initial_value": 20},
                    {"name": "ann", "initial_value": 0},
                    {"name": "bob", "initial_value": 0},
                ],
                "operations": [
                    {"type": "transfer", "source": "lisa", "target": "ann", "amount": 5},
                    {"type": "transfer", "source": "lisa", "target": "bob", "amount": 3},
                ],
                "query": {"target": "lisa", "question": "how_many"},
            },
            indent=2,
        )
    )
