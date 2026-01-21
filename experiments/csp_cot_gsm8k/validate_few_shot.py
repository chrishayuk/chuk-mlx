#!/usr/bin/env python3
"""
Few-Shot Validation for CSP-CoT Math Expert.

Validates that the math word problem expert works with LLM few-shot
prompting before investing in fine-tuning.

Usage:
    # Quick test (5 problems)
    python validate_few_shot.py --n 5

    # With specific model
    python validate_few_shot.py --model mlx-community/gemma-3-4b-it-bf16 --n 10

    # Verbose output
    python validate_few_shot.py --n 5 --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.csp_cot_gsm8k.expert_standalone import MathSolver
from experiments.csp_cot_gsm8k.evaluation.gsm8k_loader import get_sample_problems


# =============================================================================
# VALIDATION RESULT
# =============================================================================

@dataclass
class ValidationResult:
    """Result of validating a single problem."""

    question: str
    expected_answer: float
    raw_output: str = ""

    # Pipeline stages
    parsed: bool = False
    parse_error: str | None = None
    routed: bool = False
    executed: bool = False
    exec_error: str | None = None
    verified: bool = False
    answer: float | None = None
    correct: bool = False

    @property
    def error(self) -> str | None:
        if self.parse_error:
            return f"parse:{self.parse_error}"
        if not self.routed:
            return "not_routed"
        if self.exec_error:
            return f"exec:{self.exec_error}"
        if not self.verified:
            return "invalid_trace"
        if not self.correct:
            return f"wrong:{self.answer}"
        return None


@dataclass
class ValidationSummary:
    """Summary of validation run."""

    total: int = 0
    parsed: int = 0
    routed: int = 0
    executed: int = 0
    verified: int = 0
    correct: int = 0
    errors: dict[str, int] = field(default_factory=dict)
    results: list[ValidationResult] = field(default_factory=list)

    @property
    def parse_rate(self) -> float:
        return self.parsed / self.total if self.total > 0 else 0

    @property
    def route_rate(self) -> float:
        return self.routed / self.total if self.total > 0 else 0

    @property
    def valid_rate(self) -> float:
        return self.verified / self.total if self.total > 0 else 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0

    def print_summary(self):
        print(f"\n{'='*60}")
        print("FEW-SHOT VALIDATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total problems: {self.total}")
        print()
        print("Pipeline metrics:")
        print(f"  Parsed:   {self.parsed:3d}/{self.total} ({self.parse_rate:5.1%})")
        print(f"  Routed:   {self.routed:3d}/{self.total} ({self.route_rate:5.1%})")
        print(f"  Executed: {self.executed:3d}/{self.total} ({self.executed/self.total if self.total else 0:5.1%})")
        print(f"  Valid:    {self.verified:3d}/{self.total} ({self.valid_rate:5.1%})")
        print(f"  Correct:  {self.correct:3d}/{self.total} ({self.accuracy:5.1%})")

        if self.errors:
            print(f"\nError breakdown:")
            for error, count in sorted(self.errors.items(), key=lambda x: -x[1]):
                print(f"  {error}: {count}")

        print(f"\n{'='*60}")
        print("DECISION GUIDANCE")
        print(f"{'='*60}")

        if self.parse_rate < 0.5:
            print("⚠️  Parse rate <50%: Redesign JSON schema or simplify")
        elif self.valid_rate < 0.5:
            print("⚠️  Valid rate <50%: Simplify trace schema or add examples")
        elif self.accuracy < 0.3:
            print("📈 Format works but reasoning weak → Fine-tuning will help")
        elif self.accuracy < 0.6:
            print("✓ Few-shot shows promise → Fine-tune for production")
        elif self.accuracy < 0.8:
            print("✓ Few-shot works well → Fine-tune for polish")
        else:
            print("✓ Few-shot sufficient → May skip fine-tuning")


# =============================================================================
# FEW-SHOT PROMPT
# =============================================================================

def load_cot_examples() -> list[dict]:
    """Load CoT examples from JSON file."""
    path = Path(__file__).parent / "cot_examples.json"
    with open(path) as f:
        data = json.load(f)
    return [ex for ex in data["examples"] if ex["action"]["expert"] == "math_word_problem"]


def build_prompt(question: str, max_examples: int = 3, style: str = "instruct") -> str:
    """Build few-shot prompt for math word problems."""
    examples = load_cot_examples()[:max_examples]

    if style == "simple":
        # Ultra-minimal format for base models
        # Just answer extraction - no complex JSON
        examples_text = """Q: John has 10 apples. He eats 3. How many left?
A: 7

Q: Mary has 5 books. She buys 2 more. How many total?
A: 7

Q: A store sold 20 items at $5 each. What is the total revenue?
A: 100

"""
        return f"""{examples_text}Q: {question}
A: """

    elif style == "completion":
        # Completion style for base models (gpt-oss, etc.)
        # Use compact JSON on single lines for better pattern matching
        examples_text = ""
        for ex in examples:
            action_compact = json.dumps(ex["action"], separators=(',', ':'))
            examples_text += f'Q: {ex["query"]}\nA: {action_compact}\n\n'

        return f"""{examples_text}Q: {question}
A: """

    else:
        # Instruct style for chat models
        examples_text = ""
        for ex in examples:
            examples_text += f'Problem: "{ex["query"]}"\n'
            examples_text += f'Action: {json.dumps(ex["action"], indent=2)}\n\n'

        return f"""You convert math word problems to structured solver actions.

## Schema
- problem_type: entity_tracking | arithmetic_chain | comparison | allocation
- entities: [{{"name": "<name>", "initial_value": <NUMBER>}}, ...]
- operations: [{{"type": "add|subtract|multiply|divide", "target": "<entity>", "amount": <NUMBER>}}, ...]
- query: {{"target": "<entity>"}}

## Key Rules
1. "amount" must be a NUMBER, not an entity name
2. Use "factor" for multiply/divide, "amount" for add/subtract
3. Chain operations on a SINGLE entity
4. Query target = entity with the final answer

## Examples

{examples_text}
Problem: "{question}"
Action:"""


# =============================================================================
# VALIDATION
# =============================================================================

def extract_json(text: str) -> dict | None:
    """Extract JSON object from text."""
    try:
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        end = start
        in_string = False
        escape_next = False

        for i, char in enumerate(text[start:], start):
            if escape_next:
                escape_next = False
                continue
            if char == "\\":
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        return json.loads(text[start:end])
    except (json.JSONDecodeError, IndexError):
        return None


def extract_number(text: str) -> float | None:
    """Extract the first number from text."""
    import re
    # Match integers or decimals with optional commas, possibly negative
    match = re.search(r'-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?', text.strip())
    if match:
        try:
            # Remove commas before converting
            return float(match.group().replace(',', ''))
        except ValueError:
            return None
    return None


def validate_single(
    question: str,
    expected: float,
    generate_fn: Callable[[str, int], str],
    solver: MathSolver,
    max_examples: int = 3,
    verbose: bool = False,
    style: str = "instruct",
) -> ValidationResult:
    """Validate a single problem."""
    result = ValidationResult(question=question, expected_answer=expected)

    # Generate
    prompt = build_prompt(question, max_examples, style=style)
    result.raw_output = generate_fn(prompt, 500 if style != "simple" else 50)

    if verbose:
        print(f"\nQ: {question[:60]}...")
        print(f"Output: {result.raw_output[:150]}...")

    # For simple style, just extract the number directly
    if style == "simple":
        result.answer = extract_number(result.raw_output)
        if result.answer is not None:
            result.parsed = True
            result.routed = True
            result.executed = True
            result.verified = True
            result.correct = abs(result.answer - expected) < 0.01
        else:
            result.parse_error = "no_number"
        if verbose:
            status = "✓" if result.correct else f"✗ got {result.answer}, expected {expected}"
            print(f"Status: {status}")
        return result

    # Parse JSON for other styles
    action = extract_json(result.raw_output)
    if action is None:
        result.parse_error = "no_json"
        return result

    result.parsed = True

    # Check routing
    if action.get("expert") != "math_word_problem":
        result.routed = False
        return result

    result.routed = True

    # Execute
    try:
        params = action.get("parameters", {})
        exec_result = solver.solve(**params)

        if not exec_result.get("success"):
            result.exec_error = exec_result.get("error", "unknown")[:50]
            return result

        result.executed = True
        result.verified = exec_result.get("verified", False)
        result.answer = exec_result.get("answer")

    except Exception as e:
        result.exec_error = str(e)[:50]
        return result

    # Check correctness
    if result.answer is not None:
        result.correct = abs(result.answer - expected) < 0.01

    if verbose:
        status = "✓" if result.correct else f"✗ {result.error}"
        print(f"Status: {status}")

    return result


def run_validation(
    problems: list,
    generate_fn: Callable[[str, int], str],
    max_examples: int = 3,
    verbose: bool = False,
    style: str = "instruct",
) -> ValidationSummary:
    """Run validation on a set of problems."""
    solver = MathSolver()
    summary = ValidationSummary()

    for prob in problems:
        result = validate_single(
            question=prob.question,
            expected=prob.answer,
            generate_fn=generate_fn,
            solver=solver,
            max_examples=max_examples,
            verbose=verbose,
            style=style,
        )

        summary.results.append(result)
        summary.total += 1

        if result.parsed:
            summary.parsed += 1
        if result.routed:
            summary.routed += 1
        if result.executed:
            summary.executed += 1
        if result.verified:
            summary.verified += 1
        if result.correct:
            summary.correct += 1

        if result.error:
            summary.errors[result.error] = summary.errors.get(result.error, 0) + 1

    return summary


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Few-shot validation for CSP-CoT math expert"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Llama-3.2-1B-Instruct-4bit",
        help="MLX model ID",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Number of problems to test",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=3,
        help="Number of few-shot examples",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output",
    )
    parser.add_argument(
        "--style",
        type=str,
        default="instruct",
        choices=["instruct", "completion", "simple"],
        help="Prompt style: instruct (chat models), completion (base), or simple (minimal JSON)",
    )

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("CSP-CoT FEW-SHOT VALIDATION")
    print(f"{'='*60}")

    # Load problems
    print(f"Loading {args.n} GSM-8K problems...")
    problems = get_sample_problems(args.n)
    print(f"Loaded {len(problems)} problems")

    # Load model via Lazarus UnifiedPipeline
    print(f"\nLoading model: {args.model}")
    from chuk_lazarus.inference.unified import UnifiedPipeline

    pipeline = UnifiedPipeline.from_pretrained(args.model, verbose=True)
    print(f"Model loaded! Family: {pipeline.family_type.value}")

    def generate_fn(prompt: str, max_tokens: int) -> str:
        result = pipeline.generate(
            prompt,
            max_new_tokens=max_tokens,
            temperature=0.0,  # Greedy for consistency
        )
        return result.text

    # Run validation
    print(f"\nRunning validation with {args.max_examples} few-shot examples (style={args.style})...")
    summary = run_validation(
        problems=problems,
        generate_fn=generate_fn,
        max_examples=args.max_examples,
        verbose=args.verbose,
        style=args.style,
    )

    # Print results
    summary.print_summary()

    # Show sample outputs
    print(f"\n{'='*60}")
    print("SAMPLE RESULTS")
    print(f"{'='*60}")

    for i, result in enumerate(summary.results[:5]):
        print(f"\n[{i+1}] {result.question[:50]}...")
        print(f"    Expected: {result.expected_answer}")
        print(f"    Got: {result.answer}")
        print(f"    Parsed: {result.parsed}, Routed: {result.routed}, Valid: {result.verified}, Correct: {result.correct}")
        if result.error:
            print(f"    Error: {result.error}")


if __name__ == "__main__":
    main()
