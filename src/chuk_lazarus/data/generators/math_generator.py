"""
Math Problem Data Generator for Lazarus Training.

Generates synthetic math problems with:
- Tool call traces (what the model should output)
- Correct answers (for verification)
- Negative examples (for DPO preference learning)

This creates the training data for:
1. SFT - Learn tool-calling syntax
2. DPO - Learn to prefer correct tool use over guessing

Problem types:
- Arithmetic (addition, subtraction, multiplication, division)
- Fractions and percentages
- Word problems with distractors
- Multi-step calculations
"""

import json
import logging
import random
from pathlib import Path
from typing import Any

from .types import (
    MathProblem,
    ProblemType,
    ToolCallTrace,
    TrainingSample,
)

logger = logging.getLogger(__name__)


class MathProblemGenerator:
    """
    Generates math problems for training.

    Usage:
        gen = MathProblemGenerator(seed=42)
        problems = gen.generate_batch(100)

        # Generate SFT data
        sft_data = gen.to_sft_format(problems)

        # Generate DPO data
        dpo_data = gen.to_dpo_format(problems)
    """

    # Word problem templates
    WORD_TEMPLATES = [
        "Alice has {a} apples. She buys {b} more. How many apples does she have now?",
        "A train travels {a} miles per hour. How far will it travel in {b} hours?",
        "John has ${a}. He spends ${b}. How much money does he have left?",
        "There are {a} students in a class. {b} more students join. How many students are there now?",
        "A baker makes {a} cookies. He sells {b} of them. How many cookies are left?",
        "A box contains {a} red balls and {b} blue balls. How many balls are there in total?",
        "If you have {a} pencils and give away {b}, how many pencils do you have?",
        "A shop sells {a} items on Monday and {b} items on Tuesday. What is the total?",
    ]

    DISTRACTOR_TEMPLATES = [
        "{prefix} {distractor} {core}",
        "{core} {distractor}",
        "{distractor} However, {core}",
    ]

    DISTRACTORS = [
        "The sky is blue today.",
        "It was raining yesterday.",
        "The store is closed on Sundays.",
        "Birds were singing outside.",
        "The answer might seem obvious.",
        "This is a common problem.",
    ]

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.problem_counter = 0

    def generate_batch(
        self,
        num_problems: int,
        difficulty_range: tuple[int, int] = (1, 3),
        problem_types: list[ProblemType] | None = None,
    ) -> list[TrainingSample]:
        """Generate a batch of training samples."""
        if problem_types is None:
            problem_types = list(ProblemType)

        samples = []
        for _ in range(num_problems):
            problem_type = self.rng.choice(problem_types)
            difficulty = self.rng.randint(*difficulty_range)

            problem = self._generate_problem(problem_type, difficulty)
            trace = self._generate_trace(problem)
            correct_response = self._format_correct_response(problem, trace)
            incorrect_responses = self._generate_incorrect_responses(problem)

            samples.append(
                TrainingSample(
                    problem=problem,
                    correct_trace=trace,
                    correct_response=correct_response,
                    incorrect_responses=incorrect_responses,
                )
            )

        return samples

    def _generate_problem(self, problem_type: ProblemType, difficulty: int) -> MathProblem:
        """Generate a single problem."""
        self.problem_counter += 1
        problem_id = f"math_{self.problem_counter:06d}"

        if problem_type == ProblemType.ARITHMETIC:
            return self._generate_arithmetic(problem_id, difficulty)
        elif problem_type == ProblemType.FRACTIONS:
            return self._generate_fractions(problem_id, difficulty)
        elif problem_type == ProblemType.PERCENTAGES:
            return self._generate_percentages(problem_id, difficulty)
        elif problem_type == ProblemType.WORD_PROBLEM:
            return self._generate_word_problem(problem_id, difficulty)
        elif problem_type == ProblemType.MULTI_STEP:
            return self._generate_multi_step(problem_id, difficulty)
        elif problem_type == ProblemType.COMPARISON:
            return self._generate_comparison(problem_id, difficulty)
        else:
            return self._generate_arithmetic(problem_id, difficulty)

    def _generate_arithmetic(self, problem_id: str, difficulty: int) -> MathProblem:
        """Generate arithmetic problem."""
        max_val = 10**difficulty

        ops = ["+", "-", "*"]
        if difficulty >= 2:
            ops.append("/")

        op = self.rng.choice(ops)

        if op == "/":
            # Ensure clean division
            b = self.rng.randint(1, max_val // 10 or 1)
            result = self.rng.randint(1, max_val // 10 or 1)
            a = b * result
        else:
            a = self.rng.randint(1, max_val)
            b = self.rng.randint(1, max_val)

        expression = f"{a} {op} {b}"
        answer = eval(expression)

        return MathProblem(
            id=problem_id,
            problem_type=ProblemType.ARITHMETIC,
            problem_text=f"Calculate: {expression}",
            expression=expression,
            answer=float(answer),
            difficulty=difficulty,
        )

    def _generate_fractions(self, problem_id: str, difficulty: int) -> MathProblem:
        """Generate fraction problem."""
        # Generate fraction components
        num1 = self.rng.randint(1, 10)
        den1 = self.rng.randint(2, 10)
        num2 = self.rng.randint(1, 10)
        den2 = self.rng.randint(2, 10)

        op = self.rng.choice(["+", "-", "*"])

        if op == "+":
            result = (num1 * den2 + num2 * den1) / (den1 * den2)
        elif op == "-":
            result = (num1 * den2 - num2 * den1) / (den1 * den2)
        else:
            result = (num1 * num2) / (den1 * den2)

        expression = f"({num1}/{den1}) {op} ({num2}/{den2})"

        return MathProblem(
            id=problem_id,
            problem_type=ProblemType.FRACTIONS,
            problem_text=f"Calculate: {expression}",
            expression=expression,
            answer=float(result),
            difficulty=difficulty,
        )

    def _generate_percentages(self, problem_id: str, difficulty: int) -> MathProblem:
        """Generate percentage problem."""
        base = self.rng.randint(10, 100) * (difficulty)
        percent = self.rng.choice([5, 10, 15, 20, 25, 30, 50, 75])

        answer = base * percent / 100

        return MathProblem(
            id=problem_id,
            problem_type=ProblemType.PERCENTAGES,
            problem_text=f"What is {percent}% of {base}?",
            expression=f"{base} * {percent} / 100",
            answer=float(answer),
            difficulty=difficulty,
        )

    def _generate_word_problem(self, problem_id: str, difficulty: int) -> MathProblem:
        """Generate word problem with optional distractors."""
        template = self.rng.choice(self.WORD_TEMPLATES)

        a = self.rng.randint(5, 50 * difficulty)
        b = self.rng.randint(1, 30 * difficulty)

        # Ensure subtraction doesn't go negative
        if (
            "spends" in template
            or "give away" in template
            or "sells" in template
            or "left" in template
        ):
            if b > a:
                a, b = b, a

        problem_text = template.format(a=a, b=b)

        # Add distractor for higher difficulty
        if difficulty >= 3:
            distractor = self.rng.choice(self.DISTRACTORS)
            dist_template = self.rng.choice(self.DISTRACTOR_TEMPLATES)
            problem_text = dist_template.format(
                prefix="", distractor=distractor, core=problem_text
            ).strip()

        # Determine operation from template
        if "more" in template or "join" in template or "total" in template or "buys" in template:
            answer = a + b
            expression = f"{a} + {b}"
        elif (
            "left" in template
            or "spends" in template
            or "give away" in template
            or "sells" in template
        ):
            answer = a - b
            expression = f"{a} - {b}"
        elif "per" in template:
            answer = a * b
            expression = f"{a} * {b}"
        else:
            answer = a + b
            expression = f"{a} + {b}"

        return MathProblem(
            id=problem_id,
            problem_type=ProblemType.WORD_PROBLEM,
            problem_text=problem_text,
            expression=expression,
            answer=float(answer),
            difficulty=difficulty,
        )

    def _generate_multi_step(self, problem_id: str, difficulty: int) -> MathProblem:
        """Generate multi-step problem."""
        a = self.rng.randint(10, 100)
        b = self.rng.randint(5, 50)
        c = self.rng.randint(2, 10)

        templates = [
            (f"Calculate ({a} + {b}) * {c}", f"({a} + {b}) * {c}", (a + b) * c),
            (f"Calculate {a} + {b} * {c}", f"{a} + {b} * {c}", a + b * c),
            (f"Calculate ({a} - {b}) * {c}", f"({a} - {b}) * {c}", (a - b) * c),
            (f"If x = {a} + {b}, what is x * {c}?", f"({a} + {b}) * {c}", (a + b) * c),
        ]

        text, expr, answer = self.rng.choice(templates)

        return MathProblem(
            id=problem_id,
            problem_type=ProblemType.MULTI_STEP,
            problem_text=text,
            expression=expr,
            answer=float(answer),
            difficulty=difficulty,
        )

    def _generate_comparison(self, problem_id: str, difficulty: int) -> MathProblem:
        """Generate comparison problem."""
        a = self.rng.randint(10, 100 * difficulty)
        b = self.rng.randint(10, 100 * difficulty)

        if self.rng.random() < 0.5:
            # Ask which is larger
            problem_text = f"Which is larger: {a} or {b}?"
            answer = max(a, b)
        else:
            # Ask for difference
            problem_text = f"What is the difference between {a} and {b}?"
            answer = abs(a - b)

        return MathProblem(
            id=problem_id,
            problem_type=ProblemType.COMPARISON,
            problem_text=problem_text,
            expression=f"max({a}, {b})" if "larger" in problem_text else f"abs({a} - {b})",
            answer=float(answer),
            difficulty=difficulty,
        )

    def _generate_trace(self, problem: MathProblem) -> list[ToolCallTrace]:
        """Generate the tool call trace for solving the problem."""
        return [
            ToolCallTrace(
                tool_name="math_solve",
                tool_args={"expression": problem.expression},
                tool_result=problem.answer,
                thought=f"I need to calculate: {problem.expression}",
            )
        ]

    def _format_correct_response(self, problem: MathProblem, trace: list[ToolCallTrace]) -> str:
        """Format the correct response with tool calls."""
        lines = []

        for step in trace:
            if step.thought:
                lines.append(f"THINK: {step.thought}")
            lines.append(f"TOOL: {step.tool_name}({self._format_args(step.tool_args)})")
            lines.append(f"Result: {step.tool_result}")

        # Format answer
        answer = problem.answer
        if answer == int(answer):
            answer_str = str(int(answer))
        else:
            answer_str = f"{answer:.4f}".rstrip("0").rstrip(".")

        if problem.unit:
            answer_str = f"{answer_str} {problem.unit}"

        lines.append(f"ANSWER: {answer_str}")

        return "\n".join(lines)

    def _format_args(self, args: dict[str, Any]) -> str:
        """Format tool arguments."""
        parts = []
        for k, v in args.items():
            if isinstance(v, str):
                parts.append(f'{k}="{v}"')
            else:
                parts.append(f"{k}={v}")
        return ", ".join(parts)

    def _generate_incorrect_responses(self, problem: MathProblem) -> list[str]:
        """Generate incorrect responses for DPO training."""
        incorrect = []

        # Type 1: Wrong answer (guessing)
        wrong_answer = problem.answer + self.rng.randint(-10, 10)
        if wrong_answer == problem.answer:
            wrong_answer += 5
        incorrect.append(f"Let me think... I believe the answer is {wrong_answer}.")

        # Type 2: Wrong tool call (missing/wrong args)
        incorrect.append(
            f"TOOL: math_solve()\nHmm, that didn't work. The answer is probably {wrong_answer}."
        )

        # Type 3: Verbose but wrong
        incorrect.append(
            f"This is an interesting problem. Let me analyze it carefully. "
            f"Based on my understanding, I think the answer is {wrong_answer}. "
            f"However, I'm not entirely sure about this."
        )

        # Type 4: Tool spam (multiple unnecessary calls)
        incorrect.append(
            f'TOOL: math_solve(expression="1+1")\n'
            f'TOOL: math_solve(expression="2+2")\n'
            f'TOOL: math_solve(expression="{problem.expression}")\n'
            f"ANSWER: {wrong_answer}"
        )

        return incorrect

    def to_sft_format(self, samples: list[TrainingSample]) -> list[dict]:
        """Convert to SFT training format."""
        data = []
        for sample in samples:
            data.append(
                {
                    "prompt": sample.problem.problem_text,
                    "response": sample.correct_response,
                    "metadata": {
                        "problem_id": sample.problem.id,
                        "problem_type": sample.problem.problem_type.value,
                        "difficulty": sample.problem.difficulty,
                        "answer": sample.problem.answer,
                    },
                }
            )
        return data

    def to_dpo_format(self, samples: list[TrainingSample]) -> list[dict]:
        """Convert to DPO preference format."""
        data = []
        for sample in samples:
            for incorrect in sample.incorrect_responses:
                data.append(
                    {
                        "prompt": sample.problem.problem_text,
                        "chosen": sample.correct_response,
                        "rejected": incorrect,
                        "metadata": {
                            "problem_id": sample.problem.id,
                            "problem_type": sample.problem.problem_type.value,
                            "difficulty": sample.problem.difficulty,
                        },
                    }
                )
        return data

    def save_sft_dataset(self, samples: list[TrainingSample], path: str):
        """Save as SFT JSONL dataset."""
        data = self.to_sft_format(samples)
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        logger.info(f"Saved {len(data)} SFT samples to {path}")

    def save_dpo_dataset(self, samples: list[TrainingSample], path: str):
        """Save as DPO JSONL dataset."""
        data = self.to_dpo_format(samples)
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        logger.info(f"Saved {len(data)} DPO samples to {path}")


def generate_lazarus_dataset(
    output_dir: str, sft_samples: int = 10000, dpo_samples: int = 5000, seed: int = 42
):
    """
    Generate complete training dataset for Lazarus training.

    Creates:
    - train_sft.jsonl: SFT training data
    - train_dpo.jsonl: DPO preference data
    - eval_sft.jsonl: SFT evaluation data
    - eval_dpo.jsonl: DPO evaluation data

    Args:
        output_dir: Directory to save datasets
        sft_samples: Number of SFT samples
        dpo_samples: Number of DPO samples
        seed: Random seed
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    gen = MathProblemGenerator(seed=seed)

    # Generate SFT data
    logger.info(f"Generating {sft_samples} SFT samples...")
    sft_samples_list = gen.generate_batch(sft_samples, difficulty_range=(1, 3))

    # Split into train/eval (90/10)
    split_idx = int(len(sft_samples_list) * 0.9)
    train_sft = sft_samples_list[:split_idx]
    eval_sft = sft_samples_list[split_idx:]

    gen.save_sft_dataset(train_sft, str(output_path / "train_sft.jsonl"))
    gen.save_sft_dataset(eval_sft, str(output_path / "eval_sft.jsonl"))

    # Generate DPO data
    logger.info(f"Generating {dpo_samples} DPO samples...")
    dpo_samples_list = gen.generate_batch(dpo_samples, difficulty_range=(1, 4))

    split_idx = int(len(dpo_samples_list) * 0.9)
    train_dpo = dpo_samples_list[:split_idx]
    eval_dpo = dpo_samples_list[split_idx:]

    gen.save_dpo_dataset(train_dpo, str(output_path / "train_dpo.jsonl"))
    gen.save_dpo_dataset(eval_dpo, str(output_path / "eval_dpo.jsonl"))

    logger.info(f"Dataset generation complete. Files saved to {output_dir}")

    return {
        "sft_train": str(output_path / "train_sft.jsonl"),
        "sft_eval": str(output_path / "eval_sft.jsonl"),
        "dpo_train": str(output_path / "train_dpo.jsonl"),
        "dpo_eval": str(output_path / "eval_dpo.jsonl"),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate math training data")
    parser.add_argument(
        "--output", type=str, default="./data/lazarus_math", help="Output directory"
    )
    parser.add_argument("--sft-samples", type=int, default=10000, help="Number of SFT samples")
    parser.add_argument("--dpo-samples", type=int, default=5000, help="Number of DPO samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    generate_lazarus_dataset(
        output_dir=args.output,
        sft_samples=args.sft_samples,
        dpo_samples=args.dpo_samples,
        seed=args.seed,
    )
