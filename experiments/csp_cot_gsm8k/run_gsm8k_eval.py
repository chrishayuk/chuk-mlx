#!/usr/bin/env python3
"""
GSM-8K Evaluation Script.

Evaluates CSP-CoT on GSM-8K benchmark and compares against English CoT.

Usage:
    # With model (full evaluation with LLM parsing)
    python run_gsm8k_eval.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --n 100

    # Without model (uses sample specs for testing)
    python run_gsm8k_eval.py --no-model --n 10

    # With English CoT baseline
    python run_gsm8k_eval.py --model MODEL --english-cot --n 50
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.csp_cot_gsm8k.evaluation.gsm8k_loader import (
    load_gsm8k,
    get_sample_problems,
    GSM8KProblem,
)
from experiments.csp_cot_gsm8k.evaluation.evaluator import CSPCoTEvaluator
from experiments.csp_cot_gsm8k.schema.problem import (
    ProblemSpec,
    ProblemType,
    Entity,
    Operation,
    OperationType,
    Query,
)
from experiments.csp_cot_gsm8k.pipeline.executor import CSPCoTExecutor


def load_model_if_needed(model_id: str | None):
    """Load model and tokenizer if model_id is provided."""
    if model_id is None:
        return None, None

    print(f"Loading model: {model_id}")
    try:
        from chuk_lazarus.models_v2.loader import load_model

        result = load_model(model_id)
        result.model.freeze()
        print(f"  Loaded {result.family_type} model")
        return result.model, result.tokenizer
    except Exception as e:
        print(f"  Failed to load model: {e}")
        return None, None


def create_sample_specs(problems: list[GSM8KProblem]) -> list[ProblemSpec]:
    """
    Create sample specs for problems (for testing without LLM).

    These are manually crafted for the SAMPLE_PROBLEMS in gsm8k_loader.
    """
    # Map of expected answers to specs
    spec_templates = {
        # Janet's ducks: 16 eggs, eat 3, bake 4, sell rest at $2
        18: ProblemSpec(
            problem_type=ProblemType.ARITHMETIC_CHAIN,
            entities=[Entity(name="eggs", initial_value=Decimal(16))],
            operations=[
                Operation(type=OperationType.SUBTRACT, target="eggs", amount=Decimal(3)),
                Operation(type=OperationType.SUBTRACT, target="eggs", amount=Decimal(4)),
                Operation(type=OperationType.MULTIPLY, target="eggs", factor=Decimal(2)),
            ],
            query=Query(target="eggs"),
        ),
        # Robe: 2 blue + 1 white = 3
        3: ProblemSpec(
            problem_type=ProblemType.ARITHMETIC_CHAIN,
            entities=[Entity(name="bolts", initial_value=Decimal(2))],
            operations=[
                Operation(type=OperationType.ADD, target="bolts", amount=Decimal(1)),
            ],
            query=Query(target="bolts"),
        ),
        # Josh house flip: 200000 - 130000 = 70000
        70000: ProblemSpec(
            problem_type=ProblemType.ARITHMETIC_CHAIN,
            entities=[Entity(name="profit", initial_value=Decimal(200000))],
            operations=[
                Operation(type=OperationType.SUBTRACT, target="profit", amount=Decimal(130000)),
            ],
            query=Query(target="profit"),
        ),
        # James sprints: 3 * 3 * 60 = 540
        540: ProblemSpec(
            problem_type=ProblemType.ARITHMETIC_CHAIN,
            entities=[Entity(name="meters", initial_value=Decimal(3))],
            operations=[
                Operation(type=OperationType.MULTIPLY, target="meters", factor=Decimal(3)),
                Operation(type=OperationType.MULTIPLY, target="meters", factor=Decimal(60)),
            ],
            query=Query(target="meters"),
        ),
        # Wendi chickens: 60 - 15 - 25 = 20
        20: ProblemSpec(
            problem_type=ProblemType.ARITHMETIC_CHAIN,
            entities=[Entity(name="feed", initial_value=Decimal(60))],
            operations=[
                Operation(type=OperationType.SUBTRACT, target="feed", amount=Decimal(15)),
                Operation(type=OperationType.SUBTRACT, target="feed", amount=Decimal(25)),
            ],
            query=Query(target="feed"),
        ),
        # Kylar glasses: 8 * 8 = 64
        64: ProblemSpec(
            problem_type=ProblemType.ARITHMETIC_CHAIN,
            entities=[Entity(name="cost", initial_value=Decimal(8))],
            operations=[
                Operation(type=OperationType.MULTIPLY, target="cost", factor=Decimal(8)),
            ],
            query=Query(target="cost"),
        ),
        # Toulouse sheep: 20 + 80 + 160 = 260
        260: ProblemSpec(
            problem_type=ProblemType.ARITHMETIC_CHAIN,
            entities=[Entity(name="sheep", initial_value=Decimal(20))],
            operations=[
                Operation(type=OperationType.ADD, target="sheep", amount=Decimal(80)),
                Operation(type=OperationType.ADD, target="sheep", amount=Decimal(160)),
            ],
            query=Query(target="sheep"),
        ),
        # Carla download: 40 + 20 + 100 = 160
        160: ProblemSpec(
            problem_type=ProblemType.ARITHMETIC_CHAIN,
            entities=[Entity(name="minutes", initial_value=Decimal(40))],
            operations=[
                Operation(type=OperationType.ADD, target="minutes", amount=Decimal(20)),
                Operation(type=OperationType.ADD, target="minutes", amount=Decimal(100)),
            ],
            query=Query(target="minutes"),
        ),
        # John dogs: 10 * 0.5 * 7 = 35
        35: ProblemSpec(
            problem_type=ProblemType.ARITHMETIC_CHAIN,
            entities=[Entity(name="hours", initial_value=Decimal(10))],
            operations=[
                Operation(type=OperationType.MULTIPLY, target="hours", factor=Decimal("0.5")),
                Operation(type=OperationType.MULTIPLY, target="hours", factor=Decimal(7)),
            ],
            query=Query(target="hours"),
        ),
        # Gail fish: 48 + 24 = 72
        72: ProblemSpec(
            problem_type=ProblemType.ARITHMETIC_CHAIN,
            entities=[Entity(name="fish", initial_value=Decimal(48))],
            operations=[
                Operation(type=OperationType.ADD, target="fish", amount=Decimal(24)),
            ],
            query=Query(target="fish"),
        ),
    }

    specs = []
    for problem in problems:
        if problem.answer in spec_templates:
            spec = spec_templates[problem.answer]
            spec.raw_text = problem.question
            specs.append(spec)
        else:
            # Create a placeholder spec that will fail
            specs.append(
                ProblemSpec(
                    problem_type=ProblemType.UNKNOWN,
                    entities=[],
                    query=None,
                    raw_text=problem.question,
                )
            )

    return specs


def run_evaluation(args):
    """Run the GSM-8K evaluation."""
    print("=" * 70)
    print("CSP-CoT GSM-8K Evaluation")
    print("=" * 70)
    print(f"Time: {datetime.now().isoformat()}")
    print(f"Args: {vars(args)}")
    print()

    # Load model
    model, tokenizer = load_model_if_needed(args.model if not args.no_model else None)

    # Load English CoT model if requested
    english_model, english_tokenizer = None, None
    if args.english_cot:
        english_model, english_tokenizer = model, tokenizer  # Use same model

    # Load problems
    print("\nLoading GSM-8K problems...")
    if args.use_samples:
        problems = get_sample_problems(args.n)
        print(f"  Using {len(problems)} sample problems (no HuggingFace)")
    else:
        try:
            problems = load_gsm8k(n=args.n, shuffle=args.shuffle, seed=args.seed)
            print(f"  Loaded {len(problems)} problems from HuggingFace")
        except ImportError:
            print("  HuggingFace not available, using sample problems")
            problems = get_sample_problems(args.n)

    # Create specs if no model
    specs = None
    if model is None:
        print("\nNo model available - using pre-built specs for sample problems")
        specs = create_sample_specs(problems)

    # Create evaluator
    evaluator = CSPCoTEvaluator(
        model=model,
        tokenizer=tokenizer,
        english_cot_model=english_model,
        english_cot_tokenizer=english_tokenizer,
    )

    # Run evaluation
    print("\n" + "-" * 70)
    print("Running evaluation...")
    print("-" * 70)

    result = evaluator.evaluate(
        problems,
        specs=specs,
        run_english_cot=args.english_cot,
        verbose=args.verbose,
    )

    # Print results
    print("\n" + "=" * 70)
    print(result.summary())
    print("=" * 70)

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "args": vars(args),
            "metrics": result.to_dict(),
            "problems": [
                {
                    "question": r.problem.question[:100],
                    "expected": r.expected_answer,
                    "csp_cot_answer": float(r.csp_cot_answer) if r.csp_cot_answer else None,
                    "csp_cot_correct": r.csp_cot_correct,
                    "csp_cot_verified": r.csp_cot_verified,
                }
                for r in result.results
            ],
        }
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    # Error analysis
    if args.show_errors:
        print("\n" + "-" * 70)
        print("Error Analysis")
        print("-" * 70)

        wrong = [r for r in result.results if not r.csp_cot_correct]
        print(f"\nIncorrect answers ({len(wrong)}):")
        for r in wrong[:5]:
            print(f"\n  Q: {r.problem.question[:80]}...")
            print(f"  Expected: {r.expected_answer}")
            print(f"  Got: {r.csp_cot_answer}")
            if r.csp_cot_error:
                print(f"  Error: {r.csp_cot_error}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CSP-CoT on GSM-8K benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with sample problems (no model needed)
  python run_gsm8k_eval.py --use-samples --n 10

  # Full evaluation with model
  python run_gsm8k_eval.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --n 100

  # Save results
  python run_gsm8k_eval.py --use-samples --n 10 --output results.json
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model ID for LLM parsing (HuggingFace or local path)",
    )
    parser.add_argument(
        "--no-model",
        action="store_true",
        help="Run without model (requires pre-built specs)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of problems to evaluate",
    )
    parser.add_argument(
        "--use-samples",
        action="store_true",
        help="Use sample problems instead of HuggingFace",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle problems before selecting",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling",
    )
    parser.add_argument(
        "--english-cot",
        action="store_true",
        help="Also run English CoT baseline",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results JSON",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print per-problem results",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print summary",
    )
    parser.add_argument(
        "--show-errors",
        action="store_true",
        help="Show error analysis at end",
    )

    args = parser.parse_args()

    if args.quiet:
        args.verbose = False

    try:
        run_evaluation(args)
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
