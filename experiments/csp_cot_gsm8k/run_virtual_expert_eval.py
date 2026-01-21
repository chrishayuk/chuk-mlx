#!/usr/bin/env python3
"""
CSP-CoT Virtual Expert Evaluation.

Tests the MathWordProblemExpert with LLM-based action extraction on GSM-8K.

Usage:
    # Quick test with sample problems
    python run_virtual_expert_eval.py --n 3

    # With specific model
    python run_virtual_expert_eval.py --model mlx-community/Qwen2.5-0.5B-Instruct-4bit --n 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.csp_cot_gsm8k.virtual_expert import (
    MathWordProblemExpert,
    COT_EXAMPLES,
    SCHEMA_DOCS,
)
from experiments.csp_cot_gsm8k.evaluation.gsm8k_loader import get_sample_problems

# Virtual expert imports
from chuk_virtual_expert.models import VirtualExpertAction
from chuk_virtual_expert.dispatch import FewShotExtractor, Dispatcher
from chuk_virtual_expert.registry_v2 import ExpertRegistry


class LLMActionExtractor:
    """
    Uses an LLM to extract structured actions from queries.

    Built specifically for GSM-8K math word problems.
    """

    def __init__(self, model, tokenizer, expert: MathWordProblemExpert):
        self.model = model
        self.tokenizer = tokenizer
        self.expert = expert
        self._build_prompt_template()

    def _build_prompt_template(self):
        """Build the few-shot prompt template."""
        examples_text = ""
        for ex in COT_EXAMPLES[:5]:  # Use first 5 examples (more GSM-8K style)
            examples_text += f'Query: "{ex["query"]}"\n'
            examples_text += f'Action: {json.dumps(ex["action"], indent=2)}\n\n'

        # Note: Using %s instead of .format() to avoid issues with JSON braces
        self.prompt_template = """You are a math problem parser. Extract structured parameters for solving.

## Available Expert: math_word_problem

Solves GSM-8K style word problems with verifiable traces.

%s

## Examples

%s
Query: "%s"
Action:"""
        self.schema_docs = SCHEMA_DOCS
        self.examples_text = examples_text

    def extract(
        self,
        query: str,
        available_experts: list[str],
    ) -> VirtualExpertAction:
        """Extract a structured action using the LLM."""
        from mlx_lm import generate

        prompt = self.prompt_template % (self.schema_docs, self.examples_text, query)

        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=300,
            verbose=False,
        )

        print(f"  LLM response: {response[:500]}...")
        return self._parse_response(response)

    def _parse_response(self, response: str) -> VirtualExpertAction:
        """Parse LLM response into VirtualExpertAction."""
        try:
            # Find JSON in response
            start = response.find("{")
            if start == -1:
                return VirtualExpertAction.none_action("No JSON found")

            # Count braces to find end
            depth = 0
            end = start
            in_string = False
            escape_next = False

            for i, char in enumerate(response[start:], start):
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

            json_str = response[start:end]
            data = json.loads(json_str)

            return VirtualExpertAction(**data)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            return VirtualExpertAction.none_action(f"Parse error: {e}")


def run_evaluation(args):
    """Run the virtual expert evaluation."""
    print("=" * 70)
    print("CSP-CoT Virtual Expert Evaluation")
    print("=" * 70)
    print(f"Time: {datetime.now().isoformat()}")
    print()

    # Create expert
    expert = MathWordProblemExpert()
    print(f"Expert: {expert.name}")
    print(f"Operations: {expert.get_operations()}")

    # Load model if specified
    model, tokenizer = None, None
    if args.model:
        print(f"\nLoading model: {args.model}")
        try:
            from mlx_lm import load
            model, tokenizer = load(args.model)
            print("Model loaded!")
        except Exception as e:
            print(f"Failed to load model: {e}")
            return

    # Get sample problems
    print(f"\nLoading {args.n} sample problems...")
    problems = get_sample_problems(args.n)
    print(f"Loaded {len(problems)} problems")

    # Create manual specs for comparison (ground truth)
    from experiments.csp_cot_gsm8k.run_gsm8k_eval import create_sample_specs
    ground_truth_specs = create_sample_specs(problems)

    print("\n" + "-" * 70)
    print("Running evaluation...")
    print("-" * 70)

    results = []
    correct = 0
    verified = 0
    parse_success = 0

    for i, (problem, gt_spec) in enumerate(zip(problems, ground_truth_specs)):
        print(f"\n[{i+1}/{len(problems)}] {problem.question[:60]}...")
        print(f"  Expected: {problem.answer}")

        if model and tokenizer:
            # Use LLM to extract action
            extractor = LLMActionExtractor(model, tokenizer, expert)
            action = extractor.extract(problem.question, ["math_word_problem"])

            print(f"  Extracted: expert={action.expert}, op={action.operation}")
            if action.parameters:
                print(f"  Parameters: {json.dumps(action.parameters, default=str)[:100]}...")

            if action.expert == "math_word_problem":
                parse_success += 1
                result = expert.execute(action)

                if result.success and result.data:
                    answer = result.data.get("answer")
                    is_verified = result.data.get("verified", False)

                    print(f"  Answer: {answer}")
                    print(f"  Verified: {is_verified}")

                    if answer is not None and abs(answer - problem.answer) < 0.01:
                        correct += 1
                        print("  Status: CORRECT")
                    else:
                        print("  Status: WRONG")

                    if is_verified:
                        verified += 1
                else:
                    print(f"  Error: {result.error if hasattr(result, 'error') else 'Unknown'}")
            else:
                print("  Status: Not routed to math expert")
        else:
            # Use ground truth spec directly
            from chuk_virtual_expert.models import VirtualExpertAction

            if gt_spec.is_valid():
                parse_success += 1
                action = VirtualExpertAction(
                    expert="math_word_problem",
                    operation="solve",
                    parameters={
                        "problem_type": gt_spec.problem_type.value,
                        "entities": [
                            {
                                "name": e.name,
                                "initial_value": float(e.initial_value) if e.initial_value else None,
                            }
                            for e in gt_spec.entities
                        ],
                        "operations": [
                            {
                                "type": o.type.value,
                                "target": o.target,
                                "source": o.source,
                                "amount": float(o.amount) if o.amount else None,
                                "factor": float(o.factor) if o.factor else None,
                            }
                            for o in gt_spec.operations
                        ],
                        "query": {
                            "target": gt_spec.query.target,
                        } if gt_spec.query else None,
                    },
                )

                result = expert.execute(action)

                if result.success and result.data:
                    answer = result.data.get("answer")
                    is_verified = result.data.get("verified", False)

                    print(f"  Answer: {answer}")
                    print(f"  Verified: {is_verified}")

                    if answer is not None and abs(answer - problem.answer) < 0.01:
                        correct += 1
                        print("  Status: CORRECT")
                    else:
                        print("  Status: WRONG")

                    if is_verified:
                        verified += 1

    # Summary
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)
    print(f"Total:        {len(problems)}")
    print(f"Parse rate:   {parse_success}/{len(problems)} ({parse_success/len(problems)*100:.1f}%)")
    print(f"Correct:      {correct}/{len(problems)} ({correct/len(problems)*100:.1f}%)")
    print(f"Verified:     {verified}/{len(problems)} ({verified/len(problems)*100:.1f}%)")

    if model:
        print(f"\nModel: {args.model}")
    else:
        print("\nNote: Using ground truth specs (no LLM parsing)")


def main():
    parser = argparse.ArgumentParser(
        description="CSP-CoT Virtual Expert Evaluation",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="MLX model ID for LLM parsing",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Number of problems to evaluate",
    )

    args = parser.parse_args()

    try:
        run_evaluation(args)
    except KeyboardInterrupt:
        print("\nInterrupted")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
