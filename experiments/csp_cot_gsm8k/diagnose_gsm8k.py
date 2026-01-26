#!/usr/bin/env python3
"""
Diagnostic script for GSM-8K evaluation.

Shows full traces for all 10 sample problems to identify failure modes.
"""

from __future__ import annotations

import asyncio
import functools
import sys
from pathlib import Path

print = functools.partial(print, flush=True)

import yaml
import mlx.core as mx

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from chuk_lazarus.models_v2.loader import load_model
from chuk_virtual_expert import ExpertRegistry, TraceVerifier
from chuk_virtual_expert_arithmetic import (
    ArithmeticExpert,
    ComparisonExpert,
    EntityTrackExpert,
    PercentageExpert,
    RateEquationExpert,
)
from experiments.csp_cot_gsm8k.evaluation.gsm8k_loader import SAMPLE_PROBLEMS


def build_registry() -> ExpertRegistry:
    registry = ExpertRegistry()
    registry.register(EntityTrackExpert())
    registry.register(ArithmeticExpert())
    registry.register(PercentageExpert())
    registry.register(RateEquationExpert())
    registry.register(ComparisonExpert())
    return registry


_registry = build_registry()
_verifier = TraceVerifier(_registry)


def load_checkpoint(model, path: str):
    """Load model weights from directory."""
    import mlx.utils
    ckpt_path = Path(path) / "weights.npz"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    weights = dict(mx.load(str(ckpt_path)))
    model_params = dict(mlx.utils.tree_flatten(model.parameters()))
    updated = 0
    for key, value in weights.items():
        if key in model_params:
            parts = key.split(".")
            obj = model
            for part in parts[:-1]:
                if part.isdigit():
                    obj = obj[int(part)]
                else:
                    obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
            updated += 1

    mx.eval(model.parameters())
    print(f"Checkpoint loaded: {path} ({updated} arrays)")


SYSTEM_PROMPT = """You are a helpful assistant with access to the following experts: entity_track, arithmetic, rate_equation, comparison, percentage"""


def format_chat_prompt(question: str) -> str:
    return f"<|system|>\n{SYSTEM_PROMPT}</s>\n<|user|>\n{question}</s>\n<|assistant|>\n```yaml\n"


def generate(model, tokenizer, prompt: str, max_tokens: int = 300) -> str:
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    generated = []
    for _ in range(max_tokens):
        output = model(input_ids)
        logits = output.logits if hasattr(output, "logits") else output
        logits = logits[:, -1, :]

        next_token = int(mx.argmax(logits).item())

        if next_token == tokenizer.eos_token_id:
            break

        generated.append(next_token)
        input_ids = mx.concatenate([input_ids, mx.array([[next_token]])], axis=1)

        text = tokenizer.decode(generated)
        if "</s>" in text:
            break
        if "```" in text and text.count("```") >= 1 and not text.strip().endswith("```yaml"):
            break

    output = tokenizer.decode(generated).strip()
    output = output.replace("</s>", "").strip()
    return output


def extract_yaml(text: str) -> str | None:
    text = text.strip()
    if text.startswith("```yaml"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if "```" in text:
        text = text.split("```")[0]
    text = text.strip()
    if not text:
        return None
    try:
        parsed = yaml.safe_load(text)
        if isinstance(parsed, dict) and "expert" in parsed:
            return text
        if isinstance(parsed, list) and all(
            isinstance(sub, dict) and "expert" in sub for sub in parsed
        ):
            return text
    except yaml.YAMLError:
        pass
    return None


def verify_yaml(yaml_str: str, expected_answer=None) -> dict:
    result = asyncio.run(_verifier.verify(yaml_str, expected_answer, tolerance=0.01))
    return {
        "parsed": result.parsed,
        "expert": result.expert,
        "trace_valid": result.trace_valid,
        "trace_error": result.trace_error,
        "computed_answer": result.computed_answer,
        "answer_correct": result.answer_correct,
        "final_state": result.final_state,
    }


def classify_failure(yaml_str: str | None, result: dict | None, expected: int) -> str:
    """Classify the failure mode."""
    if yaml_str is None:
        return "parse_fail"
    if result is None:
        return "verify_fail"
    if not result["parsed"]:
        return "parse_fail"
    if not result["trace_valid"]:
        return f"invalid_trace:{result['trace_error']}"
    if not result["answer_correct"]:
        return f"wrong_answer:{result['computed_answer']} (expected {expected})"
    return "correct"


# Problem metadata for analysis
PROBLEM_INFO = [
    {"name": "Janet's ducks", "type": "single", "pattern": "sub-sub-mul"},
    {"name": "Robe fiber", "type": "single", "pattern": "div-add"},
    {"name": "House flipping", "type": "composition", "pattern": "arith→pct→arith"},
    {"name": "James sprints", "type": "single", "pattern": "mul-mul"},
    {"name": "Wendi's chickens", "type": "single", "pattern": "mul-add-sub"},
    {"name": "Kylar's glasses", "type": "composition", "pattern": "pct→arith"},
    {"name": "Toulouse's sheep", "type": "single", "pattern": "mul-mul-add-add"},
    {"name": "Carla download", "type": "composition", "pattern": "pct→arith"},
    {"name": "John's dogs", "type": "single", "pattern": "mul-mul"},
    {"name": "Fish tanks", "type": "single", "pattern": "div-add"},
]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                       default="experiments/csp_cot_gsm8k/checkpoints/gsm8k_yaml_schema_run_16")
    args = parser.parse_args()

    print("=" * 80)
    print("  GSM-8K DIAGNOSTIC EVALUATION")
    print("=" * 80)

    # Load model
    print("\nLoading model...")
    result = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model, tokenizer = result.model, result.tokenizer

    print(f"\nLoading checkpoint: {args.checkpoint}")
    load_checkpoint(model, args.checkpoint)
    model.freeze()

    # Results tracking
    results = []
    single_correct = 0
    single_total = 0
    comp_correct = 0
    comp_total = 0

    print("\n" + "=" * 80)
    print("  DETAILED PROBLEM ANALYSIS")
    print("=" * 80)

    for idx, (problem, info) in enumerate(zip(SAMPLE_PROBLEMS, PROBLEM_INFO)):
        print(f"\n{'='*80}")
        print(f"Problem {idx+1}: {info['name']}")
        print(f"Type: {info['type']} | Pattern: {info['pattern']}")
        print(f"{'='*80}")

        print(f"\nQuestion: {problem.question}")
        print(f"Expected answer: {problem.answer}")

        # Generate
        prompt = format_chat_prompt(problem.question)
        output = generate(model, tokenizer, prompt)

        print(f"\n--- Model Output ---")
        print(output)
        print(f"--- End Output ---")

        # Verify
        yaml_str = extract_yaml(output)
        if yaml_str:
            result = verify_yaml(yaml_str, problem.answer)
            failure_mode = classify_failure(yaml_str, result, problem.answer)

            print(f"\nVerification:")
            print(f"  Expert: {result['expert']}")
            print(f"  Trace valid: {result['trace_valid']}")
            if result['trace_error']:
                print(f"  Trace error: {result['trace_error']}")
            print(f"  Computed: {result['computed_answer']}")
            print(f"  Correct: {result['answer_correct']}")
        else:
            result = None
            failure_mode = classify_failure(None, None, problem.answer)
            print(f"\nVerification: FAILED TO PARSE YAML")

        is_correct = failure_mode == "correct"
        status = "✓ CORRECT" if is_correct else f"✗ FAILED ({failure_mode})"
        print(f"\n>>> STATUS: {status}")

        # Track stats
        if info["type"] == "single":
            single_total += 1
            if is_correct:
                single_correct += 1
        else:
            comp_total += 1
            if is_correct:
                comp_correct += 1

        results.append({
            "idx": idx + 1,
            "name": info["name"],
            "type": info["type"],
            "pattern": info["pattern"],
            "expected": problem.answer,
            "computed": result["computed_answer"] if result else None,
            "correct": is_correct,
            "failure_mode": failure_mode,
        })

    # Summary
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)

    total_correct = single_correct + comp_correct
    total = single_total + comp_total

    print(f"\nOverall: {total_correct}/{total} ({100*total_correct/total:.0f}%)")
    print(f"Single-expert: {single_correct}/{single_total} ({100*single_correct/single_total:.0f}%)")
    print(f"Composition: {comp_correct}/{comp_total} ({100*comp_correct/comp_total:.0f}%)")

    print("\n--- Results Table ---")
    print(f"{'#':>2} | {'Name':<20} | {'Type':<12} | {'Pattern':<18} | {'Expected':>8} | {'Got':>8} | Status")
    print("-" * 95)
    for r in results:
        status = "✓" if r["correct"] else "✗"
        got = str(r["computed"]) if r["computed"] is not None else "FAIL"
        print(f"{r['idx']:>2} | {r['name']:<20} | {r['type']:<12} | {r['pattern']:<18} | {r['expected']:>8} | {got:>8} | {status}")

    print("\n--- Failure Analysis ---")
    failures = [r for r in results if not r["correct"]]
    if failures:
        for r in failures:
            print(f"\nProblem {r['idx']} ({r['name']}):")
            print(f"  Type: {r['type']}")
            print(f"  Pattern: {r['pattern']}")
            print(f"  Failure: {r['failure_mode']}")
    else:
        print("No failures!")


if __name__ == "__main__":
    main()
