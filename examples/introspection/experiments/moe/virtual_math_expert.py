#!/usr/bin/env python3
"""
Virtual Math Expert Demo - Three Approaches

This script demonstrates three different approaches for creating a "virtual
math expert" that can intercept MoE routing and delegate math computations
to Python.

The key insight: Instead of relying on the model to do arithmetic (which it
often gets wrong), we can make the router learn to delegate to an external
"expert" that actually computes correctly.

Approaches:

1. **Expert Hijacking**: Intercept an existing expert's forward pass when
   math is detected, replacing its output with computed results.

2. **Virtual Expert Slot**: Add a virtual "tool" expert that the router can
   select. When selected, triggers Python computation instead of neural
   expert weights.

3. **Hybrid Embedding Injection**: Use introspection to detect model
   confidence. When low, compute externally and inject the result as an
   embedding into the residual stream.

Usage:
    # Run demo with default model
    uv run python examples/introspection/experiments/moe/virtual_math_expert.py

    # Specify model
    uv run python examples/introspection/experiments/moe/virtual_math_expert.py \
        --model mlx-community/gemma-3-4b-it-bf16

    # Test single approach
    uv run python examples/introspection/experiments/moe/virtual_math_expert.py \
        --approach hybrid \
        --prompt "127 * 89 = "

    # Run full benchmark
    uv run python examples/introspection/experiments/moe/virtual_math_expert.py \
        --benchmark
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))


def load_model(model_id: str):
    """Load model and tokenizer."""
    from chuk_lazarus.inference.loader import DType, HFLoader
    from chuk_lazarus.models_v2.families.registry import detect_model_family, get_family_info

    print(f"Loading model: {model_id}")

    result = HFLoader.download(model_id)
    model_path = result.model_path

    with open(model_path / "config.json") as f:
        config_data = json.load(f)

    family_type = detect_model_family(config_data)
    if family_type is None:
        raise ValueError(f"Unsupported model: {model_id}")

    family_info = get_family_info(family_type)
    config = family_info.config_class.from_hf_config(config_data)
    model = family_info.model_class(config)

    HFLoader.apply_weights_to_model(model, model_path, config, dtype=DType.BFLOAT16)
    tokenizer = HFLoader.load_tokenizer(model_path)

    print(f"Model loaded: {len(list(model.model.layers))} layers")

    return model, tokenizer


def demo_single_problem(
    model,
    tokenizer,
    model_id: str,
    approach: str,
    prompt: str,
):
    """Demo a single problem with one approach."""
    from chuk_lazarus.introspection.virtual_expert import create_virtual_expert

    expert = create_virtual_expert(model, tokenizer, approach, model_id)
    expert.compare(prompt)


def demo_approach_comparison(
    model,
    tokenizer,
    model_id: str,
    prompt: str,
):
    """Compare model-only vs virtual expert on a single problem."""
    from chuk_lazarus.introspection.virtual_expert import (
        MathExpertPlugin,
        VirtualMoEWrapper,
    )

    math_eval = MathExpertPlugin()
    _, correct = math_eval.extract_and_evaluate(prompt)

    print("\n" + "=" * 70)
    print("VIRTUAL EXPERT COMPARISON")
    print("=" * 70)
    print(f"Prompt: {prompt}")
    print(f"Correct answer: {correct}")
    print("-" * 70)

    wrapper = VirtualMoEWrapper(model, tokenizer, model_id)
    wrapper.calibrate()

    # Model alone
    print("\n[Model Only]")
    model_answer = wrapper._generate_direct(prompt)
    print(f"  Answer: {model_answer}")

    # With virtual expert
    print("\n[Virtual Expert]")
    result = wrapper.solve(prompt)
    print(f"  Answer: {result.answer}")
    print(f"  Correct: {result.is_correct}")
    print(f"  Plugin: {result.plugin_name}")
    print(
        f"  Virtual selected: {result.virtual_expert_selected_count}/{result.total_tokens} tokens"
    )
    print(f"  Used virtual: {result.used_virtual_expert}")

    print("=" * 70)


def run_benchmark(
    model,
    tokenizer,
    model_id: str,
):
    """Run full benchmark comparing all approaches."""
    from chuk_lazarus.introspection.virtual_expert import demo_all_approaches

    problems = [
        # Trivial (model should get these right)
        "2 + 2 = ",
        "5 * 5 = ",
        "10 - 3 = ",
        # Easy
        "6 * 7 = ",
        "25 + 17 = ",
        "100 - 37 = ",
        # Medium
        "23 * 17 = ",
        "156 + 287 = ",
        "500 - 123 = ",
        # Hard (model will likely fail)
        "127 * 89 = ",
        "456 * 78 = ",
        "999 * 888 = ",
        "1234 + 5678 = ",
        # Very hard
        "999 * 999 = ",
        "12345 + 67890 = ",
    ]

    results = demo_all_approaches(model, tokenizer, model_id, problems)

    # Detailed per-problem breakdown
    print("\n" + "=" * 70)
    print("PER-PROBLEM BREAKDOWN")
    print("=" * 70)

    for name, analysis in results.items():
        print(f"\n{name.upper()}")
        print("-" * 50)
        for result in analysis.results:
            status = "✓" if result.is_correct else "✗"
            virtual = "V" if result.used_virtual_expert else "M"
            print(
                f"  {status} [{virtual}] {result.prompt:<20} -> {result.answer:<15} (expected: {result.correct_answer})"
            )


def interactive_mode(model, tokenizer, model_id: str):
    """Interactive REPL for testing virtual experts."""
    from chuk_lazarus.introspection.virtual_expert import (
        ExpertHijacker,
        HybridEmbeddingInjector,
        VirtualExpertSlot,
    )

    print("\n" + "=" * 70)
    print("VIRTUAL MATH EXPERT - INTERACTIVE MODE")
    print("=" * 70)
    print("Commands:")
    print("  <expression>     - Evaluate with all approaches")
    print("  !approach <n>    - Set default approach (1, 2, or 3)")
    print("  !threshold <f>   - Set confidence threshold")
    print("  !quit            - Exit")
    print("=" * 70)

    # Initialize all approaches
    hijacker = ExpertHijacker(model, tokenizer, model_id)
    slot = VirtualExpertSlot(model, tokenizer, model_id)
    hybrid = HybridEmbeddingInjector(model, tokenizer, model_id)

    current_approach = 3  # Default to hybrid

    while True:
        try:
            prompt = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not prompt:
            continue

        if prompt.startswith("!"):
            parts = prompt[1:].split()
            cmd = parts[0].lower()

            if cmd == "quit":
                print("Goodbye!")
                break
            elif cmd == "approach" and len(parts) > 1:
                try:
                    current_approach = int(parts[1])
                    print(f"Set approach to {current_approach}")
                except ValueError:
                    print("Invalid approach number")
            elif cmd == "threshold" and len(parts) > 1:
                try:
                    t = float(parts[1])
                    hybrid.confidence_threshold = t
                    slot.routing_threshold = t
                    print(f"Set threshold to {t}")
                except ValueError:
                    print("Invalid threshold")
            else:
                print("Unknown command")
            continue

        # Add "= " if missing
        if not prompt.endswith("= ") and not prompt.endswith("="):
            prompt = prompt + " = "

        # Solve with all approaches
        print()
        print(f"Prompt: {prompt}")
        print("-" * 40)

        # Model alone
        model_answer = hijacker._generate_direct(prompt)
        print(f"Model only:        {model_answer}")

        # Selected approach
        if current_approach == 1:
            result = hijacker.solve(prompt)
            print(f"Expert Hijack:     {result.answer} (expert {result.hijacked_expert_idx})")
        elif current_approach == 2:
            result = slot.solve(prompt)
            print(f"Virtual Slot:      {result.answer} (score: {result.routing_score:.3f})")
        else:
            result = hybrid.solve(prompt)
            conf = f"{result.confidence_before:.1%}" if result.confidence_before else "N/A"
            print(f"Hybrid Injection:  {result.answer} (conf: {conf})")

        print(f"Correct:           {result.is_correct}")


def main():
    parser = argparse.ArgumentParser(
        description="Virtual Math Expert Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        "-m",
        default="mlx-community/gemma-3-4b-it-bf16",
        help="Model ID to use",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        default=None,
        help="Single prompt to test",
    )
    parser.add_argument(
        "--approach",
        "-a",
        choices=["hijack", "virtual_slot", "hybrid", "all"],
        default="all",
        help="Which approach to use",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run full benchmark",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Interactive REPL mode",
    )

    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model(args.model)

    if args.interactive:
        interactive_mode(model, tokenizer, args.model)
    elif args.benchmark:
        run_benchmark(model, tokenizer, args.model)
    elif args.prompt:
        if args.approach == "all":
            demo_approach_comparison(model, tokenizer, args.model, args.prompt)
        else:
            demo_single_problem(model, tokenizer, args.model, args.approach, args.prompt)
    else:
        # Default: compare on a hard problem
        demo_approach_comparison(model, tokenizer, args.model, "127 * 89 = ")


if __name__ == "__main__":
    main()
