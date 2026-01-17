"""Handler for expert interference analysis.

Analyzes how multiple experts interact when activated together:
- Compare k=1 vs k=4 outputs
- Measure linearity of multi-expert combination
- Identify cases where multi-expert hurts quality
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argparse import Namespace

logger = logging.getLogger(__name__)

DEFAULT_PROMPTS = [
    "127 * 89 = ",
    "def fibonacci(n):",
    "The capital of France is",
    "Once upon a time in a",
    "To solve this equation",
]


def handle_expert_interference(args: Namespace) -> None:
    """Handle expert interference analysis.

    Args:
        args: Command arguments with model, k values, etc.
    """
    asyncio.run(_async_handle_expert_interference(args))


async def _async_handle_expert_interference(args: Namespace) -> dict:
    """Async implementation of expert interference analysis."""
    from chuk_lazarus.introspection.moe import ExpertRouter

    model_id = args.model
    max_tokens = getattr(args, "max_tokens", 20)
    k_values = getattr(args, "k_values", [1, 2, 4])

    print(f"Loading model: {model_id}")
    router = await ExpertRouter.from_pretrained(model_id)

    prompts = DEFAULT_PROMPTS
    if hasattr(args, "prompt") and args.prompt:
        prompts = [args.prompt]

    print(f"Testing {len(prompts)} prompts with k values: {k_values}")

    # Collect results for each k value
    k_results: dict[int, list[str]] = {k: [] for k in k_values}
    baseline_results: list[str] = []

    for prompt in prompts:
        # Get baseline (default k)
        baseline = router._generate_normal_sync(prompt, max_tokens=max_tokens)
        baseline_results.append(baseline)

        # Test each k value
        for k in k_values:
            result = await router.generate_with_topk(prompt, k, max_tokens=max_tokens)
            k_results[k].append(result.response)

    # Analyze interference
    print("\n" + "=" * 70)
    print("EXPERT INTERFERENCE ANALYSIS")
    print("=" * 70)

    print(f"\nDefault k: {router.info.num_experts_per_tok}")
    print(f"Tested k values: {k_values}")

    # Compare outputs
    print("\n" + "-" * 70)
    print("OUTPUT COMPARISON")
    print("-" * 70)

    for i, prompt in enumerate(prompts):
        print(f"\nPrompt: {prompt[:40]}...")
        print(f"  Baseline: {baseline_results[i][:50]}...")
        for k in k_values:
            output = k_results[k][i]
            match = "✓" if output == baseline_results[i] else "≠"
            print(f"  k={k}: {output[:50]}... {match}")

    # Compute quality metrics
    quality_scores: dict[int, float] = {}
    for k in k_values:
        # Simple metric: average response length ratio to baseline
        total_ratio = 0.0
        for i in range(len(prompts)):
            baseline_len = len(baseline_results[i]) or 1
            k_len = len(k_results[k][i])
            total_ratio += min(k_len / baseline_len, 2.0)
        quality_scores[k] = total_ratio / len(prompts) if prompts else 0.0

    print("\n" + "-" * 70)
    print("QUALITY BY K VALUE")
    print("-" * 70)

    for k, score in sorted(quality_scores.items()):
        bar_len = int(score * 10)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"k={k}: {bar} {score:.2f}")

    # Identify interference cases
    interference_cases = []
    for i, prompt in enumerate(prompts):
        for k in k_values:
            if k > 1 and k_results[k][i] != k_results[1][i]:
                # Multi-expert gives different result than single expert
                k1_quality = len(k_results[1][i])
                kn_quality = len(k_results[k][i])
                if kn_quality < k1_quality * 0.8:
                    interference_cases.append({
                        "prompt": prompt[:50],
                        "k": k,
                        "quality_drop": 1.0 - (kn_quality / k1_quality),
                    })

    if interference_cases:
        print("\n" + "-" * 70)
        print("INTERFERENCE CASES (quality drop when using more experts)")
        print("-" * 70)
        for case in interference_cases:
            print(f"  k={case['k']}: {case['quality_drop']:.1%} drop - {case['prompt']}...")

    return {
        "k_values": k_values,
        "quality_scores": quality_scores,
        "num_interference_cases": len(interference_cases),
        "interference_cases": interference_cases,
    }


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze expert interference")
    parser.add_argument("-m", "--model", required=True, help="Model ID")
    parser.add_argument("-p", "--prompt", help="Single prompt to test")
    parser.add_argument(
        "-k", "--k-values", nargs="+", type=int, default=[1, 2, 4],
        help="K values to test"
    )
    parser.add_argument(
        "-t", "--max-tokens", type=int, default=20, help="Max tokens to generate"
    )

    args = parser.parse_args()
    result = asyncio.run(_async_handle_expert_interference(args))
    return result


if __name__ == "__main__":
    main()
