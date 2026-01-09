"""Handler for 'ablate' action - ablate (remove) experts from routing.

This module is a thin CLI wrapper - business logic is in AblationBenchmarkService.
"""

from __future__ import annotations

import asyncio
from argparse import Namespace

from ......introspection.datasets import get_arithmetic_benchmarks
from ......introspection.moe import ExpertRouter
from ......introspection.moe.ablation_service import (
    AblationBenchmarkResult,
    AblationBenchmarkService,
)
from ..formatters import format_ablation_result, format_header


def handle_ablate(args: Namespace) -> None:
    """Handle the 'ablate' action - remove experts from routing.

    Args:
        args: Parsed CLI arguments. Required:
            - model: Model ID
            - expert: Expert index to ablate (or --experts for multiple)
            - prompt: Input prompt

    Example:
        lazarus introspect moe-expert ablate -m openai/gpt-oss-20b -e 6 -p "127 * 89 = "
    """
    asyncio.run(_async_ablate(args))


async def _async_ablate(args: Namespace) -> None:
    """Async implementation of ablate handler."""
    # Parse expert indices - support both single and multiple
    expert_indices: list[int] = []

    if hasattr(args, "experts") and args.experts:
        try:
            expert_indices = [int(e.strip()) for e in args.experts.split(",")]
        except ValueError:
            print(f"Error: Invalid experts format: {args.experts}")
            return
    elif hasattr(args, "expert") and args.expert is not None:
        expert_indices = [args.expert]
    else:
        print("Error: --expert/-e or --experts is required for ablate action")
        return

    if not hasattr(args, "prompt") or args.prompt is None:
        print("Error: --prompt/-p is required for ablate action")
        return

    model_id = args.model
    prompt = args.prompt
    max_tokens = getattr(args, "max_tokens", 100)
    run_benchmark = getattr(args, "benchmark", False)

    print(f"Loading model: {model_id}")

    async with await ExpertRouter.from_pretrained(model_id) as router:
        # Get normal output first
        normal_output = router._generate_normal_sync(prompt, max_tokens)

        # Get ablated output
        ablated_output, stats = await router.generate_with_ablation(
            prompt,
            expert_indices,
            max_tokens=max_tokens,
        )

        output = format_ablation_result(
            normal_output,
            ablated_output,
            expert_indices,
            prompt,
            model_id,
        )
        print(output)

        # Run benchmark if requested
        if run_benchmark:
            await _run_ablation_benchmark(router, expert_indices, max_tokens)


async def _run_ablation_benchmark(
    router: ExpertRouter,
    expert_indices: list[int],
    max_tokens: int,
) -> None:
    """Run ablation on benchmark problems."""
    benchmarks = get_arithmetic_benchmarks()
    problems = benchmarks.get_all_problems()

    experts_str = ", ".join(str(e) for e in expert_indices)
    print(format_header(f"ABLATION BENCHMARK - Expert(s) {experts_str}"))

    # Build result using service
    benchmark_result = AblationBenchmarkResult(expert_indices=expert_indices)

    for problem in problems:
        # Normal generation
        normal = router._generate_normal_sync(problem.prompt, max_tokens)

        # Ablated generation
        ablated, _ = await router.generate_with_ablation(
            problem.prompt,
            expert_indices,
            max_tokens=max_tokens,
        )

        # Create result using service
        problem_result = AblationBenchmarkService.create_problem_result(
            prompt=problem.prompt,
            expected_answer=problem.answer,
            normal_output=normal,
            ablated_output=ablated,
        )
        benchmark_result.problems.append(problem_result)

        # Print row
        status = f"<- {problem_result.status}" if problem_result.status else ""
        print(f"{problem.prompt:<20} Normal: {normal:<12} Ablated: {ablated:<12} {status}")

    # Print summary using service
    print()
    print(AblationBenchmarkService.format_summary(benchmark_result))
    print("=" * 70)
