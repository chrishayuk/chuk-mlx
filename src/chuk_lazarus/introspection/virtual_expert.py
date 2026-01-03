"""
Virtual Expert Introspection and Demo Tools.

This module re-exports the core virtual expert classes from inference
and provides demo/analysis functions for introspection purposes.

For production use, import directly from chuk_lazarus.inference:

    from chuk_lazarus.inference import (
        VirtualMoEWrapper,
        VirtualExpertPlugin,
        MathExpertPlugin,
    )

For demos and analysis:

    from chuk_lazarus.introspection import (
        demo_virtual_expert,
        demo_all_approaches,
    )
"""

from __future__ import annotations

import re
from typing import Any

import mlx.nn as nn
import numpy as np

# Re-export core classes from inference
from chuk_lazarus.inference.virtual_expert import (
    MathExpertPlugin,
    SafeMathEvaluator,
    VirtualExpertAnalysis,
    VirtualExpertApproach,
    VirtualExpertPlugin,
    VirtualExpertRegistry,
    VirtualExpertResult,
    VirtualMoEWrapper,
    VirtualRouter,
    create_virtual_expert_wrapper,
    get_default_registry,
)

# Legacy compatibility aliases
ExpertHijacker = VirtualMoEWrapper
VirtualExpertSlot = VirtualMoEWrapper
HybridEmbeddingInjector = VirtualMoEWrapper


def demo_virtual_expert(
    model: nn.Module,
    tokenizer: Any,
    model_id: str = "unknown",
    problems: list[str] | None = None,
) -> VirtualExpertAnalysis:
    """
    Demo the virtual expert system.

    Args:
        model: The MoE model
        tokenizer: The tokenizer
        model_id: Model identifier
        problems: List of problems to test (defaults to arithmetic)

    Returns:
        VirtualExpertAnalysis with results
    """
    if problems is None:
        problems = [
            "2 + 2 = ",
            "5 * 5 = ",
            "10 - 3 = ",
            "6 * 7 = ",
            "25 + 17 = ",
            "100 - 37 = ",
            "23 * 17 = ",
            "127 * 89 = ",
            "456 * 78 = ",
            "999 * 888 = ",
        ]

    print("\n" + "=" * 70)
    print("VIRTUAL EXPERT DEMO")
    print("=" * 70)

    wrapper = VirtualMoEWrapper(model, tokenizer, model_id)

    print("\nCalibrating virtual expert routing...")
    wrapper.calibrate()
    print("Calibration complete.")

    print("\nRunning benchmark...\n")
    analysis = wrapper.benchmark(problems)

    print(f"{'Prompt':<25} {'Model':<15} {'Virtual':<15} {'Plugin':<10} {'V?':<5}")
    print("-" * 75)

    for result in analysis.results:
        model_answer = wrapper._generate_direct(result.prompt)[:12]
        virtual_answer = result.answer[:12]
        plugin = result.plugin_name or "N/A"
        used = "YES" if result.used_virtual_expert else "no"
        correct = "✓" if result.is_correct else "✗"

        print(f"{result.prompt:<25} {model_answer:<15} {virtual_answer:<15} {plugin:<10} {used:<5} {correct}")

    print("\n" + "-" * 75)
    print(f"Model-only accuracy:   {analysis.model_accuracy:.1%}")
    print(f"With virtual expert:   {analysis.virtual_accuracy:.1%}")
    print(f"Improvement:           {analysis.virtual_accuracy - analysis.model_accuracy:+.1%}")
    print(f"Virtual expert used:   {analysis.times_virtual_used}/{analysis.total_problems}")

    if analysis.plugins_used:
        print(f"Plugins used:")
        for name, count in analysis.plugins_used.items():
            print(f"  - {name}: {count}")

    print("=" * 70)

    return analysis


def demo_all_approaches(
    model: nn.Module,
    tokenizer: Any,
    model_id: str = "unknown",
    problems: list[str] | None = None,
) -> dict[str, VirtualExpertAnalysis]:
    """
    Demo the virtual expert system.

    Note: This now uses the unified plugin-based approach.
    The "approaches" terminology is kept for backwards compatibility.

    Returns:
        Dict with single key "virtual_slot" containing analysis
    """
    analysis = demo_virtual_expert(model, tokenizer, model_id, problems)
    return {"virtual_slot": analysis}


def create_virtual_expert(
    model: nn.Module,
    tokenizer: Any,
    approach: str = "virtual_slot",
    model_id: str = "unknown",
    **kwargs,
) -> VirtualMoEWrapper:
    """
    Factory function for backwards compatibility.

    Note: The 'approach' parameter is ignored - all approaches now use
    the unified VirtualMoEWrapper with plugins.
    """
    return VirtualMoEWrapper(model, tokenizer, model_id, **kwargs)


__all__ = [
    # Core classes (re-exported from inference)
    "VirtualExpertPlugin",
    "VirtualExpertRegistry",
    "VirtualExpertResult",
    "VirtualExpertAnalysis",
    "VirtualExpertApproach",
    "VirtualMoEWrapper",
    "VirtualRouter",
    "MathExpertPlugin",
    "SafeMathEvaluator",
    "create_virtual_expert_wrapper",
    "get_default_registry",
    # Legacy aliases
    "ExpertHijacker",
    "VirtualExpertSlot",
    "HybridEmbeddingInjector",
    # Demo functions
    "demo_virtual_expert",
    "demo_all_approaches",
    "create_virtual_expert",
]
