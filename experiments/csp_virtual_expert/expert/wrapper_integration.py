"""
Integration of CSP Virtual Expert with VirtualMoEWrapper.

This module provides utilities for integrating the CSP Virtual Expert
into the Lazarus inference pipeline.

Usage:
    from experiments.csp_virtual_expert.expert.wrapper_integration import (
        create_csp_wrapper,
        register_csp_expert,
    )

    # Option 1: Create new wrapper with CSP expert
    wrapper = create_csp_wrapper(model, tokenizer)
    result = wrapper.solve("Schedule meetings: ...")

    # Option 2: Add CSP expert to existing wrapper
    register_csp_expert(existing_wrapper)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import mlx.nn as nn


def register_csp_expert(wrapper: Any) -> None:
    """
    Register CSP expert plugin with an existing VirtualMoEWrapper.

    Args:
        wrapper: VirtualMoEWrapper instance
    """
    from .csp_plugin import CSPVirtualExpertPlugin

    plugin = CSPVirtualExpertPlugin()
    wrapper.register_plugin(plugin)
    print(f"Registered CSP expert plugin (priority={plugin.priority})")


def create_csp_wrapper(
    model: "nn.Module",
    tokenizer: Any,
    model_id: str = "unknown",
    include_math: bool = True,
    include_compiler: bool = False,
) -> Any:
    """
    Create a VirtualMoEWrapper with CSP expert pre-registered.

    Args:
        model: The model to wrap
        tokenizer: The tokenizer
        model_id: Model identifier
        include_math: Include math expert (default True)
        include_compiler: Include compiler expert (default False)

    Returns:
        Configured VirtualMoEWrapper
    """
    from chuk_lazarus.inference.virtual_experts import VirtualMoEWrapper
    from chuk_lazarus.inference.virtual_experts.registry import VirtualExpertRegistry

    from .csp_plugin import CSPVirtualExpertPlugin

    # Create registry
    registry = VirtualExpertRegistry()

    # Register experts in priority order
    if include_math:
        from chuk_lazarus.inference.virtual_experts.plugins.math import MathExpertPlugin
        registry.register(MathExpertPlugin())

    # CSP expert (priority 9)
    registry.register(CSPVirtualExpertPlugin())

    if include_compiler:
        try:
            from experiments.compiler_virtual_expert.compiler_plugin import CompilerExpertPlugin
            registry.register(CompilerExpertPlugin())
        except ImportError:
            print("Warning: Compiler expert not available")

    # Create wrapper
    wrapper = VirtualMoEWrapper(
        model=model,
        tokenizer=tokenizer,
        model_id=model_id,
        registry=registry,
    )

    return wrapper


def test_integration(model_path: str) -> None:
    """
    Test the CSP expert integration.

    Args:
        model_path: Path to model directory
    """
    print("=" * 70)
    print("CSP Virtual Expert - Wrapper Integration Test")
    print("=" * 70)

    # Load model
    print(f"\nLoading model: {model_path}")
    from chuk_lazarus.models_v2.loader import load_model

    loaded = load_model(model_path)

    # Create wrapper with CSP expert
    print("Creating wrapper with CSP expert...")
    wrapper = create_csp_wrapper(
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        model_id=model_path,
    )

    # List registered plugins
    print("\nRegistered plugins:")
    for plugin in wrapper.registry.get_all():
        print(f"  - {plugin.name} (priority={plugin.priority})")

    # Calibrate
    print("\nCalibrating...")
    wrapper.calibrate()
    print("Calibration complete")

    # Test cases
    test_cases = [
        {
            "name": "Math (should use math expert)",
            "prompt": "127 * 89 = ",
        },
        {
            "name": "CSP Scheduling (should use CSP expert)",
            "prompt": """TASKS: [Alice:2hr, Bob:1hr, Carol:1.5hr]
CONSTRAINTS: [no_overlap(Alice, Bob)]
OBJECTIVE: minimize_makespan
SOLVE:""",
        },
        {
            "name": "Factual (should use model)",
            "prompt": "The capital of France is",
        },
    ]

    print("\n" + "=" * 70)
    print("Running Tests")
    print("=" * 70)

    for test in test_cases:
        print(f"\n--- {test['name']} ---")
        print(f"Prompt: {test['prompt'][:50]}...")

        result = wrapper.solve(test["prompt"], verbose=True)

        print(f"Answer: {result.answer}")
        print(f"Used virtual expert: {result.used_virtual_expert}")
        if result.used_virtual_expert:
            print(f"Plugin: {result.plugin_name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    test_integration(args.model)
