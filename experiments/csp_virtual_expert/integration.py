"""
CSP Virtual Expert Integration Examples.

Shows how to integrate the CSP Virtual Expert with the Lazarus MoE wrapper
and standalone usage.

Usage:
    # Standalone test (no model required)
    python -m experiments.csp_virtual_expert.integration --standalone

    # With Lazarus model
    python -m experiments.csp_virtual_expert.integration --model openai/gpt-oss-7b
"""

from __future__ import annotations

import argparse
from typing import Any


def standalone_demo():
    """
    Demonstrate CSP virtual expert without a model.

    Tests extraction and solving directly.
    """
    print("=" * 60)
    print("CSP Virtual Expert - Standalone Demo")
    print("=" * 60)

    from .expert.csp_plugin import CSPVirtualExpertPlugin

    plugin = CSPVirtualExpertPlugin()

    # Test cases
    test_cases = [
        {
            "name": "Format A - Declarative Blocks",
            "input": """
Let me structure this scheduling problem:

TASKS: [Alice:2hr, Bob:1hr, Carol:1.5hr]
WINDOW: [9:00, 17:00]
CONSTRAINTS: [no_overlap(Alice, Bob), Carol before Alice]
OBJECTIVE: minimize_makespan
SOLVE:
""",
        },
        {
            "name": "Format C - Natural Structured",
            "input": """
Let me formalize the constraints:
- Tasks: Meeting (1hr), Coding (2hr), Review (30min)
- Time window: 9am to 12pm
- Constraint: Meeting before Coding
- Goal: Minimize total time
Solution:
""",
        },
        {
            "name": "Loose Format - Natural Language",
            "input": """
Schedule my day: gym (1hr), lunch meeting (1.5hr), dentist at 2pm (1hr).
The dentist appointment is fixed. I want to minimize the total time needed.
Finding optimal schedule...
""",
        },
        {
            "name": "Non-CSP - Should Not Trigger",
            "input": "What is 127 * 89?",
        },
    ]

    for case in test_cases:
        print(f"\n--- {case['name']} ---")
        print(f"Input: {case['input'][:100].strip()}...")

        can_handle = plugin.can_handle(case["input"])
        print(f"Can handle: {can_handle}")

        if can_handle:
            result = plugin.execute(case["input"])
            print(f"Result:\n{result}")
        print()


def lazarus_integration_demo(model_path: str):
    """
    Demonstrate CSP virtual expert with Lazarus MoE wrapper.

    Shows the full pipeline: model generates CoT → CSP detected → solver invoked.
    """
    print("=" * 60)
    print("CSP Virtual Expert - Lazarus Integration Demo")
    print("=" * 60)

    try:
        from chuk_lazarus.inference import load_model
        from chuk_lazarus.inference.virtual_experts import VirtualMoEWrapper
    except ImportError:
        print("chuk_lazarus not available. Install or use --standalone mode.")
        return

    from .expert.csp_plugin import CSPVirtualExpertPlugin

    # Load model
    print(f"\nLoading model: {model_path}")
    model, tokenizer, config = load_model(model_path)

    # Create wrapper with CSP expert
    print("Creating VirtualMoEWrapper with CSP expert...")
    wrapper = VirtualMoEWrapper(model, tokenizer)

    # Register CSP plugin
    csp_plugin = CSPVirtualExpertPlugin()
    wrapper.register_plugin(csp_plugin)

    # Calibrate router
    print("Calibrating router...")
    wrapper.calibrate()

    # Test prompts that should trigger CSP CoT
    test_prompts = [
        "Schedule these meetings: Alice needs 2 hours, Bob needs 1 hour, Carol needs 1.5 hours. They can't overlap. Minimize total time.",
        "I have three tasks: gym (1hr), lunch (1.5hr), and a dentist appointment fixed at 2pm (1hr). Schedule my day optimally.",
        "What is 127 * 89?",  # Should NOT trigger CSP
    ]

    print("\n--- Running Tests ---")
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt[:60]}...")

        # Compare model-only vs virtual expert
        print("\nModel only:")
        model_result = wrapper.solve(prompt, use_virtual_experts=False)
        print(f"  {model_result.answer[:200]}...")

        print("\nWith CSP expert:")
        expert_result = wrapper.solve(prompt, use_virtual_experts=True)
        print(f"  {expert_result.answer[:200]}...")

        if expert_result.used_virtual_expert:
            print(f"  [Virtual expert: {expert_result.plugin_name}]")


def registry_demo():
    """
    Demonstrate adding CSP expert to a registry with other experts.
    """
    print("=" * 60)
    print("CSP Virtual Expert - Registry Demo")
    print("=" * 60)

    try:
        from chuk_lazarus.inference.virtual_experts import VirtualExpertRegistry
        from chuk_lazarus.inference.virtual_experts.plugins import MathExpertPlugin
    except ImportError:
        print("Using standalone registry simulation")
        return

    from .expert.csp_plugin import CSPVirtualExpertPlugin

    # Try to import compiler plugin too
    try:
        from experiments.compiler_virtual_expert import CompilerExpertPlugin
        has_compiler = True
    except ImportError:
        has_compiler = False

    # Create registry
    registry = VirtualExpertRegistry()

    # Register plugins
    registry.register(MathExpertPlugin())
    registry.register(CSPVirtualExpertPlugin())
    if has_compiler:
        registry.register(CompilerExpertPlugin())

    # List registered plugins
    print("\nRegistered Plugins:")
    for plugin in registry.list_plugins():
        print(f"  - {plugin.name} (priority={plugin.priority}): {plugin.description}")

    # Test routing
    print("\n--- Routing Tests ---")
    test_cases = [
        ("What is 127 * 89?", "math"),
        ("Schedule meetings: Alice 2hr, Bob 1hr, no overlap", "csp"),
        ("```python\nprint('hello')\n```", "compiler" if has_compiler else "none"),
        ("What is the capital of France?", "none"),
    ]

    for prompt, expected in test_cases:
        matched = registry.find_handler(prompt)
        matched_name = matched.name if matched else "none"
        status = "PASS" if matched_name == expected else "FAIL"
        print(f"  [{status}] '{prompt[:40]}...' -> {matched_name} (expected: {expected})")


def benchmark_demo():
    """
    Run a simple benchmark comparing neural-only vs virtual expert.
    """
    print("=" * 60)
    print("CSP Virtual Expert - Benchmark Demo")
    print("=" * 60)

    from .expert.csp_plugin import CSPVirtualExpertPlugin
    from .data.prompts import CSP_PROMPTS

    plugin = CSPVirtualExpertPlugin()

    # Pre-formatted test cases with known solutions
    test_cases = [
        {
            "prompt": """
TASKS: [A:1hr, B:1hr, C:1hr]
WINDOW: [9:00, 12:00]
CONSTRAINTS: []
OBJECTIVE: minimize_makespan
SOLVE:
""",
            "expected_makespan": 180,  # 3 hours sequential
        },
        {
            "prompt": """
TASKS: [Meeting:1hr, Coding:2hr]
CONSTRAINTS: [Meeting before Coding]
OBJECTIVE: minimize_makespan
SOLVE:
""",
            "expected_makespan": 180,  # 3 hours
        },
    ]

    print("\nRunning benchmark...")
    correct = 0
    total = len(test_cases)

    for i, case in enumerate(test_cases):
        result = plugin.execute(case["prompt"])
        if result and "Schedule:" in result:
            correct += 1
            status = "PASS"
        else:
            status = "FAIL"

        print(f"  [{status}] Test {i+1}: {'Got schedule' if result else 'No result'}")

    print(f"\nResults: {correct}/{total} ({correct/total:.0%})")


def main():
    parser = argparse.ArgumentParser(description="CSP Virtual Expert Integration")
    parser.add_argument("--standalone", action="store_true",
                       help="Run standalone demo (no model required)")
    parser.add_argument("--model", type=str,
                       help="Model path for Lazarus integration")
    parser.add_argument("--registry", action="store_true",
                       help="Run registry demo")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run simple benchmark")
    args = parser.parse_args()

    if args.standalone or (not args.model and not args.registry and not args.benchmark):
        standalone_demo()

    if args.registry:
        registry_demo()

    if args.benchmark:
        benchmark_demo()

    if args.model:
        lazarus_integration_demo(args.model)


if __name__ == "__main__":
    main()
