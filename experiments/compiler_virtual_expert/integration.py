"""
Integration with VirtualMoEWrapper.

Shows how to register the CompilerExpertPlugin alongside the existing
MathExpertPlugin for a unified virtual expert system.
"""

from __future__ import annotations


def demo_integration():
    """
    Demonstrate integrating compiler expert with existing virtual expert system.
    """
    print("=" * 70)
    print("COMPILER EXPERT - Integration Demo")
    print("=" * 70)

    # Import existing infrastructure
    from chuk_lazarus.inference.virtual_experts import (
        VirtualMoEWrapper,
        VirtualExpertRegistry,
        get_default_registry,
    )
    from chuk_lazarus.inference.virtual_experts.plugins.math import MathExpertPlugin
    from compiler_plugin import CompilerExpertPlugin

    # Create registry with both experts
    registry = VirtualExpertRegistry()
    registry.register(MathExpertPlugin())  # priority=10
    registry.register(CompilerExpertPlugin())  # priority=8

    print("\nRegistered experts:")
    for expert in registry.get_all():
        print(f"  - {expert.name} (priority={expert.priority})")

    # Test routing
    test_prompts = [
        ("127 * 89 = ", "math"),
        ("```python\nprint(2+2)\n```", "compiler"),
        ("Hello world", None),
        ("def add(a, b):\n    return a + b\n```", "compiler"),
        ("Calculate 50 - 17 = ", "math"),
    ]

    print("\nRouting test:")
    print("-" * 50)
    for prompt, expected in test_prompts:
        handler = registry.find_handler(prompt)
        handler_name = handler.name if handler else "none"
        status = "✓" if handler_name == (expected or "none") else "✗"
        print(f"{status} '{prompt[:40]}...' -> {handler_name}")

    # Execute examples
    print("\nExecution test:")
    print("-" * 50)

    math_expert = registry.get("math")
    compiler_expert = registry.get("compiler")

    if math_expert:
        result = math_expert.execute("127 * 89 = ")
        print(f"Math: 127 * 89 = {result}")

    if compiler_expert:
        code = "```python\ndef greet(name):\n    return f'Hello, {name}!'\nprint(greet('World'))\n```"
        result = compiler_expert.execute(code)
        print(f"Compiler: {result}")


def demo_with_model(model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """
    Full integration with model and VirtualMoEWrapper.
    """
    print("=" * 70)
    print("COMPILER EXPERT - Full Model Integration")
    print("=" * 70)

    import mlx.core as mx
    from chuk_lazarus.models_v2.loader import load_model
    from chuk_lazarus.inference.virtual_experts import VirtualMoEWrapper
    from compiler_plugin import CompilerExpertPlugin

    # Load model
    print(f"\n1. Loading model: {model_name}...")
    loaded = load_model(model_name)
    model = loaded.model
    tokenizer = loaded.tokenizer
    mx.eval(model.parameters())
    print("   Model loaded.")

    # Create wrapper with compiler expert
    print("\n2. Creating VirtualMoEWrapper with compiler expert...")

    # Note: VirtualMoEWrapper expects MoE model
    # For dense models, use VirtualDenseWrapper
    try:
        wrapper = VirtualMoEWrapper(
            model=model,
            tokenizer=tokenizer,
            model_id=model_name,
        )

        # Register compiler expert
        compiler = CompilerExpertPlugin()
        wrapper.register_plugin(compiler)

        # Calibrate
        print("\n3. Calibrating experts...")
        wrapper.calibrate()

        # Test
        print("\n4. Testing...")
        print("-" * 50)

        # Math test
        result = wrapper.solve("Calculate 99 * 99 = ", verbose=True)
        print(f"\nMath: {result.prompt} -> {result.answer}")
        print(f"  Used virtual expert: {result.used_virtual_expert}")

        # Code test
        code_prompt = "Run this:\n```python\nprint('Hello from compiler expert!')\n```"
        result = wrapper.solve(code_prompt, verbose=True)
        print(f"\nCode: {code_prompt[:50]}...")
        print(f"  Result: {result.answer}")
        print(f"  Used virtual expert: {result.used_virtual_expert}")

    except Exception as e:
        print(f"\nNote: Full integration requires MoE model. Error: {e}")
        print("For dense models, use VirtualDenseWrapper instead.")

        # Fallback: direct plugin usage
        print("\nFallback: Direct plugin usage")
        print("-" * 50)

        from chuk_lazarus.inference.virtual_experts.plugins.math import MathExpertPlugin

        math = MathExpertPlugin()
        compiler = CompilerExpertPlugin()

        print(f"Math: 99 * 99 = {math.execute('99 * 99 = ')}")
        print(f"Compiler: {compiler.execute('```python\\nprint(2+2)\\n```')}")


def show_architecture():
    """Print the unified architecture diagram."""
    print(
        """
╔══════════════════════════════════════════════════════════════════════╗
║                    UNIFIED VIRTUAL EXPERT SYSTEM                      ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Input: "127 * 89 =" OR "```python\\nprint(2+2)\\n```"                 ║
║         │                                                            ║
║         ▼                                                            ║
║  ┌──────────────────────────────────────────────────────────────┐   ║
║  │  ROUTER (Learned Directions in Activation Space)              │   ║
║  │                                                               │   ║
║  │  For each registered expert:                                  │   ║
║  │    - Compute: score = dot(hidden_state, direction)            │   ║
║  │    - Apply softmax with learned scale/bias                    │   ║
║  │    - Select highest scoring expert above threshold            │   ║
║  └──────────────────────────────────────────────────────────────┘   ║
║         │                                                            ║
║         ├────────────────────┬───────────────────┐                  ║
║         ▼                    ▼                   ▼                  ║
║  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐        ║
║  │ Math Expert    │  │ Compiler Expert│  │ Neural Experts │        ║
║  │ (priority=10)  │  │ (priority=8)   │  │ (fallback)     │        ║
║  │                │  │                │  │                │        ║
║  │ Python eval    │  │ Sandbox exec   │  │ Model forward  │        ║
║  │ 100% accurate  │  │ + feedback     │  │ ~70% accurate  │        ║
║  └────────────────┘  └────────────────┘  └────────────────┘        ║
║         │                    │                   │                  ║
║         └────────────────────┴───────────────────┘                  ║
║                              │                                      ║
║                              ▼                                      ║
║  Output: "11303" OR "✓ Executed\\nOutput: 4" OR model generation    ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝

EXPERT COMPARISON:
┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│ Property        │ Math Expert     │ Compiler Expert │ Neural Experts  │
├─────────────────┼─────────────────┼─────────────────┼─────────────────┤
│ Trigger         │ "NUM OP NUM ="  │ "```...```"     │ Everything else │
│ Execution       │ AST + eval      │ Sandbox exec    │ Forward pass    │
│ Accuracy        │ 100%            │ 100% syntax     │ ~70%            │
│ Output          │ Number          │ Result/error    │ Tokens          │
│ Self-correct    │ N/A             │ Yes (feedback)  │ No              │
│ Calibration     │ +/- prompts     │ +/- prompts     │ Pre-trained     │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘
"""
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--model":
        model = sys.argv[2] if len(sys.argv) > 2 else None
        demo_with_model(model) if model else demo_with_model()
    elif len(sys.argv) > 1 and sys.argv[1] == "--arch":
        show_architecture()
    else:
        show_architecture()
        print("\n")
        demo_integration()
