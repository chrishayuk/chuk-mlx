#!/usr/bin/env python3
"""
Memory-optimized GPT-OSS-Lite loader with Virtual Math Expert.

Combines the optimized loader with the Lazarus virtual expert system
to provide accurate math computation through a virtual calculator expert.

Usage:
    model, tokenizer, wrapper = load_gpt_oss_lite_with_virtual_math(".")
    result = wrapper.solve("127 * 89 = ")
    print(result.answer)  # "11303"
"""

import gc
import json
import sys
import time
from pathlib import Path

import mlx.core as mx

# Add src to path for lazarus imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from lite_loader_optimized import load_gpt_oss_lite_optimized

from chuk_lazarus.inference.virtual_experts.wrapper import VirtualMoEWrapper
from chuk_lazarus.inference.virtual_experts.plugins.math import MathExpertPlugin
from chuk_lazarus.inference.virtual_experts.registry import VirtualExpertRegistry


def _patch_routers_for_virtual_expert(model, model_path: str, num_experts_per_tok: int = 4):
    """
    Patch model routers to add attributes required by VirtualMoEWrapper.

    The VirtualMoEWrapper expects routers to have:
    - num_experts: number of experts in this layer
    - num_experts_per_tok: top-k routing count
    - weight: the router weight matrix

    The mlx_lm gpt_oss model uses nn.Linear for routers which doesn't have
    these attributes by default.
    """
    # Load config to get experts_per_layer
    path = Path(model_path)
    with open(path / "config.json") as f:
        config = json.load(f)

    experts_per_layer = config.get("experts_per_layer", {})

    # Patch each layer's router
    for layer_idx, layer in enumerate(model.model.layers):
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "router"):
            router = layer.mlp.router
            # Get num_experts from config or infer from router output size
            num_experts = experts_per_layer.get(str(layer_idx), router.weight.shape[0])

            # Add the required attributes
            router.num_experts = num_experts
            router.num_experts_per_tok = min(num_experts_per_tok, num_experts)

    return model


def load_gpt_oss_lite_with_virtual_math(model_path: str = "."):
    """
    Load GPT-OSS-Lite with virtual math expert enabled.

    This uses the optimized loader for memory efficiency and wraps the model
    with the virtual expert system for accurate math computation.

    Args:
        model_path: Path to the model directory

    Returns:
        (model, tokenizer, wrapper): The model, tokenizer, and VirtualMoEWrapper
    """
    from transformers import AutoTokenizer

    print("=" * 70)
    print("GPT-OSS-LITE with Virtual Math Expert")
    print("=" * 70)
    print()

    # Reset memory tracking
    try:
        mx.reset_peak_memory()
    except:
        pass

    # Load model with optimized loader
    model, args = load_gpt_oss_lite_optimized(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Patch routers to add required attributes for VirtualMoEWrapper
    print("Patching routers for virtual expert compatibility...")
    model = _patch_routers_for_virtual_expert(model, model_path, args.num_experts_per_tok)

    # Create registry with math plugin
    print("Setting up virtual math expert...")
    registry = VirtualExpertRegistry()
    registry.register(MathExpertPlugin())

    # Create wrapper
    wrapper = VirtualMoEWrapper(
        model=model,
        tokenizer=tokenizer,
        model_id="gpt-oss-lite",
        registry=registry,
    )

    # Calibrate the virtual expert
    print("Calibrating math expert routing...")
    wrapper.calibrate()
    print("Calibration complete!")

    # Report memory
    try:
        active = mx.get_active_memory() / 1024 / 1024
        peak = mx.get_peak_memory() / 1024 / 1024
        print(f"Memory: active={active:.0f}MB, peak={peak:.0f}MB")
    except:
        pass

    print("\nModel ready with virtual math expert")
    print()

    return model, tokenizer, wrapper


def generate_with_math(wrapper, prompt: str, verbose: bool = False):
    """
    Generate response using virtual expert for math.

    Args:
        wrapper: VirtualMoEWrapper instance
        prompt: Input prompt
        verbose: Show routing details

    Returns:
        VirtualExpertResult with answer and metadata
    """
    return wrapper.solve(prompt, verbose=verbose)


def main():
    """Test the optimized loader with virtual math expert."""
    print("=" * 70)
    print("GPT-OSS-LITE with Virtual Math Expert - Test")
    print("=" * 70)
    print()

    model, tokenizer, wrapper = load_gpt_oss_lite_with_virtual_math(".")

    # Test non-math prompts with standard generation (same as lite_loader_optimized.py)
    print("=" * 70)
    print("Testing Non-Math Generation")
    print("=" * 70)

    non_math_prompts = [
        "The capital of France is",
        "def fibonacci(n):",
        "2 + 2 =",
    ]

    for prompt in non_math_prompts:
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])

        start = time.time()
        for _ in range(30):
            logits = model(input_ids)
            next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
            input_ids = mx.concatenate([input_ids, next_token], axis=1)
            mx.eval(input_ids)

        elapsed = time.time() - start
        output = tokenizer.decode(input_ids[0].tolist())

        print(f"\nPrompt: '{prompt}'")
        print(f"Output ({elapsed:.2f}s): {output[:100]}...")

    # Test math prompts with virtual expert (seamless output)
    math_prompts = [
        ("127 * 89 = ", 11303),
        ("456 + 789 = ", 1245),
        ("1000 - 250 = ", 750),
        ("144 / 12 = ", 12),
        ("25 * 25 = ", 625),
    ]

    print()
    print("=" * 70)
    print("Testing Math Generation")
    print("=" * 70)

    correct_count = 0
    math_count = len(math_prompts)

    for prompt, expected in math_prompts:
        start = time.time()
        result = wrapper.solve(prompt, verbose=False)
        elapsed = time.time() - start

        # Show seamless output like non-math prompts
        output = f"{prompt}{result.answer}"
        print(f"\nPrompt: '{prompt}'")
        print(f"Output ({elapsed:.2f}s): {output}")

        if result.is_correct:
            correct_count += 1

    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Math problems: {correct_count}/{math_count} correct")
    print(f"Accuracy: {100*correct_count/math_count:.1f}%")
    print()

    # Run benchmark with more problems
    print("=" * 70)
    print("Running Benchmark")
    print("=" * 70)

    benchmark_problems = [
        "127 * 89 = ",
        "456 + 789 = ",
        "1000 - 250 = ",
        "144 / 12 = ",
        "25 * 25 = ",
        "99 * 99 = ",
        "100 + 200 = ",
        "50 - 17 = ",
        "36 * 47 = ",
        "888 - 111 = ",
    ]

    analysis = wrapper.benchmark(benchmark_problems)
    print()
    print(analysis.summary())

    # Final memory report
    try:
        active = mx.get_active_memory() / 1024 / 1024
        peak = mx.get_peak_memory() / 1024 / 1024
        print()
        print(f"Final memory: active={active:.0f}MB, peak={peak:.0f}MB")
    except:
        pass


if __name__ == "__main__":
    main()
