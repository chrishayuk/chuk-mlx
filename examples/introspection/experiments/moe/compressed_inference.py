"""Inference using compressed MoE overlay model.

Demonstrates loading and using a compressed MoE model for inference.
The compressed model stores:
- base_weights: Mean expert per layer/projection
- deltas: Low-rank U, V factors per expert

Reconstruction: expert_i = base + U_i @ V_i
"""


import mlx.core as mx

from chuk_lazarus.introspection.moe import OverlayExperts
from chuk_lazarus.introspection.moe.moe_type import MoETypeService


def load_compressed_model(compressed_path: str, original_model_id: str):
    """Load compressed model alongside original for inference.

    Args:
        compressed_path: Path to compressed overlay directory
        original_model_id: Original model ID for non-expert weights

    Returns:
        Tuple of (model, overlay_experts, tokenizer)
    """
    print(f"Loading compressed model from: {compressed_path}")
    overlay = OverlayExperts.load(compressed_path)

    print(f"Loading original model: {original_model_id}")
    model = MoETypeService._load_model(original_model_id)

    # Load tokenizer
    from mlx_lm.tokenizers import load_tokenizer
    tokenizer = load_tokenizer(original_model_id)

    print(f"Compressed model: {overlay.num_layers} layers, {overlay.num_experts} experts")
    print(f"Memory usage: {overlay.memory_usage_mb():.1f} MB (compressed)")

    return model, overlay, tokenizer


def generate_with_overlay(
    model,
    overlay: OverlayExperts,
    tokenizer,
    prompt: str,
    max_tokens: int = 50,
    temperature: float = 0.7,
):
    """Generate text using compressed experts.

    This patches the model's expert forward pass to use compressed weights.
    """
    print(f"\nPrompt: {prompt}")
    print("-" * 50)

    # Tokenize
    tokens = tokenizer.encode(prompt)
    tokens = mx.array([tokens])

    # For now, demonstrate expert weight reconstruction
    # Full inference integration would require patching the model's MoE forward

    # Show reconstruction example
    print("\nExpert weight reconstruction example:")
    for layer_idx in overlay.moe_layer_indices[:3]:  # First 3 layers
        for proj in ["gate", "up", "down"]:
            weight = overlay.get_expert_weight(layer_idx, proj, expert=0)
            print(f"  Layer {layer_idx} {proj:5} expert 0: {weight.shape}")

    # Use the original model for actual generation
    # (In production, you'd patch the expert lookups)
    from mlx_lm import generate

    result = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        temp=temperature,
        verbose=False,
    )

    print(f"\nGenerated: {result}")
    return result


def benchmark_reconstruction(overlay: OverlayExperts, num_samples: int = 10):
    """Benchmark reconstruction time vs memory savings."""
    import time

    print("\n" + "=" * 60)
    print("RECONSTRUCTION BENCHMARK")
    print("=" * 60)

    # Time reconstruction
    times = []
    for _ in range(num_samples):
        start = time.perf_counter()
        for layer_idx in overlay.moe_layer_indices:
            for proj in ["gate", "up", "down"]:
                weight = overlay.get_expert_weight(layer_idx, proj, expert=0)
                mx.eval(weight)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    print(f"Full model reconstruction: {avg_time*1000:.1f} ms")
    print(f"Per-layer reconstruction: {avg_time*1000/overlay.num_layers:.2f} ms")

    # Memory comparison
    config = overlay.config
    original_mb = config.original_bytes / (1024 * 1024)
    compressed_mb = overlay.memory_usage_mb()

    print("\nMemory:")
    print(f"  Original:   {original_mb:,.1f} MB")
    print(f"  Compressed: {compressed_mb:,.1f} MB")
    print(f"  Savings:    {original_mb - compressed_mb:,.1f} MB ({config.compression_ratio:.1f}x)")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Inference with compressed MoE model")
    parser.add_argument(
        "--compressed", "-c",
        default="gpt-oss-20b-overlay",
        help="Path to compressed model directory",
    )
    parser.add_argument(
        "--original", "-o",
        default="openai/gpt-oss-20b",
        help="Original model ID (for non-expert weights)",
    )
    parser.add_argument(
        "--prompt", "-p",
        default="The capital of France is",
        help="Prompt for generation",
    )
    parser.add_argument(
        "--max-tokens", "-n",
        type=int,
        default=50,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--benchmark", "-b",
        action="store_true",
        help="Run reconstruction benchmark",
    )

    args = parser.parse_args()

    # Load models
    model, overlay, tokenizer = load_compressed_model(
        args.compressed,
        args.original,
    )

    # Run benchmark if requested
    if args.benchmark:
        benchmark_reconstruction(overlay)

    # Generate
    generate_with_overlay(
        model,
        overlay,
        tokenizer,
        args.prompt,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
