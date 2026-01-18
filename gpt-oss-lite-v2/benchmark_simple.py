#!/usr/bin/env python3
"""Simple benchmark of GPT-OSS-Lite configurations - memory and file sizes."""

import gc
import json
import os
from pathlib import Path

import mlx.core as mx


def get_dir_size(path: Path) -> float:
    """Get total size of directory in GB."""
    total = 0
    for f in path.rglob('*'):
        if f.is_file():
            total += f.stat().st_size
    return total / (1024**3)


def benchmark_model(model_path: str):
    """Benchmark memory usage for a single model."""
    model_path = Path(model_path)

    if not model_path.exists():
        return None

    # Load config
    with open(model_path / "config.json") as f:
        config = json.load(f)

    # File size
    file_size = get_dir_size(model_path)

    # Memory for loading weights
    gc.collect()
    mx.reset_peak_memory()

    weights = mx.load(str(model_path / "model.safetensors"))
    mx.eval(weights)

    mem_gb = mx.get_active_memory() / (1024**3)

    # Count parameters (for reference)
    total_elements = 0
    for k, v in weights.items():
        total_elements += v.size

    return {
        'name': model_path.name,
        'num_experts': config['num_local_experts'],
        'num_layers': config['num_hidden_layers'],
        'total_experts': config.get('total_experts', config['num_local_experts'] * config['num_hidden_layers']),
        'reduction_pct': config.get('reduction_percent', 'N/A'),
        'file_size_gb': file_size,
        'memory_gb': mem_gb,
        'original_experts': config.get('original_experts', 768),
    }


def main():
    print("=" * 70)
    print("GPT-OSS-Lite Benchmark: Memory & Storage Comparison")
    print("=" * 70)

    base = Path(__file__).parent
    models = [
        base / "gpt-oss-lite-minimal",   # 4 experts/layer
        base / "gpt-oss-lite-6exp",       # 6 experts/layer
        base / "gpt-oss-lite-8exp",       # 8 experts/layer
        base / "gpt-oss-lite-16exp",      # 16 experts/layer
    ]

    results = []
    for model_path in models:
        print(f"\nLoading: {model_path.name}...")
        result = benchmark_model(str(model_path))
        if result:
            results.append(result)
            print(f"  Experts: {result['num_experts']}/layer")
            print(f"  File size: {result['file_size_gb']:.2f} GB")
            print(f"  Memory: {result['memory_gb']:.2f} GB")
        gc.collect()

    # Summary table
    print("\n" + "=" * 80)
    print("MEASURED RESULTS")
    print("=" * 80)

    print(f"\n{'Configuration':<22} {'Experts':<10} {'File Size':<12} {'Memory':<12} {'Reduction'}")
    print("-" * 80)

    for r in results:
        print(f"{r['name']:<22} {r['num_experts']}/layer    {r['file_size_gb']:.2f} GB      {r['memory_gb']:.2f} GB      {r['reduction_pct']}")

    # Calculate comparisons
    if len(results) >= 2:
        baseline = results[-1]  # 16 experts as baseline (highest quality)
        smallest = results[0]   # 4 experts (smallest)

        print("\n" + "=" * 80)
        print("ANALYSIS: Compared to 16-expert model (baseline)")
        print("=" * 80)

        for r in results[:-1]:
            file_savings = (1 - r['file_size_gb'] / baseline['file_size_gb']) * 100
            mem_savings = (1 - r['memory_gb'] / baseline['memory_gb']) * 100
            print(f"\n{r['name']}:")
            print(f"  Storage savings: {file_savings:.1f}%")
            print(f"  Memory savings: {mem_savings:.1f}%")
            print(f"  Expert reduction: {r['reduction_pct']}")

    # Theoretical inference speed analysis
    print("\n" + "=" * 80)
    print("THEORETICAL INFERENCE ANALYSIS")
    print("=" * 80)
    print("""
Inference speed in MoE models is primarily determined by:
1. Memory bandwidth (loading expert weights)
2. Compute (matrix multiplications)

Since we use top-k=4 routing (always activate 4 experts per token),
the compute per token is IDENTICAL across all configurations.

The speed difference comes from:
- Smaller router lookup table (fewer expert choices)
- Smaller memory footprint = better cache utilization
- No speed benefit from "eliminated" experts (they were never computed)

Expected speed characteristics:
- 4 experts/layer: Slightly faster router, same per-token compute
- 16 experts/layer: Full router, same per-token compute
- Difference: ~5-10% from router overhead, not significant

The REAL benefit is memory savings, enabling:
- Running on devices with less RAM
- Larger batch sizes
- Longer context lengths
""")

    # Key findings
    print("=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print(f"""
FILE SIZE:
  - 4 experts/layer:  {results[0]['file_size_gb']:.2f} GB ({results[0]['reduction_pct']} reduction)
  - 16 experts/layer: {results[-1]['file_size_gb']:.2f} GB ({results[-1]['reduction_pct']} reduction)
  - Savings: {(1 - results[0]['file_size_gb']/results[-1]['file_size_gb'])*100:.1f}%

MEMORY (GPU RAM):
  - 4 experts/layer:  {results[0]['memory_gb']:.2f} GB
  - 16 experts/layer: {results[-1]['memory_gb']:.2f} GB
  - Savings: {(1 - results[0]['memory_gb']/results[-1]['memory_gb'])*100:.1f}%

QUALITY THRESHOLD (from previous tests):
  - 4-8 experts/layer: DEGRADED (repetition, math errors)
  - 16 experts/layer: FULL QUALITY (matches original)

RECOMMENDATION:
  Use 16 experts/layer (50% expert reduction) for production.
  This achieves maximum memory savings while maintaining full quality.
""")

    # Save results
    with open(base / "benchmark_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to benchmark_results.json")


if __name__ == "__main__":
    main()
