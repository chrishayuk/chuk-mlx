#!/usr/bin/env python3
"""
Example 6: Batching Analysis and Optimization

Demonstrates:
- Analyzing length distributions with histograms
- Computing bucket efficiency metrics
- Getting optimal bucket edge suggestions
- Creating complete efficiency reports

Run:
    python examples/batching/06_analyze.py

CLI equivalents:
    lazarus data batching histogram -c lengths.jsonl --bins 15
    lazarus data batching analyze -c lengths.jsonl --bucket-edges 128,256,512
    lazarus data batching suggest -c lengths.jsonl --num-buckets 4 --goal waste
"""

from chuk_lazarus.data.batching import (
    BucketSpec,
    OptimizationGoal,
    analyze_bucket_efficiency,
    compute_length_histogram,
    create_efficiency_report,
    suggest_bucket_edges,
)


def create_sample_lengths(num_samples: int = 200) -> dict[str, int]:
    """Create synthetic length data with realistic distribution."""
    import random

    random.seed(42)
    lengths = {}

    for i in range(num_samples):
        # Realistic distribution: many short, fewer long
        r = random.random()
        if r < 0.4:
            length = random.randint(30, 100)  # Short
        elif r < 0.7:
            length = random.randint(100, 250)  # Medium-short
        elif r < 0.9:
            length = random.randint(250, 500)  # Medium
        else:
            length = random.randint(500, 1000)  # Long

        lengths[f"sample_{i:04d}"] = length

    return lengths


def main():
    print("=" * 70)
    print("Batching Analysis Demo")
    print("=" * 70)

    # 1. Create sample data
    print("\n1. Creating sample dataset...")
    lengths = create_sample_lengths(num_samples=200)
    print(f"   Created {len(lengths)} samples")
    print(f"   Length range: {min(lengths.values())} - {max(lengths.values())}")

    # 2. Compute length histogram
    print("\n2. Computing length histogram...")
    histogram = compute_length_histogram(lengths, num_bins=10)

    print(histogram.to_ascii(width=50))

    print("\n   Key percentiles:")
    print(f"   P25: {histogram.p25} (25% of samples shorter)")
    print(f"   P50: {histogram.p50} (median)")
    print(f"   P75: {histogram.p75} (75% of samples shorter)")
    print(f"   P90: {histogram.p90} (90% of samples shorter)")
    print(f"   P99: {histogram.p99} (99% of samples shorter)")

    # 3. Analyze bucket efficiency with default edges
    print("\n" + "=" * 70)
    print("3. Analyzing bucket efficiency...")
    bucket_spec = BucketSpec(
        edges=(128, 256, 512),
        overflow_max=1024,
    )
    print(f"   Testing bucket edges: {bucket_spec.edges}")
    print(f"   Overflow max: {bucket_spec.overflow_max}")

    analysis = analyze_bucket_efficiency(lengths, bucket_spec)
    print(analysis.to_ascii())

    # 4. Get bucket edge suggestions
    print("\n" + "=" * 70)
    print("4. Getting bucket edge suggestions...")

    print("\n   Optimization goals:")
    for goal in OptimizationGoal:
        suggestion = suggest_bucket_edges(
            lengths,
            num_buckets=4,
            goal=goal,
            max_length=1024,
        )
        print(f"\n   Goal: {goal.value}")
        print(f"   Suggested edges: {suggestion.edges}")
        print(f"   Overflow max: {suggestion.overflow_max}")
        print(f"   Est. efficiency: {suggestion.estimated_efficiency:.1%}")
        print(f"   Rationale: {suggestion.rationale}")

    # 5. Create complete efficiency report
    print("\n" + "=" * 70)
    print("5. Creating complete efficiency report...")

    # Use the suggested edges from minimize_waste goal
    best_suggestion = suggest_bucket_edges(
        lengths,
        num_buckets=4,
        goal=OptimizationGoal.MINIMIZE_WASTE,
        max_length=1024,
    )
    optimized_spec = BucketSpec(
        edges=best_suggestion.edges,
        overflow_max=best_suggestion.overflow_max,
    )

    report = create_efficiency_report(lengths, optimized_spec)
    print(report.to_ascii())

    # 6. Compare original vs optimized
    print("\n" + "=" * 70)
    print("6. Comparing original vs optimized bucket edges...")

    original_analysis = analyze_bucket_efficiency(lengths, bucket_spec)
    optimized_analysis = analyze_bucket_efficiency(lengths, optimized_spec)

    print(f"\n   Original edges:  {bucket_spec.edges}")
    print(f"   Optimized edges: {optimized_spec.edges}")
    print()
    print(f"   Original efficiency:  {original_analysis.overall_efficiency:.1%}")
    print(f"   Optimized efficiency: {optimized_analysis.overall_efficiency:.1%}")
    print()
    improvement = optimized_analysis.overall_efficiency - original_analysis.overall_efficiency
    if improvement > 0:
        print(f"   Improvement: +{improvement:.1%}")
    else:
        print(f"   Change: {improvement:.1%}")

    # 7. Show CLI commands
    print("\n" + "=" * 70)
    print("7. Equivalent CLI commands:")
    print()
    print("   # View length histogram")
    print("   lazarus data batching histogram -c lengths.jsonl --bins 15")
    print()
    print("   # Analyze bucket efficiency")
    print(f"   lazarus data batching analyze -c lengths.jsonl --bucket-edges {','.join(map(str, bucket_spec.edges))}")
    print()
    print("   # Get bucket suggestions")
    print("   lazarus data batching suggest -c lengths.jsonl --num-buckets 4 --goal waste")
    print()
    edges_str = ",".join(map(str, optimized_spec.edges))
    print("   # Build batch plan with optimized edges")
    print(f"   lazarus data batchplan build -l lengths.jsonl --bucket-edges {edges_str} -o batch_plan/")

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
