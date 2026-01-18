#!/usr/bin/env python3
"""
Validate Experimental MoE Architectures.

Compares the proposed MoE variants against standard MoE:
1. Parameter count comparison
2. Forward pass timing
3. Quality comparison (if training available)

Usage:
    python experiments/moe_dynamics/validate_architectures.py
    python experiments/moe_dynamics/validate_architectures.py --variant tiered
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import mlx.core as mx

from chuk_lazarus.models_v2.components.ffn.moe_experimental import (
    ExperimentalMoEConfig,
    create_experimental_moe,
    CompactNonlinearRouter,
    LightweightTeam,
    ExpertTeam,
)


@dataclass
class ArchitectureStats:
    """Statistics for an architecture variant."""

    variant: str
    total_params: int
    router_params: int
    expert_params: int
    forward_time_ms: float
    memory_mb: float


def count_parameters(module) -> int:
    """Count total parameters in a module."""
    def _count_recursive(params) -> int:
        total = 0
        if isinstance(params, dict):
            for v in params.values():
                total += _count_recursive(v)
        elif isinstance(params, list):
            for v in params:
                total += _count_recursive(v)
        elif isinstance(params, mx.array):
            total += params.size
        return total

    return _count_recursive(module.parameters())


def measure_forward_time(module, x: mx.array, num_runs: int = 10) -> float:
    """Measure average forward pass time in milliseconds."""
    # Warmup
    for _ in range(3):
        _ = module(x)
        mx.eval(_)

    # Timed runs
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        output = module(x)
        mx.eval(output)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return sum(times) / len(times)


def analyze_variant(
    variant: str,
    config: ExperimentalMoEConfig,
    x: mx.array,
    layer_idx: int = 0,
) -> ArchitectureStats:
    """Analyze a specific MoE variant."""
    print(f"\nAnalyzing: {variant}")

    # Create module
    config.variant = variant
    module = create_experimental_moe(config, layer_idx)

    # Count parameters
    total_params = count_parameters(module)

    # Separate router and expert params (approximate)
    def _classify_params(params, prefix="") -> tuple[int, int]:
        """Recursively classify params as router or expert."""
        router_count = 0
        expert_count = 0
        if isinstance(params, dict):
            for k, v in params.items():
                r, e = _classify_params(v, f"{prefix}.{k}" if prefix else k)
                router_count += r
                expert_count += e
        elif isinstance(params, list):
            for i, v in enumerate(params):
                r, e = _classify_params(v, f"{prefix}[{i}]")
                router_count += r
                expert_count += e
        elif isinstance(params, mx.array):
            if "router" in prefix.lower() or "gate" in prefix.lower():
                router_count += params.size
            else:
                expert_count += params.size
        return router_count, expert_count

    router_params, expert_params = _classify_params(module.parameters())

    # Measure forward time
    forward_time = measure_forward_time(module, x)

    # Estimate memory (rough approximation)
    memory_mb = total_params * 4 / (1024 * 1024)  # Assuming float32

    return ArchitectureStats(
        variant=variant,
        total_params=total_params,
        router_params=router_params,
        expert_params=expert_params,
        forward_time_ms=forward_time,
        memory_mb=memory_mb,
    )


def print_comparison_table(stats_list: list[ArchitectureStats], baseline: ArchitectureStats) -> None:
    """Print comparison table."""
    print("\n" + "=" * 90)
    print("ARCHITECTURE COMPARISON")
    print("=" * 90)
    print()
    print(f"{'Variant':<25} {'Params':>12} {'Router':>10} {'Expert':>10} {'Time(ms)':>10} {'vs Base':>10}")
    print("-" * 90)

    for stats in stats_list:
        param_ratio = stats.total_params / baseline.total_params
        time_ratio = stats.forward_time_ms / baseline.forward_time_ms

        param_str = f"{stats.total_params:,}"
        router_str = f"{stats.router_params:,}"
        expert_str = f"{stats.expert_params:,}"
        time_str = f"{stats.forward_time_ms:.2f}"
        ratio_str = f"{param_ratio:.2f}x"

        print(f"{stats.variant:<25} {param_str:>12} {router_str:>10} {expert_str:>10} {time_str:>10} {ratio_str:>10}")

    print()


def print_detailed_analysis(variant: str, config: ExperimentalMoEConfig) -> None:
    """Print detailed analysis for a specific variant."""
    print(f"\n{'=' * 70}")
    print(f"DETAILED ANALYSIS: {variant.upper()}")
    print("=" * 70)

    if variant == "compact_router":
        # Compare router sizes
        standard_params = config.hidden_size * config.num_experts
        compact_router = CompactNonlinearRouter(
            config.hidden_size,
            config.num_experts,
            config.num_experts_per_tok,
            config.router_bottleneck,
        )
        compact_params = compact_router.param_count
        savings = compact_router.param_savings_vs_standard

        print(f"\nRouter Comparison:")
        print(f"  Standard router params:  {standard_params:,}")
        print(f"  Compact router params:   {compact_params:,}")
        print(f"  Savings:                 {savings:.1%}")

    elif variant == "tiered":
        # Show expert allocation
        print(f"\nTiered Expert Allocation:")
        print(f"  Early layers (L0-L7):    {config.tiered_early_experts} experts")
        print(f"  Middle layers (L8-L17):  {config.tiered_middle_experts} experts")
        print(f"  Late layers (L18+):      {config.tiered_late_experts} experts")

        total_tiered = (
            8 * config.tiered_early_experts +
            10 * config.tiered_middle_experts +
            6 * config.tiered_late_experts
        )
        total_standard = 24 * config.num_experts
        savings = 1 - (total_tiered / total_standard)
        print(f"\n  Total tiered experts:    {total_tiered}")
        print(f"  Total standard experts:  {total_standard}")
        print(f"  Expert reduction:        {savings:.1%}")

    elif variant in ("team", "lightweight_team"):
        # Compare team implementations
        print(f"\nTeam Comparison:")

        full_team = ExpertTeam(
            config.hidden_size,
            config.intermediate_size,
            config.team_size,
            config.bias,
        )
        light_team = LightweightTeam(
            config.hidden_size,
            config.intermediate_size,
            config.team_size,
            config.bias,
        )

        full_params = count_parameters(full_team)
        light_params = count_parameters(light_team)

        print(f"  ExpertTeam params:       {full_params:,}")
        print(f"  LightweightTeam params:  {light_params:,}")
        print(f"  Combiner savings:        {(full_params - light_params):,} params")

    elif variant == "circuit":
        print(f"\nCircuit-Based Routing:")
        print(f"  Number of circuits:      {config.num_circuits}")
        print(f"  Routing decisions:       1 (at layer 0)")
        print(f"  Standard routing:        24 (per layer)")
        print(f"  Routing reduction:       {(1 - 1/24):.1%}")

    elif variant == "adaptive_k":
        print(f"\nAdaptive-k Routing:")
        print(f"  Min k:                   {config.min_k}")
        print(f"  Max k:                   {config.max_k}")
        print(f"  Standard k:              {config.num_experts_per_tok}")

    print()


def main():
    """Run architecture validation."""
    parser = argparse.ArgumentParser(description="Validate MoE architectures")
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Specific variant to analyze (default: all)",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=4096,
        help="Hidden dimension",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for timing",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
        help="Sequence length for timing",
    )

    args = parser.parse_args()

    # Create config
    config = ExperimentalMoEConfig(
        hidden_size=args.hidden_size,
        intermediate_size=args.hidden_size * 4,
        num_experts=32,
        num_experts_per_tok=4,
        router_bottleneck=64,
        tiered_early_experts=16,
        tiered_middle_experts=32,
        tiered_late_experts=24,
        num_circuits=15,
        team_size=4,
        num_teams=8,
        min_k=2,
        max_k=8,
    )

    # Create test input
    x = mx.random.normal((args.batch_size, args.seq_len, args.hidden_size))

    variants = [
        "standard",
        "compact_router",
        "compact_router_16",  # New: with actual savings
        "tiered",
        "tiered_lightweight",  # New: hybrid combining both
        "team",
        "lightweight_team",
        "layer_pair_circuit",  # New: uses 87.5% layer-pair consistency
        "adaptive_k",
    ]

    if args.variant:
        variants = [args.variant]

    # Analyze all variants
    stats_list = []
    baseline = None

    for variant in variants:
        try:
            stats = analyze_variant(variant, config, x)
            stats_list.append(stats)
            if variant == "standard":
                baseline = stats
        except Exception as e:
            print(f"  Error: {e}")

    # Print comparison
    if baseline and len(stats_list) > 1:
        print_comparison_table(stats_list, baseline)

    # Detailed analysis for each variant
    for variant in variants:
        print_detailed_analysis(variant, config)

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("Recommended experiment order:")
    print("  1. Tiered MoE       - Low risk, 23% expert reduction")
    print("  2. Circuit MoE      - High potential if circuits transfer")
    print("  3. Lightweight Team - Cooperation by design")
    print("  4. Compact Router   - Easy to test")
    print("  5. Adaptive-k       - Most complex")
    print()
    print("Next steps:")
    print("  1. Run circuit_transfer_test.py to validate circuit viability")
    print("  2. Test TieredMoE on actual inference")
    print("  3. Compare quality vs standard MoE")
    print()


if __name__ == "__main__":
    main()
