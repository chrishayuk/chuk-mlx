"""
SVD Analysis of Expert Weight Deltas

This script uses the MoETypeService and MoECompressionService to analyze
whether MoE experts can be represented as:
    expert_i = base_expert + low_rank_delta_i

If the effective rank of deltas is low (e.g., 64-128 vs hidden_dim 2880),
then full experts are massive overkill and LoRA-style experts would work.

Usage:
    # Analyze a model's MoE type
    python analyze_expert_deltas.py --model openai/gpt-oss-20b

    # Analyze specific layer
    python analyze_expert_deltas.py --model allenai/OLMoE-1B-7B-0924 --layer 0

    # Full analysis with compression estimates
    python analyze_expert_deltas.py --model openai/gpt-oss-20b --full
"""

from __future__ import annotations

import argparse
import asyncio


async def analyze_model(model_id: str, layer: int | None = None, full: bool = False):
    """Analyze a model's MoE type and compression potential."""
    from chuk_lazarus.introspection.moe import (
        MoECompressionService,
        MoEType,
        MoETypeService,
    )

    print("=" * 80)
    print("EXPERT DELTA SVD ANALYSIS")
    print("=" * 80)
    print(f"\nModel: {model_id}")

    # Phase 1: MoE Type Analysis
    print("\n" + "-" * 80)
    print("PHASE 1: MoE Type Analysis")
    print("-" * 80)

    try:
        type_result = await MoETypeService.analyze(model_id, layer=layer)

        print(f"\nModel: {type_result.model_id}")
        print(f"Layer: {type_result.layer_idx}")
        print(f"Experts: {type_result.num_experts}")
        print(f"Architecture: {type_result.architecture.value}")

        print(f"\nMoE Type: {type_result.moe_type.value}")
        print(f"Confidence: {type_result.confidence:.1%} ({type_result.confidence_label})")
        print(f"Training Origin: {type_result.training_origin}")

        print("\nProjection Analysis (effective rank at 95% variance):")
        print(f"  {'Projection':<8} | {'Rank':>6} | {'Ratio':>8} | {'Compression':>11}")
        print(f"  {'-'*8} | {'-'*6} | {'-'*8} | {'-'*11}")
        print(
            f"  {'Gate':<8} | {type_result.gate.effective_rank_95:>6} | "
            f"{type_result.gate.rank_ratio:>7.1%} | {type_result.gate.compression_ratio:>10.1f}x"
        )
        print(
            f"  {'Up':<8} | {type_result.up.effective_rank_95:>6} | "
            f"{type_result.up.rank_ratio:>7.1%} | {type_result.up.compression_ratio:>10.1f}x"
        )
        print(
            f"  {'Down':<8} | {type_result.down.effective_rank_95:>6} | "
            f"{type_result.down.rank_ratio:>7.1%} | {type_result.down.compression_ratio:>10.1f}x"
        )

        print(f"\nExpert Similarity:")
        print(f"  Mean cosine similarity: {type_result.mean_cosine_similarity:.4f}")
        print(f"  Std cosine similarity:  {type_result.std_cosine_similarity:.4f}")

        print(f"\nCompressibility:")
        print(f"  Compressible via SVD: {type_result.is_compressible}")
        print(f"  Estimated Compression: {type_result.estimated_compression:.1f}x")

        if type_result.training_signals:
            signals = type_result.training_signals
            print(f"\nTraining Origin Signals:")
            print(f"  Upcycled score:   {signals.upcycled_score:.2f}")
            print(f"  Pretrained score: {signals.pretrained_score:.2f}")

    except Exception as e:
        print(f"Error in type analysis: {e}")
        return

    # Skip remaining phases if model is not compressible
    if not type_result.is_compressible and not full:
        print("\n" + "-" * 80)
        print("CONCLUSION")
        print("-" * 80)
        print(f"\nModel {type_result.moe_type.value} is NOT compressible via SVD overlay.")
        print("Experts are orthogonal - use quantization or pruning instead.")
        return

    if not full:
        print("\n(Use --full for compression estimates and verification)")
        return

    # Phase 2: Compression Estimates
    print("\n" + "-" * 80)
    print("PHASE 2: Storage Savings Estimate")
    print("-" * 80)

    try:
        savings = await MoECompressionService.estimate_savings(model_id)

        print(f"\nModel: {savings.model_id}")
        print(f"MoE Layers: {savings.num_layers}")
        print(f"Experts per layer: {savings.num_experts}")
        print(f"\nRanks: gate={savings.gate_rank}, up={savings.up_rank}, down={savings.down_rank}")
        print(f"\nStorage:")
        print(f"  Original:   {savings.original_mb:>8.1f} MB")
        print(f"  Compressed: {savings.compressed_mb:>8.1f} MB")
        print(f"  Savings:    {savings.savings_mb:>8.1f} MB ({savings.compression_ratio:.1f}x)")

    except Exception as e:
        print(f"Error estimating savings: {e}")

    # Phase 3: Reconstruction Verification
    print("\n" + "-" * 80)
    print("PHASE 3: Reconstruction Verification")
    print("-" * 80)

    try:
        verification = await MoECompressionService.verify_reconstruction(model_id, layer=layer)

        print(f"\nLayer: {verification.layer_idx}")
        print(f"Ranks: gate={verification.gate_rank}, up={verification.up_rank}, down={verification.down_rank}")

        print(f"\nWeight-level errors (relative):")
        print(f"  Gate: mean={verification.gate.mean_relative_error:.6f}, max={verification.gate.max_relative_error:.6f}")
        print(f"  Up:   mean={verification.up.mean_relative_error:.6f}, max={verification.up.max_relative_error:.6f}")
        print(f"  Down: mean={verification.down.mean_relative_error:.6f}, max={verification.down.max_relative_error:.6f}")

        print(f"\nOutput-level errors:")
        print(f"  Mean: {verification.mean_output_error:.6f}")
        print(f"  Max:  {verification.max_output_error:.6f}")

        print(f"\nVerification: {'PASSED' if verification.passed else 'FAILED'}")

    except Exception as e:
        print(f"Error in verification: {e}")

    # Summary
    print("\n" + "-" * 80)
    print("CONCLUSION")
    print("-" * 80)

    if type_result.moe_type == MoEType.UPCYCLED:
        print(f"\nModel is UPCYCLED (dense->MoE conversion)")
        print(f"  - Experts share a common base with low-rank deltas")
        print(f"  - Compressible via SVD overlay: {savings.compression_ratio:.1f}x")
        print(f"  - Reconstruction quality: {'GOOD' if verification.passed else 'POOR'}")
    else:
        print(f"\nModel is {type_result.moe_type.value}")
        print(f"  - Experts are orthogonal, not compressible via SVD")
        print(f"  - Consider quantization or pruning instead")


def main():
    parser = argparse.ArgumentParser(description="Analyze expert weight deltas via SVD")
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-oss-20b",
        help="Model ID or path",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Specific layer to analyze (default: first MoE layer)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full analysis including compression estimates and verification",
    )
    args = parser.parse_args()

    asyncio.run(analyze_model(args.model, layer=args.layer, full=args.full))


if __name__ == "__main__":
    main()
