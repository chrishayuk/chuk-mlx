"""Handler for expert merging analysis.

Analyzes opportunities to merge similar experts:
- Compute expert weight similarity
- Compute activation overlap
- Identify merge candidates
- Estimate compression potential
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argparse import Namespace

logger = logging.getLogger(__name__)

DEFAULT_PROMPTS = [
    "127 * 89 = ",
    "456 + 789 = ",
    "def fibonacci(n):",
    "import numpy as np",
    "The capital of France is",
    "A synonym for happy is",
    "Once upon a time",
    "The quick brown fox",
]


def handle_expert_merging(args: Namespace) -> None:
    """Handle expert merging analysis.

    Args:
        args: Command arguments with model, thresholds, etc.
    """
    asyncio.run(_async_handle_expert_merging(args))


async def _async_handle_expert_merging(args: Namespace) -> dict:
    """Async implementation of expert merging analysis."""
    from chuk_lazarus.introspection.moe import ExpertRouter
    from chuk_lazarus.introspection.moe.compression import (
        compute_similarity_matrix,
        find_merge_candidates,
        print_compression_summary,
    )

    model_id = args.model
    similarity_threshold = getattr(args, "threshold", 0.8)
    layers_to_analyze = getattr(args, "layers", None)

    print(f"Loading model: {model_id}")
    router = await ExpertRouter.from_pretrained(model_id)

    model = router._model
    moe_layers = list(router.info.moe_layers)

    if layers_to_analyze:
        moe_layers = [l for l in moe_layers if l in layers_to_analyze]

    print(f"Analyzing {len(moe_layers)} MoE layers with threshold {similarity_threshold}")

    # Analyze each layer
    all_candidates: list[dict] = []
    layer_stats: dict[int, dict] = {}

    for layer_idx in moe_layers:
        print(f"  Analyzing layer {layer_idx}...")

        # Compute similarity matrix
        similarities = compute_similarity_matrix(model, layer_idx)

        # Find merge candidates
        candidates = find_merge_candidates(similarities, threshold=similarity_threshold)

        layer_stats[layer_idx] = {
            "num_pairs_analyzed": len(similarities),
            "num_merge_candidates": len(candidates),
        }

        for exp_a, exp_b in candidates:
            # Find the similarity for this pair
            sim_score = 0.0
            for sim in similarities:
                if (sim.expert_a == exp_a and sim.expert_b == exp_b) or \
                   (sim.expert_a == exp_b and sim.expert_b == exp_a):
                    sim_score = sim.weight_cosine_similarity
                    break

            all_candidates.append({
                "layer": layer_idx,
                "expert_a": exp_a,
                "expert_b": exp_b,
                "similarity": sim_score,
            })

    # Print results
    print("\n" + "=" * 70)
    print("EXPERT MERGING ANALYSIS")
    print("=" * 70)

    print(f"\nModel: {model_id}")
    print(f"Similarity threshold: {similarity_threshold}")
    print(f"Total merge candidates: {len(all_candidates)}")

    print("\n" + "-" * 70)
    print("CANDIDATES BY LAYER")
    print("-" * 70)

    for layer_idx, stats in sorted(layer_stats.items()):
        num_cand = stats["num_merge_candidates"]
        print(f"Layer {layer_idx:2d}: {num_cand:3d} merge candidates")

    if all_candidates:
        print("\n" + "-" * 70)
        print("TOP MERGE CANDIDATES")
        print("-" * 70)

        sorted_candidates = sorted(
            all_candidates, key=lambda x: x["similarity"], reverse=True
        )

        for cand in sorted_candidates[:20]:
            print(
                f"  L{cand['layer']:2d}: E{cand['expert_a']:2d} + E{cand['expert_b']:2d} "
                f"(similarity: {cand['similarity']:.3f})"
            )

    # Estimate compression potential
    total_experts = router.info.num_experts * len(moe_layers)
    mergeable_experts = len(all_candidates)  # Each candidate can save 1 expert
    compression_ratio = 1.0 - (mergeable_experts / total_experts) if total_experts > 0 else 1.0

    print("\n" + "-" * 70)
    print("COMPRESSION ESTIMATE")
    print("-" * 70)
    print(f"Total experts: {total_experts}")
    print(f"Mergeable pairs: {mergeable_experts}")
    print(f"Potential reduction: {mergeable_experts}/{total_experts} = {1-compression_ratio:.1%}")

    return {
        "model": model_id,
        "threshold": similarity_threshold,
        "total_experts": total_experts,
        "merge_candidates": len(all_candidates),
        "compression_ratio": compression_ratio,
        "top_candidates": sorted_candidates[:10] if all_candidates else [],
        "layer_stats": {str(k): v for k, v in layer_stats.items()},
    }


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze expert merging")
    parser.add_argument("-m", "--model", required=True, help="Model ID")
    parser.add_argument(
        "-t", "--threshold", type=float, default=0.8,
        help="Similarity threshold (default: 0.8)"
    )
    parser.add_argument(
        "-l", "--layers", nargs="+", type=int,
        help="Specific layers to analyze"
    )

    args = parser.parse_args()
    result = asyncio.run(_async_handle_expert_merging(args))
    return result


if __name__ == "__main__":
    main()
