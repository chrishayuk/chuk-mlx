#!/usr/bin/env python3
"""
Convert trained GPT-OSS to TieredLightweightMoE using discovered structure.

Uses findings from MoE dynamics analysis:
- Cold experts (12.9%) → exclude from team formation
- Circuit pipelines (87.5% consistency) → group co-activating experts
- Layer differentiation → allocate more teams to middle layers (4/8/6)
- Routing frequencies → initialize mixing weights

Usage:
    python experiments/moe_dynamics/convert_to_tiered_lightweight.py
    python experiments/moe_dynamics/convert_to_tiered_lightweight.py --evaluate
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ConversionConfig:
    """Configuration for model conversion."""

    model_id: str = "openai/gpt-oss-20b"

    # Tiered allocation
    early_teams: int = 4   # L0-L7
    middle_teams: int = 8  # L8-L17
    late_teams: int = 6    # L18-L23
    team_size: int = 4

    # Output
    output_dir: str = "converted_tiered_lightweight"


@dataclass
class ExpertClusteringResult:
    """Result of clustering experts into teams."""

    layer_idx: int
    num_teams: int
    teams: list[list[int]]  # team_idx -> [expert_indices]
    mixing_weights: list[list[float]]  # team_idx -> [weights]
    excluded_experts: list[int]  # cold experts not assigned


async def analyze_expert_coactivation(
    model_id: str,
    prompts: list[str],
) -> dict[int, mx.array]:
    """
    Analyze expert co-activation patterns across layers.

    Returns coactivation matrix per layer: coact[layer][i,j] =
    frequency that experts i and j are both in top-k for same token.
    """
    from chuk_lazarus.introspection.moe import ExpertRouter

    logger.info(f"Analyzing co-activation patterns for {model_id}")
    router = await ExpertRouter.from_pretrained(model_id)

    info = router.info
    num_experts = info.num_experts
    num_layers = len(info.moe_layers)

    # Initialize coactivation matrices
    coactivation = {
        layer: mx.zeros((num_experts, num_experts))
        for layer in info.moe_layers
    }

    # Track activation frequencies
    activation_freq = {
        layer: mx.zeros((num_experts,))
        for layer in info.moe_layers
    }

    total_tokens = 0

    for prompt in prompts:
        weights = await router.capture_router_weights(prompt, layers=list(info.moe_layers))

        for layer_idx, layer_weights in zip(info.moe_layers, weights):
            if not layer_weights.positions:
                continue

            for pos in layer_weights.positions:
                total_tokens += 1
                experts = pos.expert_indices

                # Update activation frequency
                for e in experts:
                    activation_freq[layer_idx] = activation_freq[layer_idx].at[e].add(1)

                # Update coactivation (all pairs in top-k)
                for i, e1 in enumerate(experts):
                    for e2 in experts[i:]:
                        coactivation[layer_idx] = coactivation[layer_idx].at[e1, e2].add(1)
                        if e1 != e2:
                            coactivation[layer_idx] = coactivation[layer_idx].at[e2, e1].add(1)

    # Normalize
    for layer in info.moe_layers:
        activation_freq[layer] = activation_freq[layer] / total_tokens
        coactivation[layer] = coactivation[layer] / total_tokens

    return coactivation, activation_freq


def identify_cold_experts(
    activation_freq: dict[int, mx.array],
    threshold: float = 0.01,
) -> set[tuple[int, int]]:
    """
    Identify cold experts (activation rate < threshold).

    Returns set of (layer_idx, expert_idx) tuples.
    """
    cold = set()

    for layer_idx, freq in activation_freq.items():
        for expert_idx in range(freq.shape[0]):
            if float(freq[expert_idx]) < threshold:
                cold.add((layer_idx, expert_idx))

    logger.info(f"Identified {len(cold)} cold experts (threshold={threshold})")
    return cold


def cluster_experts_to_teams(
    layer_idx: int,
    num_teams: int,
    team_size: int,
    coactivation: mx.array,
    activation_freq: mx.array,
    cold_experts: set[tuple[int, int]],
) -> ExpertClusteringResult:
    """
    Cluster experts into teams based on co-activation patterns.

    Strategy:
    1. Exclude cold experts
    2. Use co-activation similarity for clustering
    3. Assign most frequently co-activating experts to same team
    4. Initialize mixing weights from activation frequencies
    """
    num_experts = coactivation.shape[0]

    # Get non-cold experts for this layer
    cold_at_layer = {e for l, e in cold_experts if l == layer_idx}
    hot_experts = [i for i in range(num_experts) if i not in cold_at_layer]

    if len(hot_experts) < num_teams * team_size:
        logger.warning(f"Layer {layer_idx}: Only {len(hot_experts)} hot experts for {num_teams * team_size} slots")

    # Convert to numpy for clustering
    coact_np = np.array(coactivation)
    freq_np = np.array(activation_freq)

    # Simple greedy clustering based on co-activation
    # Start with highest-frequency expert, add most co-active partners
    teams = []
    assigned = set()

    # Sort hot experts by frequency (descending)
    sorted_experts = sorted(hot_experts, key=lambda e: -freq_np[e])

    for _ in range(num_teams):
        team = []

        # Find seed (highest freq unassigned)
        seed = None
        for e in sorted_experts:
            if e not in assigned:
                seed = e
                break

        if seed is None:
            break

        team.append(seed)
        assigned.add(seed)

        # Add most co-active partners
        while len(team) < team_size:
            best_partner = None
            best_coact = -1

            for e in sorted_experts:
                if e in assigned:
                    continue

                # Sum of co-activation with current team members
                coact_score = sum(coact_np[e, t] for t in team)

                if coact_score > best_coact:
                    best_coact = coact_score
                    best_partner = e

            if best_partner is None:
                # No more unassigned experts, wrap around
                for e in range(num_experts):
                    if e not in assigned:
                        best_partner = e
                        break

            if best_partner is not None:
                team.append(best_partner)
                assigned.add(best_partner)
            else:
                break

        teams.append(team)

    # Compute mixing weights from activation frequencies
    mixing_weights = []
    for team in teams:
        weights = [freq_np[e] for e in team]
        total = sum(weights) + 1e-10
        weights = [w / total for w in weights]
        mixing_weights.append(weights)

    return ExpertClusteringResult(
        layer_idx=layer_idx,
        num_teams=num_teams,
        teams=teams,
        mixing_weights=mixing_weights,
        excluded_experts=list(cold_at_layer),
    )


def get_num_teams_for_layer(layer_idx: int, config: ConversionConfig) -> int:
    """Get number of teams for a layer based on tiered allocation."""
    if layer_idx < 8:
        return config.early_teams
    elif layer_idx < 18:
        return config.middle_teams
    else:
        return config.late_teams


async def convert_model(
    config: ConversionConfig,
    prompts: list[str],
) -> dict:
    """
    Convert GPT-OSS to TieredLightweight architecture.

    Returns conversion metadata including clustering results.
    """
    logger.info(f"Converting {config.model_id} to TieredLightweight")

    # Step 1: Analyze co-activation patterns
    coactivation, activation_freq = await analyze_expert_coactivation(
        config.model_id, prompts
    )

    # Step 2: Identify cold experts
    cold_experts = identify_cold_experts(activation_freq)

    # Step 3: Cluster experts into teams for each layer
    layer_clusters = {}

    for layer_idx in sorted(coactivation.keys()):
        num_teams = get_num_teams_for_layer(layer_idx, config)

        cluster_result = cluster_experts_to_teams(
            layer_idx=layer_idx,
            num_teams=num_teams,
            team_size=config.team_size,
            coactivation=coactivation[layer_idx],
            activation_freq=activation_freq[layer_idx],
            cold_experts=cold_experts,
        )

        layer_clusters[layer_idx] = cluster_result

        logger.info(
            f"Layer {layer_idx}: {num_teams} teams, "
            f"{len(cluster_result.excluded_experts)} cold experts excluded"
        )

    # Step 4: Save conversion metadata
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metadata = {
        "source_model": config.model_id,
        "config": {
            "early_teams": config.early_teams,
            "middle_teams": config.middle_teams,
            "late_teams": config.late_teams,
            "team_size": config.team_size,
        },
        "cold_experts": list(cold_experts),
        "layers": {
            str(layer_idx): {
                "num_teams": result.num_teams,
                "teams": result.teams,
                "mixing_weights": result.mixing_weights,
                "excluded_experts": result.excluded_experts,
            }
            for layer_idx, result in layer_clusters.items()
        },
    }

    metadata_path = output_path / "conversion_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info(f"Conversion metadata saved to {metadata_path}")

    return metadata, layer_clusters


def print_conversion_report(metadata: dict) -> None:
    """Print a summary of the conversion."""
    print()
    print("=" * 70)
    print("CONVERSION REPORT")
    print("=" * 70)
    print()
    print(f"Source model: {metadata['source_model']}")
    print()
    print("Tiered allocation:")
    print(f"  Early (L0-L7):   {metadata['config']['early_teams']} teams")
    print(f"  Middle (L8-L17): {metadata['config']['middle_teams']} teams")
    print(f"  Late (L18-L23):  {metadata['config']['late_teams']} teams")
    print(f"  Team size:       {metadata['config']['team_size']}")
    print()
    print(f"Cold experts excluded: {len(metadata['cold_experts'])}")
    print()

    # Summary by phase
    phases = {"early": [], "middle": [], "late": []}
    for layer_str, layer_data in metadata["layers"].items():
        layer_idx = int(layer_str)
        if layer_idx < 8:
            phases["early"].append(layer_data)
        elif layer_idx < 18:
            phases["middle"].append(layer_data)
        else:
            phases["late"].append(layer_data)

    print("Team assignments by phase:")
    for phase, layers in phases.items():
        avg_excluded = sum(len(l["excluded_experts"]) for l in layers) / len(layers)
        print(f"  {phase:6}: {len(layers)} layers, avg {avg_excluded:.1f} cold experts/layer")

    print()
    print("=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print()
    print("1. Transfer weights from source experts to teams")
    print("2. Evaluate initial quality (expect some degradation)")
    print("3. Distill from source to close quality gap")
    print("4. Final evaluation")
    print()


# Default prompts for analysis (same as circuit_transfer_test.py)
DEFAULT_PROMPTS = [
    "127 * 89 = ",
    "456 + 789 = ",
    "1000 / 25 = ",
    "Calculate the square root of 144",
    "What is 15% of 200?",
    "def fibonacci(n):",
    "import numpy as np",
    "class DataProcessor:",
    "for item in items:",
    "if x > 0:",
    "The capital of France is",
    "A synonym for happy is",
    "Shakespeare wrote many",
    "The opposite of hot is",
    "In the beginning there was",
    "If all cats are mammals, then",
    "To solve this equation, first",
    "The pattern continues as",
    "Therefore, we can conclude that",
    "Given the evidence, it follows that",
    "Once upon a time",
    "The quick brown fox",
    "Hello, my name is",
    "To summarize the main points",
    "In conclusion,",
]


async def main():
    """Run conversion."""
    parser = argparse.ArgumentParser(description="Convert GPT-OSS to TieredLightweight")
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-oss-20b",
        help="Source model ID",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="converted_tiered_lightweight",
        help="Output directory",
    )
    parser.add_argument(
        "--early-teams",
        type=int,
        default=4,
        help="Number of teams for early layers",
    )
    parser.add_argument(
        "--middle-teams",
        type=int,
        default=8,
        help="Number of teams for middle layers",
    )
    parser.add_argument(
        "--late-teams",
        type=int,
        default=6,
        help="Number of teams for late layers",
    )

    args = parser.parse_args()

    config = ConversionConfig(
        model_id=args.model,
        output_dir=args.output,
        early_teams=args.early_teams,
        middle_teams=args.middle_teams,
        late_teams=args.late_teams,
    )

    metadata, _ = await convert_model(config, DEFAULT_PROMPTS)
    print_conversion_report(metadata)


if __name__ == "__main__":
    asyncio.run(main())
