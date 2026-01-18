#!/usr/bin/env python3
"""
Circuit Transfer Test.

Tests whether expert circuits discovered on one model transfer to another.
This is the key gate for whether Circuit-MoE is viable at scale.

Research Questions:
1. Do circuits discovered on small models transfer to larger models?
2. Do circuits discovered on one dataset work on different datasets?
3. What's the minimum data needed to discover transferable circuits?

Usage:
    python experiments/moe_dynamics/circuit_transfer_test.py
    python experiments/moe_dynamics/circuit_transfer_test.py --models openai/gpt-oss-20b,meta-llama/Llama-3.1-8B
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CircuitDefinition:
    """A discovered circuit (expert path across layers)."""

    circuit_id: int
    expert_sequence: list[int]  # expert_idx at each layer
    consistency: float  # How consistent this circuit is across prompts
    activation_count: int  # How often this circuit activates
    category: str = "unknown"  # e.g., "arithmetic", "code", "language"


@dataclass
class CircuitDiscoveryResult:
    """Results of circuit discovery on a model."""

    model_id: str
    num_circuits: int
    circuits: list[CircuitDefinition]
    global_consistency: float
    discovery_prompts: list[str] = field(default_factory=list)


@dataclass
class CircuitTransferResult:
    """Results of testing circuit transfer between models."""

    source_model: str
    target_model: str
    overlap_score: float  # Jaccard similarity
    aligned_circuits: int  # Circuits that transfer exactly
    partial_matches: int  # Circuits with >50% layer overlap
    mismatch_layers: dict[int, int] = field(default_factory=dict)  # layer -> mismatch count


async def discover_circuits_from_model(
    model_id: str,
    prompts: list[str],
    num_circuits: int = 15,
    consistency_threshold: float = 0.8,
) -> CircuitDiscoveryResult:
    """
    Discover expert circuits from a trained MoE model.

    Args:
        model_id: HuggingFace model ID
        prompts: Prompts to run through model
        num_circuits: Number of top circuits to return
        consistency_threshold: Minimum consistency to include

    Returns:
        CircuitDiscoveryResult with discovered circuits
    """
    from chuk_lazarus.introspection.moe import ExpertRouter

    logger.info(f"Loading model: {model_id}")
    router = await ExpertRouter.from_pretrained(model_id)

    info = router.info
    num_layers = len(info.moe_layers)
    num_experts = info.num_experts

    # Track expert co-occurrence across layers
    # co_occurrence[layer][expert_a][expert_b] = count of prompts where both active
    layer_expert_sequences: list[list[list[int]]] = []  # [prompt_idx][layer_idx] = [experts]

    logger.info(f"Running {len(prompts)} prompts to discover circuits...")

    for prompt in prompts:
        weights = await router.capture_router_weights(prompt, layers=list(info.moe_layers))

        prompt_sequence: list[list[int]] = []
        for layer_weights in weights:
            if layer_weights.positions:
                # Use last position (typically the target token)
                last_pos = layer_weights.positions[-1]
                prompt_sequence.append(list(last_pos.expert_indices))
            else:
                prompt_sequence.append([])

        layer_expert_sequences.append(prompt_sequence)

    # Find consistent expert paths
    # For each prompt, create a "circuit signature" = tuple of primary experts
    circuit_signatures: dict[tuple[int, ...], int] = {}

    for prompt_seq in layer_expert_sequences:
        if all(len(experts) > 0 for experts in prompt_seq):
            # Primary expert at each layer
            signature = tuple(experts[0] for experts in prompt_seq)
            circuit_signatures[signature] = circuit_signatures.get(signature, 0) + 1

    # Sort by frequency
    sorted_circuits = sorted(circuit_signatures.items(), key=lambda x: x[1], reverse=True)

    # Build circuit definitions
    circuits = []
    total_prompts = len(prompts)

    for i, (signature, count) in enumerate(sorted_circuits[:num_circuits]):
        consistency = count / total_prompts
        if consistency >= consistency_threshold or i < 5:  # Always include top 5
            circuits.append(
                CircuitDefinition(
                    circuit_id=i,
                    expert_sequence=list(signature),
                    consistency=consistency,
                    activation_count=count,
                )
            )

    # Global consistency = average consistency of top circuits
    global_consistency = sum(c.consistency for c in circuits) / len(circuits) if circuits else 0

    return CircuitDiscoveryResult(
        model_id=model_id,
        num_circuits=len(circuits),
        circuits=circuits,
        global_consistency=global_consistency,
        discovery_prompts=prompts,
    )


def compute_circuit_transfer(
    source: CircuitDiscoveryResult,
    target: CircuitDiscoveryResult,
) -> CircuitTransferResult:
    """
    Compute how well circuits transfer from source to target model.

    Args:
        source: Circuits discovered on source model
        target: Circuits discovered on target model

    Returns:
        CircuitTransferResult with overlap metrics
    """
    source_circuits = {tuple(c.expert_sequence) for c in source.circuits}
    target_circuits = {tuple(c.expert_sequence) for c in target.circuits}

    # Exact matches
    intersection = source_circuits & target_circuits
    union = source_circuits | target_circuits
    overlap_score = len(intersection) / len(union) if union else 0

    # Partial matches (>50% layer agreement)
    partial_matches = 0
    mismatch_layers: dict[int, int] = {}

    for source_circuit in source.circuits:
        best_overlap = 0
        for target_circuit in target.circuits:
            # Count matching layers
            matches = sum(
                1 for s, t in zip(source_circuit.expert_sequence, target_circuit.expert_sequence)
                if s == t
            )
            overlap = matches / len(source_circuit.expert_sequence)
            best_overlap = max(best_overlap, overlap)

            # Track which layers mismatch
            for layer_idx, (s, t) in enumerate(
                zip(source_circuit.expert_sequence, target_circuit.expert_sequence)
            ):
                if s != t:
                    mismatch_layers[layer_idx] = mismatch_layers.get(layer_idx, 0) + 1

        if best_overlap >= 0.5:
            partial_matches += 1

    return CircuitTransferResult(
        source_model=source.model_id,
        target_model=target.model_id,
        overlap_score=overlap_score,
        aligned_circuits=len(intersection),
        partial_matches=partial_matches,
        mismatch_layers=mismatch_layers,
    )


def print_transfer_report(
    source: CircuitDiscoveryResult,
    target: CircuitDiscoveryResult,
    transfer: CircuitTransferResult,
) -> None:
    """Print a formatted report of circuit transfer results."""
    print()
    print("=" * 70)
    print("CIRCUIT TRANSFER TEST")
    print("=" * 70)
    print()
    print(f"Source model: {source.model_id}")
    print(f"  Circuits discovered: {source.num_circuits}")
    print(f"  Global consistency: {source.global_consistency:.1%}")
    print()
    print(f"Target model: {target.model_id}")
    print(f"  Circuits discovered: {target.num_circuits}")
    print(f"  Global consistency: {target.global_consistency:.1%}")
    print()
    print("=" * 70)
    print("TRANSFER RESULTS")
    print("=" * 70)
    print()
    print(f"  Overlap score (Jaccard): {transfer.overlap_score:.1%}")
    print(f"  Exact matches: {transfer.aligned_circuits}")
    print(f"  Partial matches (>50%): {transfer.partial_matches}")
    print()

    if transfer.mismatch_layers:
        print("  Mismatch by layer:")
        sorted_layers = sorted(transfer.mismatch_layers.items(), key=lambda x: x[1], reverse=True)
        for layer, count in sorted_layers[:5]:
            print(f"    L{layer}: {count} mismatches")
    print()

    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print()

    if transfer.overlap_score >= 0.8:
        print("  STRONG TRANSFER: Circuits are highly transferable.")
        print("  Circuit-MoE is viable - can use discovered circuits as priors.")
    elif transfer.overlap_score >= 0.5:
        print("  MODERATE TRANSFER: Some circuits transfer.")
        print("  Circuit-MoE may work with fine-tuning or layer-specific circuits.")
    elif transfer.partial_matches > source.num_circuits * 0.5:
        print("  PARTIAL TRANSFER: Circuit structure transfers but expert IDs differ.")
        print("  May need expert alignment or relative circuit definitions.")
    else:
        print("  WEAK TRANSFER: Circuits don't transfer well.")
        print("  Circuit-MoE requires per-model circuit discovery.")

    print()
    print("=" * 70)


# Default prompts for circuit discovery
DEFAULT_PROMPTS = [
    # Arithmetic
    "127 * 89 = ",
    "456 + 789 = ",
    "1000 / 25 = ",
    "Calculate the square root of 144",
    "What is 15% of 200?",
    # Code
    "def fibonacci(n):",
    "import numpy as np",
    "class DataProcessor:",
    "for item in items:",
    "if x > 0:",
    # Language
    "The capital of France is",
    "A synonym for happy is",
    "Shakespeare wrote many",
    "The opposite of hot is",
    "In the beginning there was",
    # Reasoning
    "If all cats are mammals, then",
    "To solve this equation, first",
    "The pattern continues as",
    "Therefore, we can conclude that",
    "Given the evidence, it follows that",
    # Mixed
    "Once upon a time",
    "The quick brown fox",
    "Hello, my name is",
    "To summarize the main points",
    "In conclusion,",
]


async def main():
    """Run circuit transfer test."""
    parser = argparse.ArgumentParser(description="Circuit Transfer Test")
    parser.add_argument(
        "--models",
        type=str,
        default="openai/gpt-oss-20b",
        help="Comma-separated model IDs to compare",
    )
    parser.add_argument(
        "--num-circuits",
        type=int,
        default=15,
        help="Number of circuits to discover",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results",
    )

    args = parser.parse_args()
    model_ids = [m.strip() for m in args.models.split(",")]

    if len(model_ids) < 2:
        # If only one model, compare it against itself with different prompt subsets
        logger.info("Single model mode: comparing circuit discovery consistency")
        model_ids = [model_ids[0], model_ids[0]]

    # Discover circuits on each model
    results = []
    for model_id in model_ids:
        result = await discover_circuits_from_model(
            model_id,
            DEFAULT_PROMPTS,
            num_circuits=args.num_circuits,
        )
        results.append(result)
        logger.info(f"Discovered {result.num_circuits} circuits on {model_id}")

    # Compare all pairs
    for i, source in enumerate(results):
        for j, target in enumerate(results):
            if i >= j:
                continue

            transfer = compute_circuit_transfer(source, target)
            print_transfer_report(source, target, transfer)

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_data = {
            "models": model_ids,
            "discovery_results": [
                {
                    "model_id": r.model_id,
                    "num_circuits": r.num_circuits,
                    "global_consistency": r.global_consistency,
                    "circuits": [
                        {
                            "id": c.circuit_id,
                            "sequence": c.expert_sequence,
                            "consistency": c.consistency,
                        }
                        for c in r.circuits
                    ],
                }
                for r in results
            ],
        }
        output_path.write_text(json.dumps(output_data, indent=2))
        logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
