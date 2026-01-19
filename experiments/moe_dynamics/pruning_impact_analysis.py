#!/usr/bin/env python3
"""
Pruning Impact Analysis for Fact Retrieval

This script answers the critical question:
  "If we prune/virtualize experts, do we lose facts?"

Tests:
1. ATTENTION HYPOTHESIS: Facts survive pruning (stored in attention)
2. EXPERT HYPOTHESIS: Facts are lost with pruning (stored in expert weights)
3. DISTRIBUTED HYPOTHESIS: Partial degradation (facts spread across both)

Methodology:
- Run fact queries on full model → measure accuracy
- Simulate pruning by zeroing expert outputs → measure accuracy
- Compare fact types: static vs dynamic vs computed

Usage:
    python pruning_impact_analysis.py --model openai/gpt-oss-120b
    python pruning_impact_analysis.py --model openai/gpt-oss-120b --prune-rate 0.5
"""

from __future__ import annotations

import argparse
import gc
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np


# Import fact queries from the localization probe
from fact_localization_probe import FACTUAL_QUERIES, get_model_layers, create_causal_mask


@dataclass
class PruningExperiment:
    """Results from a single pruning experiment."""
    prune_rate: float  # 0.0 = no pruning, 1.0 = prune all
    prune_strategy: str  # "cold", "random", "by_layer"
    pruned_experts: list[tuple[int, int]]  # (layer, expert) pruned

    # Accuracy before/after
    accuracy_before: dict[str, float]  # fact_type -> accuracy
    accuracy_after: dict[str, float]

    # Per-category impact
    category_impact: dict[str, float]  # category -> delta accuracy

    # Key metrics
    total_facts_lost: int
    facts_degraded: int  # Probability dropped but still correct
    facts_recovered: int  # Wrong before, correct after (unlikely but possible)


@dataclass
class FactResult:
    """Result for a single fact query."""
    category: str
    fact_type: str
    prompt: str
    expected: str
    predicted: str
    correct: bool
    probability: float


def run_fact_query(
    model,
    tokenizer,
    prompt: str,
    expected: str,
    alternatives: list[str],
) -> FactResult:
    """Run a single fact query and return result."""
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    output = model(input_ids)
    if hasattr(output, 'logits'):
        output = output.logits

    last_logits = output[0, -1, :]
    probs = mx.softmax(last_logits, axis=-1)

    # Get prediction
    pred_idx = int(mx.argmax(last_logits))
    predicted = tokenizer.decode([pred_idx]).strip()

    # Check if correct
    all_targets = [expected] + alternatives
    correct = False
    max_prob = 0.0

    for target in all_targets:
        # Encode target
        try:
            encoded = tokenizer.encode(target)
            if encoded:
                tid = encoded[0] if len(encoded) == 1 else encoded[-1]
                if tid < probs.shape[0]:
                    prob = float(probs[tid])
                    max_prob = max(max_prob, prob)

                    # Check correctness
                    pred_lower = predicted.lower().strip()
                    target_lower = target.lower().strip()
                    if target_lower in pred_lower or pred_lower in target_lower:
                        correct = True
        except Exception:
            pass

    # Also check by string matching
    for target in all_targets:
        pred_lower = predicted.lower().strip()
        target_lower = target.lower().strip()
        if target_lower in pred_lower or pred_lower in target_lower:
            correct = True
            break

    return FactResult(
        category="",  # Will be set by caller
        fact_type="",  # Will be set by caller
        prompt=prompt,
        expected=expected,
        predicted=predicted,
        correct=correct,
        probability=max_prob,
    )


def run_all_facts(
    model,
    tokenizer,
    categories: list[str] | None = None,
    verbose: bool = False,
) -> list[FactResult]:
    """Run all fact queries and return results."""
    results = []

    for category, config in FACTUAL_QUERIES.items():
        if categories and category not in categories:
            continue

        fact_type = config["type"]
        queries = config["queries"]

        for query in queries:
            prompt = query["prompt"]
            expected = query["answer"]
            alternatives = query.get("alternatives", [])

            if expected == "varies":
                continue  # Skip dynamic facts with no fixed answer

            result = run_fact_query(model, tokenizer, prompt, expected, alternatives)
            result.category = category
            result.fact_type = fact_type
            results.append(result)

            if verbose:
                status = "✓" if result.correct else "✗"
                print(f"  {status} {prompt} → {result.predicted} (expected: {expected})")

    return results


def identify_cold_experts(
    model,
    tokenizer,
    sample_prompts: list[str] | None = None,
    threshold: float = 0.01,
) -> dict[int, list[int]]:
    """
    Identify cold (rarely activated) experts per layer.

    Returns:
        Dict mapping layer_idx -> list of cold expert indices
    """
    if sample_prompts is None:
        # Use a mix of fact prompts as sample
        sample_prompts = []
        for config in FACTUAL_QUERIES.values():
            for q in config["queries"][:2]:
                sample_prompts.append(q["prompt"])

    layers = get_model_layers(model)
    num_layers = len(layers)

    if hasattr(model.args, 'num_local_experts'):
        num_experts = model.args.num_local_experts
        top_k = model.args.num_experts_per_tok
    else:
        num_experts = 32
        top_k = 4

    hidden_size = model.args.hidden_size

    # Track activations
    activation_counts = defaultdict(lambda: defaultdict(int))
    total_tokens = 0

    for prompt in sample_prompts:
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        seq_len = len(tokens)

        # Get embeddings
        if hasattr(model, 'model'):
            h = model.model.embed_tokens(input_ids)
        else:
            h = model.embed_tokens(input_ids)

        mask = create_causal_mask(seq_len, h.dtype)

        for layer_idx, layer in enumerate(layers):
            if hasattr(layer, 'input_layernorm'):
                normed = layer.input_layernorm(h)
            else:
                normed = h

            mlp = layer.mlp
            if hasattr(mlp, 'router'):
                x_flat = normed.reshape(-1, hidden_size)
                logits = mlp.router(x_flat)
                top_k_indices = mx.argsort(logits, axis=-1)[:, -top_k:]

                for tok_idx in range(seq_len):
                    for k in range(top_k):
                        expert_idx = int(top_k_indices[tok_idx, k])
                        activation_counts[layer_idx][expert_idx] += 1

            h = layer(h, mask=mask)

        total_tokens += seq_len
        mx.eval(h)
        gc.collect()

    # Identify cold experts (activated < threshold * expected)
    expected_per_expert = total_tokens * top_k / num_experts
    cold_threshold = threshold * expected_per_expert

    cold_experts = {}
    for layer_idx in range(num_layers):
        cold_in_layer = []
        for expert_idx in range(num_experts):
            count = activation_counts[layer_idx].get(expert_idx, 0)
            if count < cold_threshold:
                cold_in_layer.append(expert_idx)
        cold_experts[layer_idx] = cold_in_layer

    return cold_experts


class ExpertPruner:
    """Context manager for temporarily pruning experts."""

    def __init__(
        self,
        model,
        experts_to_prune: dict[int, list[int]],  # layer -> [expert_indices]
    ):
        self.model = model
        self.experts_to_prune = experts_to_prune
        self.original_weights = {}
        self.layers = get_model_layers(model)

    def __enter__(self):
        """Zero out the specified experts."""
        for layer_idx, expert_indices in self.experts_to_prune.items():
            if layer_idx >= len(self.layers):
                continue

            layer = self.layers[layer_idx]
            mlp = layer.mlp

            if not hasattr(mlp, 'experts'):
                continue

            experts = mlp.experts

            for expert_idx in expert_indices:
                if expert_idx >= len(experts):
                    continue

                expert = experts[expert_idx]

                # Store original weights
                self.original_weights[(layer_idx, expert_idx)] = {}

                # Zero out the down projection (output)
                if hasattr(expert, 'down_proj'):
                    original = expert.down_proj.weight
                    self.original_weights[(layer_idx, expert_idx)]['down_proj'] = original
                    expert.down_proj.weight = mx.zeros_like(original)

        return self

    def __exit__(self, *args):
        """Restore original weights."""
        for (layer_idx, expert_idx), weights in self.original_weights.items():
            layer = self.layers[layer_idx]
            expert = layer.mlp.experts[expert_idx]

            if 'down_proj' in weights:
                expert.down_proj.weight = weights['down_proj']


def run_pruning_experiment(
    model,
    tokenizer,
    prune_rate: float,
    prune_strategy: str = "cold",
    verbose: bool = False,
) -> PruningExperiment:
    """
    Run a pruning experiment and measure impact on facts.

    Args:
        model: The MoE model
        tokenizer: Tokenizer
        prune_rate: Fraction of experts to prune (0-1)
        prune_strategy: "cold" (prune cold experts), "random", "by_layer"
        verbose: Show per-fact results

    Returns:
        PruningExperiment with before/after comparison
    """
    print(f"\nRunning pruning experiment: {prune_rate:.0%} prune rate, strategy={prune_strategy}")
    print("-" * 60)

    # Get baseline results (no pruning)
    print("Running baseline (no pruning)...")
    baseline_results = run_all_facts(model, tokenizer, verbose=verbose)

    # Compute baseline accuracy by type
    accuracy_before = {}
    for fact_type in ["static", "dynamic", "computed"]:
        type_results = [r for r in baseline_results if r.fact_type == fact_type]
        if type_results:
            accuracy_before[fact_type] = sum(1 for r in type_results if r.correct) / len(type_results)

    # Identify experts to prune
    layers = get_model_layers(model)
    num_layers = len(layers)

    if hasattr(model.args, 'num_local_experts'):
        num_experts = model.args.num_local_experts
    else:
        num_experts = 32

    experts_to_prune: dict[int, list[int]] = {}

    if prune_strategy == "cold":
        # Prune cold experts
        cold_experts = identify_cold_experts(model, tokenizer)

        # Take the coldest fraction
        all_cold = []
        for layer_idx, experts in cold_experts.items():
            for exp_idx in experts:
                all_cold.append((layer_idx, exp_idx))

        # Prune up to prune_rate of total experts
        total_experts = num_layers * num_experts
        num_to_prune = int(total_experts * prune_rate)
        num_to_prune = min(num_to_prune, len(all_cold))

        # Select coldest experts
        selected = all_cold[:num_to_prune]
        for layer_idx, exp_idx in selected:
            if layer_idx not in experts_to_prune:
                experts_to_prune[layer_idx] = []
            experts_to_prune[layer_idx].append(exp_idx)

    elif prune_strategy == "random":
        # Random pruning
        import random
        total_experts = num_layers * num_experts
        num_to_prune = int(total_experts * prune_rate)

        all_experts = [(l, e) for l in range(num_layers) for e in range(num_experts)]
        selected = random.sample(all_experts, num_to_prune)

        for layer_idx, exp_idx in selected:
            if layer_idx not in experts_to_prune:
                experts_to_prune[layer_idx] = []
            experts_to_prune[layer_idx].append(exp_idx)

    elif prune_strategy == "by_layer":
        # Prune middle layers (keep early + late)
        early_cutoff = num_layers // 4
        late_start = 3 * num_layers // 4

        for layer_idx in range(early_cutoff, late_start):
            # Prune random fraction of experts in middle layers
            num_to_prune = int(num_experts * prune_rate)
            experts_to_prune[layer_idx] = list(range(num_to_prune))

    # Count pruned
    total_pruned = sum(len(e) for e in experts_to_prune.values())
    print(f"Pruning {total_pruned} experts across {len(experts_to_prune)} layers")

    # Run with pruning
    print("Running with pruning applied...")
    with ExpertPruner(model, experts_to_prune):
        pruned_results = run_all_facts(model, tokenizer, verbose=verbose)

    # Compute pruned accuracy by type
    accuracy_after = {}
    for fact_type in ["static", "dynamic", "computed"]:
        type_results = [r for r in pruned_results if r.fact_type == fact_type]
        if type_results:
            accuracy_after[fact_type] = sum(1 for r in type_results if r.correct) / len(type_results)

    # Compute per-category impact
    category_impact = {}
    for category in FACTUAL_QUERIES:
        baseline_cat = [r for r in baseline_results if r.category == category]
        pruned_cat = [r for r in pruned_results if r.category == category]

        if baseline_cat and pruned_cat:
            baseline_acc = sum(1 for r in baseline_cat if r.correct) / len(baseline_cat)
            pruned_acc = sum(1 for r in pruned_cat if r.correct) / len(pruned_cat)
            category_impact[category] = pruned_acc - baseline_acc

    # Count facts lost/degraded
    facts_lost = 0
    facts_degraded = 0
    facts_recovered = 0

    for base, pruned in zip(baseline_results, pruned_results):
        if base.correct and not pruned.correct:
            facts_lost += 1
        elif base.correct and pruned.correct and pruned.probability < base.probability * 0.5:
            facts_degraded += 1
        elif not base.correct and pruned.correct:
            facts_recovered += 1

    # Convert to list for storage
    pruned_list = []
    for layer_idx, experts in experts_to_prune.items():
        for exp_idx in experts:
            pruned_list.append((layer_idx, exp_idx))

    return PruningExperiment(
        prune_rate=prune_rate,
        prune_strategy=prune_strategy,
        pruned_experts=pruned_list,
        accuracy_before=accuracy_before,
        accuracy_after=accuracy_after,
        category_impact=category_impact,
        total_facts_lost=facts_lost,
        facts_degraded=facts_degraded,
        facts_recovered=facts_recovered,
    )


def print_pruning_report(experiments: list[PruningExperiment]):
    """Print comprehensive pruning impact report."""
    print()
    print("=" * 80)
    print("PRUNING IMPACT ON FACT RETRIEVAL")
    print("=" * 80)
    print()

    # Summary table
    print("-" * 80)
    print(f"{'Prune Rate':<12} {'Strategy':<12} {'Static Δ':<12} {'Dynamic Δ':<12} {'Computed Δ':<12} {'Lost':<8}")
    print("-" * 80)

    for exp in experiments:
        static_delta = exp.accuracy_after.get("static", 0) - exp.accuracy_before.get("static", 0)
        dynamic_delta = exp.accuracy_after.get("dynamic", 0) - exp.accuracy_before.get("dynamic", 0)
        computed_delta = exp.accuracy_after.get("computed", 0) - exp.accuracy_before.get("computed", 0)

        print(f"{exp.prune_rate:>10.0%}  {exp.prune_strategy:<12} {static_delta:>+10.1%} {dynamic_delta:>+10.1%} {computed_delta:>+10.1%} {exp.total_facts_lost:>6}")

    print()

    # Interpretation
    print("-" * 80)
    print("INTERPRETATION")
    print("-" * 80)
    print()

    # Find most impactful experiment
    if experiments:
        worst_exp = max(experiments, key=lambda e: e.total_facts_lost)

        if worst_exp.total_facts_lost == 0:
            print("FINDING: Facts SURVIVED pruning!")
            print("  → Supports ATTENTION HYPOTHESIS: Facts are stored in attention, not experts")
            print("  → Virtual experts are SAFE for knowledge preservation")
            print()
        else:
            # Check which fact types were affected
            static_drop = worst_exp.accuracy_before.get("static", 0) - worst_exp.accuracy_after.get("static", 0)
            computed_drop = worst_exp.accuracy_before.get("computed", 0) - worst_exp.accuracy_after.get("computed", 0)

            if static_drop > 0.1:
                print("FINDING: Static facts DEGRADED with pruning")
                print("  → Facts ARE stored in expert weights")
                print("  → Virtual experts may lose knowledge")
                print("  → Consider: RAG for fact retrieval, keep knowledge-critical experts")
                print()

            if computed_drop > static_drop:
                print("FINDING: Computed facts more affected than static facts")
                print("  → Computation happens in experts, facts may be in attention")
                print("  → Virtualize computation (calculator) but keep fact retrieval")
                print()

    # Category-level insights
    print("-" * 80)
    print("CATEGORY-LEVEL IMPACT")
    print("-" * 80)
    print()

    for exp in experiments:
        if exp.category_impact:
            print(f"Prune rate: {exp.prune_rate:.0%} ({exp.prune_strategy})")
            for cat, delta in sorted(exp.category_impact.items(), key=lambda x: x[1]):
                bar = "█" * int(abs(delta) * 20) if delta < 0 else "░" * int(abs(delta) * 20)
                direction = "↓" if delta < 0 else "↑"
                print(f"  {cat:<20} {direction} {bar} {delta:+.1%}")
            print()

    # Recommendations
    print("=" * 80)
    print("RECOMMENDATIONS FOR VIRTUAL EXPERTS")
    print("=" * 80)
    print()

    print("Based on pruning impact analysis:")
    print()
    print("1. SAFE TO VIRTUALIZE:")
    print("   - Computed facts (arithmetic, dates) → Use calculator/tools")
    print("   - Cold experts (rarely activated)")
    print()
    print("2. CAUTION NEEDED:")
    print("   - Static knowledge experts (if identified)")
    print("   - High-activation experts in early layers")
    print()
    print("3. ALTERNATIVE STRATEGIES:")
    print("   - RAG for fact retrieval (always accurate, updatable)")
    print("   - Keep knowledge-critical experts, virtualize computation")
    print("   - Use attention analysis to identify fact storage layers")
    print()


def save_results(experiments: list[PruningExperiment], output_path: Path):
    """Save experiment results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "timestamp": datetime.now().isoformat(),
        "num_experiments": len(experiments),
        "experiments": [
            {
                "prune_rate": exp.prune_rate,
                "prune_strategy": exp.prune_strategy,
                "num_pruned": len(exp.pruned_experts),
                "accuracy_before": exp.accuracy_before,
                "accuracy_after": exp.accuracy_after,
                "category_impact": exp.category_impact,
                "facts_lost": exp.total_facts_lost,
                "facts_degraded": exp.facts_degraded,
            }
            for exp in experiments
        ],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Pruning Impact Analysis")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-120b")
    parser.add_argument("--output", type=str, default="results/pruning_impact.json")
    parser.add_argument("--prune-rates", type=float, nargs="+",
                        default=[0.1, 0.3, 0.5, 0.7],
                        help="Pruning rates to test")
    parser.add_argument("--strategies", type=str, nargs="+",
                        default=["cold", "random"],
                        help="Pruning strategies to test")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # Load model
    from mlx_lm import load
    print(f"Loading {args.model}...")
    model, tokenizer = load(args.model)

    # Run experiments
    experiments = []

    for strategy in args.strategies:
        for prune_rate in args.prune_rates:
            try:
                exp = run_pruning_experiment(
                    model, tokenizer,
                    prune_rate=prune_rate,
                    prune_strategy=strategy,
                    verbose=args.verbose,
                )
                experiments.append(exp)
                gc.collect()
            except Exception as e:
                print(f"Error with {strategy}/{prune_rate}: {e}")
                continue

    # Report and save
    print_pruning_report(experiments)
    save_results(experiments, Path(args.output))


if __name__ == "__main__":
    main()
