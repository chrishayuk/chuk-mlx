#!/usr/bin/env python3
"""
Fact Localization Probe for MoE Models

This script investigates WHERE factual knowledge is stored in MoE architectures:
- In expert weights (MLP/FFN)?
- In attention patterns?
- Distributed across layers?

Key questions:
1. When completing "The capital of France is ___", where does "Paris" come from?
2. Do static facts (capitals) vs dynamic facts (CEOs) live in different places?
3. What happens to fact retrieval when we prune experts?

Uses logit lens to trace token probabilities through layers, combined with
expert activation analysis to identify "knowledge storage" locations.

Usage:
    python fact_localization_probe.py --model openai/gpt-oss-120b
    python fact_localization_probe.py --model openai/gpt-oss-20b --detailed
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


# =============================================================================
# Fact Categories - Static vs Dynamic vs Computed
# =============================================================================

FACTUAL_QUERIES = {
    # =========================================================================
    # STATIC FACTS - Should be stored in model weights
    # These never change, model should "know" them
    # =========================================================================

    "geography": {
        "type": "static",
        "description": "Immutable geographic facts",
        "queries": [
            {"prompt": "The capital of France is", "answer": "Paris", "alternatives": ["paris", " Paris", " paris"]},
            {"prompt": "The capital of Japan is", "answer": "Tokyo", "alternatives": ["tokyo", " Tokyo", " tokyo"]},
            {"prompt": "The capital of Brazil is", "answer": "Brasília", "alternatives": ["Brasilia", "brasilia", " Brasília"]},
            {"prompt": "The capital of Australia is", "answer": "Canberra", "alternatives": ["canberra", " Canberra"]},
            {"prompt": "The capital of Germany is", "answer": "Berlin", "alternatives": ["berlin", " Berlin"]},
            {"prompt": "The largest ocean is the", "answer": "Pacific", "alternatives": ["pacific", " Pacific"]},
            {"prompt": "The longest river in the world is the", "answer": "Nile", "alternatives": ["nile", " Nile", "Amazon", "amazon"]},
            {"prompt": "The highest mountain is Mount", "answer": "Everest", "alternatives": ["everest", " Everest"]},
        ],
    },

    "science_constants": {
        "type": "static",
        "description": "Physical and mathematical constants",
        "queries": [
            {"prompt": "Water boils at", "answer": "100", "alternatives": ["212", " 100", "100°"]},
            {"prompt": "The speed of light is approximately", "answer": "299", "alternatives": ["300", "3", " 299"]},
            {"prompt": "Absolute zero is", "answer": "-273", "alternatives": ["-459", "0", " -273"]},
            {"prompt": "Pi is approximately", "answer": "3.14", "alternatives": ["3", " 3.14", "3.1"]},
            {"prompt": "The atomic number of carbon is", "answer": "6", "alternatives": [" 6", "six"]},
            {"prompt": "Humans have", "answer": "46", "alternatives": ["23", " 46", "46 chromosomes"]},
            {"prompt": "The speed of sound in air is approximately", "answer": "343", "alternatives": ["340", "330", " 343"]},
        ],
    },

    "historical_dates": {
        "type": "static",
        "description": "Immutable historical events",
        "queries": [
            {"prompt": "World War 2 ended in", "answer": "1945", "alternatives": [" 1945", "1945."]},
            {"prompt": "The first moon landing was in", "answer": "1969", "alternatives": [" 1969", "1969."]},
            {"prompt": "Shakespeare was born in", "answer": "1564", "alternatives": [" 1564", "1564."]},
            {"prompt": "The French Revolution began in", "answer": "1789", "alternatives": [" 1789", "1789."]},
            {"prompt": "World War 1 started in", "answer": "1914", "alternatives": [" 1914", "1914."]},
            {"prompt": "The Declaration of Independence was signed in", "answer": "1776", "alternatives": [" 1776", "1776."]},
        ],
    },

    # =========================================================================
    # DYNAMIC FACTS - Model will try but often be wrong/outdated
    # =========================================================================

    "current_entities": {
        "type": "dynamic",
        "description": "Facts that change over time",
        "queries": [
            {"prompt": "The CEO of Apple is", "answer": "Tim Cook", "alternatives": ["tim cook", " Tim", "Cook"]},
            {"prompt": "The president of the United States is", "answer": "varies", "alternatives": []},  # Changes!
            {"prompt": "The president of France is", "answer": "varies", "alternatives": []},  # Changes!
            {"prompt": "The tallest building in the world is", "answer": "Burj Khalifa", "alternatives": ["burj", "Burj"]},
            {"prompt": "The current year is", "answer": "varies", "alternatives": []},  # Changes!
            {"prompt": "The world population is approximately", "answer": "8", "alternatives": ["7", " 8", "8 billion"]},
        ],
    },

    # =========================================================================
    # COMPUTED FACTS - Lookup tables / procedural knowledge
    # =========================================================================

    "arithmetic": {
        "type": "computed",
        "description": "Facts the model computes rather than retrieves",
        "queries": [
            {"prompt": "2 + 2 =", "answer": "4", "alternatives": [" 4", "4.", "four"]},
            {"prompt": "7 * 8 =", "answer": "56", "alternatives": [" 56", "56."]},
            {"prompt": "100 / 4 =", "answer": "25", "alternatives": [" 25", "25."]},
            {"prompt": "15 - 9 =", "answer": "6", "alternatives": [" 6", "6."]},
            {"prompt": "3^2 =", "answer": "9", "alternatives": [" 9", "9."]},
            {"prompt": "sqrt(144) =", "answer": "12", "alternatives": [" 12", "12."]},
        ],
    },

    "language_rules": {
        "type": "computed",
        "description": "Procedural language knowledge",
        "queries": [
            {"prompt": "The plural of 'mouse' is", "answer": "mice", "alternatives": [" mice", "'mice'"]},
            {"prompt": "The past tense of 'go' is", "answer": "went", "alternatives": [" went", "'went'"]},
            {"prompt": "The opposite of 'hot' is", "answer": "cold", "alternatives": [" cold", "'cold'"]},
            {"prompt": "The comparative form of 'good' is", "answer": "better", "alternatives": [" better", "'better'"]},
        ],
    },
}


@dataclass
class TokenProbabilityTrace:
    """Probability of target token at each layer."""
    prompt: str
    target_token: str
    target_token_id: int
    layer_probs: list[float]  # P(target) at each layer
    layer_ranks: list[int]    # Rank of target at each layer
    emergence_layer: int      # First layer where target enters top-10
    dominant_layer: int       # Layer where target becomes top-1
    final_prob: float
    final_rank: int


@dataclass
class ExpertContributionTrace:
    """Which experts activate when retrieving a fact."""
    prompt: str
    target_token: str
    layer_experts: dict[int, list[tuple[int, float]]]  # layer -> [(expert_idx, weight), ...]
    key_layers: list[int]  # Layers where target prob increases most
    key_experts: list[tuple[int, int, float]]  # (layer, expert, correlation) for fact retrieval


@dataclass
class FactLocalizationResult:
    """Complete analysis for a single fact."""
    category: str
    fact_type: str  # static, dynamic, computed
    prompt: str
    expected_answer: str
    actual_prediction: str
    prediction_correct: bool
    probability_trace: TokenProbabilityTrace
    expert_trace: ExpertContributionTrace

    # Key insights
    knowledge_layer_range: tuple[int, int]  # Where fact "emerges"
    primary_storage: str  # "early", "middle", "late"
    expert_dependency: float  # How much experts matter vs attention (0-1)


@dataclass
class FactTypeAnalysis:
    """Aggregate analysis for a fact type."""
    fact_type: str
    num_queries: int
    accuracy: float
    avg_emergence_layer: float
    avg_dominant_layer: float
    layer_distribution: dict[str, float]  # early/middle/late percentages
    expert_dependency: float
    common_experts: list[tuple[int, int, int]]  # (layer, expert, count)


def get_model_layers(model):
    """Get transformer layers from model."""
    if hasattr(model, 'model'):
        return model.model.layers
    return model.layers


def create_causal_mask(seq_len: int, dtype) -> mx.array:
    """Create causal attention mask."""
    return mx.triu(mx.full((seq_len, seq_len), float('-inf'), dtype=dtype), k=1)


def trace_token_probability(
    model,
    tokenizer,
    prompt: str,
    target_tokens: list[str],
    verbose: bool = False,
) -> TokenProbabilityTrace | None:
    """
    Trace the probability of a target token through all layers using logit lens.

    At each layer, we project the hidden state to vocabulary space and
    measure P(target_token).
    """
    # Encode prompt
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    # Get target token IDs (handle multiple possible tokenizations)
    target_token_ids = []
    for target in target_tokens:
        try:
            encoded = tokenizer.encode(target)
            if encoded:
                # Get first token after any prefix
                if len(encoded) > 0:
                    target_token_ids.append(encoded[0] if len(encoded) == 1 else encoded[-1])
        except Exception:
            pass

    if not target_token_ids:
        return None

    layers = get_model_layers(model)
    num_layers = len(layers)

    # Get LM head for projecting to vocabulary
    if hasattr(model, 'lm_head'):
        lm_head = model.lm_head
    elif hasattr(model, 'model') and hasattr(model, 'lm_head'):
        lm_head = model.lm_head
    else:
        # Try tied embeddings
        if hasattr(model, 'model'):
            lm_head = model.model.embed_tokens
        else:
            lm_head = model.embed_tokens

    # Get embeddings
    if hasattr(model, 'model'):
        h = model.model.embed_tokens(input_ids)
    else:
        h = model.embed_tokens(input_ids)

    batch_size, seq_len, hidden_size = h.shape
    mask = create_causal_mask(seq_len, h.dtype)

    layer_probs = []
    layer_ranks = []

    # Track hidden states through layers
    for layer_idx, layer in enumerate(layers):
        h = layer(h, mask=mask)

        # Apply final norm if we're projecting mid-network
        if hasattr(model, 'model') and hasattr(model.model, 'norm'):
            h_normed = model.model.norm(h)
        elif hasattr(model, 'norm'):
            h_normed = model.norm(h)
        else:
            h_normed = h

        # Project to vocabulary (logit lens)
        # Get last token's hidden state
        last_hidden = h_normed[0, -1, :]

        if hasattr(lm_head, 'weight'):
            logits = last_hidden @ lm_head.weight.T
        else:
            logits = lm_head(last_hidden.reshape(1, -1))[0]

        # Get probabilities
        probs = mx.softmax(logits, axis=-1)

        # Get max probability across target tokens
        max_prob = 0.0
        min_rank = float('inf')
        for tid in target_token_ids:
            if tid < probs.shape[0]:
                prob = float(probs[tid])
                max_prob = max(max_prob, prob)

                # Compute rank
                rank = int(mx.sum(probs > probs[tid]))
                min_rank = min(min_rank, rank)

        layer_probs.append(max_prob)
        layer_ranks.append(int(min_rank) if min_rank != float('inf') else 100000)

        mx.eval(h)

    # Find emergence and dominant layers
    emergence_layer = num_layers  # Default: never enters top-10
    for i, rank in enumerate(layer_ranks):
        if rank < 10:
            emergence_layer = i
            break

    dominant_layer = num_layers  # Default: never becomes top-1
    for i, rank in enumerate(layer_ranks):
        if rank == 0:
            dominant_layer = i
            break

    return TokenProbabilityTrace(
        prompt=prompt,
        target_token=target_tokens[0],
        target_token_id=target_token_ids[0] if target_token_ids else -1,
        layer_probs=layer_probs,
        layer_ranks=layer_ranks,
        emergence_layer=emergence_layer,
        dominant_layer=dominant_layer,
        final_prob=layer_probs[-1] if layer_probs else 0.0,
        final_rank=layer_ranks[-1] if layer_ranks else 100000,
    )


def trace_expert_contributions(
    model,
    tokenizer,
    prompt: str,
    verbose: bool = False,
) -> ExpertContributionTrace:
    """
    Track which experts activate during fact retrieval.
    """
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    layers = get_model_layers(model)
    num_layers = len(layers)

    if hasattr(model.args, 'num_local_experts'):
        num_experts = model.args.num_local_experts
        top_k = model.args.num_experts_per_tok
    else:
        num_experts = 32  # Default
        top_k = 4

    hidden_size = model.args.hidden_size

    # Get embeddings
    if hasattr(model, 'model'):
        h = model.model.embed_tokens(input_ids)
    else:
        h = model.embed_tokens(input_ids)

    batch_size, seq_len, _ = h.shape
    mask = create_causal_mask(seq_len, h.dtype)

    layer_experts: dict[int, list[tuple[int, float]]] = {}

    for layer_idx, layer in enumerate(layers):
        # Get pre-MLP hidden state for routing
        if hasattr(layer, 'input_layernorm'):
            normed = layer.input_layernorm(h)
        elif hasattr(layer, 'post_attention_layernorm'):
            normed = layer.post_attention_layernorm(h)
        else:
            normed = h

        mlp = layer.mlp
        if hasattr(mlp, 'router'):
            # Get routing for last token only (next token prediction)
            last_hidden = normed[0, -1:, :]
            x_flat = last_hidden.reshape(-1, hidden_size)

            logits = mlp.router(x_flat)
            weights = mx.softmax(logits, axis=-1)
            top_k_indices = mx.argsort(logits, axis=-1)[:, -top_k:]

            # Extract selected experts and their weights
            experts_weights = []
            for k in range(top_k):
                expert_idx = int(top_k_indices[0, k])
                weight = float(weights[0, expert_idx])
                experts_weights.append((expert_idx, weight))

            layer_experts[layer_idx] = sorted(experts_weights, key=lambda x: x[1], reverse=True)

        h = layer(h, mask=mask)
        mx.eval(h)

    return ExpertContributionTrace(
        prompt=prompt,
        target_token="",  # Will be filled by caller
        layer_experts=layer_experts,
        key_layers=[],  # Computed later
        key_experts=[],  # Computed later
    )


def analyze_fact_localization(
    model,
    tokenizer,
    category: str,
    config: dict,
    verbose: bool = False,
) -> list[FactLocalizationResult]:
    """
    Analyze where facts from a category are stored.
    """
    fact_type = config["type"]
    queries = config["queries"]
    results = []

    num_layers = len(get_model_layers(model))

    for query in queries:
        prompt = query["prompt"]
        answer = query["answer"]
        alternatives = query.get("alternatives", [])

        all_targets = [answer] + alternatives

        if verbose:
            print(f"  Analyzing: {prompt}")

        # Trace token probability through layers
        prob_trace = trace_token_probability(
            model, tokenizer, prompt, all_targets, verbose
        )

        if prob_trace is None:
            continue

        # Trace expert contributions
        expert_trace = trace_expert_contributions(
            model, tokenizer, prompt, verbose
        )
        expert_trace.target_token = answer

        # Get actual model prediction
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array([tokens])
        output = model(input_ids)
        if hasattr(output, 'logits'):
            output = output.logits

        last_logits = output[0, -1, :]
        pred_idx = int(mx.argmax(last_logits))
        actual_prediction = tokenizer.decode([pred_idx])

        # Check if prediction is correct
        prediction_correct = False
        pred_lower = actual_prediction.lower().strip()
        for target in all_targets:
            if target.lower().strip() in pred_lower or pred_lower in target.lower().strip():
                prediction_correct = True
                break

        # Determine knowledge layer range and primary storage
        emergence = prob_trace.emergence_layer
        dominant = prob_trace.dominant_layer

        # Classify by layer position
        early_cutoff = num_layers // 3
        late_cutoff = 2 * num_layers // 3

        if emergence < early_cutoff:
            primary_storage = "early"
        elif emergence < late_cutoff:
            primary_storage = "middle"
        else:
            primary_storage = "late"

        # Find key layers (where probability increases most)
        prob_increases = []
        for i in range(1, len(prob_trace.layer_probs)):
            increase = prob_trace.layer_probs[i] - prob_trace.layer_probs[i-1]
            if increase > 0.01:  # Significant increase
                prob_increases.append((i, increase))

        key_layers = [l for l, _ in sorted(prob_increases, key=lambda x: x[1], reverse=True)[:5]]
        expert_trace.key_layers = key_layers

        # Identify key experts at key layers
        key_experts = []
        for layer in key_layers:
            if layer in expert_trace.layer_experts:
                for expert_idx, weight in expert_trace.layer_experts[layer]:
                    key_experts.append((layer, expert_idx, weight))

        expert_trace.key_experts = sorted(key_experts, key=lambda x: x[2], reverse=True)[:10]

        # Estimate expert dependency
        # Higher if probability increases correlate with specific expert activations
        expert_dependency = 0.5  # Placeholder - could compute correlation

        results.append(FactLocalizationResult(
            category=category,
            fact_type=fact_type,
            prompt=prompt,
            expected_answer=answer,
            actual_prediction=actual_prediction.strip(),
            prediction_correct=prediction_correct,
            probability_trace=prob_trace,
            expert_trace=expert_trace,
            knowledge_layer_range=(emergence, dominant),
            primary_storage=primary_storage,
            expert_dependency=expert_dependency,
        ))

        gc.collect()

    return results


def aggregate_by_fact_type(results: list[FactLocalizationResult]) -> dict[str, FactTypeAnalysis]:
    """
    Aggregate results by fact type (static/dynamic/computed).
    """
    by_type: dict[str, list[FactLocalizationResult]] = defaultdict(list)
    for r in results:
        by_type[r.fact_type].append(r)

    analyses = {}
    for fact_type, type_results in by_type.items():
        num_queries = len(type_results)
        accuracy = sum(1 for r in type_results if r.prediction_correct) / num_queries

        avg_emergence = np.mean([r.probability_trace.emergence_layer for r in type_results])
        avg_dominant = np.mean([r.probability_trace.dominant_layer for r in type_results])

        # Layer distribution
        storage_counts = {"early": 0, "middle": 0, "late": 0}
        for r in type_results:
            storage_counts[r.primary_storage] += 1

        layer_dist = {k: v / num_queries for k, v in storage_counts.items()}

        # Common experts across facts of this type
        expert_counts: dict[tuple[int, int], int] = defaultdict(int)
        for r in type_results:
            for layer, expert, _ in r.expert_trace.key_experts:
                expert_counts[(layer, expert)] += 1

        common_experts = [
            (l, e, c) for (l, e), c in sorted(
                expert_counts.items(), key=lambda x: x[1], reverse=True
            )[:20]
        ]

        avg_expert_dep = np.mean([r.expert_dependency for r in type_results])

        analyses[fact_type] = FactTypeAnalysis(
            fact_type=fact_type,
            num_queries=num_queries,
            accuracy=accuracy,
            avg_emergence_layer=avg_emergence,
            avg_dominant_layer=avg_dominant,
            layer_distribution=layer_dist,
            expert_dependency=avg_expert_dep,
            common_experts=common_experts,
        )

    return analyses


def print_fact_localization_report(
    results: list[FactLocalizationResult],
    analyses: dict[str, FactTypeAnalysis],
    num_layers: int,
):
    """Print comprehensive fact localization report."""
    print()
    print("=" * 80)
    print("FACT LOCALIZATION ANALYSIS")
    print("=" * 80)
    print()
    print("Key Question: Where are facts stored in the model?")
    print("=" * 80)
    print()

    # Summary by fact type
    print("-" * 80)
    print("FACT TYPE COMPARISON")
    print("-" * 80)
    print()
    print(f"{'Type':<12} {'Count':<8} {'Accuracy':<10} {'Emergence':<12} {'Dominant':<12} {'Storage'}")
    print("-" * 80)

    for fact_type in ["static", "dynamic", "computed"]:
        if fact_type not in analyses:
            continue
        a = analyses[fact_type]
        storage = f"E:{a.layer_distribution.get('early', 0):.0%} M:{a.layer_distribution.get('middle', 0):.0%} L:{a.layer_distribution.get('late', 0):.0%}"
        print(f"{fact_type:<12} {a.num_queries:<8} {a.accuracy:>8.1%}  L{a.avg_emergence_layer:>5.1f}       L{a.avg_dominant_layer:>5.1f}       {storage}")

    print()
    print("Legend: E=Early, M=Middle, L=Late layers")
    print()

    # Layer distribution visualization
    print("-" * 80)
    print("LAYER-WISE KNOWLEDGE EMERGENCE")
    print("-" * 80)
    print()

    # Create histogram of emergence layers
    early_cutoff = num_layers // 3
    late_cutoff = 2 * num_layers // 3

    print(f"Layer ranges: Early (0-{early_cutoff}), Middle ({early_cutoff+1}-{late_cutoff}), Late ({late_cutoff+1}-{num_layers-1})")
    print()

    for fact_type in ["static", "dynamic", "computed"]:
        type_results = [r for r in results if r.fact_type == fact_type]
        if not type_results:
            continue

        emergence_counts = [0] * num_layers
        for r in type_results:
            if r.probability_trace.emergence_layer < num_layers:
                emergence_counts[r.probability_trace.emergence_layer] += 1

        # Simple ASCII histogram
        max_count = max(emergence_counts) if emergence_counts else 1
        print(f"\n{fact_type.upper()} facts emergence distribution:")

        # Group into 6 bins for readability
        bin_size = num_layers // 6
        for i in range(6):
            start = i * bin_size
            end = min((i + 1) * bin_size, num_layers)
            bin_count = sum(emergence_counts[start:end])
            bar = "█" * int(20 * bin_count / max(max_count * (end - start) / bin_size, 1))
            print(f"  L{start:2d}-L{end-1:2d}: {bar} ({bin_count})")

    print()

    # Per-category details
    print("-" * 80)
    print("CATEGORY ANALYSIS")
    print("-" * 80)

    for category in FACTUAL_QUERIES:
        cat_results = [r for r in results if r.category == category]
        if not cat_results:
            continue

        print(f"\n{category.upper()} ({cat_results[0].fact_type})")
        print("-" * 40)

        for r in cat_results[:3]:  # Show top 3 examples
            emergence = r.probability_trace.emergence_layer
            dominant = r.probability_trace.dominant_layer
            correct = "✓" if r.prediction_correct else "✗"

            print(f"  {correct} {r.prompt}")
            print(f"    Expected: {r.expected_answer}, Got: {r.actual_prediction}")
            print(f"    Emergence: L{emergence}, Dominant: L{dominant}, Storage: {r.primary_storage}")

            if r.expert_trace.key_experts:
                top_experts = r.expert_trace.key_experts[:3]
                exp_str = ", ".join(f"L{l}/E{e}({w:.2f})" for l, e, w in top_experts)
                print(f"    Key experts: {exp_str}")

    print()

    # Key insights
    print("=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print()

    static_analysis = analyses.get("static")
    dynamic_analysis = analyses.get("dynamic")
    computed_analysis = analyses.get("computed")

    if static_analysis:
        print(f"STATIC FACTS (capitals, science constants, historical dates):")
        print(f"  - Emerge around layer {static_analysis.avg_emergence_layer:.0f} (of {num_layers})")
        print(f"  - Primarily stored in: {max(static_analysis.layer_distribution, key=static_analysis.layer_distribution.get)} layers")
        print(f"  - Accuracy: {static_analysis.accuracy:.1%}")
        if static_analysis.common_experts:
            top_exp = static_analysis.common_experts[0]
            print(f"  - Most common expert: L{top_exp[0]}/E{top_exp[1]} (appears in {top_exp[2]} facts)")
        print()

    if dynamic_analysis:
        print(f"DYNAMIC FACTS (CEOs, current events):")
        print(f"  - Emerge around layer {dynamic_analysis.avg_emergence_layer:.0f} (of {num_layers})")
        print(f"  - Primarily stored in: {max(dynamic_analysis.layer_distribution, key=dynamic_analysis.layer_distribution.get)} layers")
        print(f"  - Accuracy: {dynamic_analysis.accuracy:.1%} (expected low - facts may be outdated)")
        print()

    if computed_analysis:
        print(f"COMPUTED FACTS (arithmetic, language rules):")
        print(f"  - Emerge around layer {computed_analysis.avg_emergence_layer:.0f} (of {num_layers})")
        print(f"  - Primarily stored in: {max(computed_analysis.layer_distribution, key=computed_analysis.layer_distribution.get)} layers")
        print(f"  - Accuracy: {computed_analysis.accuracy:.1%}")
        print()

    # Implications for virtual experts
    print("-" * 80)
    print("IMPLICATIONS FOR VIRTUAL EXPERTS")
    print("-" * 80)
    print()

    if static_analysis and static_analysis.avg_emergence_layer < num_layers // 3:
        print("FINDING: Static facts emerge EARLY (layers 0-12)")
        print("  → Facts may be stored in embeddings + early attention")
        print("  → Pruning middle/late experts may not affect static knowledge")
        print()

    if computed_analysis and computed_analysis.avg_emergence_layer > num_layers // 2:
        print("FINDING: Computed facts emerge LATE (after layer 12)")
        print("  → Computation happens in middle/late layers")
        print("  → These are good candidates for virtualization to tools")
        print()

    print("HYPOTHESIS TEST:")
    print("  If facts are in expert weights → pruning hurts fact retrieval")
    print("  If facts are in attention → pruning preserves fact retrieval")
    print("  → Run this analysis before/after pruning to confirm")
    print()


def save_results(results: list[FactLocalizationResult], analyses: dict[str, FactTypeAnalysis], output_path: Path):
    """Save results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "timestamp": datetime.now().isoformat(),
        "num_results": len(results),
        "analyses": {
            fact_type: {
                "num_queries": a.num_queries,
                "accuracy": a.accuracy,
                "avg_emergence_layer": float(a.avg_emergence_layer),
                "avg_dominant_layer": float(a.avg_dominant_layer),
                "layer_distribution": a.layer_distribution,
                "expert_dependency": float(a.expert_dependency),
                "common_experts": [{"layer": l, "expert": e, "count": c} for l, e, c in a.common_experts[:10]],
            }
            for fact_type, a in analyses.items()
        },
        "detailed_results": [
            {
                "category": r.category,
                "fact_type": r.fact_type,
                "prompt": r.prompt,
                "expected": r.expected_answer,
                "predicted": r.actual_prediction,
                "correct": r.prediction_correct,
                "emergence_layer": r.probability_trace.emergence_layer,
                "dominant_layer": r.probability_trace.dominant_layer,
                "final_prob": r.probability_trace.final_prob,
                "final_rank": r.probability_trace.final_rank,
                "primary_storage": r.primary_storage,
                "key_experts": [
                    {"layer": l, "expert": e, "weight": w}
                    for l, e, w in r.expert_trace.key_experts[:5]
                ],
            }
            for r in results
        ],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Fact Localization Probe")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-120b")
    parser.add_argument("--output", type=str, default="results/fact_localization.json")
    parser.add_argument("--detailed", action="store_true", help="Show detailed per-fact output")
    parser.add_argument("--categories", type=str, nargs="+",
                        help="Specific categories to analyze (default: all)")

    args = parser.parse_args()

    # Load model
    from mlx_lm import load
    print(f"Loading {args.model}...")
    model, tokenizer = load(args.model)

    num_layers = len(get_model_layers(model))
    print(f"Model has {num_layers} layers")
    print()

    # Select categories
    categories = args.categories or list(FACTUAL_QUERIES.keys())

    # Run analysis
    all_results = []
    for category in categories:
        if category not in FACTUAL_QUERIES:
            print(f"Unknown category: {category}")
            continue

        config = FACTUAL_QUERIES[category]
        print(f"Analyzing {category} ({config['type']} facts)...")

        results = analyze_fact_localization(
            model, tokenizer, category, config, verbose=args.detailed
        )
        all_results.extend(results)

        gc.collect()

    # Aggregate and report
    analyses = aggregate_by_fact_type(all_results)
    print_fact_localization_report(all_results, analyses, num_layers)

    # Save results
    save_results(all_results, analyses, Path(args.output))


if __name__ == "__main__":
    main()
