"""Handler for 'full-taxonomy' action - semantic trigram pattern analysis.

This implements the validated semantic trigram methodology for expert analysis.
This module is a thin CLI wrapper - token classification and test data
are centralized in the MoE introspection module.
"""

from __future__ import annotations

import asyncio
from argparse import Namespace
from collections import Counter, defaultdict

from ......introspection.moe import ExpertRouter
from ......introspection.moe.analysis_service import classify_token
from ......introspection.moe.test_data import TAXONOMY_TEST_PROMPTS
from .._types import FullTaxonomyConfig
from ..formatters import format_header

# =============================================================================
# PATTERN DEFINITIONS
# =============================================================================

PATTERN_CATEGORIES = {
    "arithmetic": ["NUM→OP", "OP→WS→NUM", "OP→NUM", "NUM→WS→NUM"],
    "code": ["^→KW", "KW→CW→BR", "KW→VAR", "BR→VAR→BR", "VAR→OP→VAR", "KW→BR", "CW→OP→CW"],
    "synonym": ["→SYN→", "ADJ→SYN", "NOUN→SYN"],
    "antonym": ["→ANT→", "ADJ→ANT", "NOUN→ANT"],
    "analogy": ["→AS→", "→TO→", "NOUN→AS", "FUNC→TO→NOUN"],
    "hypernym": ["NOUN→FUNC→NOUN", "FUNC→FUNC→NOUN", "→FUNC→NOUN"],
    "comparison": ["→THAN→", "ADJ→THAN", "COMP→THAN"],
    "causation": ["→CAUSE→", "PN→CAUSE", "VERB→CAUSE"],
    "conditional": ["→COND→", "^→COND", "COND→CW"],
    "question": ["^→QW", "QW→VERB", "QW→FUNC"],
    "negation": ["→NEG→", "VERB→NEG", "FUNC→NEG"],
    "temporal": ["^→TIME", "→TIME→", "VERB→TIME"],
    "quantification": ["^→QUANT", "QUANT→NOUN", "QUANT→FUNC"],
}


def handle_full_taxonomy(args: Namespace) -> None:
    """Handle the 'full-taxonomy' action - semantic trigram pattern analysis.

    Analyzes expert routing using semantic trigram patterns to reveal:
    - What token sequence patterns each expert specializes in
    - How specialization evolves across layers
    - Which categories (arithmetic, code, semantic relations) peak at which layers

    Args:
        args: Parsed CLI arguments. Required:
            - model: Model ID
        Optional:
            - categories: Comma-separated list of categories (default: all)
            - verbose: Show detailed per-pattern breakdown

    Example:
        lazarus introspect moe-expert full-taxonomy -m openai/gpt-oss-20b
        lazarus introspect moe-expert full-taxonomy -m openai/gpt-oss-20b --categories code,arithmetic
    """
    asyncio.run(_async_full_taxonomy(args))


async def _async_full_taxonomy(args: Namespace) -> None:
    """Async implementation of semantic trigram taxonomy analysis."""
    config = FullTaxonomyConfig.from_args(args)

    print(f"Loading model: {config.model}")
    async with await ExpertRouter.from_pretrained(config.model) as router:
        info = router.info
        print(f"Model: {info.num_experts} experts, {len(info.moe_layers)} MoE layers\n")

        # Select categories to analyze
        if config.categories:
            categories = [c.strip() for c in config.categories.split(",")]
        else:
            categories = list(TAXONOMY_TEST_PROMPTS.keys())

        # Collect all prompts using centralized test data
        all_prompts = []
        for cat in categories:
            if cat in TAXONOMY_TEST_PROMPTS:
                for prompt in TAXONOMY_TEST_PROMPTS[cat]:
                    all_prompts.append((cat, prompt))

        print(f"Analyzing {len(all_prompts)} prompts across {len(categories)} categories...\n")

        # Track trigrams
        expert_trigrams: dict[tuple[int, int], Counter] = defaultdict(Counter)
        trigram_examples: dict[tuple[int, int, str], list] = defaultdict(list)
        category_layer_experts: dict[str, dict[int, set]] = defaultdict(lambda: defaultdict(set))

        for cat, prompt in all_prompts:
            weights = await router.capture_router_weights(prompt)

            for layer_weights in weights:
                layer = layer_weights.layer_idx
                positions = layer_weights.positions
                tokens = [p.token for p in positions]

                # Classify tokens using centralized function
                sem_types = [classify_token(p.token).value for p in positions]

                for i, pos in enumerate(positions):
                    prev_t = sem_types[i - 1] if i > 0 else "^"
                    curr_t = sem_types[i]
                    next_t = sem_types[i + 1] if i < len(sem_types) - 1 else "$"
                    trigram = f"{prev_t}→{curr_t}→{next_t}"

                    # Build context
                    prev_tok = tokens[i - 1] if i > 0 else "^"
                    curr_tok = tokens[i]
                    next_tok = tokens[i + 1] if i < len(tokens) - 1 else "$"
                    context = f"{prev_tok}[{curr_tok}]{next_tok}"

                    for exp in pos.expert_indices:
                        key = (layer, exp)
                        expert_trigrams[key][trigram] += 1

                        ex_key = (layer, exp, trigram)
                        if len(trigram_examples[ex_key]) < 3:
                            trigram_examples[ex_key].append(context)

                        # Track which experts handle this category's patterns
                        if cat in PATTERN_CATEGORIES:
                            for pattern in PATTERN_CATEGORIES[cat]:
                                if pattern in trigram:
                                    category_layer_experts[cat][layer].add(exp)

        # =================================================================
        # OUTPUT RESULTS
        # =================================================================

        print(format_header("SEMANTIC TRIGRAM TAXONOMY ANALYSIS"))

        # Per-category pattern analysis
        for cat in categories:
            if cat not in PATTERN_CATEGORIES:
                continue

            print(f"\n{'=' * 60}")
            print(f"{cat.upper()}")
            print(f"{'=' * 60}")

            patterns = PATTERN_CATEGORIES[cat]

            for pattern in patterns:
                print(f"\n  Pattern: {pattern}")
                print(f"  {'-' * 50}")

                # Find top experts for this pattern
                pattern_experts = []
                for (layer, exp), counts in expert_trigrams.items():
                    for trigram, count in counts.items():
                        if pattern in trigram:
                            examples = trigram_examples[(layer, exp, trigram)]
                            pattern_experts.append(
                                {
                                    "layer": layer,
                                    "expert": exp,
                                    "trigram": trigram,
                                    "count": count,
                                    "examples": examples,
                                }
                            )

                pattern_experts.sort(key=lambda x: (-x["count"], x["layer"]))

                for pe in pattern_experts[:4]:
                    ex = pe["examples"][0] if pe["examples"] else ""
                    print(
                        f"    L{pe['layer']:02d} E{pe['expert']:02d}: "
                        f"{pe['trigram']:<24} (n={pe['count']:2d})  {ex}"
                    )

        # Layer evolution summary
        print("\n" + format_header("LAYER EVOLUTION BY CATEGORY"))
        layer_labels = " ".join(f"L{i:02d}" for i in range(0, 24, 4))
        print(f"\n{'Category':<16} | {layer_labels}")
        print("-" * 80)

        for cat in categories:
            if cat not in PATTERN_CATEGORIES:
                continue
            counts = []
            for layer in range(0, 24, 4):
                count = len(category_layer_experts[cat].get(layer, set()))
                counts.append(count)

            bars = " ".join(f"{c:3d}" for c in counts)
            print(f"{cat:<16} | {bars}")

        # Find peak layers for each category
        print("\n" + format_header("PEAK LAYERS BY CATEGORY"))

        for cat in categories:
            if cat not in PATTERN_CATEGORIES:
                continue

            layer_counts = [
                (layer, len(experts)) for layer, experts in category_layer_experts[cat].items()
            ]
            if layer_counts:
                layer_counts.sort(key=lambda x: -x[1])
                peak_layers = layer_counts[:3]
                peak_str = ", ".join(f"L{layer}({cnt})" for layer, cnt in peak_layers)
                print(f"  {cat:<16}: {peak_str}")

        # Expert specialization summary (verbose mode)
        if config.verbose:
            print("\n" + format_header("TOP EXPERT SPECIALIZATIONS"))

            # Aggregate trigrams across all layers for top experts
            expert_total: Counter[tuple[int, int]] = Counter()
            for key, counts in expert_trigrams.items():
                expert_total[key] = sum(counts.values())

            print(f"\n{'Expert':<10} {'Activations':<12} {'Top Pattern':<24} {'Category'}")
            print("-" * 70)

            for (layer, exp), total in expert_total.most_common(20):
                top_trigram = expert_trigrams[(layer, exp)].most_common(1)
                if top_trigram:
                    pattern, count = top_trigram[0]
                    # Find which category this pattern belongs to
                    cat_match = "general"
                    for cat, patterns in PATTERN_CATEGORIES.items():
                        if any(p in pattern for p in patterns):
                            cat_match = cat
                            break
                    print(f"L{layer:02d} E{exp:02d}   {total:<12} {pattern:<24} {cat_match}")

        print("\n" + "=" * 80)
