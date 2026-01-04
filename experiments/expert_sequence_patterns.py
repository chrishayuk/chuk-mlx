"""Experiment: Do MoE experts specialize on token sequence patterns?

Hypothesis: Experts specialize on patterns like:
- number → operator → number (math expressions)
- keyword → identifier → bracket (function definitions)
- article → noun → verb (sentence structure)

And these patterns should vary by layer (syntax in early, semantics in late).
"""

import asyncio
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chuk_lazarus.introspection.moe import ExpertRouter, get_prompts_flat


def classify_token(token: str) -> str:
    """Simple token classifier."""
    clean = token.strip()
    lower = clean.lower()

    if not clean:
        return "WS"  # whitespace

    # Numbers
    if re.match(r"^-?\d+\.?\d*$", clean):
        return "NUM"

    # Code keywords (simplified)
    code_keywords = {
        "def", "class", "import", "return", "if", "else", "for", "while",
        "function", "const", "let", "var", "async", "await",
        "SELECT", "FROM", "WHERE", "INSERT", "CREATE",
        "fn", "mut", "impl", "struct", "enum",
    }
    if clean in code_keywords or lower in code_keywords:
        return "KW"  # keyword

    # Brackets
    if clean in "()[]{}":
        return "BR"  # bracket

    # Operators
    if clean in "+-*/=<>!&|^~" or clean in ["==", "!=", "<=", ">=", "+=", "-=", "->", "=>"]:
        return "OP"  # operator

    # Punctuation
    if re.match(r"^[^\w\s]+$", clean):
        return "PN"  # punctuation

    # Function words
    func_words = {"the", "a", "an", "in", "on", "at", "to", "for", "with", "by", "of",
                  "and", "or", "but", "is", "are", "was", "were", "be", "been",
                  "i", "you", "he", "she", "it", "we", "they", "this", "that"}
    if lower in func_words:
        return "FW"  # function word

    # Capitalized
    if clean and clean[0].isupper():
        return "CAP"

    # Single letter
    if len(clean) == 1 and clean.isalpha():
        return "VAR"

    # Default content word
    return "CW"  # content word


async def analyze_expert_patterns(model_id: str, num_prompts: int = 30):
    """Analyze what sequence patterns each expert specializes on."""

    print(f"Loading model: {model_id}")
    async with await ExpertRouter.from_pretrained(model_id) as router:
        info = router.info
        print(f"Model: {info.num_experts} experts, {len(info.moe_layers)} MoE layers")

        # Get prompts
        all_prompts = get_prompts_flat()[:num_prompts]
        print(f"Using {len(all_prompts)} prompts")

        # Track trigrams per expert per layer
        # Key: (layer, expert) -> Counter of trigram patterns
        expert_trigrams: dict[tuple[int, int], Counter] = defaultdict(Counter)
        expert_examples: dict[tuple[int, int], list] = defaultdict(list)

        for cat, prompt in all_prompts:
            weights = await router.capture_router_weights(prompt)

            for layer_weights in weights:
                layer_idx = layer_weights.layer_idx
                positions = layer_weights.positions

                # Classify all tokens
                types = [classify_token(p.token) for p in positions]

                for i, pos in enumerate(positions):
                    # Build trigram
                    prev_t = types[i-1] if i > 0 else "^"  # start marker
                    curr_t = types[i]
                    next_t = types[i+1] if i < len(types)-1 else "$"  # end marker
                    trigram = f"{prev_t}_{curr_t}_{next_t}"

                    for exp in pos.expert_indices:
                        key = (layer_idx, exp)
                        expert_trigrams[key][trigram] += 1
                        if len(expert_examples[key]) < 5:
                            expert_examples[key].append((trigram, pos.token))

        # Analyze results
        print("\n" + "="*70)
        print("EXPERT SEQUENCE PATTERN ANALYSIS")
        print("="*70)

        # Group by layer
        by_layer: dict[int, list] = defaultdict(list)
        for (layer, exp), trigram_counts in expert_trigrams.items():
            total = sum(trigram_counts.values())
            top_trigrams = trigram_counts.most_common(3)
            top_trigram = top_trigrams[0] if top_trigrams else ("none", 0)
            concentration = top_trigram[1] / total if total > 0 else 0

            by_layer[layer].append({
                "expert": exp,
                "top_pattern": top_trigram[0],
                "concentration": concentration,
                "top_3": top_trigrams,
                "examples": expert_examples[(layer, exp)],
                "total": total,
            })

        # Show per-layer analysis
        for layer in sorted(by_layer.keys()):
            experts = by_layer[layer]

            # Find specialists (high concentration on specific patterns)
            specialists = [e for e in experts if e["concentration"] > 0.15]
            specialists.sort(key=lambda x: -x["concentration"])

            # Count pattern types across layer
            layer_patterns = Counter()
            for e in experts:
                layer_patterns[e["top_pattern"]] += 1

            print(f"\n--- Layer {layer} ---")
            print(f"Top patterns in layer: {layer_patterns.most_common(5)}")

            if specialists[:5]:
                print("Specialists (>15% concentration on one pattern):")
                for e in specialists[:5]:
                    pattern = e["top_pattern"]
                    conc = e["concentration"]
                    examples = [ex[1] for ex in e["examples"][:3]]
                    print(f"  E{e['expert']:02d}: {pattern} ({conc:.0%}) - {examples}")

        # Cross-layer pattern analysis
        print("\n" + "="*70)
        print("PATTERN EVOLUTION ACROSS LAYERS")
        print("="*70)

        # Track how common patterns change
        patterns_to_track = ["^_CW_CW", "CW_CW_CW", "NUM_OP_NUM", "KW_CW_BR", "FW_CW_PN"]

        for pattern in patterns_to_track:
            print(f"\n{pattern}:")
            layer_counts = []
            for layer in sorted(by_layer.keys()):
                count = sum(1 for e in by_layer[layer] if e["top_pattern"] == pattern)
                layer_counts.append(count)

            # Simple sparkline
            max_count = max(layer_counts) if layer_counts else 1
            for i, (layer, count) in enumerate(zip(sorted(by_layer.keys()), layer_counts)):
                bar = "█" * int(count / max_count * 20) if max_count > 0 else ""
                print(f"  L{layer:02d}: {bar} ({count})")


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "openai/gpt-oss-20b"
    num = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    asyncio.run(analyze_expert_patterns(model, num))
