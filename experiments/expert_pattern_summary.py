"""Experiment: Summarize expert pattern specializations across all layers.

Focus on finding the most interesting/specialized experts.
"""

import asyncio
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chuk_lazarus.introspection.moe import ExpertRouter, get_prompts_flat


def classify_token(token: str) -> str:
    """Token classifier with meaningful short codes."""
    clean = token.strip()
    lower = clean.lower()

    if not clean:
        return "WS"

    if re.match(r"^-?\d+\.?\d*$", clean):
        return "NUM"

    code_keywords = {
        "def", "class", "import", "return", "if", "else", "for", "while",
        "function", "const", "let", "var", "async", "await",
        "SELECT", "FROM", "WHERE", "INSERT", "CREATE",
        "fn", "mut", "impl", "struct", "enum",
    }
    if clean in code_keywords or lower in code_keywords:
        return "KW"

    if clean in "()[]{}":
        return "BR"

    if clean in "+-*/=<>!&|^~" or clean in ["==", "!=", "<=", ">=", "+=", "-=", "->", "=>"]:
        return "OP"

    if re.match(r"^[^\w\s]+$", clean):
        return "PN"

    func_words = {"the", "a", "an", "in", "on", "at", "to", "for", "with", "by", "of",
                  "and", "or", "but", "is", "are", "was", "were", "be", "been",
                  "i", "you", "he", "she", "it", "we", "they", "this", "that"}
    if lower in func_words:
        return "FW"

    if clean and clean[0].isupper():
        return "CAP"

    if len(clean) == 1 and clean.isalpha():
        return "VAR"

    return "CW"


async def find_specialists(model_id: str, num_prompts: int = 100):
    """Find the most specialized experts across all layers."""

    print(f"Loading model: {model_id}")
    async with await ExpertRouter.from_pretrained(model_id) as router:
        info = router.info
        print(f"Model: {info.num_experts} experts, {len(info.moe_layers)} MoE layers")

        all_prompts = get_prompts_flat()[:num_prompts]
        print(f"Analyzing {len(all_prompts)} prompts...\n")

        # Track patterns per expert
        expert_trigrams: dict[tuple[int, int], Counter] = defaultdict(Counter)
        expert_examples: dict[tuple[int, int], list] = defaultdict(list)

        for cat, prompt in all_prompts:
            weights = await router.capture_router_weights(prompt)

            for layer_weights in weights:
                layer_idx = layer_weights.layer_idx
                positions = layer_weights.positions
                types = [classify_token(p.token) for p in positions]

                for i, pos in enumerate(positions):
                    prev_t = types[i-1] if i > 0 else "^"
                    curr_t = types[i]
                    next_t = types[i+1] if i < len(types)-1 else "$"
                    trigram = f"{prev_t}→{curr_t}→{next_t}"

                    for exp in pos.expert_indices:
                        key = (layer_idx, exp)
                        expert_trigrams[key][trigram] += 1
                        if len(expert_examples[key]) < 8:
                            expert_examples[key].append((trigram, pos.token, prompt[:30]))

        # Find specialists
        all_specialists = []
        for (layer, exp), counts in expert_trigrams.items():
            total = sum(counts.values())
            if total < 5:  # Skip low-activity experts
                continue

            top_pattern, top_count = counts.most_common(1)[0]
            concentration = top_count / total

            # Get top 3 patterns
            top_3 = counts.most_common(3)

            # Calculate entropy (lower = more specialized)
            probs = [c/total for _, c in counts.items()]
            entropy = -sum(p * (p and (p > 0 and __import__('math').log2(p) or 0) or 0) for p in probs)

            all_specialists.append({
                "layer": layer,
                "expert": exp,
                "top_pattern": top_pattern,
                "concentration": concentration,
                "top_3": top_3,
                "entropy": entropy,
                "total": total,
                "examples": expert_examples[(layer, exp)][:5],
            })

        # Sort by concentration (most specialized first)
        all_specialists.sort(key=lambda x: (-x["concentration"], x["layer"]))

        # Print results
        print("="*80)
        print("TOP PATTERN SPECIALISTS (sorted by specialization)")
        print("="*80)

        # Group interesting patterns
        pattern_groups = {
            "Arithmetic": ["NUM→OP→", "OP→NUM", "→NUM→OP", "OP→WS→NUM", "NUM→OP→WS"],
            "Code Structure": ["KW→", "→BR→", "BR→", "→KW→"],
            "Sequence Position": ["^→", "→$"],
            "Punctuation": ["→PN→", "PN→"],
        }

        for group_name, patterns in pattern_groups.items():
            group_experts = [
                s for s in all_specialists
                if any(p in s["top_pattern"] for p in patterns) and s["concentration"] > 0.15
            ]
            if group_experts:
                print(f"\n{group_name} Specialists:")
                print("-" * 40)
                for s in group_experts[:10]:
                    examples = [e[1] for e in s["examples"][:3]]
                    print(f"  L{s['layer']:02d} E{s['expert']:02d}: {s['top_pattern']:<20} "
                          f"({s['concentration']:.0%}, n={s['total']}) {examples}")

        # Show layer evolution for key patterns
        print("\n" + "="*80)
        print("PATTERN FREQUENCY BY LAYER")
        print("="*80)

        interesting_patterns = ["NUM→OP→WS", "^→KW→CW", "KW→CW→BR", "CW→PN→$"]
        for pattern in interesting_patterns:
            print(f"\n{pattern}:")
            layer_freq = defaultdict(int)
            for s in all_specialists:
                if pattern in [p[0] for p in s["top_3"]]:
                    layer_freq[s["layer"]] += 1

            for layer in range(24):
                count = layer_freq[layer]
                bar = "█" * count
                print(f"  L{layer:02d}: {bar} ({count})")


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "openai/gpt-oss-20b"
    num = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    asyncio.run(find_specialists(model, num))
