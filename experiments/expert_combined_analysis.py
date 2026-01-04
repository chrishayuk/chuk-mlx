"""Combined analysis: Sequence patterns + Semantic content.

The hypothesis: Experts specialize on COMBINATIONS like:
- "ADJ same ADJ" (synonym pattern)
- "ADJ opposite ADJ" (antonym pattern)
- "NOUN is NOUN" (hypernym pattern)
- "NUM OP NUM" (arithmetic pattern)
"""

import asyncio
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chuk_lazarus.introspection.moe import ExpertRouter


def get_semantic_type(token: str) -> str:
    """Classify token by semantic type."""
    # Handle leading/trailing whitespace - strip for matching
    clean = token.strip().lower()

    # Pure whitespace token
    if not clean:
        return "WS"

    # Numbers
    if re.match(r"^-?\d+\.?\d*$", clean):
        return "NUM"

    # Operators (check both with and without the symbol)
    if clean in ["+", "-", "*", "/", "=", "<", ">", "==", "!=", "+=", "-="]:
        return "OP"

    # Relationship markers
    if clean in ["same", "similar", "like", "equals", "means"]:
        return "SYN"
    if clean in ["opposite", "contrary", "versus", "against", "opposed"]:
        return "ANT"

    # Analogy markers
    if clean == "as":
        return "AS"
    if clean == "to":
        return "TO"

    # Articles/function words
    if clean in ["the", "a", "an", "of", "is", "are", "was", "were"]:
        return "FUNC"

    # Common adjectives (for testing synonym/antonym content)
    adjectives = {"happy", "sad", "hot", "cold", "big", "small", "fast", "slow",
                  "good", "bad", "old", "new", "light", "dark", "high", "low",
                  "warm", "cool", "large", "quick", "great", "young", "rich", "poor"}
    if clean in adjectives:
        return "ADJ"

    # Common nouns
    nouns = {"dog", "cat", "car", "tree", "book", "house", "person", "animal",
             "king", "queen", "man", "woman", "doctor", "teacher", "student",
             "patient", "vehicle", "pet", "object", "puppy", "kitten", "bird",
             "fish", "song", "pen", "brush", "eye", "ear", "sun", "moon",
             "hand", "foot", "glove", "shoe", "day", "night"}
    if clean in nouns:
        return "NOUN"

    # Verbs related to relationships
    if clean in ["read", "listen", "fly", "swim", "write", "paint", "see", "hear"]:
        return "VERB"

    return "OTHER"


# Test prompts with clear patterns
PATTERN_PROMPTS = [
    # Synonym: ADJ SYN_MARKER ADJ
    "Happy means happy.",
    "Hot is similar to warm.",
    "Big equals large.",
    "Fast is like quick.",
    "Good means great.",

    # Antonym: ADJ ANT_MARKER ADJ
    "Hot is opposite of cold.",
    "Big versus small.",
    "Happy against sad.",
    "Fast contrary to slow.",
    "Good opposite bad.",

    # Hypernym: NOUN FUNC NOUN
    "Dog is an animal.",
    "Cat is a pet.",
    "Car is a vehicle.",
    "King is a person.",
    "Book is an object.",

    # Arithmetic: NUM OP NUM
    "5 + 3 = 8",
    "10 - 4 = 6",
    "7 * 2 = 14",
    "20 / 4 = 5",
    "100 + 50 = 150",

    # Analogy: NOUN ANALOGY NOUN ANALOGY NOUN ANALOGY NOUN
    "King is to queen as man is to woman.",
    "Dog is to cat as big is to small.",
    "Hot is to cold as up is to down.",
    "Teacher is to student as doctor is to patient.",
    "Book is to read as song is to hear.",
]


async def analyze_combined_patterns(model_id: str):
    """Analyze experts on combined semantic+syntactic patterns."""

    print(f"Loading model: {model_id}")
    async with await ExpertRouter.from_pretrained(model_id) as router:
        info = router.info
        print(f"Model: {info.num_experts} experts\n")

        # Track trigrams with semantic types
        expert_semantic_trigrams: dict[tuple[int, int], Counter] = defaultdict(Counter)
        # Store examples per (layer, expert, trigram) so we show correct examples
        trigram_examples: dict[tuple[int, int, str], list] = defaultdict(list)

        for prompt in PATTERN_PROMPTS:
            weights = await router.capture_router_weights(prompt)

            for layer_weights in weights:
                layer = layer_weights.layer_idx
                positions = layer_weights.positions

                # Get semantic types and tokens for context
                sem_types = [get_semantic_type(p.token) for p in positions]
                tokens = [p.token for p in positions]

                for i, pos in enumerate(positions):
                    prev_t = sem_types[i-1] if i > 0 else "^"
                    curr_t = sem_types[i]
                    next_t = sem_types[i+1] if i < len(sem_types)-1 else "$"
                    trigram = f"{prev_t}→{curr_t}→{next_t}"

                    # Build context string: prev_token [CURRENT] next_token
                    prev_tok = tokens[i-1] if i > 0 else "^"
                    curr_tok = tokens[i]
                    next_tok = tokens[i+1] if i < len(tokens)-1 else "$"
                    context = f"{prev_tok}[{curr_tok}]{next_tok}"

                    for exp in pos.expert_indices:
                        key = (layer, exp)
                        expert_semantic_trigrams[key][trigram] += 1
                        # Store example with trigram as key
                        ex_key = (layer, exp, trigram)
                        if len(trigram_examples[ex_key]) < 3:
                            trigram_examples[ex_key].append(context)

        # Find interesting patterns
        print("="*80)
        print("SEMANTIC TRIGRAM SPECIALISTS")
        print("="*80)

        # Patterns of interest (accounting for WS tokens between)
        interesting_patterns = {
            "Synonym (ADJ→SYN)": "ADJ→SYN",
            "Antonym (ADJ→ANT)": "ADJ→ANT",
            "Arithmetic (NUM→OP)": "NUM→OP",
            "Arithmetic (OP→WS→NUM)": "OP→WS→NUM",
            "Analogy (→AS→)": "→AS→",
            "Analogy (→TO→)": "→TO→",
            "Hypernym (NOUN→FUNC)": "NOUN→FUNC",
            "Hypernym (FUNC→NOUN)": "FUNC→NOUN",
        }

        for pattern_name, pattern_match in interesting_patterns.items():
            print(f"\n{pattern_name}:")
            print("-" * 70)

            # Find experts with this pattern across layers
            pattern_experts = []
            for (layer, exp), counts in expert_semantic_trigrams.items():
                for trigram, count in counts.items():
                    if pattern_match in trigram:
                        # Get examples for this specific trigram
                        examples = trigram_examples[(layer, exp, trigram)]
                        pattern_experts.append({
                            "layer": layer,
                            "expert": exp,
                            "trigram": trigram,
                            "count": count,
                            "examples": examples,
                        })

            # Sort by count
            pattern_experts.sort(key=lambda x: (-x["count"], x["layer"]))

            # Show top experts for this pattern
            for pe in pattern_experts[:6]:
                ex_str = " | ".join(pe["examples"][:2]) if pe["examples"] else ""
                print(f"  L{pe['layer']:02d} E{pe['expert']:02d}: {pe['trigram']:<22} "
                      f"(n={pe['count']:2d})  {ex_str}")

        # Layer-by-layer evolution
        print("\n" + "="*80)
        print("PATTERN EVOLUTION BY LAYER")
        print("="*80)

        for pattern_name, pattern_match in interesting_patterns.items():
            print(f"\n{pattern_name}:")
            for layer in range(0, 24, 4):  # Every 4th layer
                layer_experts = [
                    exp for (l, exp), counts in expert_semantic_trigrams.items()
                    if l == layer and any(pattern_match in t for t in counts)
                ]
                print(f"  L{layer:02d}: {len(layer_experts)} experts")


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "openai/gpt-oss-20b"
    asyncio.run(analyze_combined_patterns(model))
