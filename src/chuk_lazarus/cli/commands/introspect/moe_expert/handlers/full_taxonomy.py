"""Handler for 'full-taxonomy' action - semantic trigram pattern analysis.

This implements the validated semantic trigram methodology for expert analysis:
- Classifies tokens into 20+ semantic types (NUM, OP, KW, NOUN, ADJ, VERB, etc.)
- Analyzes trigram patterns (PREV→CURR→NEXT) for expert specialization
- Groups patterns into categories (arithmetic, code, semantic relationships, etc.)
- Shows layer evolution for each category
"""

from __future__ import annotations

import asyncio
import re
from argparse import Namespace
from collections import Counter, defaultdict

from ......introspection.moe import ExpertRouter
from ..formatters import format_header

# =============================================================================
# TOKEN TYPE CLASSIFICATION
# =============================================================================

# Keywords by category
CODE_KEYWORDS = {
    "def",
    "class",
    "if",
    "for",
    "while",
    "return",
    "import",
    "from",
    "function",
    "const",
    "let",
    "var",
    "async",
    "await",
    "try",
    "except",
    "yield",
    "lambda",
    "with",
    "else",
    "elif",
    "switch",
    "case",
    "break",
}

BOOL_LITERALS = {"true", "false", "True", "False", "null", "None", "nil"}

TYPE_KEYWORDS = {"int", "str", "float", "bool", "list", "dict", "string", "number"}

# Parts of speech
NOUNS = {
    "cat",
    "dog",
    "man",
    "woman",
    "king",
    "queen",
    "doctor",
    "student",
    "teacher",
    "patient",
    "car",
    "house",
    "tree",
    "animal",
    "pet",
    "bird",
    "fish",
    "book",
    "song",
    "pen",
    "brush",
    "eye",
    "ear",
    "sun",
    "moon",
    "hand",
    "foot",
    "glove",
    "shoe",
    "day",
    "night",
    "puppy",
    "kitten",
    "vehicle",
    "city",
    "country",
    "river",
    "mountain",
    "ocean",
    "server",
    "computer",
    "person",
    "child",
    "parent",
    "friend",
    "enemy",
    "world",
}

ADJECTIVES = {
    "big",
    "small",
    "happy",
    "sad",
    "fast",
    "slow",
    "hot",
    "cold",
    "good",
    "bad",
    "old",
    "new",
    "light",
    "dark",
    "high",
    "low",
    "warm",
    "cool",
    "large",
    "quick",
    "great",
    "young",
    "rich",
    "poor",
    "tall",
    "short",
    "long",
    "deep",
    "wide",
    "narrow",
    "thick",
    "thin",
    "heavy",
    "soft",
    "hard",
    "easy",
    "difficult",
    "simple",
    "complex",
}

VERBS = {
    "run",
    "walk",
    "think",
    "eat",
    "make",
    "go",
    "come",
    "see",
    "know",
    "read",
    "write",
    "listen",
    "fly",
    "swim",
    "paint",
    "hear",
    "speak",
    "work",
    "play",
    "sleep",
    "wake",
    "start",
    "stop",
    "open",
    "close",
    "give",
    "take",
    "find",
    "lose",
    "win",
    "fail",
    "pass",
    "grow",
}

FUNCTION_WORDS = {
    "the",
    "a",
    "an",
    "in",
    "on",
    "at",
    "to",
    "for",
    "with",
    "by",
    "of",
    "and",
    "or",
    "but",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "has",
    "have",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "must",
    "can",
    "shall",
    "this",
    "that",
}

# Relationship markers
SYNONYM_MARKERS = {"means", "equals", "same", "like", "similar", "equivalent"}
ANTONYM_MARKERS = {"versus", "against", "opposite", "unlike", "different", "contrary"}
CAUSE_MARKERS = {"because", "since", "therefore", "so", "thus", "hence", "consequently"}
CONDITION_MARKERS = {"if", "unless", "when", "while", "although", "though", "whether"}

# Question words
QUESTION_WORDS = {"what", "who", "where", "when", "why", "how", "which", "whose"}
ANSWER_WORDS = {"yes", "no", "maybe", "probably", "definitely", "certainly"}

# Negation
NEGATION_WORDS = {"not", "never", "none", "nothing", "neither", "nor", "nobody"}

# Temporal
TEMPORAL_WORDS = {"now", "then", "before", "after", "yesterday", "today", "tomorrow", "always"}

# Quantifiers
QUANTIFIERS = {"all", "some", "many", "few", "most", "every", "each", "any", "none", "both"}

# Comparatives
COMPARISON_WORDS = {"than", "more", "less", "most", "least", "better", "worse", "bigger", "smaller"}

# Coordination
COORD_WORDS = {"and", "or", "but", "nor", "yet", "so"}


def classify_token(token: str) -> str:
    """Classify a token into semantic type."""
    clean = token.strip().lower()

    # Empty/whitespace
    if not clean:
        return "WS"

    # Numbers (including decimals and negatives)
    if re.match(r"^-?\d+\.?\d*$", clean):
        return "NUM"

    # Operators
    if clean in [
        "+",
        "-",
        "*",
        "/",
        "=",
        "<",
        ">",
        "==",
        "!=",
        "<=",
        ">=",
        "+=",
        "-=",
        "*=",
        "/=",
        "&&",
        "||",
        "!",
        "&",
        "|",
        "^",
        "%",
    ]:
        return "OP"

    # Brackets
    if clean in "()[]{}":
        return "BR"

    # Punctuation
    if clean in ".,;:!?":
        return "PN"

    # Quotes
    if clean in ["'", '"', "`", "''", '""']:
        return "QUOTE"

    # Code keywords
    if clean in CODE_KEYWORDS:
        return "KW"

    # Boolean/null literals
    if clean in BOOL_LITERALS:
        return "BOOL"

    # Type keywords
    if clean in TYPE_KEYWORDS:
        return "TYPE"

    # Relationship markers (check before function words)
    if clean in SYNONYM_MARKERS:
        return "SYN"
    if clean in ANTONYM_MARKERS:
        return "ANT"
    if clean == "as":
        return "AS"
    if clean == "to" and token.strip() == "to":  # standalone "to"
        return "TO"
    if clean in CAUSE_MARKERS:
        return "CAUSE"
    if clean in CONDITION_MARKERS:
        return "COND"
    if clean == "than":
        return "THAN"

    # Question/answer
    if clean in QUESTION_WORDS:
        return "QW"
    if clean in ANSWER_WORDS:
        return "ANS"

    # Negation
    if clean in NEGATION_WORDS:
        return "NEG"

    # Temporal
    if clean in TEMPORAL_WORDS:
        return "TIME"

    # Quantifiers
    if clean in QUANTIFIERS:
        return "QUANT"

    # Comparatives
    if clean in COMPARISON_WORDS:
        return "COMP"

    # Coordination
    if clean in COORD_WORDS:
        return "COORD"

    # Parts of speech
    if clean in NOUNS:
        return "NOUN"
    if clean in ADJECTIVES:
        return "ADJ"
    if clean in VERBS:
        return "VERB"
    if clean in FUNCTION_WORDS:
        return "FUNC"

    # Capitalized (proper noun or sentence start)
    original = token.strip()
    if original and original[0].isupper() and len(original) > 1:
        return "CAP"

    # Single letter variable
    if len(clean) == 1 and clean.isalpha():
        return "VAR"

    # Default content word
    return "CW"


# =============================================================================
# TEST PROMPTS BY CATEGORY
# =============================================================================

TEST_PROMPTS = {
    "arithmetic": [
        "2 + 3 = 5",
        "127 * 89 = 11303",
        "45 - 12 = 33",
        "100 / 4 = 25",
        "5 + 3 = 8",
        "10 - 4 = 6",
        "7 * 2 = 14",
        "20 / 4 = 5",
    ],
    "code": [
        "def hello(name):",
        "for i in range(10):",
        "if x > 0: return x",
        "async def fetch():",
        "class MyClass:",
        "while True: break",
        "try: x = 1",
        "lambda x: x * 2",
        "import os",
        "from sys import path",
    ],
    "synonym": [
        "Happy means joyful.",
        "Big equals large.",
        "Fast is similar to quick.",
        "Good means great.",
        "Old equals ancient.",
    ],
    "antonym": [
        "Big versus small.",
        "Hot against cold.",
        "Light opposite dark.",
        "Good versus bad.",
        "Fast against slow.",
    ],
    "analogy": [
        "King is to queen as man is to woman.",
        "Doctor is to patient as teacher is to student.",
        "Cat is to kitten as dog is to puppy.",
        "Hot is to cold as big is to small.",
        "Book is to read as song is to listen.",
    ],
    "hypernym": [
        "Dog is an animal.",
        "Cat is a pet.",
        "Oak is a tree.",
        "Car is a vehicle.",
        "Paris is a city.",
    ],
    "comparison": [
        "Dogs are bigger than cats.",
        "Cheetahs run faster than lions.",
        "5 is greater than 3.",
        "This is better than that.",
        "More people than expected.",
    ],
    "causation": [
        "It failed because the server crashed.",
        "She won since she practiced daily.",
        "Therefore the result is zero.",
        "The test passed because it works.",
        "Thus we conclude success.",
    ],
    "conditional": [
        "If it rains then stay inside.",
        "Unless you study you will fail.",
        "When ready please start.",
        "While running check status.",
        "Although difficult it worked.",
    ],
    "question": [
        "What is the capital of France?",
        "Who wrote this book?",
        "How does it work?",
        "Where is the cat?",
        "Why did it fail?",
    ],
    "negation": [
        "The dog is not big.",
        "I have never seen that.",
        "There is nothing here.",
        "Nobody knows the answer.",
        "Neither option works.",
    ],
    "temporal": [
        "Yesterday I went home.",
        "Now the cat sleeps.",
        "Before that it was sunny.",
        "Tomorrow we will start.",
        "Then it happened suddenly.",
    ],
    "quantification": [
        "All dogs are mammals.",
        "Some cats like water.",
        "Every student passed.",
        "Many people attended.",
        "Few options remain.",
    ],
    "context_switch": [
        "The result is 42.",
        "Calculate x + y first.",
        "Answer: 127 * 89 = 11303",
        "Set x = 5 and y = 3.",
        "The dog weighs 15 pounds.",
        "See page 42 for details.",
        "He scored 98 on the test.",
        "Mix 2 cups with 3 eggs.",
    ],
    "position": [
        "The quick brown fox.",
        "Hello world today.",
        "First item here.",
        "Last thing done.",
        "Start here now.",
        "End of the story.",
        "Beginning of time.",
        "Finally we finished.",
    ],
    "coordination": [
        "Dogs and cats play.",
        "Hot or cold drinks.",
        "Big but gentle.",
        "Run and jump fast.",
        "Happy and sad times.",
        "Read or write code.",
        "Young and old people.",
        "Fast but careful.",
    ],
}


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
    "position": ["^→CW", "^→NOUN", "^→NUM", "CW→PN→$", "NUM→PN→$", "^→CAP", "^→FUNC"],
    "context_switch": ["CW→WS→NUM", "NUM→WS→CW", "FUNC→NUM", "NUM→FUNC", "PN→WS→NUM", "CW→PN→NUM"],
    "coordination": [
        "NOUN→COORD→NOUN",
        "ADJ→COORD→ADJ",
        "VERB→COORD→VERB",
        "CW→COORD→CW",
        "→COORD→",
    ],
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
    model_id = args.model
    verbose = getattr(args, "verbose", False)
    categories_arg = getattr(args, "categories", None)

    print(f"Loading model: {model_id}")
    async with await ExpertRouter.from_pretrained(model_id) as router:
        info = router.info
        print(f"Model: {info.num_experts} experts, {len(info.moe_layers)} MoE layers\n")

        # Select categories to analyze
        if categories_arg:
            categories = [c.strip() for c in categories_arg.split(",")]
        else:
            categories = list(TEST_PROMPTS.keys())

        # Collect all prompts
        all_prompts = []
        for cat in categories:
            if cat in TEST_PROMPTS:
                for prompt in TEST_PROMPTS[cat]:
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

                # Classify all tokens
                sem_types = [classify_token(p.token) for p in positions]
                tokens = [p.token for p in positions]

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

        # Expert specialization summary
        if verbose:
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
