"""Test data for MoE expert analysis.

This module centralizes all test prompts, contexts, and domain data
used across MoE expert CLI handlers.
"""

from __future__ import annotations

from ...cli.commands._constants import Domain, PatternCategory


# =============================================================================
# Token Routing Test Contexts
# =============================================================================

TOKEN_CONTEXTS: dict[str, tuple[str, str]] = {
    # Token: (context_string, description)
    "127": ("The number is 127", "After article and number word"),
    "is": ("What is the answer?", "Question word context"),
    "function": ("def function(x):", "Code definition context"),
    "King": ("King is to queen", "Analogy start"),
    "=": ("2 + 2 = 4", "Arithmetic equality"),
    "+": ("2 + 3", "Addition operator"),
    "*": ("5 * 6", "Multiplication operator"),
    "if": ("if x > 0:", "Conditional code"),
    "the": ("The quick brown fox", "Article start"),
    "dog": ("The dog runs fast", "After article noun"),
}


# =============================================================================
# Context Window Test Data
# =============================================================================

CONTEXT_WINDOW_TESTS: dict[str, tuple[str, str, str]] = {
    # Test name: (trigram_context, extended_context, description)
    "number_after_article": (
        "a 7",
        "The value is a 7 digit number",
        "Number after indefinite article",
    ),
    "operator_in_expression": (
        "5 + 3",
        "Calculate 5 + 3 equals",
        "Operator in arithmetic",
    ),
    "word_after_question": (
        "is the",
        "What is the answer to",
        "Word after question word",
    ),
    "token_after_code": (
        "def foo",
        "def foo(x): return x * 2",
        "Token in code context",
    ),
    "noun_after_adjective": (
        "red ball",
        "I see a big red ball rolling",
        "Noun after adjective",
    ),
    "verb_after_noun": (
        "dog runs",
        "The small dog runs very fast",
        "Verb after noun",
    ),
}


# =============================================================================
# Context Type Test Data
# =============================================================================

DEFAULT_CONTEXTS: dict[str, str] = {
    "numeric": "The number 127 is prime",
    "after_word": "Calculate the sum",
    "after_article": "A large building",
    "standalone": "Hello",
    "after_operator": "5 + 3 = 8",
}


# =============================================================================
# Domain Test Data
# =============================================================================

DOMAIN_PROMPTS: dict[str, list[str]] = {
    Domain.MATH.value: [
        "2 + 2 =",
        "5 * 7 =",
        "12 - 4 =",
        "sqrt(16) =",
        "3^2 =",
    ],
    Domain.CODE.value: [
        "def factorial(n):",
        "for i in range(10):",
        "if x > 0:",
        "class MyClass:",
        "import numpy as",
    ],
    Domain.LANGUAGE.value: [
        "The quick brown",
        "Once upon a time",
        "In conclusion,",
        "To summarize,",
        "However,",
    ],
    Domain.REASONING.value: [
        "If A then B. A is true, so",
        "All cats are mammals. Fluffy is a cat, so Fluffy",
        "The premise implies that",
        "Therefore, we can conclude",
        "Based on the evidence,",
    ],
}


# =============================================================================
# Taxonomy Test Data by Category
# =============================================================================

TAXONOMY_TEST_PROMPTS: dict[str, list[str]] = {
    PatternCategory.ARITHMETIC.value: [
        "2 + 2 =",
        "5 * 7 =",
        "12 - 4 =",
        "100 / 10 =",
        "15 + 23 =",
    ],
    PatternCategory.CODE.value: [
        "def hello():",
        "for i in range(10):",
        "if x > 0:",
        "class Foo(Bar):",
        "return x * 2",
    ],
    PatternCategory.SYNONYM.value: [
        "happy means joyful",
        "big is like large",
        "fast equals quick",
        "smart similar to intelligent",
        "begin same as start",
    ],
    PatternCategory.ANTONYM.value: [
        "hot is opposite of cold",
        "up versus down",
        "good but not bad",
        "light contrasts dark",
        "happy unlike sad",
    ],
    PatternCategory.ANALOGY.value: [
        "king is to queen as man is to",
        "dog is to puppy as cat is to",
        "hand is to glove as foot is to",
        "bird is to nest as bee is to",
        "teacher is to student as doctor is to",
    ],
    PatternCategory.COMPARISON.value: [
        "bigger than the",
        "smaller compared to",
        "more important than",
        "less expensive than",
        "as tall as the",
    ],
    PatternCategory.CAUSATION.value: [
        "because of the rain",
        "therefore the result",
        "so the outcome was",
        "since the beginning",
        "thus we conclude",
    ],
    PatternCategory.CONDITIONAL.value: [
        "if it rains then",
        "when the sun sets",
        "unless you try",
        "provided that you",
        "assuming that we",
    ],
    PatternCategory.QUESTION.value: [
        "What is the answer?",
        "How does it work?",
        "Why did that happen?",
        "Where is the location?",
        "When will it start?",
    ],
    PatternCategory.NEGATION.value: [
        "not the same as",
        "never going to",
        "no one knows",
        "nothing is certain",
        "without any doubt",
    ],
    PatternCategory.TEMPORAL.value: [
        "yesterday was sunny",
        "tomorrow will be",
        "before the meeting",
        "after the event",
        "during the process",
    ],
    PatternCategory.QUANTIFICATION.value: [
        "all of the items",
        "some of the people",
        "none of the above",
        "most of the time",
        "few of the students",
    ],
}


# =============================================================================
# Attention Routing Test Data
# =============================================================================

ATTENTION_ROUTING_CONTEXTS: dict[str, str] = {
    "analogy": "King is to queen as man is to woman",
    "arithmetic": "5 + 3 = 8",
    "code_def": "def calculate(x, y): return x + y",
    "question": "What is the capital of France?",
    "comparison": "The red ball is bigger than the blue ball",
}


# =============================================================================
# Token Classification Lexicons
# =============================================================================

CODE_KEYWORDS: frozenset[str] = frozenset({
    "def", "class", "if", "else", "elif", "for", "while", "return",
    "import", "from", "try", "except", "finally", "with", "as", "assert",
    "break", "continue", "pass", "raise", "yield", "lambda", "global",
    "nonlocal", "async", "await", "in", "not", "and", "or", "is", "None",
})

BOOLEAN_LITERALS: frozenset[str] = frozenset({"True", "False", "true", "false"})

TYPE_KEYWORDS: frozenset[str] = frozenset({
    "int", "float", "str", "bool", "list", "dict", "set", "tuple",
    "bytes", "None", "Any", "Optional", "Union", "List", "Dict",
})

QUESTION_WORDS: frozenset[str] = frozenset({
    "what", "who", "where", "when", "why", "how", "which", "whose",
})

ANSWER_WORDS: frozenset[str] = frozenset({
    "yes", "no", "maybe", "probably", "perhaps", "definitely",
})

NEGATION_WORDS: frozenset[str] = frozenset({
    "not", "no", "never", "none", "nothing", "neither", "nobody", "nowhere",
})

TIME_WORDS: frozenset[str] = frozenset({
    "yesterday", "today", "tomorrow", "now", "then", "before", "after",
    "during", "always", "never", "sometimes", "often", "usually",
})

QUANTIFIER_WORDS: frozenset[str] = frozenset({
    "all", "some", "none", "many", "few", "most", "any", "each", "every",
})

COMPARISON_WORDS: frozenset[str] = frozenset({
    "more", "less", "most", "least", "bigger", "smaller", "larger",
    "better", "worse", "higher", "lower", "faster", "slower",
})

COORDINATION_WORDS: frozenset[str] = frozenset({
    "and", "or", "but", "yet", "so", "nor", "for",
})

CAUSATION_WORDS: frozenset[str] = frozenset({
    "because", "since", "therefore", "thus", "hence", "so",
    "consequently", "accordingly",
})

CONDITIONAL_WORDS: frozenset[str] = frozenset({
    "if", "unless", "when", "whenever", "provided", "assuming",
    "supposing", "given",
})


__all__ = [
    # Test contexts
    "TOKEN_CONTEXTS",
    "CONTEXT_WINDOW_TESTS",
    "DEFAULT_CONTEXTS",
    "DOMAIN_PROMPTS",
    "TAXONOMY_TEST_PROMPTS",
    "ATTENTION_ROUTING_CONTEXTS",
    # Lexicons
    "CODE_KEYWORDS",
    "BOOLEAN_LITERALS",
    "TYPE_KEYWORDS",
    "QUESTION_WORDS",
    "ANSWER_WORDS",
    "NEGATION_WORDS",
    "TIME_WORDS",
    "QUANTIFIER_WORDS",
    "COMPARISON_WORDS",
    "COORDINATION_WORDS",
    "CAUSATION_WORDS",
    "CONDITIONAL_WORDS",
]
