"""
Runtime tokenizer utilities for dynamic vocabulary and special tokens.

Modules:
- special_registry: Special token registry with collision detection
- dynamic_vocab: Runtime vocabulary extension
- semantics: Token semantics mapping
"""

from .dynamic_vocab import (
    DynamicVocab,
    VocabExtension,
    create_embedding_slot,
    extend_vocab_runtime,
)
from .semantics import (
    SemanticDomain,
    SemanticMapping,
    TokenSemantics,
    create_standard_semantics,
    get_semantic_group,
    map_token_to_semantic,
)
from .special_registry import (
    CollisionReport,
    ReservedRange,
    SpecialTokenEntry,
    SpecialTokenRegistry,
    TokenCategory,
    check_collisions,
    create_standard_registry,
    get_reserved_ranges,
    register_special_token,
)

__all__ = [
    # Special registry
    "CollisionReport",
    "ReservedRange",
    "SpecialTokenRegistry",
    "SpecialTokenEntry",
    "TokenCategory",
    "register_special_token",
    "check_collisions",
    "create_standard_registry",
    "get_reserved_ranges",
    # Dynamic vocab
    "DynamicVocab",
    "VocabExtension",
    "extend_vocab_runtime",
    "create_embedding_slot",
    # Semantics
    "SemanticDomain",
    "TokenSemantics",
    "SemanticMapping",
    "create_standard_semantics",
    "map_token_to_semantic",
    "get_semantic_group",
]
