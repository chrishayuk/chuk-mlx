"""
Runtime Vocabulary Extension Example

Demonstrates runtime tokenizer utilities:
- Special token registry with collision detection
- Dynamic vocabulary extension
- Token semantics mapping for tools/agents

Uses Pydantic models for all data structures.
"""

from chuk_lazarus.data.tokenizers.runtime import (
    # Special token registry
    CollisionReport,
    ReservedRange,
    SpecialTokenEntry,
    SpecialTokenRegistry,
    TokenCategory,
    check_collisions,
    create_standard_registry,
    get_reserved_ranges,
    register_special_token,
    # Dynamic vocabulary
    DynamicVocab,
    VocabExtension,
    create_embedding_slot,
    extend_vocab_runtime,
    # Token semantics
    SemanticDomain,
    SemanticMapping,
    TokenSemantics,
    create_standard_semantics,
    get_semantic_group,
    map_token_to_semantic,
)


class MockTokenizer:
    """Simple mock tokenizer for demonstration."""

    def __init__(self):
        self.vocab = {
            "<pad>": 0,
            "<unk>": 1,
            "<bos>": 2,
            "<eos>": 3,
            "hello": 4,
            "world": 5,
        }
        self._vocab_size = len(self.vocab)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def get_vocab(self) -> dict[str, int]:
        return self.vocab.copy()

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [self.vocab.get(w, 1) for w in text.lower().split()]

    def decode(self, ids: list[int]) -> str:
        id_to_token = {v: k for k, v in self.vocab.items()}
        return " ".join(id_to_token.get(i, "<unk>") for i in ids)

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.vocab.get(token, 1)


def demo_special_token_registry():
    """Demonstrate special token registry."""
    print("=" * 60)
    print("Special Token Registry")
    print("=" * 60)

    # Create a new registry
    registry = SpecialTokenRegistry()

    # Register tool-related tokens
    registry.register(
        token_str="<TOOL_CALL>",
        token_id=50001,
        category=TokenCategory.TOOL_CALL,
        description="Start of tool invocation",
    )
    registry.register(
        token_str="<TOOL_RESULT>",
        token_id=50002,
        category=TokenCategory.TOOL_RESULT,
        description="Tool execution result",
    )

    # Register memory tokens
    registry.register(
        token_str="<MEM_READ>",
        token_id=50003,
        category=TokenCategory.MEMORY_LOAD,
        description="Read from memory",
    )
    registry.register(
        token_str="<MEM_WRITE>",
        token_id=50004,
        category=TokenCategory.MEMORY_STORE,
        description="Write to memory",
    )

    print(f"\nRegistry has {len(registry.tokens)} tokens:")
    for token_str, entry in registry.tokens.items():
        print(f"  {entry.token_id}: {entry.token_str:20s} [{entry.category.value}]")
        if entry.description:
            print(f"      {entry.description}")

    # Check for collisions with reserved ranges
    ranges = get_reserved_ranges(registry)
    print(f"\nReserved ranges: {len(ranges)}")
    for r in ranges[:3]:
        print(f"  {r.start}-{r.end}: {r.category.value} - {r.description}")

    # Check collisions
    report: CollisionReport = check_collisions(registry)
    print(f"\nCollision check: {'No collisions' if not report.has_collisions else 'Collisions found!'}")


def demo_standard_registry():
    """Demonstrate standard registry creation."""
    print("\n" + "=" * 60)
    print("Standard Token Registry")
    print("=" * 60)

    # Create standard registry with common special tokens
    registry = create_standard_registry()

    print(f"\nStandard registry has {len(registry.tokens)} tokens:")

    # Group by category
    by_category: dict[TokenCategory, list[SpecialTokenEntry]] = {}
    for token_str, entry in registry.tokens.items():
        if entry.category not in by_category:
            by_category[entry.category] = []
        by_category[entry.category].append(entry)

    for category, entries in by_category.items():
        print(f"\n  {category.value}:")
        for entry in entries[:3]:
            print(f"    {entry.token_id}: {entry.token_str}")


def demo_dynamic_vocab():
    """Demonstrate dynamic vocabulary extension."""
    print("\n" + "=" * 60)
    print("Dynamic Vocabulary Extension")
    print("=" * 60)

    tokenizer = MockTokenizer()
    print(f"\nOriginal vocab size: {tokenizer.vocab_size}")

    # Create dynamic vocab from tokenizer
    vocab = DynamicVocab.from_tokenizer(tokenizer)
    print(f"Base vocab size:     {vocab.base_vocab_size}")

    # Add new tokens
    ext1 = vocab.add_token("<SPECIAL_1>", tokenizer)
    ext2 = vocab.add_token("<SPECIAL_2>", tokenizer)
    ext3 = vocab.add_token("<DOMAIN_TOKEN>", tokenizer)

    print(f"\nAdded tokens:")
    print(f"  {ext1.token_str}: id={ext1.token_id}")
    print(f"  {ext2.token_str}: id={ext2.token_id}")
    print(f"  {ext3.token_str}: id={ext3.token_id}")

    print(f"\nTotal vocab size:    {vocab.total_vocab_size}")
    print(f"Extensions:          {len(vocab.extensions)}")

    # Get all added tokens
    all_tokens = vocab.get_all_tokens()
    print(f"\nAll extended tokens:")
    for ext in all_tokens:
        print(f"  {ext.token_str}: {ext.token_id}")


def demo_extend_vocab_batch():
    """Demonstrate batch vocabulary extension."""
    print("\n" + "=" * 60)
    print("Batch Vocabulary Extension")
    print("=" * 60)

    tokenizer = MockTokenizer()

    # Extend with multiple tokens at once
    new_tokens = [
        "<TOOL_START>",
        "<TOOL_END>",
        "<THINK>",
        "<ANSWER>",
        "<MEMORY>",
    ]

    vocab = DynamicVocab.from_tokenizer(tokenizer)
    extensions = extend_vocab_runtime(vocab, new_tokens, tokenizer)

    print(f"\nExtended vocab with {len(extensions)} new tokens:")
    print(f"  Base size:  {vocab.base_vocab_size}")
    print(f"  Total size: {vocab.total_vocab_size}")

    for ext in extensions:
        print(f"  {ext.token_str}: id={ext.token_id}, init={ext.init_method}")


def demo_embedding_slots():
    """Demonstrate embedding slot creation."""
    print("\n" + "=" * 60)
    print("Embedding Slot Creation")
    print("=" * 60)

    tokenizer = MockTokenizer()
    vocab = DynamicVocab.from_tokenizer(tokenizer)
    ext = vocab.add_token("<NEW_TOKEN>", tokenizer)

    # Create embedding slot for the new token
    embedding_dim = 768
    embedding = create_embedding_slot(ext, embedding_dim, init_method="mean")

    print(f"\nCreated embedding for '{ext.token_str}':")
    print(f"  Token ID:     {ext.token_id}")
    print(f"  Dimension:    {len(embedding)}")
    print(f"  Mean value:   {sum(embedding) / len(embedding):.6f}")

    # Create with different init methods
    for method in ["zero", "random"]:
        embedding = create_embedding_slot(ext, 128, init_method=method)
        mean_val = sum(embedding) / len(embedding)
        print(f"\n  Init '{method}' (dim=128): mean={mean_val:.6f}")


def demo_token_semantics():
    """Demonstrate token semantics mapping."""
    print("\n" + "=" * 60)
    print("Token Semantics Mapping")
    print("=" * 60)

    # Create semantics registry
    semantics = TokenSemantics()

    # Register tokens with semantic meaning
    semantics.register(
        token_str="<TOOL_CALL>",
        token_id=50001,
        domain=SemanticDomain.TOOL,
        operation="call_start",
        description="Initiates a tool call",
    )
    semantics.register(
        token_str="<TOOL_RESULT>",
        token_id=50002,
        domain=SemanticDomain.TOOL,
        operation="result",
        description="Contains tool result",
    )
    semantics.register(
        token_str="<MEM_READ>",
        token_id=50003,
        domain=SemanticDomain.MEMORY,
        operation="read",
        description="Read from context memory",
    )
    semantics.register(
        token_str="<THINK>",
        token_id=50004,
        domain=SemanticDomain.SOLVER,
        operation="think",
        description="Chain-of-thought reasoning",
    )

    print(f"\nRegistered {len(semantics.mappings)} semantic mappings:")

    # Lookup by ID
    mapping = semantics.get_by_id(50001)
    if mapping:
        print(f"\nToken 50001:")
        print(f"  Domain:    {mapping.domain.value}")
        print(f"  Operation: {mapping.operation}")
        print(f"  Full path: {mapping.full_path}")
        print(f"  Desc:      {mapping.description}")

    # Lookup by path
    mapping = semantics.get_by_path("memory.op.read")
    if mapping:
        print(f"\nPath 'memory.op.read':")
        print(f"  Token ID: {mapping.token_id}")
        print(f"  Domain:   {mapping.domain.value}")

    # Get all tokens in a domain
    tool_ids = semantics.domains.get(SemanticDomain.TOOL, [])
    print(f"\nTool domain tokens: {len(tool_ids)}")
    for tid in tool_ids:
        m = semantics.get_by_id(tid)
        if m:
            print(f"  {m.token_id}: {m.full_path}")


def demo_standard_semantics():
    """Demonstrate standard semantics creation."""
    print("\n" + "=" * 60)
    print("Standard Semantics Registry")
    print("=" * 60)

    # Create standard semantics with common agent tokens
    semantics = create_standard_semantics()

    print(f"\nStandard semantics has {len(semantics.mappings)} mappings:")

    # Show by domain
    for domain in SemanticDomain:
        domain_ids = semantics.domains.get(domain, [])
        if domain_ids:
            print(f"\n  {domain.value}:")
            for tid in domain_ids[:3]:
                m = semantics.get_by_id(tid)
                if m:
                    print(f"    {m.token_id}: {m.full_path}")


def demo_map_token_to_semantic():
    """Demonstrate convenience function for semantic mapping."""
    print("\n" + "=" * 60)
    print("Semantic Mapping Convenience Function")
    print("=" * 60)

    semantics = TokenSemantics()

    # Use convenience function
    mapping = map_token_to_semantic(
        semantics=semantics,
        token_str="<STOP>",
        token_id=60001,
        domain=SemanticDomain.CONTROL,
        operation="stop",
        description="Stop generation",
        arguments=["reason"],
        returns="acknowledgment",
    )

    print(f"\nMapped token:")
    print(f"  ID:          {mapping.token_id}")
    print(f"  Domain:      {mapping.domain.value}")
    print(f"  Operation:   {mapping.operation}")
    print(f"  Full path:   {mapping.full_path}")
    print(f"  Arguments:   {mapping.arguments}")
    print(f"  Returns:     {mapping.returns}")
    print(f"  Description: {mapping.description}")

    # Get semantic group
    control_ids = get_semantic_group(semantics, SemanticDomain.CONTROL)
    print(f"\nControl group has {len(control_ids)} token(s)")


def main():
    """Run all runtime vocabulary demos."""
    print("Runtime Vocabulary Extension Demo")
    print("=" * 60)

    demo_special_token_registry()
    demo_standard_registry()
    demo_dynamic_vocab()
    demo_extend_vocab_batch()
    demo_embedding_slots()
    demo_token_semantics()
    demo_standard_semantics()
    demo_map_token_to_semantic()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
