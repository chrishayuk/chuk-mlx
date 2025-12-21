"""Special token registry with collision detection and reserved ranges."""

from enum import Enum

from pydantic import BaseModel, Field


class TokenCategory(str, Enum):
    """Categories for special tokens."""

    # Standard special tokens
    PADDING = "padding"
    UNKNOWN = "unknown"
    BEGINNING = "beginning"
    END = "end"
    SEPARATOR = "separator"

    # Tool/agent tokens
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"

    # Memory tokens
    MEMORY_LOAD = "memory_load"
    MEMORY_STORE = "memory_store"
    PAGE_IN = "page_in"
    PAGE_OUT = "page_out"

    # Reasoning tokens
    THINK_START = "think_start"
    THINK_END = "think_end"

    # Solver tokens
    SOLVER_OP = "solver_op"

    # Reference tokens
    REFERENCE = "reference"

    # Custom
    CUSTOM = "custom"


class SpecialTokenEntry(BaseModel):
    """A registered special token."""

    token_str: str = Field(description="Token string representation")
    token_id: int = Field(ge=0, description="Token ID")
    category: TokenCategory = Field(description="Token category")
    description: str = Field(default="", description="Human description")
    is_trainable: bool = Field(default=True, description="Whether embedding is trainable")
    reserved: bool = Field(default=False, description="Whether ID is in reserved range")


class ReservedRange(BaseModel):
    """A reserved ID range for special tokens."""

    start: int = Field(ge=0, description="Start of range (inclusive)")
    end: int = Field(ge=0, description="End of range (exclusive)")
    category: TokenCategory = Field(description="Category this range is for")
    description: str = Field(default="", description="Range description")


class CollisionReport(BaseModel):
    """Report of detected token ID collisions."""

    has_collisions: bool = Field(description="Whether collisions were found")
    collisions: list[tuple[str, str, int]] = Field(
        default_factory=list,
        description="List of (token1, token2, shared_id) collisions",
    )
    reserved_violations: list[tuple[str, int, str]] = Field(
        default_factory=list,
        description="Tokens using reserved IDs: (token, id, reserved_for)",
    )


class SpecialTokenRegistry(BaseModel):
    """Registry for managing special tokens with collision detection."""

    tokens: dict[str, SpecialTokenEntry] = Field(
        default_factory=dict, description="Token string -> entry mapping"
    )
    id_to_token: dict[int, str] = Field(
        default_factory=dict, description="Token ID -> string mapping"
    )
    reserved_ranges: list[ReservedRange] = Field(
        default_factory=list, description="Reserved ID ranges"
    )
    next_dynamic_id: int = Field(default=50000, description="Next ID for dynamic tokens")

    def register(
        self,
        token_str: str,
        token_id: int | None = None,
        category: TokenCategory = TokenCategory.CUSTOM,
        description: str = "",
        is_trainable: bool = True,
    ) -> SpecialTokenEntry:
        """
        Register a special token.

        Args:
            token_str: Token string representation
            token_id: Token ID (auto-assigned if None)
            category: Token category
            description: Human description
            is_trainable: Whether embedding is trainable

        Returns:
            SpecialTokenEntry for the registered token

        Raises:
            ValueError: If token already registered or ID collision
        """
        if token_str in self.tokens:
            raise ValueError(f"Token '{token_str}' already registered")

        if token_id is None:
            token_id = self.next_dynamic_id
            self.next_dynamic_id += 1

        if token_id in self.id_to_token:
            existing = self.id_to_token[token_id]
            raise ValueError(f"ID {token_id} already used by '{existing}'")

        # Check reserved ranges
        reserved = False
        for range_def in self.reserved_ranges:
            if range_def.start <= token_id < range_def.end:
                reserved = True
                break

        entry = SpecialTokenEntry(
            token_str=token_str,
            token_id=token_id,
            category=category,
            description=description,
            is_trainable=is_trainable,
            reserved=reserved,
        )

        self.tokens[token_str] = entry
        self.id_to_token[token_id] = token_str
        return entry

    def reserve_range(
        self,
        start: int,
        end: int,
        category: TokenCategory,
        description: str = "",
    ) -> ReservedRange:
        """
        Reserve an ID range for a category.

        Args:
            start: Start of range (inclusive)
            end: End of range (exclusive)
            category: Category to reserve for
            description: Range description

        Returns:
            ReservedRange entry
        """
        range_def = ReservedRange(start=start, end=end, category=category, description=description)
        self.reserved_ranges.append(range_def)
        return range_def

    def get_by_category(self, category: TokenCategory) -> list[SpecialTokenEntry]:
        """Get all tokens in a category."""
        return [e for e in self.tokens.values() if e.category == category]

    def get_by_id(self, token_id: int) -> SpecialTokenEntry | None:
        """Get token entry by ID."""
        token_str = self.id_to_token.get(token_id)
        if token_str:
            return self.tokens.get(token_str)
        return None

    def check_collisions(self) -> CollisionReport:
        """Check for any token ID collisions or reserved violations."""
        collisions = []
        reserved_violations = []

        # Check for duplicate IDs (shouldn't happen with proper registration)
        seen_ids: dict[int, str] = {}
        for token_str, entry in self.tokens.items():
            if entry.token_id in seen_ids:
                collisions.append((seen_ids[entry.token_id], token_str, entry.token_id))
            else:
                seen_ids[entry.token_id] = token_str

        # Check reserved range violations
        for token_str, entry in self.tokens.items():
            for range_def in self.reserved_ranges:
                if range_def.start <= entry.token_id < range_def.end:
                    if entry.category != range_def.category:
                        reserved_violations.append(
                            (token_str, entry.token_id, range_def.category.value)
                        )

        return CollisionReport(
            has_collisions=bool(collisions or reserved_violations),
            collisions=collisions,
            reserved_violations=reserved_violations,
        )


def register_special_token(
    registry: SpecialTokenRegistry,
    token_str: str,
    category: TokenCategory,
    token_id: int | None = None,
    description: str = "",
) -> SpecialTokenEntry:
    """
    Convenience function to register a special token.

    Args:
        registry: Registry to add token to
        token_str: Token string
        category: Token category
        token_id: Optional specific ID
        description: Token description

    Returns:
        Registered SpecialTokenEntry
    """
    return registry.register(
        token_str=token_str,
        token_id=token_id,
        category=category,
        description=description,
    )


def check_collisions(registry: SpecialTokenRegistry) -> CollisionReport:
    """Check registry for collisions."""
    return registry.check_collisions()


def get_reserved_ranges(registry: SpecialTokenRegistry) -> list[ReservedRange]:
    """Get all reserved ranges."""
    return registry.reserved_ranges


def create_standard_registry(
    vocab_size: int = 32000,
    tool_range_size: int = 100,
    memory_range_size: int = 50,
) -> SpecialTokenRegistry:
    """
    Create a registry with standard reserved ranges.

    Args:
        vocab_size: Base vocabulary size
        tool_range_size: Number of IDs to reserve for tools
        memory_range_size: Number of IDs to reserve for memory

    Returns:
        Configured SpecialTokenRegistry
    """
    registry = SpecialTokenRegistry(next_dynamic_id=vocab_size + 1000)

    # Reserve ranges at end of vocab
    tool_start = vocab_size
    memory_start = tool_start + tool_range_size
    solver_start = memory_start + memory_range_size

    registry.reserve_range(
        tool_start, tool_start + tool_range_size, TokenCategory.TOOL_CALL, "Tool tokens"
    )
    registry.reserve_range(
        memory_start, memory_start + memory_range_size, TokenCategory.MEMORY_LOAD, "Memory tokens"
    )
    registry.reserve_range(
        solver_start, solver_start + 50, TokenCategory.SOLVER_OP, "Solver tokens"
    )

    # Register standard tokens
    registry.register("<pad>", 0, TokenCategory.PADDING, "Padding token", False)
    registry.register("<unk>", 1, TokenCategory.UNKNOWN, "Unknown token")
    registry.register("<s>", 2, TokenCategory.BEGINNING, "Beginning of sequence")
    registry.register("</s>", 3, TokenCategory.END, "End of sequence")

    return registry
