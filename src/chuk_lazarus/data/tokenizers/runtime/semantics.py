"""Token semantics mapping for tool and action grounding."""

from enum import Enum

from pydantic import BaseModel, Field


class SemanticDomain(str, Enum):
    """Semantic domains for token mapping."""

    MEMORY = "memory"
    TOOL = "tool"
    SOLVER = "solver"
    CONTROL = "control"
    DATA = "data"
    CUSTOM = "custom"


class SemanticMapping(BaseModel):
    """Mapping from token to semantic meaning."""

    token_str: str = Field(description="Token string")
    token_id: int = Field(ge=0, description="Token ID")
    domain: SemanticDomain = Field(description="Semantic domain")
    operation: str = Field(description="Operation name (e.g., 'load', 'add')")
    full_path: str = Field(description="Full semantic path (e.g., 'memory.op.load')")
    arguments: list[str] = Field(default_factory=list, description="Expected argument types")
    returns: str = Field(default="", description="Return type")
    description: str = Field(default="", description="Human description")


class TokenSemantics(BaseModel):
    """Registry of token semantics mappings."""

    mappings: dict[int, SemanticMapping] = Field(
        default_factory=dict, description="Token ID -> semantic mapping"
    )
    by_path: dict[str, int] = Field(default_factory=dict, description="Semantic path -> token ID")
    domains: dict[SemanticDomain, list[int]] = Field(
        default_factory=dict, description="Domain -> token IDs"
    )

    def register(
        self,
        token_str: str,
        token_id: int,
        domain: SemanticDomain,
        operation: str,
        arguments: list[str] | None = None,
        returns: str = "",
        description: str = "",
    ) -> SemanticMapping:
        """
        Register a token-to-semantic mapping.

        Args:
            token_str: Token string
            token_id: Token ID
            domain: Semantic domain
            operation: Operation name
            arguments: Expected argument types
            returns: Return type
            description: Human description

        Returns:
            Created SemanticMapping
        """
        full_path = f"{domain.value}.op.{operation}"

        mapping = SemanticMapping(
            token_str=token_str,
            token_id=token_id,
            domain=domain,
            operation=operation,
            full_path=full_path,
            arguments=arguments or [],
            returns=returns,
            description=description,
        )

        self.mappings[token_id] = mapping
        self.by_path[full_path] = token_id

        if domain not in self.domains:
            self.domains[domain] = []
        self.domains[domain].append(token_id)

        return mapping

    def get_by_id(self, token_id: int) -> SemanticMapping | None:
        """Get semantic mapping by token ID."""
        return self.mappings.get(token_id)

    def get_by_path(self, path: str) -> SemanticMapping | None:
        """Get semantic mapping by path."""
        token_id = self.by_path.get(path)
        if token_id is not None:
            return self.mappings.get(token_id)
        return None

    def get_domain_tokens(self, domain: SemanticDomain) -> list[SemanticMapping]:
        """Get all tokens in a semantic domain."""
        token_ids = self.domains.get(domain, [])
        return [self.mappings[tid] for tid in token_ids if tid in self.mappings]


def map_token_to_semantic(
    semantics: TokenSemantics,
    token_str: str,
    token_id: int,
    domain: SemanticDomain,
    operation: str,
    **kwargs,
) -> SemanticMapping:
    """
    Convenience function to map a token to a semantic meaning.

    Args:
        semantics: TokenSemantics registry
        token_str: Token string
        token_id: Token ID
        domain: Semantic domain
        operation: Operation name
        **kwargs: Additional arguments for SemanticMapping

    Returns:
        Created SemanticMapping
    """
    return semantics.register(
        token_str=token_str,
        token_id=token_id,
        domain=domain,
        operation=operation,
        **kwargs,
    )


def get_semantic_group(
    semantics: TokenSemantics,
    domain: SemanticDomain,
) -> list[SemanticMapping]:
    """Get all mappings in a semantic domain."""
    return semantics.get_domain_tokens(domain)


def create_standard_semantics() -> TokenSemantics:
    """
    Create a semantics registry with standard mappings.

    Returns:
        Configured TokenSemantics
    """
    semantics = TokenSemantics()

    # Memory operations
    semantics.register(
        "<LOAD_PAGE>",
        100,
        SemanticDomain.MEMORY,
        "load",
        arguments=["page_id"],
        returns="page_content",
        description="Load a memory page",
    )
    semantics.register(
        "<STORE_PAGE>",
        101,
        SemanticDomain.MEMORY,
        "store",
        arguments=["page_id", "content"],
        returns="success",
        description="Store to memory page",
    )
    semantics.register(
        "<PAGE_IN>",
        102,
        SemanticDomain.MEMORY,
        "page_in",
        arguments=["page_id"],
        returns="page_content",
        description="Page in from storage",
    )
    semantics.register(
        "<PAGE_OUT>",
        103,
        SemanticDomain.MEMORY,
        "page_out",
        arguments=["page_id"],
        returns="success",
        description="Page out to storage",
    )

    # Tool operations
    semantics.register(
        "<TOOL_CALL>",
        200,
        SemanticDomain.TOOL,
        "call",
        arguments=["tool_name", "args"],
        returns="result",
        description="Call an external tool",
    )
    semantics.register(
        "<TOOL_RESULT>",
        201,
        SemanticDomain.TOOL,
        "result",
        arguments=["result"],
        returns="",
        description="Tool call result",
    )

    # Solver operations
    semantics.register(
        "<ADD>",
        300,
        SemanticDomain.SOLVER,
        "add",
        arguments=["a", "b"],
        returns="sum",
        description="Add two numbers",
    )
    semantics.register(
        "<MUL>",
        301,
        SemanticDomain.SOLVER,
        "mul",
        arguments=["a", "b"],
        returns="product",
        description="Multiply two numbers",
    )
    semantics.register(
        "<ARGMIN>",
        302,
        SemanticDomain.SOLVER,
        "argmin",
        arguments=["sequence"],
        returns="index",
        description="Find index of minimum",
    )
    semantics.register(
        "<ARGMAX>",
        303,
        SemanticDomain.SOLVER,
        "argmax",
        arguments=["sequence"],
        returns="index",
        description="Find index of maximum",
    )

    # Control operations
    semantics.register(
        "<THINK>", 400, SemanticDomain.CONTROL, "think", description="Begin reasoning block"
    )
    semantics.register(
        "</THINK>", 401, SemanticDomain.CONTROL, "end_think", description="End reasoning block"
    )

    return semantics
