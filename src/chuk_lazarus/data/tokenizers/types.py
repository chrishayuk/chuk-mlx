"""
Shared types, enums, constants, and protocols for the tokenizers module.

This module provides:
- Enums for magic strings (roles, special tokens, etc.)
- Pydantic models for structured data (messages, tools, etc.)
- The canonical TokenizerProtocol
- Type aliases for common patterns
"""

from enum import Enum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

# =============================================================================
# Enums - Eliminate magic strings
# =============================================================================


class ChatRole(str, Enum):
    """Roles in a chat conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class SpecialTokenName(str, Enum):
    """Standard special token string representations."""

    PAD = "<pad>"
    UNK = "<unk>"
    BOS = "<s>"
    EOS = "</s>"
    SEP = "<sep>"
    CLS = "<cls>"
    MASK = "<mask>"

    # Alternative representations (LLaMA/GPT style)
    BOS_ALT = "<bos>"
    EOS_ALT = "<eos>"

    # BERT-style representations
    UNK_BERT = "[UNK]"
    PAD_BERT = "[PAD]"
    CLS_BERT = "[CLS]"
    SEP_BERT = "[SEP]"
    MASK_BERT = "[MASK]"


class SpecialTokenField(str, Enum):
    """Field names for special token IDs."""

    PAD_TOKEN_ID = "pad_token_id"
    UNK_TOKEN_ID = "unk_token_id"
    BOS_TOKEN_ID = "bos_token_id"
    EOS_TOKEN_ID = "eos_token_id"
    SEP_TOKEN_ID = "sep_token_id"
    CLS_TOKEN_ID = "cls_token_id"
    MASK_TOKEN_ID = "mask_token_id"


class MessageField(str, Enum):
    """Field names in chat message dictionaries."""

    ROLE = "role"
    CONTENT = "content"
    NAME = "name"
    TOOL_CALLS = "tool_calls"
    TOOL_CALL_ID = "tool_call_id"


# Set of all known UNK token representations for lookup
UNKNOWN_TOKEN_MARKERS: frozenset[str] = frozenset(
    {
        SpecialTokenName.UNK.value,
        SpecialTokenName.UNK_BERT.value,
        "<UNK>",
        "[unk]",
    }
)


# =============================================================================
# Pydantic Models - Replace dict goop
# =============================================================================


class ChatMessage(BaseModel):
    """A message in a chat conversation."""

    model_config = {"frozen": True}

    role: ChatRole = Field(description="The role of the message sender")
    content: str = Field(description="The message content")
    name: str | None = Field(default=None, description="Optional name for the sender")
    tool_calls: list["ToolCall"] | None = Field(
        default=None, description="Tool calls made by assistant"
    )
    tool_call_id: str | None = Field(
        default=None, description="ID of tool call this message responds to"
    )


class ToolParameter(BaseModel):
    """A parameter for a tool function."""

    model_config = {"frozen": True}

    name: str = Field(description="Parameter name")
    type: str = Field(description="Parameter type (e.g., 'string', 'integer')")
    description: str = Field(default="", description="Parameter description")
    required: bool = Field(default=False, description="Whether parameter is required")


class Tool(BaseModel):
    """A tool definition for function calling."""

    model_config = {"frozen": True}

    name: str = Field(description="Tool name")
    description: str = Field(description="Tool description")
    parameters: list[ToolParameter] = Field(default_factory=list, description="Tool parameters")


class ToolCall(BaseModel):
    """A tool call made by the assistant."""

    model_config = {"frozen": True}

    id: str = Field(description="Unique ID for this tool call")
    name: str = Field(description="Name of the tool to call")
    arguments: str = Field(description="JSON-encoded arguments")


class TokenOffset(BaseModel):
    """Character offset for a token in the original text."""

    model_config = {"frozen": True}

    start: int = Field(ge=0, description="Start character position")
    end: int = Field(ge=0, description="End character position (exclusive)")


class VocabularyData(BaseModel):
    """Complete vocabulary data loaded from a file."""

    model_config = {"frozen": True}

    vocab: dict[str, int] = Field(description="Token to ID mapping")
    special_tokens: dict[str, int] = Field(
        default_factory=dict, description="Special token string to ID mapping"
    )
    added_tokens: list[str] = Field(
        default_factory=list, description="Additional tokens added to vocabulary"
    )

    def get_special_token_id(self, token: SpecialTokenName | str) -> int | None:
        """Get ID for a special token by name or string."""
        key = token.value if isinstance(token, SpecialTokenName) else token
        return self.special_tokens.get(key)


class DuplicateIdIssue(BaseModel):
    """A token ID assigned to multiple tokens."""

    model_config = {"frozen": True}

    token_id: int = Field(description="The duplicated token ID")
    tokens: list[str] = Field(description="Tokens sharing this ID")


class NegativeIdIssue(BaseModel):
    """A token with a negative ID."""

    model_config = {"frozen": True}

    token: str = Field(description="The token string")
    token_id: int = Field(lt=0, description="The negative token ID")


class TokenIdMapping(BaseModel):
    """Reverse mapping from token IDs to token strings."""

    model_config = {"frozen": True}

    mapping: dict[int, str] = Field(description="ID to token mapping")
    vocab_size: int = Field(ge=0, description="Size of the vocabulary")

    def get(self, token_id: int, default: str | None = None) -> str | None:
        """Get token string for an ID."""
        return self.mapping.get(token_id, default)

    def __getitem__(self, token_id: int) -> str:
        """Get token string for an ID."""
        return self.mapping[token_id]

    def __contains__(self, token_id: int) -> bool:
        """Check if ID is in mapping."""
        return token_id in self.mapping


# =============================================================================
# Protocol - Single source of truth for tokenizer interface
# =============================================================================


@runtime_checkable
class TokenizerProtocol(Protocol):
    """
    Protocol defining the interface for tokenizers.

    This is the canonical definition - all other modules should import from here.
    Compatible with HuggingFace tokenizers, SentencePiece, and custom implementations.
    """

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Encode text to token IDs."""
        ...

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        ...

    def get_vocab(self) -> dict[str, int]:
        """Get the vocabulary as token -> ID mapping."""
        ...

    @property
    def pad_token_id(self) -> int | None:
        """Padding token ID."""
        ...

    @property
    def unk_token_id(self) -> int | None:
        """Unknown token ID."""
        ...

    @property
    def bos_token_id(self) -> int | None:
        """Beginning of sequence token ID."""
        ...

    @property
    def eos_token_id(self) -> int | None:
        """End of sequence token ID."""
        ...


@runtime_checkable
class ExtendedTokenizerProtocol(TokenizerProtocol, Protocol):
    """
    Extended tokenizer protocol with additional optional features.

    Use this for tokenizers that support chat templates and other advanced features.
    """

    @property
    def vocab_size(self) -> int:
        """Size of the vocabulary."""
        ...

    def apply_chat_template(
        self,
        messages: list[dict],
        add_generation_prompt: bool = True,
        tokenize: bool = False,
    ) -> str | list[int]:
        """Apply chat template to messages."""
        ...


# =============================================================================
# Type Aliases
# =============================================================================

# Vocabulary types
Vocab = dict[str, int]
ReverseVocab = dict[int, str]

# Token sequence types
TokenIds = list[int]
TokenStrings = list[str]
