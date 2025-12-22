"""
HuggingFace tokenizer backend.

Primary goal: correctness + portability.
Wraps HuggingFace/SentencePiece tokenizers with a unified interface.
"""

from typing import Any

from ..types import (
    ChatMessage,
    ChatRole,
    MessageField,
    SpecialTokenField,
    TokenizerProtocol,
    Tool,
)
from .base import (
    BackendInfo,
    BackendType,
    BaseBackend,
    TokenizationResult,
)


class HuggingFaceBackend(BaseBackend):
    """
    HuggingFace tokenizer backend.

    Wraps any tokenizer that implements the HuggingFace tokenizer interface.
    Provides correctness and portability over raw speed.
    """

    def __init__(self, tokenizer: TokenizerProtocol | Any = None):
        """
        Initialize with a HuggingFace-compatible tokenizer.

        Args:
            tokenizer: A tokenizer with encode/decode methods.
                      If None, creates a minimal tokenizer.
        """
        self._tokenizer = tokenizer
        self._vocab: dict[str, int] | None = None

    # -------------------------------------------------------------------------
    # Core properties
    # -------------------------------------------------------------------------

    @property
    def backend_type(self) -> BackendType:
        return BackendType.HUGGINGFACE

    @property
    def vocab_size(self) -> int:
        if self._tokenizer is None:
            return 0
        if hasattr(self._tokenizer, "vocab_size"):
            return self._tokenizer.vocab_size
        return len(self.get_vocab())

    @property
    def tokenizer(self) -> TokenizerProtocol | Any:
        """Access the underlying tokenizer."""
        return self._tokenizer

    # -------------------------------------------------------------------------
    # Special token properties
    # -------------------------------------------------------------------------

    def _get_special_token_id(self, field: SpecialTokenField) -> int | None:
        """Get a special token ID from the underlying tokenizer."""
        if self._tokenizer is None:
            return None
        return getattr(self._tokenizer, field.value, None)

    @property
    def pad_token_id(self) -> int | None:
        return self._get_special_token_id(SpecialTokenField.PAD_TOKEN_ID)

    @property
    def eos_token_id(self) -> int | None:
        return self._get_special_token_id(SpecialTokenField.EOS_TOKEN_ID)

    @property
    def bos_token_id(self) -> int | None:
        return self._get_special_token_id(SpecialTokenField.BOS_TOKEN_ID)

    @property
    def unk_token_id(self) -> int | None:
        return self._get_special_token_id(SpecialTokenField.UNK_TOKEN_ID)

    # -------------------------------------------------------------------------
    # Core tokenization methods
    # -------------------------------------------------------------------------

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        return_offsets: bool = False,
    ) -> TokenizationResult:
        """
        Encode text using the HuggingFace tokenizer.

        Args:
            text: Text to encode
            add_special_tokens: Whether to add BOS/EOS tokens
            return_offsets: Whether to compute character offsets

        Returns:
            TokenizationResult with token IDs
        """
        if self._tokenizer is None:
            return TokenizationResult(token_ids=[], tokens=[], offsets=[])

        token_ids = self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
        tokens = self._get_token_strings(token_ids)
        offsets = self._get_offsets(text, add_special_tokens) if return_offsets else []

        return TokenizationResult(token_ids=token_ids, tokens=tokens, offsets=offsets)

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.

        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text
        """
        if self._tokenizer is None:
            return ""
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def get_vocab(self) -> dict[str, int]:
        """Return the vocabulary as token -> id mapping."""
        if self._vocab is not None:
            return self._vocab

        if self._tokenizer is None:
            return {}

        if hasattr(self._tokenizer, "get_vocab"):
            self._vocab = self._tokenizer.get_vocab()
        elif hasattr(self._tokenizer, "vocab"):
            self._vocab = dict(self._tokenizer.vocab)
        else:
            self._vocab = {}

        return self._vocab

    def get_info(self) -> BackendInfo:
        """Return backend information."""
        return BackendInfo(
            backend_type=BackendType.HUGGINGFACE,
            vocab_size=self.vocab_size,
            supports_parallel=False,
            supports_offsets=True,
            is_available=True,
        )

    # -------------------------------------------------------------------------
    # Chat template support
    # -------------------------------------------------------------------------

    def apply_chat_template(
        self,
        messages: list[ChatMessage] | list[dict],
        tools: list[Tool] | list[dict] | None = None,
        add_generation_prompt: bool = True,
        tokenize: bool = False,
    ) -> str | list[int]:
        """
        Apply chat template to messages.

        Args:
            messages: List of ChatMessage or message dicts with 'role' and 'content'
            tools: Optional list of Tool or tool definition dicts
            add_generation_prompt: Whether to add generation prompt
            tokenize: Whether to return token IDs instead of text

        Returns:
            Formatted text or token IDs
        """
        if self._tokenizer is None:
            return [] if tokenize else ""

        # Convert to dicts for HuggingFace compatibility
        message_dicts = [
            msg.model_dump() if isinstance(msg, ChatMessage) else msg for msg in messages
        ]
        tool_dicts = (
            [t.model_dump() if isinstance(t, Tool) else t for t in tools] if tools else None
        )

        if not hasattr(self._tokenizer, "apply_chat_template"):
            return self._apply_fallback_template(message_dicts, add_generation_prompt, tokenize)

        kwargs: dict[str, bool | list[dict]] = {
            "add_generation_prompt": add_generation_prompt,
            "tokenize": tokenize,
        }
        if tool_dicts is not None:
            kwargs["tools"] = tool_dicts

        return self._tokenizer.apply_chat_template(message_dicts, **kwargs)

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _get_token_strings(self, token_ids: list[int]) -> list[str]:
        """Convert token IDs to token strings if possible."""
        if hasattr(self._tokenizer, "convert_ids_to_tokens"):
            return self._tokenizer.convert_ids_to_tokens(token_ids)
        return []

    def _get_offsets(self, text: str, add_special_tokens: bool) -> list[tuple[int, int]]:
        """Get character offsets for tokens if available."""
        if not callable(self._tokenizer):
            return []

        try:
            encoding = self._tokenizer(
                text,
                add_special_tokens=add_special_tokens,
                return_offsets_mapping=True,
            )
            if "offset_mapping" not in encoding:
                return []

            return [
                (start, end)
                for start, end in encoding["offset_mapping"]
                if start is not None and end is not None
            ]
        except Exception:
            return []

    def _apply_fallback_template(
        self,
        messages: list[dict],
        add_generation_prompt: bool,
        tokenize: bool,
    ) -> str | list[int]:
        """Apply a simple fallback chat template."""
        parts = []
        for msg in messages:
            role = msg.get(MessageField.ROLE.value, ChatRole.USER.value)
            content = msg.get(MessageField.CONTENT.value, "")
            parts.append(f"<|{role}|>\n{content}\n")

        if add_generation_prompt:
            parts.append(f"<|{ChatRole.ASSISTANT.value}|>\n")

        text = "".join(parts)
        return self.encode(text).token_ids if tokenize else text
