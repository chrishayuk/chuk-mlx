"""Wrapper for tiktoken tokenizers to provide HuggingFace-compatible interface.

This allows using OpenAI's tokenizers (gpt-4, gpt-3.5-turbo, etc.) with all
the analysis and utility tools in this library.

Usage:
    from chuk_lazarus.data.tokenizers.tiktoken_wrapper import TiktokenWrapper

    # Load by model name
    tokenizer = TiktokenWrapper.from_model("gpt-4")
    tokenizer = TiktokenWrapper.from_model("gpt-3.5-turbo")

    # Load by encoding name
    tokenizer = TiktokenWrapper.from_encoding("cl100k_base")
    tokenizer = TiktokenWrapper.from_encoding("o200k_base")

    # Use like any other tokenizer
    tokens = tokenizer.encode("Hello, world!")
    text = tokenizer.decode(tokens)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


# Model name to encoding mapping
# See: https://github.com/openai/tiktoken/blob/main/tiktoken/model.py
MODEL_TO_ENCODING: dict[str, str] = {
    # GPT-4o and later models use o200k_base
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    "o1": "o200k_base",
    "o1-mini": "o200k_base",
    "o1-preview": "o200k_base",
    "o3-mini": "o200k_base",
    # GPT-4 and GPT-3.5-turbo use cl100k_base
    "gpt-4": "cl100k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-4-turbo-preview": "cl100k_base",
    "gpt-4-32k": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
    "gpt-3.5-turbo-16k": "cl100k_base",
    "gpt-3.5": "cl100k_base",
    # Text embedding models
    "text-embedding-ada-002": "cl100k_base",
    "text-embedding-3-small": "cl100k_base",
    "text-embedding-3-large": "cl100k_base",
    # Legacy models
    "text-davinci-003": "p50k_base",
    "text-davinci-002": "p50k_base",
    "code-davinci-002": "p50k_base",
    "davinci": "r50k_base",
    "curie": "r50k_base",
    "babbage": "r50k_base",
    "ada": "r50k_base",
}

# Known encoding names
KNOWN_ENCODINGS = {"o200k_base", "cl100k_base", "p50k_base", "r50k_base", "p50k_edit", "gpt2"}


def is_tiktoken_model(name: str) -> bool:
    """Check if a model name is a tiktoken/OpenAI model.

    Args:
        name: Model name or encoding name to check

    Returns:
        True if this is a tiktoken model/encoding
    """
    name_lower = name.lower()

    # Check direct encoding names
    if name_lower in KNOWN_ENCODINGS:
        return True

    # Check model names
    if name_lower in MODEL_TO_ENCODING:
        return True

    # Check prefixes for model families
    tiktoken_prefixes = ("gpt-4", "gpt-3", "o1", "o3", "text-embedding", "text-davinci", "code-")
    return any(name_lower.startswith(prefix) for prefix in tiktoken_prefixes)


class TiktokenWrapper:
    """Wrapper around tiktoken to provide HuggingFace-compatible interface.

    This allows using OpenAI tokenizers with the tokenizer analysis tools.
    Implements the TokenizerProtocol used throughout this library.
    """

    def __init__(self, encoding, model_name: str | None = None):
        """Initialize with a tiktoken Encoding object.

        Args:
            encoding: tiktoken.Encoding object
            model_name: Optional model name for display purposes
        """
        self._encoding = encoding
        self._model_name = model_name or encoding.name
        self._vocab: dict[str, int] | None = None

        # Special token IDs - tiktoken uses specific IDs for these
        # Note: tiktoken doesn't have traditional BOS/EOS/PAD in the same way
        # We'll use reasonable defaults for compatibility
        self._pad_token_id: int | None = None
        self._unk_token_id: int | None = None
        self._bos_token_id: int | None = None
        self._eos_token_id: int | None = None

        # Try to get special tokens from the encoding
        special_tokens = getattr(encoding, "_special_tokens", {})
        if special_tokens:
            # Common special tokens in OpenAI models
            self._eos_token_id = special_tokens.get("<|endoftext|>")
            self._bos_token_id = special_tokens.get("<|startoftext|>")

            # For chat models
            if "<|end|>" in special_tokens:
                self._eos_token_id = special_tokens["<|end|>"]

    @classmethod
    def from_model(cls, model_name: str) -> TiktokenWrapper:
        """Create a TiktokenWrapper from an OpenAI model name.

        Args:
            model_name: OpenAI model name (e.g., "gpt-4", "gpt-3.5-turbo")

        Returns:
            TiktokenWrapper instance

        Raises:
            ImportError: If tiktoken is not installed
            ValueError: If model name is not recognized
        """
        try:
            import tiktoken
        except ImportError as err:
            raise ImportError(
                "tiktoken is required for OpenAI tokenizers. "
                "Install with: pip install 'chuk-lazarus[openai]' or pip install tiktoken"
            ) from err

        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except KeyError as err:
            # Try as encoding name
            if model_name.lower() in KNOWN_ENCODINGS:
                encoding = tiktoken.get_encoding(model_name.lower())
            else:
                raise ValueError(
                    f"Unknown model or encoding: {model_name}. "
                    f"Known models: {list(MODEL_TO_ENCODING.keys())}. "
                    f"Known encodings: {list(KNOWN_ENCODINGS)}"
                ) from err

        return cls(encoding, model_name)

    @classmethod
    def from_encoding(cls, encoding_name: str) -> TiktokenWrapper:
        """Create a TiktokenWrapper from an encoding name.

        Args:
            encoding_name: Encoding name (e.g., "cl100k_base", "o200k_base")

        Returns:
            TiktokenWrapper instance

        Raises:
            ImportError: If tiktoken is not installed
            ValueError: If encoding name is not recognized
        """
        try:
            import tiktoken
        except ImportError as err:
            raise ImportError(
                "tiktoken is required for OpenAI tokenizers. "
                "Install with: pip install 'chuk-lazarus[openai]' or pip install tiktoken"
            ) from err

        try:
            encoding = tiktoken.get_encoding(encoding_name)
        except ValueError as err:
            raise ValueError(
                f"Unknown encoding: {encoding_name}. Known encodings: {list(KNOWN_ENCODINGS)}"
            ) from err

        return cls(encoding, encoding_name)

    @property
    def name_or_path(self) -> str:
        """Return model name for compatibility."""
        return self._model_name

    @property
    def pad_token_id(self) -> int | None:
        """Padding token ID."""
        return self._pad_token_id

    @pad_token_id.setter
    def pad_token_id(self, value: int | None):
        """Set padding token ID."""
        self._pad_token_id = value

    @property
    def unk_token_id(self) -> int | None:
        """Unknown token ID."""
        return self._unk_token_id

    @property
    def bos_token_id(self) -> int | None:
        """Beginning of sequence token ID."""
        return self._bos_token_id

    @property
    def eos_token_id(self) -> int | None:
        """End of sequence token ID."""
        return self._eos_token_id

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return self._encoding.n_vocab

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        **kwargs,
    ) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: Text to encode
            add_special_tokens: Ignored for tiktoken (included for compatibility)
            **kwargs: Additional arguments (ignored)

        Returns:
            List of token IDs
        """
        return self._encoding.encode(text)

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = False,
        **kwargs,
    ) -> str:
        """Decode token IDs to text.

        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Ignored for tiktoken (included for compatibility)
            **kwargs: Additional arguments (ignored)

        Returns:
            Decoded text
        """
        return self._encoding.decode(token_ids)

    def get_vocab(self) -> dict[str, int]:
        """Get the vocabulary as a dict mapping tokens to IDs.

        Returns:
            Dictionary mapping token strings to token IDs

        Note:
            This builds the vocab on first call and caches it.
            For large vocabularies, this can be slow.
        """
        if self._vocab is not None:
            return self._vocab

        # Build vocab from encoding
        # tiktoken doesn't expose vocab directly, so we need to decode each ID
        vocab = {}
        n_vocab = self._encoding.n_vocab

        # Decode each token ID to get the token string
        # We use errors='replace' to handle tokens that can't be decoded to valid UTF-8
        for token_id in range(n_vocab):
            try:
                token_bytes = self._encoding.decode_single_token_bytes(token_id)
                try:
                    token_str = token_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    # Use hex representation for non-UTF8 tokens
                    token_str = f"<0x{token_bytes.hex().upper()}>"
                vocab[token_str] = token_id
            except KeyError:
                # Some token IDs may not be valid
                continue

        # Add special tokens
        special_tokens = getattr(self._encoding, "_special_tokens", {})
        for token_str, token_id in special_tokens.items():
            vocab[token_str] = token_id

        self._vocab = vocab
        return vocab

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        """Convert token IDs to token strings.

        Args:
            ids: List of token IDs

        Returns:
            List of token strings
        """
        tokens = []
        for token_id in ids:
            try:
                token_bytes = self._encoding.decode_single_token_bytes(token_id)
                try:
                    token_str = token_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    token_str = f"<0x{token_bytes.hex().upper()}>"
                tokens.append(token_str)
            except KeyError:
                tokens.append(f"<UNK:{token_id}>")
        return tokens

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        """Convert token strings to token IDs.

        Args:
            tokens: List of token strings

        Returns:
            List of token IDs
        """
        vocab = self.get_vocab()
        return [vocab.get(token, 0) for token in tokens]

    def __repr__(self) -> str:
        return f"TiktokenWrapper(model={self._model_name}, vocab_size={self.vocab_size})"
