"""
Fast tokenizer backend using MLX Data CharTrie.

Primary goal: scale tokenization throughput across cores.
Especially useful for dataset preprocessing and packing.

The MLX Data CharTrie provides true parallel processing by avoiding Python's GIL.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

from ..types import (
    SpecialTokenField,
    SpecialTokenName,
    TokenizerProtocol,
    Vocab,
)
from .base import (
    BackendInfo,
    BackendType,
    BaseBackend,
    BatchTokenizationResult,
    TokenizationResult,
)

if TYPE_CHECKING:
    from mlx.data.core import CharTrie as CharTrieType
    from mlx.data.core import Tokenizer as MlxTokenizerType

# Check if MLX Data is available
_MLX_DATA_AVAILABLE = False
_CharTrie: type["CharTrieType"] | None = None
_Tokenizer: type["MlxTokenizerType"] | None = None

try:
    from mlx.data.core import CharTrie
    from mlx.data.core import Tokenizer as MlxTokenizer

    _MLX_DATA_AVAILABLE = True
    _CharTrie = CharTrie
    _Tokenizer = MlxTokenizer
except ImportError:
    pass


def is_fast_backend_available() -> bool:
    """Check if the fast backend (MLX Data) is available."""
    return _MLX_DATA_AVAILABLE


class FastBackend(BaseBackend):
    """
    MLX Data CharTrie-based tokenizer backend.

    Provides high-throughput parallel tokenization by using MLX Data's
    CharTrie implementation which avoids Python's GIL.

    Usage:
        # From vocabulary dict
        backend = FastBackend(vocab={"hello": 0, "world": 1})

        # From HuggingFace tokenizer (extracts vocab)
        backend = FastBackend.from_tokenizer(hf_tokenizer)

        # From SentencePiece model
        backend = FastBackend.from_sentencepiece("model.spm")

        # Parallel batch encoding
        result = backend.encode_batch(texts, num_workers=4)
    """

    # Subword markers to clean during decode
    SUBWORD_MARKERS = ("▁", "##")

    def __init__(
        self,
        vocab_or_tokenizer: Vocab | TokenizerProtocol | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
        unk_token_id: int | None = None,
    ):
        """
        Initialize the fast backend.

        Args:
            vocab_or_tokenizer: Either a vocab dict {token: id} or a HF tokenizer
            bos_token_id: Beginning of sequence token ID
            eos_token_id: End of sequence token ID
            pad_token_id: Padding token ID
            unk_token_id: Unknown token ID
        """
        if not _MLX_DATA_AVAILABLE:
            raise ImportError(
                "Fast backend requires mlx-data. Install with: pip install chuk-lazarus[fast]"
            )

        self._vocab: Vocab = {}
        self._id_to_token: dict[int, str] = {}
        self._trie: CharTrieType | None = None
        self._tokenizer: MlxTokenizerType | None = None

        # Special token IDs
        self._bos_token_id = bos_token_id
        self._eos_token_id = eos_token_id
        self._pad_token_id = pad_token_id
        self._unk_token_id = unk_token_id

        if vocab_or_tokenizer is not None:
            if isinstance(vocab_or_tokenizer, dict):
                self._init_from_vocab(vocab_or_tokenizer)
            elif hasattr(vocab_or_tokenizer, "get_vocab"):
                self._init_from_hf_tokenizer(vocab_or_tokenizer)
            else:
                raise TypeError(f"Expected dict or HF tokenizer, got {type(vocab_or_tokenizer)}")

    def _init_from_vocab(self, vocab: Vocab) -> None:
        """Initialize from vocabulary dict."""
        self._vocab = vocab
        self._id_to_token = {v: k for k, v in vocab.items()}

        # Build CharTrie
        assert _CharTrie is not None
        assert _Tokenizer is not None
        self._trie = _CharTrie()
        for token in vocab:
            self._trie.insert(token.encode("utf-8"))

        self._tokenizer = _Tokenizer(self._trie)

    def _init_from_hf_tokenizer(self, tokenizer: TokenizerProtocol) -> None:
        """Initialize from HuggingFace tokenizer."""
        vocab = tokenizer.get_vocab()
        self._init_from_vocab(vocab)

        # Extract special token IDs using enum
        for field in (
            SpecialTokenField.BOS_TOKEN_ID,
            SpecialTokenField.EOS_TOKEN_ID,
            SpecialTokenField.PAD_TOKEN_ID,
            SpecialTokenField.UNK_TOKEN_ID,
        ):
            current = getattr(self, f"_{field.value}")
            if current is None:
                setattr(self, f"_{field.value}", getattr(tokenizer, field.value, None))

    @classmethod
    def from_tokenizer(cls, tokenizer: TokenizerProtocol) -> "FastBackend":
        """Create FastBackend from a HuggingFace tokenizer."""
        return cls(tokenizer)

    @classmethod
    def from_vocab_file(cls, path: str | Path) -> "FastBackend":
        """
        Create FastBackend from a vocabulary file (one token per line).

        Args:
            path: Path to vocabulary file

        Returns:
            FastBackend instance
        """
        path = Path(path)
        vocab: Vocab = {}

        with open(path, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                token = line.rstrip("\n")
                if token:
                    vocab[token] = idx

        return cls(vocab)

    @classmethod
    async def from_vocab_file_async(cls, path: str | Path) -> "FastBackend":
        """
        Async version: Create FastBackend from a vocabulary file.

        Args:
            path: Path to vocabulary file

        Returns:
            FastBackend instance
        """
        import aiofiles

        path = Path(path)
        vocab: Vocab = {}

        async with aiofiles.open(path, encoding="utf-8") as f:
            idx = 0
            async for line in f:
                token = line.rstrip("\n")
                if token:
                    vocab[token] = idx
                    idx += 1

        return cls(vocab)

    @classmethod
    def from_sentencepiece(cls, model_path: str | Path) -> "FastBackend":
        """
        Create FastBackend from a SentencePiece model.

        Args:
            model_path: Path to .model or .spm file

        Returns:
            FastBackend instance

        Note:
            Requires sentencepiece to be installed.
        """
        try:
            import sentencepiece as spm
        except ImportError as err:
            raise ImportError("SentencePiece support requires: pip install sentencepiece") from err

        sp = spm.SentencePieceProcessor()
        sp.Load(str(model_path))

        vocab = {sp.IdToPiece(i): i for i in range(sp.GetPieceSize())}

        instance = cls(vocab)
        instance._bos_token_id = sp.bos_id()
        instance._eos_token_id = sp.eos_id()
        instance._pad_token_id = sp.pad_id()
        instance._unk_token_id = sp.unk_id()

        return instance

    @property
    def backend_type(self) -> BackendType:
        return BackendType.FAST

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @property
    def bos_token_id(self) -> int | None:
        return self._bos_token_id

    @property
    def eos_token_id(self) -> int | None:
        return self._eos_token_id

    @property
    def pad_token_id(self) -> int | None:
        return self._pad_token_id

    @property
    def unk_token_id(self) -> int | None:
        return self._unk_token_id

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        return_offsets: bool = False,
    ) -> TokenizationResult:
        """
        Encode text to token IDs using CharTrie.

        Args:
            text: Text to encode
            add_special_tokens: Whether to add BOS/EOS tokens
            return_offsets: Whether to compute character offsets (not supported)

        Returns:
            TokenizationResult with token IDs
        """
        if self._tokenizer is None or self._trie is None:
            return TokenizationResult(token_ids=[], tokens=[], offsets=[])

        # Tokenize using MLX Data
        text_bytes = text.encode("utf-8")
        mlx_result = self._tokenizer.tokenize_shortest(text_bytes)

        # Convert to token IDs
        token_ids: list[int] = []
        tokens: list[str] = []

        # Process MLX tokenization result
        for idx in mlx_result:
            if idx >= 0:
                token_bytes = self._trie.key_bytes(idx)
                token_str = token_bytes.decode("utf-8", errors="replace")
                if token_str in self._vocab:
                    token_ids.append(self._vocab[token_str])
                    tokens.append(token_str)
                elif self._unk_token_id is not None:
                    token_ids.append(self._unk_token_id)
                    tokens.append(SpecialTokenName.UNK.value)

        # Add special tokens
        if add_special_tokens:
            if self._bos_token_id is not None:
                token_ids = [self._bos_token_id] + token_ids
                tokens = [SpecialTokenName.BOS.value] + tokens
            if self._eos_token_id is not None:
                token_ids = token_ids + [self._eos_token_id]
                tokens = tokens + [SpecialTokenName.EOS.value]

        return TokenizationResult(
            token_ids=token_ids,
            tokens=tokens,
            offsets=[],  # Offsets not supported in fast mode
        )

    def encode_batch(
        self,
        texts: list[str],
        add_special_tokens: bool = True,
        num_workers: int = 1,
    ) -> BatchTokenizationResult:
        """
        Encode multiple texts in parallel.

        This is where the fast backend shines - it can tokenize
        across multiple cores without GIL contention.

        Args:
            texts: List of texts to encode
            add_special_tokens: Whether to add special tokens
            num_workers: Number of parallel workers

        Returns:
            BatchTokenizationResult with all results
        """
        if num_workers <= 1:
            return super().encode_batch(texts, add_special_tokens, num_workers)

        # Parallel encoding using ThreadPoolExecutor
        # MLX Data's CharTrie releases the GIL, so threads work efficiently
        def encode_one(text: str) -> TokenizationResult:
            return self.encode(text, add_special_tokens)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(encode_one, texts))

        total_tokens = sum(len(r.token_ids) for r in results)
        return BatchTokenizationResult(results=results, total_tokens=total_tokens)

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.

        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text
        """
        special_ids: set[int] = set()
        if skip_special_tokens:
            for tid in (self._bos_token_id, self._eos_token_id, self._pad_token_id):
                if tid is not None:
                    special_ids.add(tid)

        tokens = [
            self._id_to_token[tid]
            for tid in token_ids
            if tid not in special_ids and tid in self._id_to_token
        ]

        # Join tokens and clean up subword markers
        text = "".join(tokens)
        for marker in self.SUBWORD_MARKERS:
            text = text.replace(marker, " " if marker == "▁" else "")

        return text.strip()

    def get_vocab(self) -> Vocab:
        """Return the vocabulary."""
        return self._vocab

    def get_info(self) -> BackendInfo:
        """Return backend information."""
        return BackendInfo(
            backend_type=BackendType.FAST,
            vocab_size=self.vocab_size,
            supports_parallel=True,
            supports_offsets=False,
            is_available=_MLX_DATA_AVAILABLE,
        )
