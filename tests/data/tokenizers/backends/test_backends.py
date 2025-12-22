"""Tests for tokenizer backends."""

import tempfile
from pathlib import Path

import pytest

from chuk_lazarus.data.tokenizers.backends import (
    BackendType,
    CompatBackend,  # Backwards compatibility alias
    FastBackend,
    HuggingFaceBackend,
    TokenizerBackend,
    create_backend,
    get_best_backend,
    is_fast_backend_available,
)
from chuk_lazarus.data.tokenizers.backends.base import (
    BackendInfo,
    BatchTokenizationResult,
    TokenizationResult,
)


class MockTokenizer:
    """Mock HuggingFace-style tokenizer for testing."""

    def __init__(self):
        self.vocab = {
            "<pad>": 0,
            "<unk>": 1,
            "<s>": 2,
            "</s>": 3,
            "hello": 4,
            "world": 5,
            "test": 6,
            " ": 7,
        }
        self._id_to_token = {v: k for k, v in self.vocab.items()}

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        tokens = []
        for word in text.lower().split():
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                tokens.append(1)  # UNK

        if add_special_tokens:
            tokens = [2] + tokens + [3]  # BOS, EOS

        return tokens

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        tokens = []
        for tid in token_ids:
            if skip_special_tokens and tid in [0, 2, 3]:
                continue
            if tid in self._id_to_token:
                tokens.append(self._id_to_token[tid])
        return " ".join(tokens)

    def get_vocab(self) -> dict[str, int]:
        return self.vocab

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [self._id_to_token.get(i, "<unk>") for i in ids]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_token_id(self) -> int:
        return 0

    @property
    def unk_token_id(self) -> int:
        return 1

    @property
    def bos_token_id(self) -> int:
        return 2

    @property
    def eos_token_id(self) -> int:
        return 3


class TestBackendType:
    """Tests for BackendType enum."""

    def test_values(self):
        assert BackendType.HUGGINGFACE.value == "huggingface"
        assert BackendType.FAST.value == "fast"
        # Backwards compatibility
        assert BackendType.COMPAT.value == "huggingface"


class TestTokenizationResult:
    """Tests for TokenizationResult model."""

    def test_basic_result(self):
        result = TokenizationResult(
            token_ids=[1, 2, 3],
            tokens=["a", "b", "c"],
            offsets=[(0, 1), (1, 2), (2, 3)],
        )
        assert result.token_ids == [1, 2, 3]
        assert len(result.tokens) == 3

    def test_empty_result(self):
        result = TokenizationResult(token_ids=[])
        assert result.token_ids == []
        assert result.tokens == []
        assert result.offsets == []


class TestHuggingFaceBackend:
    """Tests for HuggingFaceBackend (and CompatBackend alias)."""

    def test_create_with_tokenizer(self):
        mock = MockTokenizer()
        backend = HuggingFaceBackend(mock)
        assert backend.backend_type == BackendType.HUGGINGFACE
        assert backend.vocab_size == 8

    def test_compat_alias_works(self):
        """Test CompatBackend alias still works."""
        mock = MockTokenizer()
        backend = CompatBackend(mock)
        assert backend.backend_type == BackendType.HUGGINGFACE
        assert isinstance(backend, HuggingFaceBackend)

    def test_encode(self):
        mock = MockTokenizer()
        backend = CompatBackend(mock)

        result = backend.encode("hello world")
        assert isinstance(result, TokenizationResult)
        assert len(result.token_ids) > 0
        assert 4 in result.token_ids  # "hello"
        assert 5 in result.token_ids  # "world"

    def test_encode_with_special_tokens(self):
        mock = MockTokenizer()
        backend = CompatBackend(mock)

        result = backend.encode("hello", add_special_tokens=True)
        assert result.token_ids[0] == 2  # BOS
        assert result.token_ids[-1] == 3  # EOS

    def test_encode_without_special_tokens(self):
        mock = MockTokenizer()
        backend = CompatBackend(mock)

        result = backend.encode("hello", add_special_tokens=False)
        assert 2 not in result.token_ids  # No BOS
        assert 3 not in result.token_ids  # No EOS

    def test_decode(self):
        mock = MockTokenizer()
        backend = CompatBackend(mock)

        decoded = backend.decode([4, 5])  # hello, world
        assert "hello" in decoded
        assert "world" in decoded

    def test_get_vocab(self):
        mock = MockTokenizer()
        backend = CompatBackend(mock)

        vocab = backend.get_vocab()
        assert len(vocab) == 8
        assert "hello" in vocab

    def test_batch_encode(self):
        mock = MockTokenizer()
        backend = CompatBackend(mock)

        result = backend.encode_batch(["hello", "world", "test"])
        assert isinstance(result, BatchTokenizationResult)
        assert len(result.results) == 3
        assert result.total_tokens > 0

    def test_special_token_properties(self):
        mock = MockTokenizer()
        backend = CompatBackend(mock)

        assert backend.pad_token_id == 0
        assert backend.unk_token_id == 1
        assert backend.bos_token_id == 2
        assert backend.eos_token_id == 3

    def test_get_info(self):
        mock = MockTokenizer()
        backend = CompatBackend(mock)

        info = backend.get_info()
        assert isinstance(info, BackendInfo)
        assert info.backend_type == BackendType.HUGGINGFACE
        assert info.vocab_size == 8
        assert info.supports_parallel is False
        assert info.is_available is True

    def test_empty_tokenizer(self):
        backend = CompatBackend(None)
        assert backend.vocab_size == 0
        result = backend.encode("hello")
        assert result.token_ids == []

    def test_decode_empty(self):
        backend = CompatBackend(None)
        decoded = backend.decode([1, 2, 3])
        assert decoded == ""

    def test_get_vocab_empty(self):
        backend = CompatBackend(None)
        vocab = backend.get_vocab()
        assert vocab == {}

    def test_special_tokens_none_tokenizer(self):
        backend = CompatBackend(None)
        assert backend.pad_token_id is None
        assert backend.unk_token_id is None
        assert backend.bos_token_id is None
        assert backend.eos_token_id is None

    def test_vocab_size_from_get_vocab(self):
        """Test vocab_size when tokenizer has no vocab_size property."""

        class MinimalTokenizer:
            def get_vocab(self):
                return {"a": 0, "b": 1, "c": 2}

            def encode(self, text, add_special_tokens=True):
                return [0, 1]

            def decode(self, ids, skip_special_tokens=True):
                return "ab"

        backend = CompatBackend(MinimalTokenizer())
        assert backend.vocab_size == 3

    def test_vocab_caching(self):
        """Test that vocab is cached after first access."""
        mock = MockTokenizer()
        backend = CompatBackend(mock)

        vocab1 = backend.get_vocab()
        vocab2 = backend.get_vocab()
        assert vocab1 is vocab2  # Same object

    def test_vocab_from_vocab_property(self):
        """Test vocab access via .vocab property instead of get_vocab()."""

        class VocabPropertyTokenizer:
            def __init__(self):
                self.vocab = {"x": 0, "y": 1}

            def encode(self, text, add_special_tokens=True):
                return [0]

            def decode(self, ids, skip_special_tokens=True):
                return "x"

        backend = CompatBackend(VocabPropertyTokenizer())
        vocab = backend.get_vocab()
        assert vocab == {"x": 0, "y": 1}

    def test_vocab_fallback_empty(self):
        """Test vocab fallback when tokenizer has neither get_vocab nor vocab."""

        class NoVocabTokenizer:
            def encode(self, text, add_special_tokens=True):
                return [0]

            def decode(self, ids, skip_special_tokens=True):
                return "x"

            @property
            def vocab_size(self):
                return 1

        backend = CompatBackend(NoVocabTokenizer())
        vocab = backend.get_vocab()
        assert vocab == {}

    def test_apply_chat_template_with_tokenizer(self):
        """Test chat template application."""

        class ChatTemplateTokenizer:
            def encode(self, text, add_special_tokens=True):
                return [1, 2, 3]

            def decode(self, ids, skip_special_tokens=True):
                return "test"

            def get_vocab(self):
                return {"a": 0}

            def apply_chat_template(self, messages, **kwargs):
                return f"User: {messages[0]['content']}\nAssistant:"

        backend = CompatBackend(ChatTemplateTokenizer())
        result = backend.apply_chat_template(
            [{"role": "user", "content": "Hello"}],
            add_generation_prompt=True,
        )
        assert "User:" in result
        assert "Hello" in result

    def test_apply_chat_template_fallback(self):
        """Test chat template fallback when tokenizer doesn't support it."""
        mock = MockTokenizer()
        backend = CompatBackend(mock)

        result = backend.apply_chat_template(
            [{"role": "user", "content": "Hello"}],
            add_generation_prompt=True,
        )
        assert "<|user|>" in result
        assert "Hello" in result
        assert "<|assistant|>" in result

    def test_apply_chat_template_no_generation_prompt(self):
        """Test chat template without generation prompt."""
        mock = MockTokenizer()
        backend = CompatBackend(mock)

        result = backend.apply_chat_template(
            [{"role": "user", "content": "Hello"}],
            add_generation_prompt=False,
        )
        assert "<|user|>" in result
        assert "<|assistant|>" not in result

    def test_apply_chat_template_tokenized(self):
        """Test chat template returning tokens."""
        mock = MockTokenizer()
        backend = CompatBackend(mock)

        result = backend.apply_chat_template(
            [{"role": "user", "content": "Hello"}],
            tokenize=True,
        )
        assert isinstance(result, list)
        assert all(isinstance(t, int) for t in result)

    def test_apply_chat_template_with_tools(self):
        """Test chat template with tools."""

        class ToolsTokenizer:
            def encode(self, text, add_special_tokens=True):
                return [1]

            def decode(self, ids, skip_special_tokens=True):
                return ""

            def get_vocab(self):
                return {}

            def apply_chat_template(self, messages, tools=None, **kwargs):
                if tools:
                    return f"Tools: {len(tools)}"
                return "No tools"

        backend = CompatBackend(ToolsTokenizer())
        result = backend.apply_chat_template(
            [{"role": "user", "content": "Hello"}],
            tools=[{"name": "test"}],
        )
        assert "Tools: 1" in result

    def test_apply_chat_template_empty_tokenizer(self):
        """Test chat template with None tokenizer."""
        backend = CompatBackend(None)
        result = backend.apply_chat_template([{"role": "user", "content": "Hello"}])
        assert result == ""

    def test_apply_chat_template_tokenized_empty(self):
        """Test tokenized chat template with None tokenizer."""
        backend = CompatBackend(None)
        result = backend.apply_chat_template(
            [{"role": "user", "content": "Hello"}],
            tokenize=True,
        )
        assert result == []

    def test_encode_with_convert_ids_to_tokens(self):
        """Test encode returns token strings."""
        mock = MockTokenizer()
        backend = CompatBackend(mock)

        result = backend.encode("hello world")
        assert len(result.tokens) > 0
        assert "<s>" in result.tokens  # BOS token

    def test_encode_without_convert_ids_to_tokens(self):
        """Test encode handles tokenizer without convert_ids_to_tokens."""

        class MinimalEncoderTokenizer:
            def encode(self, text, add_special_tokens=True):
                return [0, 1, 2]

            def decode(self, ids, skip_special_tokens=True):
                return "hello"

            def get_vocab(self):
                return {"a": 0, "b": 1, "c": 2}

        backend = CompatBackend(MinimalEncoderTokenizer())
        result = backend.encode("hello")
        assert result.token_ids == [0, 1, 2]
        assert result.tokens == []  # No convert_ids_to_tokens available

    def test_tokenizer_property(self):
        """Test access to underlying tokenizer."""
        mock = MockTokenizer()
        backend = CompatBackend(mock)

        assert backend.tokenizer is mock

    def test_encode_with_offsets(self):
        """Test encode with offset mapping request."""

        class OffsetsTokenizer:
            def __init__(self):
                self.vocab = {"hello": 0, "world": 1}

            def encode(self, text, add_special_tokens=True):
                return [0, 1]

            def decode(self, ids, skip_special_tokens=True):
                return "hello world"

            def get_vocab(self):
                return self.vocab

            def convert_ids_to_tokens(self, ids):
                return ["hello", "world"]

            def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False):
                result = {"input_ids": [0, 1]}
                if return_offsets_mapping:
                    result["offset_mapping"] = [(0, 5), (6, 11)]
                return result

        backend = CompatBackend(OffsetsTokenizer())
        result = backend.encode("hello world", return_offsets=True)
        assert len(result.offsets) == 2
        assert result.offsets[0] == (0, 5)
        assert result.offsets[1] == (6, 11)

    def test_encode_with_offsets_none_values(self):
        """Test encode handles None values in offset mapping."""

        class OffsetsWithNoneTokenizer:
            def __init__(self):
                self.vocab = {"<s>": 0, "hello": 1, "</s>": 2}

            def encode(self, text, add_special_tokens=True):
                return [0, 1, 2]

            def decode(self, ids, skip_special_tokens=True):
                return "hello"

            def get_vocab(self):
                return self.vocab

            def convert_ids_to_tokens(self, ids):
                return ["<s>", "hello", "</s>"]

            def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False):
                result = {"input_ids": [0, 1, 2]}
                if return_offsets_mapping:
                    # Special tokens have None offsets
                    result["offset_mapping"] = [(None, None), (0, 5), (None, None)]
                return result

        backend = CompatBackend(OffsetsWithNoneTokenizer())
        result = backend.encode("hello", return_offsets=True)
        # Only non-None offsets should be included
        assert len(result.offsets) == 1
        assert result.offsets[0] == (0, 5)

    def test_encode_offsets_not_supported(self):
        """Test encode handles tokenizers that don't support offsets."""
        mock = MockTokenizer()
        backend = CompatBackend(mock)

        # MockTokenizer is not callable with return_offsets_mapping
        result = backend.encode("hello", return_offsets=True)
        # Should still work but with empty offsets
        assert result.offsets == []

    def test_encode_offsets_exception_handling(self):
        """Test encode handles exceptions when getting offsets."""

        class FailingOffsetsTokenizer:
            def __init__(self):
                self.vocab = {"hello": 0}

            def encode(self, text, add_special_tokens=True):
                return [0]

            def decode(self, ids, skip_special_tokens=True):
                return "hello"

            def get_vocab(self):
                return self.vocab

            def convert_ids_to_tokens(self, ids):
                return ["hello"]

            def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False):
                if return_offsets_mapping:
                    raise RuntimeError("Offsets not available")
                return {"input_ids": [0]}

        backend = CompatBackend(FailingOffsetsTokenizer())
        result = backend.encode("hello", return_offsets=True)
        # Should handle exception gracefully
        assert result.offsets == []

    def test_encode_offsets_no_offset_mapping_key(self):
        """Test encode handles callable tokenizer without offset_mapping in result."""

        class NoOffsetMappingTokenizer:
            def __init__(self):
                self.vocab = {"hello": 0}

            def encode(self, text, add_special_tokens=True):
                return [0]

            def decode(self, ids, skip_special_tokens=True):
                return "hello"

            def get_vocab(self):
                return self.vocab

            def convert_ids_to_tokens(self, ids):
                return ["hello"]

            def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False):
                # Returns encoding without offset_mapping key
                return {"input_ids": [0], "attention_mask": [1]}

        backend = CompatBackend(NoOffsetMappingTokenizer())
        result = backend.encode("hello", return_offsets=True)
        assert result.offsets == []


class TestBackendInfo:
    """Tests for BackendInfo model."""

    def test_backend_info_creation(self):
        info = BackendInfo(
            backend_type=BackendType.HUGGINGFACE,
            vocab_size=1000,
            supports_parallel=False,
            supports_offsets=True,
            is_available=True,
        )
        assert info.backend_type == BackendType.HUGGINGFACE
        assert info.vocab_size == 1000
        assert info.supports_parallel is False
        assert info.supports_offsets is True
        assert info.is_available is True


class TestBatchTokenizationResult:
    """Tests for BatchTokenizationResult model."""

    def test_batch_result(self):
        results = [
            TokenizationResult(token_ids=[1, 2]),
            TokenizationResult(token_ids=[3, 4, 5]),
        ]
        batch = BatchTokenizationResult(results=results, total_tokens=5)
        assert len(batch.results) == 2
        assert batch.total_tokens == 5


class TestBaseBackend:
    """Tests for BaseBackend abstract class."""

    def test_encode_batch_sequential(self):
        """Test encode_batch with num_workers=1 (sequential)."""
        mock = MockTokenizer()
        backend = CompatBackend(mock)

        result = backend.encode_batch(["hello", "world"], num_workers=1)
        assert len(result.results) == 2

    def test_encode_batch_empty_list(self):
        """Test encode_batch with empty list."""
        mock = MockTokenizer()
        backend = CompatBackend(mock)

        result = backend.encode_batch([])
        assert len(result.results) == 0
        assert result.total_tokens == 0


class TestFastBackendAvailability:
    """Tests for fast backend availability."""

    def test_availability_check(self):
        # This should return True or False without raising
        available = is_fast_backend_available()
        assert isinstance(available, bool)


class TestFastBackend:
    """Tests for FastBackend (when mlx-data not available)."""

    def test_import_error_without_mlx_data(self):
        if is_fast_backend_available():
            pytest.skip("mlx-data is available, skipping import error test")

        with pytest.raises(ImportError, match="mlx-data"):
            FastBackend({"hello": 0})


class TestCreateBackend:
    """Tests for create_backend factory."""

    def test_create_huggingface(self):
        mock = MockTokenizer()
        backend = create_backend(BackendType.HUGGINGFACE, mock)
        assert backend.backend_type == BackendType.HUGGINGFACE

    def test_create_huggingface_from_string(self):
        mock = MockTokenizer()
        backend = create_backend("huggingface", mock)
        assert backend.backend_type == BackendType.HUGGINGFACE

    def test_create_compat_alias(self):
        """Test backwards compatibility with COMPAT enum."""
        mock = MockTokenizer()
        backend = create_backend(BackendType.COMPAT, mock)
        assert backend.backend_type == BackendType.HUGGINGFACE

    def test_create_fast_without_mlx_data(self):
        if is_fast_backend_available():
            pytest.skip("mlx-data is available")

        with pytest.raises(ImportError):
            create_backend(BackendType.FAST, {"hello": 0})

    def test_invalid_backend_type(self):
        with pytest.raises(ValueError):
            create_backend("invalid", None)


class TestGetBestBackend:
    """Tests for get_best_backend."""

    def test_returns_huggingface_when_no_fast(self):
        if is_fast_backend_available():
            pytest.skip("mlx-data is available")

        mock = MockTokenizer()
        backend = get_best_backend(mock, prefer_fast=True)
        assert backend.backend_type == BackendType.HUGGINGFACE

    def test_returns_huggingface_when_not_preferred(self):
        mock = MockTokenizer()
        backend = get_best_backend(mock, prefer_fast=False)
        assert backend.backend_type == BackendType.HUGGINGFACE


# Integration tests when mlx-data IS available
class TestFastBackendIntegration:
    """Integration tests for FastBackend (only run when mlx-data available)."""

    @pytest.fixture
    def vocab(self):
        return {
            "<pad>": 0,
            "<unk>": 1,
            "<s>": 2,
            "</s>": 3,
            "hello": 4,
            "world": 5,
            "test": 6,
            "▁hello": 7,  # With subword marker
            "##world": 8,  # With WordPiece marker
        }

    @pytest.fixture
    def fast_backend(self, vocab):
        if not is_fast_backend_available():
            pytest.skip("mlx-data not available")

        return FastBackend(
            vocab,
            bos_token_id=2,
            eos_token_id=3,
            pad_token_id=0,
            unk_token_id=1,
        )

    def test_encode(self, fast_backend):
        result = fast_backend.encode("hello world")
        assert isinstance(result, TokenizationResult)
        assert len(result.token_ids) > 0

    def test_encode_without_special_tokens(self, fast_backend):
        result = fast_backend.encode("hello", add_special_tokens=False)
        assert 2 not in result.token_ids  # No BOS
        assert 3 not in result.token_ids  # No EOS

    def test_decode(self, fast_backend):
        decoded = fast_backend.decode([4, 5])
        assert isinstance(decoded, str)

    def test_decode_with_special_tokens(self, fast_backend):
        # Decode with special tokens included
        decoded = fast_backend.decode([2, 4, 5, 3], skip_special_tokens=False)
        assert isinstance(decoded, str)

    def test_decode_skip_special(self, fast_backend):
        # Decode skipping special tokens
        decoded = fast_backend.decode([2, 4, 5, 3], skip_special_tokens=True)
        assert isinstance(decoded, str)
        # Should not contain BOS/EOS markers in output

    def test_decode_with_subword_markers(self, fast_backend):
        """Test decoding handles subword markers."""
        decoded = fast_backend.decode([7, 8])  # ▁hello, ##world
        assert isinstance(decoded, str)

    def test_batch_parallel(self, fast_backend):
        texts = ["hello", "world"] * 10
        result = fast_backend.encode_batch(texts, num_workers=2)
        assert len(result.results) == 20

    def test_batch_sequential(self, fast_backend):
        texts = ["hello", "world"]
        result = fast_backend.encode_batch(texts, num_workers=1)
        assert len(result.results) == 2

    def test_get_info(self, fast_backend):
        info = fast_backend.get_info()
        assert info.supports_parallel is True
        assert info.is_available is True
        assert info.supports_offsets is False

    def test_get_vocab(self, fast_backend, vocab):
        result_vocab = fast_backend.get_vocab()
        assert result_vocab == vocab

    def test_vocab_size(self, fast_backend, vocab):
        assert fast_backend.vocab_size == len(vocab)

    def test_special_token_properties(self, fast_backend):
        assert fast_backend.bos_token_id == 2
        assert fast_backend.eos_token_id == 3
        assert fast_backend.pad_token_id == 0
        assert fast_backend.unk_token_id == 1

    def test_backend_type(self, fast_backend):
        assert fast_backend.backend_type == BackendType.FAST

    def test_from_tokenizer(self, vocab):
        if not is_fast_backend_available():
            pytest.skip("mlx-data not available")

        mock = MockTokenizer()
        backend = FastBackend.from_tokenizer(mock)
        assert backend.backend_type == BackendType.FAST
        assert backend.vocab_size == 8

    def test_from_tokenizer_special_tokens(self, vocab):
        if not is_fast_backend_available():
            pytest.skip("mlx-data not available")

        mock = MockTokenizer()
        backend = FastBackend.from_tokenizer(mock)
        assert backend.bos_token_id == 2
        assert backend.eos_token_id == 3

    def test_from_vocab_file(self, vocab):
        if not is_fast_backend_available():
            pytest.skip("mlx-data not available")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for token in vocab.keys():
                f.write(f"{token}\n")
            f.flush()

            backend = FastBackend.from_vocab_file(f.name)
            assert backend.vocab_size == len(vocab)

    def test_from_vocab_file_path(self, vocab):
        if not is_fast_backend_available():
            pytest.skip("mlx-data not available")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for token in vocab.keys():
                f.write(f"{token}\n")
            f.flush()

            # Test with Path object
            backend = FastBackend.from_vocab_file(Path(f.name))
            assert backend.vocab_size == len(vocab)

    def test_encode_empty_tokenizer(self):
        if not is_fast_backend_available():
            pytest.skip("mlx-data not available")

        backend = FastBackend({})
        backend._tokenizer = None
        result = backend.encode("hello")
        assert result.token_ids == []

    def test_invalid_input_type(self):
        if not is_fast_backend_available():
            pytest.skip("mlx-data not available")

        with pytest.raises(TypeError, match="Expected dict or HF tokenizer"):
            FastBackend("invalid")

    def test_get_best_backend_prefers_fast(self):
        if not is_fast_backend_available():
            pytest.skip("mlx-data not available")

        mock = MockTokenizer()
        backend = get_best_backend(mock, prefer_fast=True)
        assert backend.backend_type == BackendType.FAST

    def test_create_backend_fast(self, vocab):
        if not is_fast_backend_available():
            pytest.skip("mlx-data not available")

        backend = create_backend(BackendType.FAST, vocab)
        assert backend.backend_type == BackendType.FAST


class TestTokenizerBackendProtocol:
    """Tests for TokenizerBackend protocol compliance."""

    def test_huggingface_backend_is_tokenizer_backend(self):
        mock = MockTokenizer()
        backend = HuggingFaceBackend(mock)
        assert isinstance(backend, TokenizerBackend)

    def test_fast_backend_is_tokenizer_backend(self):
        if not is_fast_backend_available():
            pytest.skip("mlx-data not available")

        vocab = {"hello": 0, "world": 1}
        backend = FastBackend(vocab)
        assert isinstance(backend, TokenizerBackend)
