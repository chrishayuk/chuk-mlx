"""Tests for dynamic_vocab module."""

from chuk_lazarus.data.tokenizers.runtime.dynamic_vocab import (
    DynamicVocab,
    VocabExtension,
    create_embedding_slot,
    extend_vocab_runtime,
)


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self, vocab: dict[str, int] | None = None):
        self._vocab = vocab or {
            "<pad>": 0,
            "<unk>": 1,
            "hello": 2,
            "world": 3,
        }

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        words = text.lower().split()
        return [self._vocab.get(w, 1) for w in words]

    def decode(self, ids: list[int]) -> str:
        id_to_token = {v: k for k, v in self._vocab.items()}
        return " ".join(id_to_token.get(i, "<unk>") for i in ids)

    def get_vocab(self) -> dict[str, int]:
        return self._vocab.copy()

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)


class TestVocabExtensionModel:
    """Tests for VocabExtension model."""

    def test_valid_extension(self):
        ext = VocabExtension(
            token_str="<TOOL>",
            token_id=100,
            original_tokens=[1, 2, 3],
        )
        assert ext.token_id == 100
        assert ext.token_str == "<TOOL>"

    def test_default_values(self):
        ext = VocabExtension(
            token_str="<NEW>",
            token_id=50,
        )
        assert ext.embedding_initialized is False
        assert ext.init_method == "mean"


class TestDynamicVocabModel:
    """Tests for DynamicVocab model."""

    def test_valid_vocab(self):
        vocab = DynamicVocab(
            base_vocab_size=1000,
            next_id=1000,
        )
        assert vocab.base_vocab_size == 1000
        assert vocab.next_id == 1000

    def test_from_tokenizer(self):
        tokenizer = MockTokenizer()
        vocab = DynamicVocab.from_tokenizer(tokenizer)
        assert vocab.base_vocab_size == 4  # 4 tokens in mock
        assert vocab.next_id == 4

    def test_add_token(self):
        tokenizer = MockTokenizer()
        vocab = DynamicVocab.from_tokenizer(tokenizer)
        ext = vocab.add_token("<NEW>", tokenizer)
        assert ext.token_str == "<NEW>"
        assert ext.token_id == 4

    def test_add_token_increments_id(self):
        tokenizer = MockTokenizer()
        vocab = DynamicVocab.from_tokenizer(tokenizer)
        vocab.add_token("<A>", tokenizer)
        vocab.add_token("<B>", tokenizer)
        assert vocab.next_id == 6

    def test_add_duplicate_raises(self):
        tokenizer = MockTokenizer()
        vocab = DynamicVocab.from_tokenizer(tokenizer)
        vocab.add_token("<NEW>", tokenizer)
        try:
            vocab.add_token("<NEW>", tokenizer)
            raise AssertionError("Should raise ValueError")
        except ValueError as e:
            assert "already added" in str(e)

    def test_add_existing_token_raises(self):
        tokenizer = MockTokenizer()
        vocab = DynamicVocab.from_tokenizer(tokenizer)
        try:
            vocab.add_token("hello", tokenizer)  # Already in base vocab
            raise AssertionError("Should raise ValueError")
        except ValueError as e:
            assert "already in base" in str(e)

    def test_get_all_tokens(self):
        tokenizer = MockTokenizer()
        vocab = DynamicVocab.from_tokenizer(tokenizer)
        vocab.add_token("<A>", tokenizer)
        vocab.add_token("<B>", tokenizer)
        tokens = vocab.get_all_tokens()
        assert len(tokens) == 2

    def test_get_token_id(self):
        tokenizer = MockTokenizer()
        vocab = DynamicVocab.from_tokenizer(tokenizer)
        vocab.add_token("<NEW>", tokenizer)
        assert vocab.get_token_id("<NEW>") == 4
        assert vocab.get_token_id("<NOT_EXIST>") is None

    def test_total_vocab_size(self):
        tokenizer = MockTokenizer()
        vocab = DynamicVocab.from_tokenizer(tokenizer)
        assert vocab.total_vocab_size == 4
        vocab.add_token("<NEW>", tokenizer)
        assert vocab.total_vocab_size == 5


class TestExtendVocabRuntime:
    """Tests for extend_vocab_runtime function."""

    def test_basic_extension(self):
        tokenizer = MockTokenizer()
        vocab = DynamicVocab.from_tokenizer(tokenizer)
        extensions = extend_vocab_runtime(vocab, ["<A>", "<B>"], tokenizer)
        assert len(extensions) == 2

    def test_skips_existing(self):
        tokenizer = MockTokenizer()
        vocab = DynamicVocab.from_tokenizer(tokenizer)
        vocab.add_token("<A>", tokenizer)
        extensions = extend_vocab_runtime(vocab, ["<A>", "<B>"], tokenizer)
        # Should skip <A> which already exists
        assert len(extensions) == 1
        assert extensions[0].token_str == "<B>"

    def test_skips_base_vocab(self):
        tokenizer = MockTokenizer()
        vocab = DynamicVocab.from_tokenizer(tokenizer)
        extensions = extend_vocab_runtime(vocab, ["hello", "<NEW>"], tokenizer)
        # Should skip "hello" which is in base vocab
        assert len(extensions) == 1
        assert extensions[0].token_str == "<NEW>"

    def test_empty_list(self):
        tokenizer = MockTokenizer()
        vocab = DynamicVocab.from_tokenizer(tokenizer)
        extensions = extend_vocab_runtime(vocab, [], tokenizer)
        assert len(extensions) == 0

    def test_custom_init_method(self):
        tokenizer = MockTokenizer()
        vocab = DynamicVocab.from_tokenizer(tokenizer)
        extensions = extend_vocab_runtime(vocab, ["<NEW>"], tokenizer, init_method="zero")
        assert extensions[0].init_method == "zero"


class TestCreateEmbeddingSlot:
    """Tests for create_embedding_slot function."""

    def test_basic_slot(self):
        ext = VocabExtension(token_str="<TOOL>", token_id=100)
        slot = create_embedding_slot(ext, embedding_dim=768)
        assert len(slot) == 768

    def test_slot_values(self):
        ext = VocabExtension(token_str="<TOOL>", token_id=100)
        slot = create_embedding_slot(ext, embedding_dim=128)
        # Values should be small (initialized)
        assert all(abs(v) < 1.0 for v in slot)

    def test_different_dims(self):
        ext = VocabExtension(token_str="<A>", token_id=0)
        slot_256 = create_embedding_slot(ext, embedding_dim=256)
        slot_512 = create_embedding_slot(ext, embedding_dim=512)
        assert len(slot_256) == 256
        assert len(slot_512) == 512

    def test_zero_init(self):
        ext = VocabExtension(token_str="<A>", token_id=0, init_method="zero")
        slot = create_embedding_slot(ext, embedding_dim=64)
        assert all(v == 0.0 for v in slot)

    def test_random_init(self):
        ext = VocabExtension(token_str="<A>", token_id=0, init_method="random")
        slot = create_embedding_slot(ext, embedding_dim=64)
        # Random init should have non-zero values
        assert any(v != 0.0 for v in slot)

    def test_override_init_method(self):
        ext = VocabExtension(token_str="<A>", token_id=0, init_method="random")
        slot = create_embedding_slot(ext, embedding_dim=64, init_method="zero")
        assert all(v == 0.0 for v in slot)

    def test_unknown_init_raises(self):
        ext = VocabExtension(token_str="<A>", token_id=0, init_method="unknown_method")
        try:
            create_embedding_slot(ext, embedding_dim=64)
            raise AssertionError("Should raise ValueError")
        except ValueError as e:
            assert "Unknown init method" in str(e)
