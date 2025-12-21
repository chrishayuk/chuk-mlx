"""Tests for batch_processing module."""

from chuk_lazarus.data.tokenizers.batch_processing import (
    BatchResult,
    ChunkConfig,
    PaddingSide,
    SequenceStats,
    chunk_text,
    create_batch,
    decode_batch,
    encode_batch,
    get_sequence_lengths,
    pad_batch,
)


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self):
        self._vocab = {"hello": 0, "world": 1, "<unk>": 2, "the": 3, "<pad>": 4}
        self.pad_token_id = 4

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        tokens = text.lower().split()
        return [self._vocab.get(t, self._vocab.get("<unk>", 2)) for t in tokens]

    def decode(self, ids: list[int]) -> str:
        id_to_token = {v: k for k, v in self._vocab.items()}
        return " ".join(id_to_token.get(i, "<unk>") for i in ids)

    def get_vocab(self) -> dict[str, int]:
        return self._vocab


class TestPaddingSideEnum:
    """Tests for PaddingSide enum."""

    def test_left_value(self):
        assert PaddingSide.LEFT == "left"
        assert PaddingSide.LEFT.value == "left"

    def test_right_value(self):
        assert PaddingSide.RIGHT == "right"
        assert PaddingSide.RIGHT.value == "right"


class TestBatchResultModel:
    """Tests for BatchResult Pydantic model."""

    def test_with_attention_mask(self):
        result = BatchResult(
            input_ids=[[1, 2, 3], [4, 5, 6]],
            attention_mask=[[1, 1, 1], [1, 1, 0]],
        )
        assert len(result.input_ids) == 2
        assert result.attention_mask is not None

    def test_without_attention_mask(self):
        result = BatchResult(input_ids=[[1, 2, 3]])
        assert result.attention_mask is None


class TestSequenceStatsModel:
    """Tests for SequenceStats Pydantic model."""

    def test_valid_stats(self):
        stats = SequenceStats(
            min_length=5,
            max_length=20,
            mean_length=12.5,
            total_tokens=100,
            count=8,
        )
        assert stats.min_length == 5
        assert stats.mean_length == 12.5


class TestChunkConfigModel:
    """Tests for ChunkConfig Pydantic model."""

    def test_valid_config(self):
        config = ChunkConfig(chunk_size=128, overlap=16, add_special_tokens=True)
        assert config.chunk_size == 128
        assert config.overlap == 16

    def test_defaults(self):
        config = ChunkConfig(chunk_size=64)
        assert config.overlap == 0
        assert config.add_special_tokens is True


class TestEncodeBatch:
    """Tests for encode_batch function."""

    def test_encode_multiple_texts(self):
        tokenizer = MockTokenizer()
        texts = ["hello world", "the world"]
        encoded = encode_batch(texts, tokenizer)
        assert len(encoded) == 2
        assert encoded[0] == [0, 1]  # hello, world
        assert encoded[1] == [3, 1]  # the, world

    def test_encode_empty_list(self):
        tokenizer = MockTokenizer()
        encoded = encode_batch([], tokenizer)
        assert len(encoded) == 0

    def test_encode_with_special_tokens(self):
        tokenizer = MockTokenizer()
        encoded = encode_batch(["hello"], tokenizer, add_special_tokens=True)
        assert len(encoded) == 1


class TestDecodeBatch:
    """Tests for decode_batch function."""

    def test_decode_multiple(self):
        tokenizer = MockTokenizer()
        token_ids = [[0, 1], [3, 1]]
        decoded = decode_batch(token_ids, tokenizer)
        assert len(decoded) == 2
        assert "hello" in decoded[0]
        assert "world" in decoded[1]

    def test_decode_empty_list(self):
        tokenizer = MockTokenizer()
        decoded = decode_batch([], tokenizer)
        assert len(decoded) == 0


class TestPadBatch:
    """Tests for pad_batch function."""

    def test_right_padding(self):
        sequences = [[1, 2], [1, 2, 3, 4]]
        result = pad_batch(sequences, pad_token_id=0, padding_side=PaddingSide.RIGHT)
        assert result.input_ids[0] == [1, 2, 0, 0]
        assert result.input_ids[1] == [1, 2, 3, 4]
        assert result.attention_mask == [[1, 1, 0, 0], [1, 1, 1, 1]]

    def test_left_padding(self):
        sequences = [[1, 2], [1, 2, 3, 4]]
        result = pad_batch(sequences, pad_token_id=0, padding_side=PaddingSide.LEFT)
        assert result.input_ids[0] == [0, 0, 1, 2]
        assert result.input_ids[1] == [1, 2, 3, 4]
        assert result.attention_mask == [[0, 0, 1, 1], [1, 1, 1, 1]]

    def test_with_max_length(self):
        sequences = [[1, 2], [1, 2, 3]]
        result = pad_batch(sequences, pad_token_id=0, max_length=5)
        assert len(result.input_ids[0]) == 5
        assert len(result.input_ids[1]) == 5

    def test_truncation(self):
        sequences = [[1, 2, 3, 4, 5]]
        result = pad_batch(sequences, pad_token_id=0, max_length=3, truncate=True)
        assert result.input_ids[0] == [1, 2, 3]

    def test_no_attention_mask(self):
        sequences = [[1, 2], [3, 4]]
        result = pad_batch(sequences, pad_token_id=0, return_attention_mask=False)
        assert result.attention_mask is None

    def test_empty_sequences(self):
        result = pad_batch([], pad_token_id=0)
        assert result.input_ids == []
        assert result.attention_mask == []


class TestCreateBatch:
    """Tests for create_batch function."""

    def test_basic_batch(self):
        tokenizer = MockTokenizer()
        texts = ["hello world", "the world hello"]
        result = create_batch(texts, tokenizer)
        assert len(result.input_ids) == 2

    def test_batch_with_padding(self):
        tokenizer = MockTokenizer()
        texts = ["hello", "hello world"]
        result = create_batch(texts, tokenizer, padding=True)
        assert len(result.input_ids[0]) == len(result.input_ids[1])

    def test_batch_without_padding(self):
        tokenizer = MockTokenizer()
        texts = ["hello", "hello world"]
        result = create_batch(texts, tokenizer, padding=False)
        assert len(result.input_ids[0]) != len(result.input_ids[1])

    def test_batch_with_max_length(self):
        tokenizer = MockTokenizer()
        texts = ["hello world the world"]
        result = create_batch(texts, tokenizer, max_length=2, truncation=True)
        assert len(result.input_ids[0]) == 2

    def test_batch_left_padding(self):
        tokenizer = MockTokenizer()
        texts = ["hello", "hello world"]
        result = create_batch(texts, tokenizer, padding_side=PaddingSide.LEFT)
        # Left padding: shorter sequence should have pad at start
        assert result.input_ids[0][0] == tokenizer.pad_token_id


class TestChunkText:
    """Tests for chunk_text function."""

    def test_short_text_no_chunking(self):
        tokenizer = MockTokenizer()
        config = ChunkConfig(chunk_size=10, overlap=0, add_special_tokens=False)
        chunks = chunk_text("hello world", tokenizer, config)
        assert len(chunks) == 1
        assert chunks[0] == [0, 1]

    def test_long_text_chunking(self):
        tokenizer = MockTokenizer()
        config = ChunkConfig(chunk_size=2, overlap=0, add_special_tokens=False)
        chunks = chunk_text("hello world the world", tokenizer, config)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk) <= 2

    def test_chunking_with_overlap(self):
        tokenizer = MockTokenizer()
        config = ChunkConfig(chunk_size=2, overlap=1, add_special_tokens=False)
        chunks = chunk_text("hello world the world", tokenizer, config)
        assert len(chunks) >= 2


class TestGetSequenceLengths:
    """Tests for get_sequence_lengths function."""

    def test_basic_stats(self):
        sequences = [[1, 2], [1, 2, 3], [1, 2, 3, 4, 5]]
        stats = get_sequence_lengths(sequences)
        assert stats.min_length == 2
        assert stats.max_length == 5
        assert stats.mean_length == (2 + 3 + 5) / 3
        assert stats.total_tokens == 10
        assert stats.count == 3

    def test_empty_sequences(self):
        stats = get_sequence_lengths([])
        assert stats.min_length == 0
        assert stats.max_length == 0
        assert stats.mean_length == 0.0
        assert stats.count == 0

    def test_single_sequence(self):
        stats = get_sequence_lengths([[1, 2, 3]])
        assert stats.min_length == 3
        assert stats.max_length == 3
        assert stats.mean_length == 3.0
        assert stats.count == 1
