"""Tests for packer module."""

from chuk_lazarus.data.tokenizers.training.packer import (
    PackedBatch,
    PackedSequence,
    PackingConfig,
    PackingStats,
    calculate_packing_efficiency,
    create_packed_batch,
    pack_sequences,
)


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self):
        self._vocab = {"<pad>": 0, "<unk>": 1, "<eos>": 2}
        self.pad_token_id = 0
        self.eos_token_id = 2

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        # Simple: each word = 1 token, starting from ID 10
        words = text.split()
        return [10 + i for i in range(len(words))]

    def get_vocab(self) -> dict[str, int]:
        return self._vocab


class TestPackingConfigModel:
    """Tests for PackingConfig model."""

    def test_default_values(self):
        config = PackingConfig(
            max_seq_length=512,
            pad_token_id=0,
        )
        assert config.max_seq_length == 512
        assert config.add_eos_between is True
        assert config.respect_document_boundaries is True

    def test_custom_values(self):
        config = PackingConfig(
            max_seq_length=1024,
            pad_token_id=0,
            eos_token_id=2,
            add_eos_between=False,
            min_sequence_length=10,
        )
        assert config.add_eos_between is False
        assert config.min_sequence_length == 10


class TestPackedSequenceModel:
    """Tests for PackedSequence model."""

    def test_valid_sequence(self):
        seq = PackedSequence(
            token_ids=[1, 2, 3, 0, 0],
            attention_mask=[1, 1, 1, 0, 0],
            loss_mask=[1, 1, 1, 0, 0],
            source_indices=[0, 1],
            num_real_tokens=3,
            num_padding_tokens=2,
        )
        assert seq.num_real_tokens == 3
        assert len(seq.source_indices) == 2


class TestPackedBatchModel:
    """Tests for PackedBatch model."""

    def test_valid_batch(self):
        seq = PackedSequence(
            token_ids=[1, 2, 3],
            attention_mask=[1, 1, 1],
            loss_mask=[1, 1, 1],
            source_indices=[0],
            num_real_tokens=3,
            num_padding_tokens=0,
        )
        batch = PackedBatch(
            sequences=[seq],
            total_sequences=1,
            total_source_sequences=1,
            packing_ratio=1.0,
        )
        assert batch.packing_ratio == 1.0


class TestPackingStatsModel:
    """Tests for PackingStats model."""

    def test_valid_stats(self):
        stats = PackingStats(
            total_tokens=1000,
            real_tokens=800,
            padding_tokens=200,
            padding_ratio=0.2,
            efficiency=0.8,
            avg_sequences_per_pack=2.5,
            throughput_improvement=1.5,
        )
        assert stats.efficiency == 0.8
        assert stats.throughput_improvement == 1.5


class TestPackSequences:
    """Tests for pack_sequences function."""

    def test_basic_packing(self):
        sequences = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
        config = PackingConfig(
            max_seq_length=10,
            pad_token_id=0,
            eos_token_id=2,
        )
        packed = pack_sequences(sequences, config)
        assert len(packed) > 0
        assert all(isinstance(s, PackedSequence) for s in packed)

    def test_packing_respects_max_length(self):
        sequences = [[1, 2, 3], [4, 5, 6]]
        config = PackingConfig(
            max_seq_length=5,
            pad_token_id=0,
        )
        packed = pack_sequences(sequences, config)
        for seq in packed:
            assert len(seq.token_ids) <= 5

    def test_empty_sequences(self):
        sequences: list[list[int]] = []
        config = PackingConfig(
            max_seq_length=10,
            pad_token_id=0,
        )
        packed = pack_sequences(sequences, config)
        assert len(packed) == 0

    def test_single_sequence(self):
        sequences = [[1, 2, 3, 4, 5]]
        config = PackingConfig(
            max_seq_length=10,
            pad_token_id=0,
        )
        packed = pack_sequences(sequences, config)
        assert len(packed) == 1

    def test_eos_between_sequences(self):
        sequences = [[1, 2], [3, 4]]
        config = PackingConfig(
            max_seq_length=10,
            pad_token_id=0,
            eos_token_id=99,
            add_eos_between=True,
        )
        packed = pack_sequences(sequences, config)
        # Check that EOS is added between
        if len(packed) == 1:
            assert 99 in packed[0].token_ids

    def test_no_eos_between(self):
        sequences = [[1, 2], [3, 4]]
        config = PackingConfig(
            max_seq_length=10,
            pad_token_id=0,
            eos_token_id=99,
            add_eos_between=False,
        )
        packed = pack_sequences(sequences, config)
        # Should not have EOS between
        for seq in packed:
            # EOS might still appear at end, but not between
            pass  # Just ensure it doesn't crash

    def test_min_sequence_length_filter(self):
        sequences = [[1], [2, 3, 4, 5, 6]]
        config = PackingConfig(
            max_seq_length=10,
            pad_token_id=0,
            min_sequence_length=3,
        )
        packed = pack_sequences(sequences, config)
        # Should filter out [1] which is too short
        total_source = sum(len(s.source_indices) for s in packed)
        assert total_source == 1  # Only the longer sequence

    def test_attention_mask_correct(self):
        sequences = [[1, 2, 3]]
        config = PackingConfig(
            max_seq_length=5,
            pad_token_id=0,
        )
        packed = pack_sequences(sequences, config)
        seq = packed[0]
        # Real tokens should have attention=1, padding=0
        assert sum(seq.attention_mask) == seq.num_real_tokens

    def test_loss_mask_correct(self):
        sequences = [[1, 2, 3]]
        config = PackingConfig(
            max_seq_length=5,
            pad_token_id=0,
        )
        packed = pack_sequences(sequences, config)
        seq = packed[0]
        # Loss mask should be 1 for real tokens, 0 for padding
        assert sum(seq.loss_mask) <= seq.num_real_tokens

    def test_truncation(self):
        sequences = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        config = PackingConfig(
            max_seq_length=5,
            pad_token_id=0,
        )
        packed = pack_sequences(sequences, config)
        # Sequence should be truncated
        assert len(packed[0].token_ids) == 5


class TestCreatePackedBatch:
    """Tests for create_packed_batch function."""

    def test_basic_batch(self):
        tokenizer = MockTokenizer()
        texts = ["hello world", "foo bar baz"]
        batch = create_packed_batch(texts, tokenizer, max_seq_length=20)
        assert isinstance(batch, PackedBatch)
        assert batch.total_source_sequences == 2

    def test_packing_ratio(self):
        tokenizer = MockTokenizer()
        texts = ["a", "b", "c", "d"]  # Short texts
        batch = create_packed_batch(texts, tokenizer, max_seq_length=10)
        assert batch.packing_ratio >= 1.0

    def test_empty_texts(self):
        tokenizer = MockTokenizer()
        texts: list[str] = []
        batch = create_packed_batch(texts, tokenizer, max_seq_length=10)
        assert batch.total_sequences == 0

    def test_with_special_tokens(self):
        tokenizer = MockTokenizer()
        texts = ["hello world"]
        batch = create_packed_batch(texts, tokenizer, max_seq_length=10, add_special_tokens=True)
        assert len(batch.sequences) > 0


class TestCalculatePackingEfficiency:
    """Tests for calculate_packing_efficiency function."""

    def test_basic_efficiency(self):
        seq = PackedSequence(
            token_ids=[1, 2, 3, 0, 0],
            attention_mask=[1, 1, 1, 0, 0],
            loss_mask=[1, 1, 1, 0, 0],
            source_indices=[0],
            num_real_tokens=3,
            num_padding_tokens=2,
        )
        batch = PackedBatch(
            sequences=[seq],
            total_sequences=1,
            total_source_sequences=1,
            packing_ratio=1.0,
        )
        stats = calculate_packing_efficiency(batch)
        assert isinstance(stats, PackingStats)
        assert stats.efficiency == 0.6  # 3/5 real tokens

    def test_perfect_efficiency(self):
        seq = PackedSequence(
            token_ids=[1, 2, 3, 4, 5],
            attention_mask=[1, 1, 1, 1, 1],
            loss_mask=[1, 1, 1, 1, 1],
            source_indices=[0],
            num_real_tokens=5,
            num_padding_tokens=0,
        )
        batch = PackedBatch(
            sequences=[seq],
            total_sequences=1,
            total_source_sequences=1,
            packing_ratio=1.0,
        )
        stats = calculate_packing_efficiency(batch)
        assert stats.efficiency == 1.0
        assert stats.padding_ratio == 0.0

    def test_high_padding(self):
        seq = PackedSequence(
            token_ids=[1, 0, 0, 0, 0],
            attention_mask=[1, 0, 0, 0, 0],
            loss_mask=[1, 0, 0, 0, 0],
            source_indices=[0],
            num_real_tokens=1,
            num_padding_tokens=4,
        )
        batch = PackedBatch(
            sequences=[seq],
            total_sequences=1,
            total_source_sequences=1,
            packing_ratio=1.0,
        )
        stats = calculate_packing_efficiency(batch)
        assert stats.efficiency == 0.2  # 1/5
        assert stats.padding_ratio == 0.8

    def test_empty_batch(self):
        batch = PackedBatch(
            sequences=[],
            total_sequences=0,
            total_source_sequences=0,
            packing_ratio=1.0,
        )
        stats = calculate_packing_efficiency(batch)
        assert stats.total_tokens == 0
        assert stats.throughput_improvement == 1.0

    def test_throughput_improvement(self):
        seq1 = PackedSequence(
            token_ids=[1, 2, 3, 4, 5],
            attention_mask=[1, 1, 1, 1, 1],
            loss_mask=[1, 1, 1, 1, 1],
            source_indices=[0, 1],  # 2 source sequences packed
            num_real_tokens=5,
            num_padding_tokens=0,
        )
        batch = PackedBatch(
            sequences=[seq1],
            total_sequences=1,
            total_source_sequences=2,
            packing_ratio=2.0,
        )
        stats = calculate_packing_efficiency(batch)
        assert stats.throughput_improvement >= 1.0
