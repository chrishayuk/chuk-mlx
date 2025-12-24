"""Tests for sequence packing (Phase 3)."""

import numpy as np
import pytest
from pydantic import ValidationError

from chuk_lazarus.data.batching import (
    PackedSequence,
    PackingConfig,
    PackingMetrics,
    PackingMode,
    SequenceToPack,
    compute_packing_metrics,
    create_segment_attention_mask,
    pack_sequences,
)
from chuk_lazarus.data.batching.planning.packing import (
    PackingBin,
    pack_sequences_best_fit,
    pack_sequences_first_fit,
    pack_sequences_greedy,
)


class TestPackingMode:
    """Tests for PackingMode enum."""

    def test_values(self):
        assert PackingMode.FIRST_FIT.value == "first_fit"
        assert PackingMode.BEST_FIT.value == "best_fit"
        assert PackingMode.GREEDY.value == "greedy"

    def test_enum_members(self):
        assert len(list(PackingMode)) == 3


class TestPackingConfig:
    """Tests for PackingConfig."""

    def test_create_default(self):
        config = PackingConfig.default()
        assert config.mode == PackingMode.FIRST_FIT
        assert config.max_length == 2048
        assert config.pad_to_max is True
        assert config.add_separator is False
        assert config.separator_token_id is None

    def test_custom_max_length(self):
        config = PackingConfig.default(max_length=1024)
        assert config.max_length == 1024

    def test_with_separator(self):
        config = PackingConfig(
            max_length=512,
            add_separator=True,
            separator_token_id=2,
        )
        assert config.add_separator is True
        assert config.separator_token_id == 2

    def test_immutable(self):
        config = PackingConfig.default()
        with pytest.raises((TypeError, AttributeError, ValidationError)):
            config.max_length = 1024


class TestSequenceToPack:
    """Tests for SequenceToPack."""

    def test_create(self):
        seq = SequenceToPack(
            sample_id="s1",
            input_ids=[1, 2, 3, 4],
            loss_mask=[0, 0, 1, 1],
        )
        assert seq.sample_id == "s1"
        assert seq.input_ids == (1, 2, 3, 4)
        assert seq.loss_mask == (0, 0, 1, 1)
        assert seq.length == 4

    def test_list_to_tuple_conversion(self):
        seq = SequenceToPack(
            sample_id="s1",
            input_ids=[1, 2, 3],
            loss_mask=[1, 1, 1],
        )
        assert isinstance(seq.input_ids, tuple)
        assert isinstance(seq.loss_mask, tuple)


class TestPackedSequence:
    """Tests for PackedSequence."""

    def test_create(self):
        packed = PackedSequence(
            input_ids=[1, 2, 3, 4, 5, 0, 0, 0],
            loss_mask=[0, 1, 0, 1, 1, 0, 0, 0],
            segment_ids=[0, 0, 1, 1, 1, 1, 1, 1],
            sample_ids=["s1", "s2"],
            sample_lengths=[2, 3],
            num_segments=2,
            total_tokens=5,
            padding_tokens=3,
        )
        assert packed.length == 8
        assert packed.num_segments == 2
        assert packed.total_tokens == 5
        assert packed.padding_tokens == 3

    def test_efficiency(self):
        packed = PackedSequence(
            input_ids=[1, 2, 3, 4, 0, 0, 0, 0],
            loss_mask=[1, 1, 1, 1, 0, 0, 0, 0],
            segment_ids=[0, 0, 0, 0, 0, 0, 0, 0],
            sample_ids=["s1"],
            sample_lengths=[4],
            num_segments=1,
            total_tokens=4,
            padding_tokens=4,
        )
        assert packed.efficiency == 0.5  # 4/8

    def test_num_loss_tokens(self):
        packed = PackedSequence(
            input_ids=[1, 2, 3, 4],
            loss_mask=[0, 1, 1, 0],
            segment_ids=[0, 0, 0, 0],
            sample_ids=["s1"],
            sample_lengths=[4],
            num_segments=1,
            total_tokens=4,
            padding_tokens=0,
        )
        assert packed.num_loss_tokens == 2


class TestPackingBin:
    """Tests for PackingBin helper class."""

    def test_create_empty(self):
        bin = PackingBin(max_length=100, pad_token_id=0)
        assert bin.max_length == 100
        assert bin.remaining_space == 100
        assert bin.current_length == 0

    def test_can_fit(self):
        bin = PackingBin(max_length=100)
        seq = SequenceToPack(
            sample_id="s1",
            input_ids=[1] * 50,
            loss_mask=[1] * 50,
        )
        assert bin.can_fit(seq) is True

        # Too long
        long_seq = SequenceToPack(
            sample_id="s2",
            input_ids=[1] * 150,
            loss_mask=[1] * 150,
        )
        assert bin.can_fit(long_seq) is False

    def test_add(self):
        bin = PackingBin(max_length=100)
        seq = SequenceToPack(
            sample_id="s1",
            input_ids=[1, 2, 3],
            loss_mask=[1, 1, 1],
        )
        bin.add(seq)
        assert bin.current_length == 3
        assert bin.remaining_space == 97
        assert len(bin.sequences) == 1

    def test_to_packed_sequence(self):
        bin = PackingBin(max_length=10, pad_token_id=0)

        seq1 = SequenceToPack(
            sample_id="s1",
            input_ids=[1, 2],
            loss_mask=[0, 1],
        )
        seq2 = SequenceToPack(
            sample_id="s2",
            input_ids=[3, 4, 5],
            loss_mask=[1, 1, 0],
        )

        bin.add(seq1)
        bin.add(seq2)

        packed = bin.to_packed_sequence(pad_to_max=True)

        assert packed.length == 10
        assert packed.num_segments == 2
        assert packed.sample_ids == ("s1", "s2")
        assert packed.sample_lengths == (2, 3)
        assert packed.total_tokens == 5
        assert packed.padding_tokens == 5

        # Check segment IDs
        assert packed.segment_ids[:2] == (0, 0)  # s1
        assert packed.segment_ids[2:5] == (1, 1, 1)  # s2

    def test_to_packed_sequence_no_padding(self):
        bin = PackingBin(max_length=10)
        seq = SequenceToPack(
            sample_id="s1",
            input_ids=[1, 2, 3],
            loss_mask=[1, 1, 1],
        )
        bin.add(seq)

        packed = bin.to_packed_sequence(pad_to_max=False)
        assert packed.length == 3
        assert packed.padding_tokens == 0

    def test_to_packed_sequence_with_separator(self):
        bin = PackingBin(max_length=20, pad_token_id=0)

        seq1 = SequenceToPack(
            sample_id="s1",
            input_ids=[1, 2],
            loss_mask=[1, 1],
        )
        seq2 = SequenceToPack(
            sample_id="s2",
            input_ids=[3, 4],
            loss_mask=[1, 1],
        )

        bin.add(seq1)
        bin.add(seq2, separator_len=1)

        packed = bin.to_packed_sequence(pad_to_max=False, separator_token_id=99)

        # Should have: s1 tokens, separator, s2 tokens
        assert packed.input_ids[:2] == (1, 2)  # s1
        assert packed.input_ids[2] == 99  # separator
        assert packed.input_ids[3:5] == (3, 4)  # s2
        assert packed.loss_mask[2] == 0  # separator has no loss


class TestPackSequencesFirstFit:
    """Tests for first-fit packing algorithm."""

    def test_basic_packing(self):
        sequences = [
            SequenceToPack(sample_id="s1", input_ids=[1, 2], loss_mask=[1, 1]),
            SequenceToPack(sample_id="s2", input_ids=[3, 4, 5], loss_mask=[1, 1, 1]),
            SequenceToPack(sample_id="s3", input_ids=[6], loss_mask=[1]),
        ]
        config = PackingConfig(max_length=10, pad_to_max=True)

        packed = pack_sequences_first_fit(sequences, config)

        # All should fit in one bin
        assert len(packed) == 1
        assert packed[0].num_segments == 3
        assert packed[0].total_tokens == 6

    def test_multiple_bins(self):
        sequences = [
            SequenceToPack(sample_id="s1", input_ids=[1] * 6, loss_mask=[1] * 6),
            SequenceToPack(sample_id="s2", input_ids=[2] * 6, loss_mask=[1] * 6),
            SequenceToPack(sample_id="s3", input_ids=[3] * 6, loss_mask=[1] * 6),
        ]
        config = PackingConfig(max_length=10, pad_to_max=True)

        packed = pack_sequences_first_fit(sequences, config)

        # Each needs its own bin (6 tokens each, max 10)
        assert len(packed) == 3
        assert all(p.num_segments == 1 for p in packed)

    def test_skip_too_long(self):
        sequences = [
            SequenceToPack(sample_id="s1", input_ids=[1, 2], loss_mask=[1, 1]),
            SequenceToPack(sample_id="s2", input_ids=[1] * 20, loss_mask=[1] * 20),  # Too long
            SequenceToPack(sample_id="s3", input_ids=[3, 4], loss_mask=[1, 1]),
        ]
        config = PackingConfig(max_length=10, pad_to_max=True)

        packed = pack_sequences_first_fit(sequences, config)

        # s2 should be skipped
        assert len(packed) == 1
        assert packed[0].num_segments == 2


class TestPackSequencesBestFit:
    """Tests for best-fit packing algorithm."""

    def test_better_packing(self):
        sequences = [
            SequenceToPack(sample_id="s1", input_ids=[1] * 5, loss_mask=[1] * 5),
            SequenceToPack(sample_id="s2", input_ids=[2] * 5, loss_mask=[1] * 5),
            SequenceToPack(sample_id="s3", input_ids=[3] * 4, loss_mask=[1] * 4),
            SequenceToPack(sample_id="s4", input_ids=[4] * 5, loss_mask=[1] * 5),
        ]
        config = PackingConfig(max_length=10, pad_to_max=False)

        packed = pack_sequences_best_fit(sequences, config)

        # Best-fit should pack s1+s2 (5+5=10), s3+s4 won't fit together
        # but s3 (4) should go with something if possible
        total_samples = sum(p.num_segments for p in packed)
        assert total_samples == 4


class TestPackSequencesGreedy:
    """Tests for greedy packing algorithm."""

    def test_sorts_by_length(self):
        sequences = [
            SequenceToPack(sample_id="s1", input_ids=[1], loss_mask=[1]),
            SequenceToPack(sample_id="s2", input_ids=[2] * 8, loss_mask=[1] * 8),
            SequenceToPack(sample_id="s3", input_ids=[3] * 3, loss_mask=[1] * 3),
        ]
        config = PackingConfig(max_length=10, pad_to_max=False)

        packed = pack_sequences_greedy(sequences, config)

        # Greedy sorts by length descending then uses first-fit
        # s2 (8) goes first, then s3 (3) can't fit, then s1 (1) can fit with s2
        # Result: bin1=[s2, s1], bin2=[s3]
        assert len(packed) == 2


class TestPackSequencesDispatch:
    """Tests for pack_sequences dispatcher."""

    def test_dispatch_first_fit(self):
        sequences = [
            SequenceToPack(sample_id="s1", input_ids=[1, 2], loss_mask=[1, 1]),
        ]
        config = PackingConfig(mode=PackingMode.FIRST_FIT, max_length=10)

        packed = pack_sequences(sequences, config)
        assert len(packed) == 1

    def test_dispatch_best_fit(self):
        sequences = [
            SequenceToPack(sample_id="s1", input_ids=[1, 2], loss_mask=[1, 1]),
        ]
        config = PackingConfig(mode=PackingMode.BEST_FIT, max_length=10)

        packed = pack_sequences(sequences, config)
        assert len(packed) == 1

    def test_dispatch_greedy(self):
        sequences = [
            SequenceToPack(sample_id="s1", input_ids=[1, 2], loss_mask=[1, 1]),
        ]
        config = PackingConfig(mode=PackingMode.GREEDY, max_length=10)

        packed = pack_sequences(sequences, config)
        assert len(packed) == 1


class TestCreateSegmentAttentionMask:
    """Tests for segment-aware attention mask generation."""

    def test_single_segment(self):
        segment_ids = [0, 0, 0, 0]
        mask = create_segment_attention_mask(segment_ids, use_mlx=False)

        # Should be causal mask (lower triangular)
        expected = np.tril(np.ones((4, 4)))
        np.testing.assert_array_equal(mask, expected)

    def test_two_segments(self):
        segment_ids = [0, 0, 1, 1]
        mask = create_segment_attention_mask(segment_ids, use_mlx=False)

        # Position 0 can attend to: [0]
        # Position 1 can attend to: [0, 1]
        # Position 2 can attend to: [2] (not 0, 1 - different segment)
        # Position 3 can attend to: [2, 3]
        expected = np.array(
            [
                [1, 0, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 1, 1],
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(mask, expected)

    def test_three_segments(self):
        segment_ids = [0, 0, 1, 2, 2]
        mask = create_segment_attention_mask(segment_ids, use_mlx=False)

        # Segment 0: positions 0, 1
        # Segment 1: position 2
        # Segment 2: positions 3, 4
        expected = np.array(
            [
                [1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 1, 1],
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(mask, expected)

    def test_tuple_input(self):
        segment_ids = (0, 0, 1, 1)
        mask = create_segment_attention_mask(segment_ids, use_mlx=False)
        assert mask.shape == (4, 4)

    def test_mask_shape(self):
        segment_ids = [0] * 10
        mask = create_segment_attention_mask(segment_ids, use_mlx=False)
        assert mask.shape == (10, 10)

    def test_mlx_backend(self):
        """Test MLX backend if available."""
        try:
            import mlx.core as mx  # noqa: F401 - import to check availability

            segment_ids = [0, 0, 1, 1]
            mask = create_segment_attention_mask(segment_ids, use_mlx=True)

            # Convert to numpy for comparison
            mask_np = np.array(mask)
            expected = np.array(
                [
                    [1, 0, 0, 0],
                    [1, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 1, 1],
                ],
                dtype=np.float32,
            )
            np.testing.assert_array_equal(mask_np, expected)
        except ImportError:
            pytest.skip("MLX not available")


class TestPackingMetrics:
    """Tests for PackingMetrics."""

    def test_create_empty(self):
        metrics = PackingMetrics()
        assert metrics.num_original_samples == 0
        assert metrics.num_packed_sequences == 0
        assert metrics.packing_ratio == 0.0
        assert metrics.efficiency == 0.0

    def test_record_packed_sequence(self):
        metrics = PackingMetrics()
        packed = PackedSequence(
            input_ids=[1, 2, 3, 4, 0, 0],
            loss_mask=[1, 1, 1, 1, 0, 0],
            segment_ids=[0, 0, 1, 1, 1, 1],
            sample_ids=["s1", "s2"],
            sample_lengths=[2, 2],
            num_segments=2,
            total_tokens=4,
            padding_tokens=2,
        )
        metrics.record_packed_sequence(packed)

        assert metrics.num_original_samples == 2
        assert metrics.num_packed_sequences == 1
        assert metrics.packing_ratio == 2.0
        assert metrics.total_tokens == 4
        assert metrics.total_padded_length == 6

    def test_efficiency(self):
        metrics = PackingMetrics()
        metrics.total_tokens = 80
        metrics.total_padded_length = 100
        assert metrics.efficiency == 0.8

    def test_loss_efficiency(self):
        metrics = PackingMetrics()
        metrics.total_loss_tokens = 60
        metrics.total_padded_length = 100
        assert metrics.loss_efficiency == 0.6

    def test_summary(self):
        metrics = PackingMetrics()
        metrics.num_original_samples = 10
        metrics.num_packed_sequences = 4
        metrics.total_tokens = 200
        metrics.total_padded_length = 256

        summary = metrics.summary()
        assert "num_original_samples" in summary
        assert "packing_ratio" in summary
        assert "efficiency" in summary


class TestComputePackingMetrics:
    """Tests for compute_packing_metrics helper."""

    def test_basic(self):
        packed = [
            PackedSequence(
                input_ids=[1, 2, 0, 0],
                loss_mask=[1, 1, 0, 0],
                segment_ids=[0, 0, 0, 0],
                sample_ids=["s1"],
                sample_lengths=[2],
                num_segments=1,
                total_tokens=2,
                padding_tokens=2,
            ),
            PackedSequence(
                input_ids=[3, 4, 5, 0],
                loss_mask=[1, 1, 1, 0],
                segment_ids=[0, 0, 0, 0],
                sample_ids=["s2"],
                sample_lengths=[3],
                num_segments=1,
                total_tokens=3,
                padding_tokens=1,
            ),
        ]

        metrics = compute_packing_metrics(packed)

        assert metrics.num_original_samples == 2
        assert metrics.num_packed_sequences == 2
        assert metrics.total_tokens == 5
        assert metrics.total_padded_length == 8

    def test_with_skipped(self):
        packed = []
        metrics = compute_packing_metrics(packed, num_skipped=5)
        assert metrics.num_skipped == 5


class TestIntegrationPacking:
    """Integration tests for packing workflow."""

    def test_pack_then_create_mask(self):
        """Test full workflow: pack sequences then create attention mask."""
        sequences = [
            SequenceToPack(sample_id="s1", input_ids=[1, 2, 3], loss_mask=[0, 1, 1]),
            SequenceToPack(sample_id="s2", input_ids=[4, 5], loss_mask=[1, 1]),
        ]
        config = PackingConfig(max_length=10, pad_to_max=True)

        packed_list = pack_sequences(sequences, config)
        assert len(packed_list) == 1

        packed = packed_list[0]

        # Create attention mask
        mask = create_segment_attention_mask(packed.segment_ids, use_mlx=False)
        assert mask.shape == (10, 10)

        # Verify segment isolation
        # s1 is positions 0-2, s2 is positions 3-4
        # Position 3 should not attend to position 2
        assert mask[3, 2] == 0
        # Position 4 should attend to position 3
        assert mask[4, 3] == 1

    def test_metrics_after_packing(self):
        """Test computing metrics after packing."""
        sequences = [
            SequenceToPack(sample_id=f"s{i}", input_ids=[1] * (i + 1), loss_mask=[1] * (i + 1))
            for i in range(5)
        ]
        config = PackingConfig(max_length=20, pad_to_max=True)

        packed = pack_sequences(sequences, config)
        metrics = compute_packing_metrics(packed)

        assert metrics.num_original_samples == 5
        assert metrics.packing_ratio > 1.0  # Multiple samples per pack
        assert 0 < metrics.efficiency <= 1.0
