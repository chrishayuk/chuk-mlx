"""Tests for base dataset."""

from typing import Any
from unittest.mock import patch

import mlx.core as mx

from chuk_lazarus.data.base_dataset import BaseDataset


class ConcreteDataset(BaseDataset):
    """Concrete implementation for testing."""

    def __init__(self, samples: list[dict]):
        super().__init__()
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._samples[idx]

    def _collate_batch(self, samples: list[dict], pad_token_id: int) -> dict[str, mx.array]:
        """Simple collate that pads input_ids."""
        input_ids = [s["input_ids"] for s in samples]
        padded = self.pad_sequences(input_ids, pad_value=pad_token_id)
        return {"input_ids": mx.array(padded)}


class TestBaseDataset:
    """Tests for BaseDataset class."""

    def test_len(self):
        """Test __len__ method."""
        samples = [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5, 6]}]
        dataset = ConcreteDataset(samples)
        assert len(dataset) == 2

    def test_getitem(self):
        """Test __getitem__ method."""
        samples = [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5, 6]}]
        dataset = ConcreteDataset(samples)
        assert dataset[0] == {"input_ids": [1, 2, 3]}
        assert dataset[1] == {"input_ids": [4, 5, 6]}


class TestIterBatches:
    """Tests for iter_batches method."""

    def test_iter_batches_basic(self):
        """Test basic batch iteration."""
        samples = [
            {"input_ids": [1, 2, 3]},
            {"input_ids": [4, 5]},
            {"input_ids": [6, 7, 8, 9]},
            {"input_ids": [10]},
        ]
        dataset = ConcreteDataset(samples)

        batches = list(dataset.iter_batches(batch_size=2, shuffle=False))

        assert len(batches) == 2
        assert "input_ids" in batches[0]

    def test_iter_batches_with_shuffle(self):
        """Test batch iteration with shuffling."""
        samples = [{"input_ids": [i]} for i in range(10)]
        dataset = ConcreteDataset(samples)

        with patch("random.shuffle") as mock_shuffle:
            list(dataset.iter_batches(batch_size=2, shuffle=True))
            mock_shuffle.assert_called_once()

    def test_iter_batches_no_shuffle(self):
        """Test batch iteration without shuffling."""
        samples = [{"input_ids": [i]} for i in range(10)]
        dataset = ConcreteDataset(samples)

        with patch("random.shuffle") as mock_shuffle:
            list(dataset.iter_batches(batch_size=2, shuffle=False))
            mock_shuffle.assert_not_called()

    def test_iter_batches_drop_last(self):
        """Test batch iteration with drop_last."""
        samples = [{"input_ids": [i]} for i in range(5)]
        dataset = ConcreteDataset(samples)

        batches = list(dataset.iter_batches(batch_size=2, shuffle=False, drop_last=True))

        # 5 samples, batch_size 2, drop_last=True -> 2 batches
        assert len(batches) == 2

    def test_iter_batches_no_drop_last(self):
        """Test batch iteration without drop_last."""
        samples = [{"input_ids": [i]} for i in range(5)]
        dataset = ConcreteDataset(samples)

        batches = list(dataset.iter_batches(batch_size=2, shuffle=False, drop_last=False))

        # 5 samples, batch_size 2, drop_last=False -> 3 batches
        assert len(batches) == 3

    def test_iter_batches_custom_pad_token(self):
        """Test batch iteration with custom pad token."""
        samples = [{"input_ids": [1, 2]}, {"input_ids": [3, 4, 5]}]
        dataset = ConcreteDataset(samples)

        batches = list(dataset.iter_batches(batch_size=2, shuffle=False, pad_token_id=999))

        # Check that padding was applied
        assert batches[0]["input_ids"].shape[1] == 3


class TestGetBatches:
    """Tests for get_batches method."""

    def test_get_batches_basic(self):
        """Test get_batches returns list."""
        samples = [{"input_ids": [i]} for i in range(4)]
        dataset = ConcreteDataset(samples)

        batches = dataset.get_batches(batch_size=2, shuffle=False)

        assert isinstance(batches, list)
        assert len(batches) == 2

    def test_get_batches_with_shuffle(self):
        """Test get_batches with shuffle."""
        samples = [{"input_ids": [i]} for i in range(4)]
        dataset = ConcreteDataset(samples)

        batches = dataset.get_batches(batch_size=2, shuffle=True)

        assert len(batches) == 2

    def test_get_batches_custom_pad(self):
        """Test get_batches with custom pad token."""
        samples = [{"input_ids": [1, 2]}, {"input_ids": [3]}]
        dataset = ConcreteDataset(samples)

        batches = dataset.get_batches(batch_size=2, shuffle=False, pad_token_id=42)

        assert len(batches) == 1


class TestPadSequences:
    """Tests for pad_sequences static method."""

    def test_pad_sequences_basic(self):
        """Test basic padding."""
        sequences = [[1, 2, 3], [4, 5], [6]]
        padded = BaseDataset.pad_sequences(sequences, pad_value=0)

        assert padded == [[1, 2, 3], [4, 5, 0], [6, 0, 0]]

    def test_pad_sequences_custom_pad_value(self):
        """Test padding with custom pad value."""
        sequences = [[1, 2], [3]]
        padded = BaseDataset.pad_sequences(sequences, pad_value=-1)

        assert padded == [[1, 2], [3, -1]]

    def test_pad_sequences_max_length(self):
        """Test padding with max_length."""
        sequences = [[1, 2, 3, 4, 5], [6, 7]]
        padded = BaseDataset.pad_sequences(sequences, pad_value=0, max_length=3)

        assert padded == [[1, 2, 3], [6, 7, 0]]

    def test_pad_sequences_pad_left(self):
        """Test left padding."""
        sequences = [[1, 2, 3], [4, 5], [6]]
        padded = BaseDataset.pad_sequences(sequences, pad_value=0, pad_left=True)

        assert padded == [[1, 2, 3], [0, 4, 5], [0, 0, 6]]

    def test_pad_sequences_empty(self):
        """Test padding empty sequences."""
        sequences = []
        padded = BaseDataset.pad_sequences(sequences)

        assert padded == []

    def test_pad_sequences_truncation(self):
        """Test that sequences longer than max_length are truncated."""
        sequences = [[1, 2, 3, 4, 5], [6, 7, 8]]
        padded = BaseDataset.pad_sequences(sequences, pad_value=0, max_length=2)

        assert padded == [[1, 2], [6, 7]]


class TestCreateAttentionMask:
    """Tests for create_attention_mask static method."""

    def test_create_attention_mask_basic(self):
        """Test basic attention mask creation."""
        sequences = [[1, 2, 3], [4, 5, 0], [6, 0, 0]]
        masks = BaseDataset.create_attention_mask(sequences, pad_value=0)

        assert masks == [[1.0, 1.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]

    def test_create_attention_mask_custom_pad(self):
        """Test attention mask with custom pad value."""
        sequences = [[1, 2, -1], [3, -1, -1]]
        masks = BaseDataset.create_attention_mask(sequences, pad_value=-1)

        assert masks == [[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]

    def test_create_attention_mask_no_padding(self):
        """Test attention mask with no padding."""
        sequences = [[1, 2, 3], [4, 5, 6]]
        masks = BaseDataset.create_attention_mask(sequences, pad_value=0)

        assert masks == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]


class TestCreateLabelsWithMask:
    """Tests for create_labels_with_mask static method."""

    def test_create_labels_with_mask_basic(self):
        """Test basic label creation."""
        input_ids = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
        response_starts = [2, 3]

        labels, loss_masks = BaseDataset.create_labels_with_mask(
            input_ids, response_starts, pad_token_id=0, ignore_index=-100
        )

        # Labels are shifted by 1
        assert len(labels) == 2
        assert len(labels[0]) == 5
        assert len(labels[1]) == 5

        # Loss masks
        assert loss_masks[0][:2] == [0.0, 0.0]  # Prompt masked
        assert loss_masks[0][2:] == [1.0, 1.0, 1.0]  # Response unmasked

        assert loss_masks[1][:3] == [0.0, 0.0, 0.0]  # Prompt masked
        assert loss_masks[1][3:] == [1.0, 1.0]  # Response unmasked

    def test_create_labels_with_mask_ignore_index(self):
        """Test that masked positions have ignore_index."""
        input_ids = [[1, 2, 3, 4]]
        response_starts = [2]

        labels, _ = BaseDataset.create_labels_with_mask(
            input_ids, response_starts, pad_token_id=0, ignore_index=-100
        )

        # First two positions should be ignore_index
        assert labels[0][0] == -100
        assert labels[0][1] == -100
        # Response positions should have actual labels (shifted)
        assert labels[0][2] == 4
        assert labels[0][3] == 0  # pad_token_id for last position

    def test_create_labels_with_mask_custom_pad(self):
        """Test label creation with custom pad token."""
        input_ids = [[1, 2, 3]]
        response_starts = [1]

        labels, _ = BaseDataset.create_labels_with_mask(
            input_ids, response_starts, pad_token_id=999, ignore_index=-100
        )

        # Last position should have pad_token_id
        assert labels[0][-1] == 999

    def test_create_labels_with_mask_response_at_start(self):
        """Test label creation when response starts at beginning."""
        input_ids = [[1, 2, 3, 4]]
        response_starts = [0]

        labels, loss_masks = BaseDataset.create_labels_with_mask(
            input_ids, response_starts, pad_token_id=0, ignore_index=-100
        )

        # All positions should be unmasked
        assert loss_masks[0] == [1.0, 1.0, 1.0, 1.0]
        # Labels should be shifted input_ids
        assert labels[0] == [2, 3, 4, 0]
