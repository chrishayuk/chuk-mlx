"""Tests for batching padding utilities."""

import numpy as np

from chuk_lazarus.data.batching import pad_sequences


class TestPadSequences:
    """Tests for pad_sequences function."""

    def test_pad_to_longest(self):
        """Test padding sequences to longest length."""
        sequences = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
        padded = pad_sequences(sequences, pad_value=0)

        assert padded.shape == (3, 4)
        assert list(padded[0]) == [1, 2, 3, 0]
        assert list(padded[1]) == [4, 5, 0, 0]
        assert list(padded[2]) == [6, 7, 8, 9]

    def test_pad_to_max_length(self):
        """Test padding to specified max length."""
        sequences = [[1, 2], [3, 4, 5]]
        padded = pad_sequences(sequences, pad_value=0, max_length=6)

        assert padded.shape == (2, 6)
        assert list(padded[0]) == [1, 2, 0, 0, 0, 0]
        assert list(padded[1]) == [3, 4, 5, 0, 0, 0]

    def test_custom_pad_value(self):
        """Test custom padding value."""
        sequences = [[1, 2], [3]]
        padded = pad_sequences(sequences, pad_value=999, max_length=4)

        assert padded[0, 2] == 999
        assert padded[0, 3] == 999
        assert padded[1, 1] == 999

    def test_dtype(self):
        """Test dtype specification."""
        sequences = [[1, 2], [3, 4]]

        padded_int32 = pad_sequences(sequences, pad_value=0, dtype=np.int32)
        assert padded_int32.dtype == np.int32

        padded_int64 = pad_sequences(sequences, pad_value=0, dtype=np.int64)
        assert padded_int64.dtype == np.int64

    def test_single_sequence(self):
        """Test padding single sequence."""
        sequences = [[1, 2, 3]]
        padded = pad_sequences(sequences, pad_value=0, max_length=5)

        assert padded.shape == (1, 5)
        assert list(padded[0]) == [1, 2, 3, 0, 0]

    def test_already_max_length(self):
        """Test sequence already at max length needs no padding."""
        sequences = [[1, 2, 3, 4]]
        padded = pad_sequences(sequences, pad_value=0, max_length=4)

        assert padded.shape == (1, 4)
        assert list(padded[0]) == [1, 2, 3, 4]

    def test_empty_sequences(self):
        """Test handling empty sequences in list."""
        sequences = [[], [1, 2], []]
        padded = pad_sequences(sequences, pad_value=0)

        assert padded.shape == (3, 2)
        assert list(padded[0]) == [0, 0]
        assert list(padded[1]) == [1, 2]
        assert list(padded[2]) == [0, 0]
