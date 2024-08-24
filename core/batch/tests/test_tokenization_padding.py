import pytest
import numpy as np
from core.batch.tokenization_utils import batch_tokenize_and_pad, tokenize_and_pad

class MockTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1

@pytest.fixture
def mock_tokenizer():
    return MockTokenizer()

# Test for tokenize_and_pad
def test_tokenize_and_pad_short_sequence(mock_tokenizer):
    seq = [1, 2, 3]
    result = tokenize_and_pad(seq, mock_tokenizer, 5)
    expected = [1, 2, 3, 0, 0]
    assert result == expected, f"Expected {expected}, but got {result}"

def test_tokenize_and_pad_exact_length(mock_tokenizer):
    seq = [1, 2, 3, 4, 5]
    result = tokenize_and_pad(seq, mock_tokenizer, 5)
    expected = [1, 2, 3, 4, 5]
    assert result == expected, f"Expected {expected}, but got {result}"

def test_tokenize_and_pad_long_sequence(mock_tokenizer):
    seq = [1, 2, 3, 4, 5, 6]
    result = tokenize_and_pad(seq, mock_tokenizer, 5)
    expected = [1, 2, 3, 4, 5]
    assert result == expected, f"Expected {expected}, but got {result}"

def test_tokenize_and_pad_empty_sequence(mock_tokenizer):
    seq = []
    result = tokenize_and_pad(seq, mock_tokenizer, 5)
    expected = [0, 0, 0, 0, 0]
    assert result == expected, f"Expected {expected}, but got {result}"

def test_tokenize_and_pad_invalid_length(mock_tokenizer):
    seq = [1, 2, 3]
    with pytest.raises(ValueError):
        tokenize_and_pad(seq, mock_tokenizer, 0)

# Test for batch_tokenize_and_pad
def test_batch_tokenize_and_pad(mock_tokenizer):
    batch_data = [[1, 2], [1, 2, 3, 4], [1]]
    result = batch_tokenize_and_pad(batch_data, mock_tokenizer, 4)  # Change to 4
    expected = [
        [1, 2, 0, 0],       # Length 4
        [1, 2, 3, 4],       # Already length 4
        [1, 0, 0, 0]        # Length 4
    ]
    assert np.array_equal(result, expected), f"Expected {expected}, but got {result}"


def test_batch_tokenize_and_pad_max_length(mock_tokenizer):
    batch_data = [[1, 2], [1, 2, 3, 4, 5, 6], [1]]
    result = batch_tokenize_and_pad(batch_data, mock_tokenizer, 5)
    expected = [
        [1, 2, 0, 0, 0],
        [1, 2, 3, 4, 5],
        [1, 0, 0, 0, 0]
    ]
    assert np.array_equal(result, expected), f"Expected {expected}, but got {result}"

def test_batch_tokenize_and_pad_empty_batch(mock_tokenizer):
    batch_data = []
    with pytest.raises(ValueError):
        batch_tokenize_and_pad(batch_data, mock_tokenizer, 5)


def test_batch_tokenize_and_pad_invalid_length(mock_tokenizer):
    batch_data = [[1, 2], [3, 4]]
    with pytest.raises(ValueError):
        batch_tokenize_and_pad(batch_data, mock_tokenizer, 0)

# New test to handle tuples (input and target sequences)
def test_batch_tokenize_and_pad_with_tuples(mock_tokenizer):
    batch_data = [
        ([1, 2, 3], [2, 3, 4]),
        ([4, 5], [5, 6]),
        ([7], [8])
    ]
    result = batch_tokenize_and_pad([seq[0] for seq in batch_data], mock_tokenizer, 3)  # Change to 3
    expected = [
        [1, 2, 3],       # Length 3
        [4, 5, 0],       # Length 3
        [7, 0, 0]        # Length 3
    ]
    assert np.array_equal(result, expected), f"Expected {expected}, but got {result}"


# Test for consistent processing across various sequence lengths
def test_batch_tokenize_and_pad_consistent_processing(mock_tokenizer):
    batch_data = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [11, 12],
        [13, 14, 15, 16]
    ]
    result = batch_tokenize_and_pad(batch_data, mock_tokenizer, 10)
    expected = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [11, 12, 0, 0, 0, 0, 0, 0, 0, 0],
        [13, 14, 15, 16, 0, 0, 0, 0, 0, 0]
    ]
    assert np.array_equal(result, expected), f"Expected {expected}, but got {result}"
