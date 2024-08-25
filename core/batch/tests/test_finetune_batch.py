import os
import pytest
import tempfile
import numpy as np
from unittest.mock import MagicMock
from core.batch.finetune_batch import FineTuneBatch

@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0  # Pad token ID is 2
    tokenizer.eos_token_id = 2  # EOS token ID is 2
    tokenizer.encode = lambda x, **kwargs: list(map(ord, x))  # Simple encode: ASCII values
    return tokenizer

@pytest.fixture
def temp_output_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

def test_finetune_batch_initialization(mock_tokenizer, temp_output_dir):
    finetune_batch = FineTuneBatch(mock_tokenizer, temp_output_dir, 'finetune', 128, 32, True)
    
    assert finetune_batch.tokenizer == mock_tokenizer
    assert finetune_batch.output_directory == temp_output_dir
    assert finetune_batch.file_prefix == 'finetune'
    assert finetune_batch.max_sequence_length == 128
    assert finetune_batch.batch_size == 32
    assert finetune_batch.print_summaries is True

def test_finetune_batch_generation(mock_tokenizer, temp_output_dir):
    class TestFineTuneBatch(FineTuneBatch):
        def tokenize_line(self, line):
            input_tokens = mock_tokenizer.encode(line.strip())
            target_tokens = mock_tokenizer.encode(line.strip()[::-1])
            return input_tokens, target_tokens

    finetune_batch = TestFineTuneBatch(mock_tokenizer, temp_output_dir, 'finetune', 5, 32, True)

    input_text = "a\nbc\n"
    input_file = os.path.join(temp_output_dir, 'input.txt')
    with open(input_file, 'w') as f:
        f.write(input_text)

    finetune_batch.tokenize_and_batch([input_file])

    batch_files = [f for f in os.listdir(temp_output_dir) if f.endswith('.npz')]
    assert len(batch_files) > 0

    with np.load(os.path.join(temp_output_dir, batch_files[0])) as data:
        assert 'input_tensor' in data
        assert 'target_tensor' in data
        assert data['input_tensor'].shape == data['target_tensor'].shape

        # Ensure the sequences are correctly padded
        max_seq_len_in_batch = 2  # Length of "bc"
        for seq in data['input_tensor']:
            assert len(seq) == max_seq_len_in_batch
            assert seq[-1] == mock_tokenizer.encode("c")[0]  # Since "bc" doesn't require padding

def test_padding_correctly_applied(mock_tokenizer):
    class TestFineTuneBatch(FineTuneBatch):
        def tokenize_line(self, line):
            input_tokens = mock_tokenizer.encode(line.strip())
            target_tokens = mock_tokenizer.encode(line.strip()[::-1])
            return input_tokens, target_tokens

    finetune_batch = TestFineTuneBatch(mock_tokenizer, '.', 'finetune', 10, 2, False)

    input_tokens = "hello"
    target_tokens = "olleh"
    batch_data = [(mock_tokenizer.encode(input_tokens), mock_tokenizer.encode(target_tokens))]

    with tempfile.NamedTemporaryFile(suffix=".npz") as tmpfile:
        padded_inputs, padded_targets = finetune_batch.save_batch(batch_data, tmpfile.name)

        # Verify that padding was applied correctly
        expected_length = 5  # Both sequences are length 5
        assert padded_inputs.shape == (1, expected_length)
        assert padded_targets.shape == (1, expected_length)
        assert np.all(padded_inputs[0, 5:] == [])  # No padding required, same with targets
