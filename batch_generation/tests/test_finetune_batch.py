import os
import pytest
import tempfile
import numpy as np
from batch_generation.finetune_batch import FineTuneBatch
from unittest.mock import MagicMock

@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 2
    tokenizer.encode = lambda x, **kwargs: list(map(ord, x))
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
            # Mock tokenization: input is line itself, target is reversed line
            input_tokens = mock_tokenizer.encode(line.strip())
            target_tokens = mock_tokenizer.encode(line.strip()[::-1])
            return input_tokens, target_tokens

    finetune_batch = TestFineTuneBatch(mock_tokenizer, temp_output_dir, 'finetune', 128, 32, True)
    
    # Create a simple input file
    input_text = "abc\ndef\n"
    input_file = os.path.join(temp_output_dir, 'input.txt')
    with open(input_file, 'w') as f:
        f.write(input_text)
    
    # Run tokenize_and_batch
    finetune_batch.tokenize_and_batch([input_file])
    
    # Check that batch files are created
    batch_files = [f for f in os.listdir(temp_output_dir) if f.endswith('.npz')]
    assert len(batch_files) > 0

    # Verify the content of one batch file
    with np.load(os.path.join(temp_output_dir, batch_files[0])) as data:
        assert 'input_tensor' in data
        assert 'target_tensor' in data
        assert data['input_tensor'].shape == data['target_tensor'].shape
