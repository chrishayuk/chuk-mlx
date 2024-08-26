import os
import numpy as np
import pytest
import tempfile
from core.batch.batch_base import BatchBase

# Mock tokenizer for testing purposes
class MockTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1

    def encode(self, text, max_length=None, truncation=True, add_special_tokens=False):
        # Simple mock encoding that returns a sequence of integers based on text length
        return list(range(1, min(len(text), max_length) + 1))

# Mock subclass for testing purposes
class MockBatchBase(BatchBase):
    def tokenize_line(self, line):
        if line is None:
            raise ValueError("Input line cannot be None.")
    
        # Simple tokenization by converting each character to an integer
        input_tokens = [ord(c) for c in line.strip()]
        target_tokens = input_tokens[::-1]  # Reverse for target just as a mock behavior
        attention_mask = [1] * len(input_tokens)
        return input_tokens, target_tokens, attention_mask

@pytest.fixture
def temp_output_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def mock_tokenizer():
    return MockTokenizer()

@pytest.fixture
def batch_base_instance(mock_tokenizer, temp_output_dir):
    return MockBatchBase(
        tokenizer=mock_tokenizer,
        output_directory=temp_output_dir,
        file_prefix='test',
        max_sequence_length=5,
        batch_size=2,
        print_summaries=False
    )

# Test padding sequences (already provided)
def test_pad_sequences(batch_base_instance):
    input_tokens_list = [[72, 101], [108, 111, 114]]
    target_tokens_list = [[101, 72], [114, 111, 108]]
    attention_masks_list = [[1, 1], [1, 1, 1]]

    input_tokens_padded, target_tokens_padded, attention_masks_padded = batch_base_instance.pad_sequences(
        input_tokens_list, target_tokens_list, attention_masks_list, batch_base_instance.tokenizer.pad_token_id
    )

    expected_input_tokens_padded = np.array([
        [72, 101, 0],  # Padded to the length of the longest sequence (3)
        [108, 111, 114]
    ], dtype=np.int32)
    expected_target_tokens_padded = np.array([
        [101, 72, 0],  # Padded to the length of the longest sequence (3)
        [114, 111, 108]
    ], dtype=np.int32)
    expected_attention_masks_padded = np.array([
        [1, 1, 0],  # Mask padded with 0
        [1, 1, 1]
    ], dtype=np.int32)

    np.testing.assert_array_equal(input_tokens_padded, expected_input_tokens_padded)
    np.testing.assert_array_equal(target_tokens_padded, expected_target_tokens_padded)
    np.testing.assert_array_equal(attention_masks_padded, expected_attention_masks_padded)

# Test tokenization and batching
def test_tokenize_and_batch(batch_base_instance, temp_output_dir):
    input_files = [os.path.join(temp_output_dir, 'mock_input.txt')]
    
    with open(input_files[0], 'w') as f:
        f.write("Hello\n")
        f.write("World\n")
    
    batch_base_instance.tokenize_and_batch(input_files)
    
    # Check that output files are generated
    output_files = [f for f in os.listdir(temp_output_dir) if f.startswith('test_batch_')]
    assert len(output_files) > 0, "No batch files were created."

# Test handling of empty input
def test_empty_input_handling(batch_base_instance):
    input_files = []
    
    # Expecting no exception, but no output as well
    batch_base_instance.tokenize_and_batch(input_files)
    
    # Assert that no processing was done
    assert not os.path.exists(os.path.join(batch_base_instance.output_directory, 'test_batch_0001.npz'))

# Test error handling for None input line
def test_tokenize_line_error_handling(batch_base_instance):
    with pytest.raises(ValueError, match="Input line cannot be None."):
        batch_base_instance.tokenize_line(None)

# Test saving batch
def test_save_batch(batch_base_instance, temp_output_dir):
    batch_data = [
        ([1, 2, 3], [3, 2, 1], [1, 1, 1]),
        ([4, 5], [5, 4], [1, 1])
    ]
    
    file_path = os.path.join(temp_output_dir, 'test_batch.npz')
    input_tensor, target_tensor, attention_mask_tensor = batch_base_instance.save_batch(batch_data, file_path)
    
    # Check that the file was created
    assert os.path.exists(file_path), "Batch file was not created."
    
    # Verify the contents of the file
    with np.load(file_path) as data:
        np.testing.assert_array_equal(data['input_tensor'], input_tensor)
        np.testing.assert_array_equal(data['target_tensor'], target_tensor)
        np.testing.assert_array_equal(data['attention_mask_tensor'], attention_mask_tensor)
