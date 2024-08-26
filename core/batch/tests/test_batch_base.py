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

# Existing tests
def test_tokenize_line(batch_base_instance):
    text = "Hello"
    input_tokens, target_tokens, attention_mask = batch_base_instance.tokenize_line(text)
    assert input_tokens == [72, 101, 108, 108, 111], "Tokenization failed."
    assert target_tokens == [111, 108, 108, 101, 72], "Target tokenization failed."
    assert attention_mask == [1, 1, 1, 1, 1], "Attention mask generation failed."

def test_save_batch(batch_base_instance, temp_output_dir):
    batch_data = [
        ([72, 101, 108], [108, 111, 1], [1, 1, 1]),
        ([108, 111], [111, 108, 1], [1, 1])
    ]
    file_path = os.path.join(temp_output_dir, "test_batch.npz")

    # Unpack the returned values to test the input, target, and attention mask tensors
    input_tensor, target_tensor, attention_mask = batch_base_instance.save_batch(batch_data, file_path)

    # The expected shape should be based on the longest sequence length in the batch
    expected_shape = (2, 3)  # Padded to the length of the longest sequence

    assert input_tensor.shape == expected_shape, f"Shape mismatch in saved input tensor: expected {expected_shape}, got {input_tensor.shape}."
    assert target_tensor.shape == expected_shape, f"Shape mismatch in saved target tensor: expected {expected_shape}, got {target_tensor.shape}."
    assert attention_mask.shape == expected_shape, f"Shape mismatch in saved attention mask: expected {expected_shape}, got {attention_mask.shape}."

    # Validate the actual content of the tensors
    expected_input_tensor = np.array([
        [72, 101, 108],  # No padding needed here
        [108, 111, 0]    # Should be padded with the tokenizer's pad token (0)
    ])
    
    expected_target_tensor = np.array([
        [108, 111, 1],   # Target sequence
        [111, 108, 1]    # Target sequence with padding
    ])
    
    expected_attention_mask = np.array([
        [1, 1, 1],  # Full sequence is valid
        [1, 1, 0]   # Padding should be masked with 0
    ])
    
    np.testing.assert_array_equal(input_tensor, expected_input_tensor, "Mismatch in input tensor content.")
    np.testing.assert_array_equal(target_tensor, expected_target_tensor, "Mismatch in target tensor content.")
    np.testing.assert_array_equal(attention_mask, expected_attention_mask, "Mismatch in attention mask content.")


def test_process_batch(batch_base_instance, temp_output_dir):
    batch_data = [
        ([72, 101, 108], [108, 111, 1], [1, 1, 1]),
        ([108, 111], [111, 108, 1], [1, 1])
    ]
    file_path = os.path.join(temp_output_dir, "test_batch.npz")

    batch_base_instance.process_batch(0, batch_data, file_path)

    assert os.path.exists(file_path), "Batch file was not created during processing."

def test_tokenize_dataset(batch_base_instance):
    with tempfile.NamedTemporaryFile(delete=False) as temp_input_file:
        temp_input_file.write(b"Hello\nWorld\n")
        input_file_name = temp_input_file.name
    
    try:
        tokenized_dataset = batch_base_instance.tokenize_dataset([input_file_name])
        
        expected_tokenized = [
            ([72, 101, 108, 108, 111], [111, 108, 108, 101, 72], [1, 1, 1, 1, 1]),
            ([87, 111, 114, 108, 100], [100, 108, 114, 111, 87], [1, 1, 1, 1, 1])
        ]
        assert tokenized_dataset == expected_tokenized, "Tokenized dataset does not match expected."
    finally:
        if os.path.exists(input_file_name):
            os.remove(input_file_name)

def test_tokenize_and_batch(batch_base_instance, temp_output_dir):
    with tempfile.NamedTemporaryFile(delete=False) as temp_input_file:
        temp_input_file.write(b"Hello\nWorld\n")
        input_file_name = temp_input_file.name
    
    try:
        batch_base_instance.tokenize_and_batch([input_file_name])
        
        batch_files = [f for f in os.listdir(temp_output_dir) if f.endswith('.npz')]
        assert len(batch_files) > 0, "No batch files were created."
    finally:
        if os.path.exists(input_file_name):
            os.remove(input_file_name)

# Additional tests
def test_infinite_loop_prevention(batch_base_instance, temp_output_dir):
    input_files = ["test_input_file.txt"]
    with open(input_files[0], 'w') as f:
        f.write("test\n" * 9)  # 9 sequences, batch size 2, should result in 5 batches

    batch_base_instance.tokenize_and_batch(input_files)

    batch_files = [f for f in os.listdir(temp_output_dir) if f.endswith('.npz')]
    assert len(batch_files) == 5, f"Expected 5 batches, got {len(batch_files)}"
    os.remove(input_files[0])

def test_bucket_emptying(batch_base_instance):
    buckets = {5: [([72, 101], [111, 108], [1, 1])] * 5}
    batch_base_instance.create_batches(buckets)
    assert all(len(bucket) == 0 for bucket in buckets.values()), "Buckets were not emptied correctly."

def test_final_batch_leftover_sequences(batch_base_instance, temp_output_dir):
    input_files = ["test_input_file.txt"]
    with open(input_files[0], 'w') as f:
        f.write("test\n" * 5)  # 5 sequences, batch size 2, should result in 3 batches

    batch_base_instance.tokenize_and_batch(input_files)

    batch_files = [f for f in os.listdir(temp_output_dir) if f.endswith('.npz')]
    assert len(batch_files) == 3, f"Expected 3 batches, got {len(batch_files)}"
    os.remove(input_files[0])

def test_proper_batch_size(batch_base_instance, temp_output_dir):
    input_files = ["test_input_file.txt"]
    with open(input_files[0], 'w') as f:
        f.write("test\n" * 7)  # 7 sequences, batch size 2, should result in 4 batches

    batch_base_instance.tokenize_and_batch(input_files)

    batch_files = [f for f in os.listdir(temp_output_dir) if f.endswith('.npz')]
    assert len(batch_files) == 4, f"Expected 4 batches, got {len(batch_files)}"
    os.remove(input_files[0])

def test_empty_input(batch_base_instance, temp_output_dir):
    input_files = ["empty_input_file.txt"]
    with open(input_files[0], 'w') as f:
        pass  # Empty file

    batch_base_instance.tokenize_and_batch(input_files)

    batch_files = [f for f in os.listdir(temp_output_dir) if f.endswith('.npz')]
    assert len(batch_files) == 0, f"Expected 0 batches, got {len(batch_files)}"
    os.remove(input_files[0])

# Error Handling Tests
def test_tokenize_line_error_handling(batch_base_instance):
    invalid_line = None  # Simulate an invalid line input
    with pytest.raises(ValueError, match="Input line cannot be None."):
        batch_base_instance.tokenize_line(invalid_line)

def test_save_batch_with_invalid_data(batch_base_instance, temp_output_dir):
    invalid_batch_data = None  # Simulate invalid batch data
    file_path = os.path.join(temp_output_dir, "test_batch.npz")
    with pytest.raises(TypeError):
        batch_base_instance.save_batch(invalid_batch_data, file_path)

def test_process_batch_with_invalid_data(batch_base_instance, temp_output_dir):
    invalid_batch_data = None  # Simulate invalid batch data
    file_path = os.path.join(temp_output_dir, "test_batch.npz")
    with pytest.raises(TypeError):
        batch_base_instance.process_batch(0, invalid_batch_data, file_path)
