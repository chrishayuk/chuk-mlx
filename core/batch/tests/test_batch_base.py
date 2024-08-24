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
        # Simple tokenization by converting each character to an integer
        return [ord(c) for c in line.strip()]

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
    tokens = batch_base_instance.tokenize_line(text)
    assert tokens == [72, 101, 108, 108, 111], "Tokenization failed."

def test_save_batch(batch_base_instance, temp_output_dir):
    batch_data = [[72, 101, 108], [108, 111]]
    file_path = os.path.join(temp_output_dir, "test_batch.npz")
    
    input_tensor = batch_base_instance.save_batch(batch_data, file_path)
    
    # Expected shape should match the length of the longest sequence
    expected_shape = (2, max(len(seq) for seq in batch_data))
    
    assert input_tensor.shape == expected_shape, f"Shape mismatch in saved batch: expected {expected_shape}, got {input_tensor.shape}."


def test_process_batch(batch_base_instance, temp_output_dir):
    batch_data = [[72, 101, 108], [108, 111]]
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
            [72, 101, 108, 108, 111],
            [87, 111, 114, 108, 100]
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
    buckets = {5: [(list(map(ord, "hello")), list(map(ord, "world")))] * 5}
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
