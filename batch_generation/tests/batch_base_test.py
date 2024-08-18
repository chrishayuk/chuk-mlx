import os
import numpy as np
import pytest
import tempfile
from batch_generation.batch_base import BatchBase

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
    # Create an instance of MockBatchBase for testing
    return MockBatchBase(
        tokenizer=mock_tokenizer,
        output_directory=temp_output_dir,
        file_prefix='test',
        max_sequence_length=5,
        batch_size=2,
        print_summaries=False
    )

def test_tokenize_line(batch_base_instance):
    text = "Hello"
    tokens = batch_base_instance.tokenize_line(text)
    assert tokens == [72, 101, 108, 108, 111], "Tokenization failed."

def test_save_batch(batch_base_instance, temp_output_dir):
    batch_data = [[72, 101, 108], [108, 111]]
    file_path = os.path.join(temp_output_dir, "test_batch.npz")

    input_tensor = batch_base_instance.save_batch(batch_data, file_path)

    assert input_tensor.shape == (2, batch_base_instance.max_sequence_length), "Shape mismatch in saved batch."
    assert os.path.exists(file_path), "Batch file was not created."

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
