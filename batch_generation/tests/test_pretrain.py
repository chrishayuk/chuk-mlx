import os
import numpy as np
import pytest
import tempfile
from batch_generation.pretrain_batch import PretrainBatchGenerator
from batch_generation.batch_base import BatchBase

class MockTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1

    def encode(self, text, max_length=None, truncation=True, add_special_tokens=False):
        return list(range(1, len(text) + 1))

@pytest.fixture
def temp_output_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def mock_tokenizer():
    return MockTokenizer()

# Mock subclass for testing purposes
class MockBatchBase(BatchBase):
    def tokenize_line(self, line):
        # Simple tokenization by converting each character to an integer
        return [ord(c) for c in line.strip()]
    
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

def test_save_batch(batch_base_instance, temp_output_dir):
    batch_data = [[72, 101, 108], [108, 111]]
    file_path = os.path.join(temp_output_dir, "test_batch.npz")
    
    input_tensor = batch_base_instance.save_batch(batch_data, file_path)
    
    # Expected shape should now be based on the longest sequence in the batch
    expected_shape = (2, max(len(seq) for seq in batch_data))
    
    assert input_tensor.shape == expected_shape, "Shape mismatch in saved batch."


def test_process_batch(temp_output_dir, mock_tokenizer):
    generator = PretrainBatchGenerator(tokenizer=mock_tokenizer, output_directory=temp_output_dir, file_prefix='test',
                                       max_sequence_length=5, batch_size=2, print_summaries=False)
    
    batch_data = [[1, 2, 3], [4, 5]]
    file_path = os.path.join(temp_output_dir, "test_batch.npz")

    generator.process_batch(0, batch_data, file_path)

    assert os.path.exists(file_path)

def test_tokenize_and_batch(temp_output_dir, mock_tokenizer):
    with tempfile.NamedTemporaryFile(delete=False) as temp_input_file:
        input_file_name = temp_input_file.name
        temp_input_file.write(b"Hello\nWorld\n")

    generator = PretrainBatchGenerator(tokenizer=mock_tokenizer, output_directory=temp_output_dir, file_prefix='test',
                                       max_sequence_length=5, batch_size=2, print_summaries=False)

    input_files = [input_file_name]
    
    try:
        generator.tokenize_and_batch(input_files)

        batch_files = [f for f in os.listdir(temp_output_dir) if f.endswith('.npz')]
        assert len(batch_files) == 1

    finally:
        if os.path.exists(input_file_name):
            os.remove(input_file_name)


def test_padding_in_batches(mock_tokenizer):
    generator = PretrainBatchGenerator(tokenizer=mock_tokenizer, output_directory=tempfile.gettempdir(), file_prefix='test',
                                       max_sequence_length=5, batch_size=3, print_summaries=False)

    batch_data = [[1, 2, 3], [4, 5], [6]]

    with tempfile.NamedTemporaryFile(delete=False) as temp_batch_file:
        file_path = temp_batch_file.name

    try:
        input_tensor = generator.save_batch(batch_data, file_path)

        # Adjust expected input to match dynamic padding
        max_length_in_batch = max(len(seq) for seq in batch_data)
        expected_input = np.array([
            seq + [0] * (max_length_in_batch - len(seq)) for seq in batch_data
        ], dtype=np.int32)

        np.testing.assert_array_equal(input_tensor, expected_input)


    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

def test_target_tensor_generation(mock_tokenizer):
    generator = PretrainBatchGenerator(
        tokenizer=mock_tokenizer,
        output_directory=tempfile.gettempdir(),
        file_prefix='test',
        max_sequence_length=5,
        batch_size=2,
        print_summaries=False
    )
    
    # Define the input batch
    batch_data = [[1, 2, 3], [4, 5]]
    
    # Manually generate the expected target tensor
    expected_target_tensor = np.array([
        [2, 3, 0],   # Shifted: [1, 2, 3] -> [2, 3, <pad>]
        [5, 0, 0]    # Shifted: [4, 5] -> [5, <pad>, <pad>]
    ], dtype=np.int32)
    
    # Generate the actual target tensor using the method
    input_tensor = generator.process_batch_data(batch_data)
    target_tensor, _ = generator.create_target_batch(input_tensor, generator.tokenizer.pad_token_id)
    
    # Ensure the target tensor matches the expected tensor
    np.testing.assert_array_equal(target_tensor, expected_target_tensor)


def test_input_and_target_tensor_consistency(mock_tokenizer):
    generator = PretrainBatchGenerator(
        tokenizer=mock_tokenizer,
        output_directory=tempfile.gettempdir(),
        file_prefix='test',
        max_sequence_length=5,
        batch_size=2,
        print_summaries=False
    )

    # Define the input batch
    batch_data = [[1, 2, 3], [4, 5]]

    # Process the batch
    input_tensor = generator.process_batch_data(batch_data)
    target_tensor, _ = generator.create_target_batch(input_tensor, generator.tokenizer.pad_token_id)

    # Verify that the input and target tensors have the same shape
    assert input_tensor.shape == target_tensor.shape, "Input and target tensor shapes do not match."

    # Check the first sequence in the input and target tensor
    assert np.array_equal(input_tensor[0, 1:], target_tensor[0, :-1]), \
        "Target tensor does not match the input tensor shifted by one."

    # Check the last element in the target tensor is the pad token
    assert target_tensor[0, -1] == mock_tokenizer.pad_token_id, \
        "Last element of the target tensor is not the pad token."
