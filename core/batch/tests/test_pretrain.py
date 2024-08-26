import os
import numpy as np
import pytest
import tempfile
from core.batch.pretrain_batch import PretrainBatchGenerator

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

def test_save_batch(mock_tokenizer, temp_output_dir):
    generator = PretrainBatchGenerator(
        tokenizer=mock_tokenizer,
        output_directory=temp_output_dir,
        file_prefix='test',
        max_sequence_length=5,
        batch_size=2,
        print_summaries=False
    )

    batch_data = [[72, 101, 108], [108, 111]]
    file_path = os.path.join(temp_output_dir, "test_batch.npz")

    # Use `save_batch` and unpack the returned values to test the input tensor
    input_tensor, target_tensor, attention_mask_tensor = generator.save_batch(batch_data, file_path)

    expected_input_shape = (2, max(len(seq) for seq in batch_data))
    assert input_tensor.shape == expected_input_shape, "Shape mismatch in saved input tensor."

    # Verify that target_tensor is correctly shifted and padded
    expected_target_tensor = np.array([
        [101, 108, 0],  # Shifted and padded target for [72, 101, 108]
        [111, 0, 0]     # Shifted and padded target for [108, 111]
    ], dtype=np.int32)

    np.testing.assert_array_equal(target_tensor, expected_target_tensor, "Target tensor was not generated correctly.")

    # Verify that attention_mask_tensor is correctly generated
    expected_attention_mask = np.array([
        [1, 1, 1],  # Attention mask for [72, 101, 108]
        [1, 1, 0]   # Attention mask for [108, 111]
    ], dtype=np.int32)

    np.testing.assert_array_equal(attention_mask_tensor, expected_attention_mask, "Attention mask tensor was not generated correctly.")

def test_tokenize_and_batch(mock_tokenizer, temp_output_dir):
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

        # Load and verify the content of the saved batch file
        batch_file_path = os.path.join(temp_output_dir, batch_files[0])
        with np.load(batch_file_path) as data:
            assert 'input_tensor' in data
            assert 'target_tensor' in data
            assert 'attention_mask_tensor' in data

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
        input_tensor, target_tensor, _ = generator.save_batch(batch_data, file_path)

        max_length_in_batch = max(len(seq) for seq in batch_data)
        expected_input = np.array([
            seq + [generator.tokenizer.pad_token_id] * (max_length_in_batch - len(seq)) for seq in batch_data
        ], dtype=np.int32)

        expected_target = np.array([
            seq[1:] + [generator.tokenizer.pad_token_id] * (max_length_in_batch - len(seq)) + [generator.tokenizer.pad_token_id]
            for seq in batch_data
        ], dtype=np.int32)

        np.testing.assert_array_equal(input_tensor, expected_input)
        np.testing.assert_array_equal(target_tensor, expected_target)

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
