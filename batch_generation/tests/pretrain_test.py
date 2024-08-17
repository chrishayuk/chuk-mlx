import os
import numpy as np
import pytest
import tempfile
from unittest.mock import MagicMock

from batch_generation.pretrain_batch import save_batch, process_batch, tokenize_and_batch

@pytest.fixture
def temp_output_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    tokenizer.encode = lambda text, max_length, truncation, add_special_tokens: list(range(1, len(text) + 1))
    return tokenizer

def test_save_batch(temp_output_dir, mock_tokenizer):
    # create some batch data
    batch_data = [[1, 2, 3], [4, 5]]
    file_path = f"{temp_output_dir}/test_batch.npz"
    max_sequence_length = 5
    pad_token_id = 0
    initial_pad_token_id = 0
    eos_token_id = 1

    # save the batch, and load into an input tensor
    input_tensor = save_batch(batch_data, file_path, max_sequence_length, pad_token_id, initial_pad_token_id, eos_token_id)
    
    # Validate the shape of the tensor
    assert input_tensor.shape == (2, max_sequence_length)

    # Validate that the saved file exists
    assert os.path.exists(file_path)

    # Load and validate the contents of the .npz file
    data = np.load(file_path)

    # ensure input, target all match
    assert 'input_tensor' in data
    assert 'target_tensor' in data
    assert data['input_tensor'].shape == (2, max_sequence_length)

def test_process_batch(temp_output_dir, mock_tokenizer, monkeypatch):
     # create some batch data
    batch_data = [[1, 2, 3], [4, 5]]
    file_path = f"{temp_output_dir}/test_batch.npz"
    max_sequence_length = 5
    pad_token_id = 0
    initial_pad_token_id = 0
    eos_token_id = 1

    # do some monkey patching
    monkeypatch.setattr('batch_generation.pretrain_target_batch_generator.create_target_batch', lambda *args, **kwargs: (np.array([[1], [1]]), [3, 2]))
    monkeypatch.setattr('batch_generation.sequence_utility.SequenceUtility.batch_sequences', lambda self, batch: batch)

    # process the batch
    process_batch(0, batch_data, file_path, max_sequence_length, pad_token_id, initial_pad_token_id, eos_token_id, print_summaries=False)

    # Validate that the saved file exists
    assert os.path.exists(file_path)

def test_tokenize_and_batch(temp_output_dir, mock_tokenizer, monkeypatch):
    # Use a temporary file for the input
    with tempfile.NamedTemporaryFile(delete=False) as temp_input_file:
        input_file_name = temp_input_file.name
        temp_input_file.write(b"Hello\nWorld\n")

    # input file list
    input_files = [input_file_name]
    file_prefix = "test"
    max_sequence_length = 5
    batch_size = 2
    print_summaries = False

    try:
        # Patch the necessary methods
        monkeypatch.setattr('batch_generation.pretrain_target_batch_generator.create_target_batch', lambda *args, **kwargs: (np.array([[1], [1]]), [3, 2]))

        # tokenize and batch using the mock tokenizer directly
        tokenize_and_batch(input_files, mock_tokenizer, temp_output_dir, file_prefix, max_sequence_length, batch_size, print_summaries)

        # Validate the number of batch files created
        batch_files = [f for f in os.listdir(temp_output_dir) if f.endswith('.npz')]
        assert len(batch_files) == 1

    finally:
        # Clean up the temporary input file
        if os.path.exists(input_file_name):
            os.remove(input_file_name)

def test_padding_in_batches(mock_tokenizer):
    # Define a batch with varying sequence lengths
    batch_data = [
        [1, 2, 3],  # Length 3
        [4, 5],     # Length 2
        [6]         # Length 1
    ]
    max_sequence_length = 5
    pad_token_id = 0
    initial_pad_token_id = 0
    eos_token_id = 1

    with tempfile.NamedTemporaryFile(delete=False) as temp_batch_file:
        file_path = temp_batch_file.name

    try:
        # Call the save_batch function directly
        input_tensor = save_batch(batch_data, file_path, max_sequence_length, pad_token_id, initial_pad_token_id, eos_token_id)

        # Expected output:
        expected_output = np.array([
            [1, 2, 3, eos_token_id, pad_token_id],
            [4, 5, eos_token_id, pad_token_id, pad_token_id],
            [6, eos_token_id, pad_token_id, pad_token_id, pad_token_id]
        ], dtype=np.int32)

        # Assert that the input tensor matches the expected output
        assert np.array_equal(input_tensor, expected_output), f"Expected {expected_output}, but got {input_tensor}"

    finally:
        # Clean up the temporary batch file
        if os.path.exists(file_path):
            os.remove(file_path)

def test_padding_and_eos_short_sequence():
    batch_data = [
        [1, 2, 3]  # Length 3
    ]
    max_sequence_length = 5
    pad_token_id = 0
    eos_token_id = 1

    with tempfile.NamedTemporaryFile(delete=False) as temp_batch_file:
        file_path = temp_batch_file.name

    try:
        # Call the save_batch function
        input_tensor = save_batch(batch_data, file_path, max_sequence_length, pad_token_id, pad_token_id, eos_token_id)

        expected_output = np.array([
            [1, 2, 3, eos_token_id, pad_token_id]  # 1, 2, 3 + EOS + PAD
        ], dtype=np.int32)

        assert np.array_equal(input_tensor, expected_output), f"Expected {expected_output}, but got {input_tensor}"

    finally:
        # Clean up the temporary batch file
        if os.path.exists(file_path):
            os.remove(file_path)

def test_padding_and_eos_exact_sequence():
    batch_data = [
        [1, 2, 3, 4]  # Length 4
    ]
    max_sequence_length = 5
    pad_token_id = 0
    eos_token_id = 1

    with tempfile.NamedTemporaryFile(delete=False) as temp_batch_file:
        file_path = temp_batch_file.name

    try:
        # Call the save_batch function
        input_tensor = save_batch(batch_data, file_path, max_sequence_length, pad_token_id, pad_token_id, eos_token_id)

        expected_output = np.array([
            [1, 2, 3, 4, eos_token_id]  # 1, 2, 3, 4 + EOS
        ], dtype=np.int32)

        assert np.array_equal(input_tensor, expected_output), f"Expected {expected_output}, but got {input_tensor}"

    finally:
        # Clean up the temporary batch file
        if os.path.exists(file_path):
            os.remove(file_path)

def test_padding_and_eos_max_length_sequence():
    batch_data = [
        [1, 2, 3, 4, 5]  # Length 5 (max_sequence_length)
    ]
    max_sequence_length = 5
    pad_token_id = 0
    eos_token_id = 1

    with tempfile.NamedTemporaryFile(delete=False) as temp_batch_file:
        file_path = temp_batch_file.name

    try:
        # Call the save_batch function
        input_tensor = save_batch(batch_data, file_path, max_sequence_length, pad_token_id, pad_token_id, eos_token_id)

        expected_output = np.array([
            [1, 2, 3, 4, eos_token_id]  # 1, 2, 3, 4, 5 -> 1, 2, 3, 4 + EOS (replace last element)
        ], dtype=np.int32)

        assert np.array_equal(input_tensor, expected_output), f"Expected {expected_output}, but got {input_tensor}"

    finally:
        # Clean up the temporary batch file
        if os.path.exists(file_path):
            os.remove(file_path)

def test_truncate_and_eos_at_max_length():
    batch_data = [
        [1, 2, 3, 4, 5, 6]  # Length 6 (more than max_sequence_length)
    ]
    max_sequence_length = 5
    pad_token_id = 0
    eos_token_id = 1

    with tempfile.NamedTemporaryFile(delete=False) as temp_batch_file:
        file_path = temp_batch_file.name

    try:
        input_tensor = save_batch(batch_data, file_path, max_sequence_length, pad_token_id, pad_token_id, eos_token_id)

        expected_output = np.array([
            [1, 2, 3, 4, eos_token_id]  # Truncated to 4 + EOS
        ], dtype=np.int32)

        assert np.array_equal(input_tensor, expected_output), f"Expected {expected_output}, but got {input_tensor}"

    finally:
        # Clean up the temporary batch file
        if os.path.exists(file_path):
            os.remove(file_path)

def test_pad_when_sequence_short():
    batch_data = [
        [1, 2]  # Length 2 (shorter than max_sequence_length)
    ]
    max_sequence_length = 5
    pad_token_id = 0
    eos_token_id = 1

    with tempfile.NamedTemporaryFile(delete=False) as temp_batch_file:
        file_path = temp_batch_file.name

    try:
        input_tensor = save_batch(batch_data, file_path, max_sequence_length, pad_token_id, pad_token_id, eos_token_id)

        expected_output = np.array([
            [1, 2, eos_token_id, pad_token_id, pad_token_id]  # 1, 2 + EOS + PAD + PAD
        ], dtype=np.int32)

        assert np.array_equal(input_tensor, expected_output), f"Expected {expected_output}, but got {input_tensor}"

    finally:
        # Clean up the temporary batch file
        if os.path.exists(file_path):
            os.remove(file_path)

def test_exact_max_length_without_eos():
    batch_data = [
        [1, 2, 3, 4, 5]  # Length 5 (max_sequence_length)
    ]
    max_sequence_length = 5
    pad_token_id = 0
    eos_token_id = 1

    with tempfile.NamedTemporaryFile(delete=False) as temp_batch_file:
        file_path = temp_batch_file.name

    try:
        input_tensor = save_batch(batch_data, file_path, max_sequence_length, pad_token_id, pad_token_id, eos_token_id)

        expected_output = np.array([
            [1, 2, 3, 4, eos_token_id]  # Replace last element with EOS
        ], dtype=np.int32)

        assert np.array_equal(input_tensor, expected_output), f"Expected {expected_output}, but got {input_tensor}"

    finally:
        # Clean up the temporary batch file
        if os.path.exists(file_path):
            os.remove(file_path)

def test_exact_max_length_with_eos():
    batch_data = [
        [1, 2, 3, 4, 1]  # Length 5 (max_sequence_length) with EOS at end
    ]
    max_sequence_length = 5
    pad_token_id = 0
    eos_token_id = 1

    with tempfile.NamedTemporaryFile(delete=False) as temp_batch_file:
        file_path = temp_batch_file.name

    try:
        input_tensor = save_batch(batch_data, file_path, max_sequence_length, pad_token_id, pad_token_id, eos_token_id)

        expected_output = np.array([
            [1, 2, 3, 4, eos_token_id]  # No change
        ], dtype=np.int32)

        assert np.array_equal(input_tensor, expected_output), f"Expected {expected_output}, but got {input_tensor}"

    finally:
        # Clean up the temporary batch file
        if os.path.exists(file_path):
            os.remove(file_path)

def test_longer_than_max_length():
    batch_data = [
        [1, 2, 3, 4, 5, 6]  # Length 6 (more than max_sequence_length)
    ]
    max_sequence_length = 5
    pad_token_id = 0
    eos_token_id = 1

    with tempfile.NamedTemporaryFile(delete=False) as temp_batch_file:
        file_path = temp_batch_file.name

    try:
        input_tensor = save_batch(batch_data, file_path, max_sequence_length, pad_token_id, pad_token_id, eos_token_id)

        expected_output = np.array([
            [1, 2, 3, 4, eos_token_id]  # Truncate to 4 + EOS
        ], dtype=np.int32)

        assert np.array_equal(input_tensor, expected_output), f"Expected {expected_output}, but got {input_tensor}"

    finally:
        # Clean up the temporary batch file
        if os.path.exists(file_path):
            os.remove(file_path)

def test_shorter_than_max_length_without_eos():
    batch_data = [
        [1, 2, 3]  # Length 3 (less than max_sequence_length)
    ]
    max_sequence_length = 5
    pad_token_id = 0
    eos_token_id = 1

    with tempfile.NamedTemporaryFile(delete=False) as temp_batch_file:
        file_path = temp_batch_file.name

    try:
        input_tensor = save_batch(batch_data, file_path, max_sequence_length, pad_token_id, pad_token_id, eos_token_id)

        expected_output = np.array([
            [1, 2, 3, eos_token_id, pad_token_id]  # Append EOS and pad
        ], dtype=np.int32)

        assert np.array_equal(input_tensor, expected_output), f"Expected {expected_output}, but got {input_tensor}"

    finally:
        # Clean up the temporary batch file
        if os.path.exists(file_path):
            os.remove(file_path)

def test_shorter_than_max_length_with_eos():
    batch_data = [
        [1, 2, 3, 1]  # Length 4 (less than max_sequence_length) with EOS at end
    ]
    max_sequence_length = 5
    pad_token_id = 0
    eos_token_id = 1

    with tempfile.NamedTemporaryFile(delete=False) as temp_batch_file:
        file_path = temp_batch_file.name

    try:
        input_tensor = save_batch(batch_data, file_path, max_sequence_length, pad_token_id, pad_token_id, eos_token_id)

        expected_output = np.array([
            [1, 2, 3, eos_token_id, pad_token_id]  # Just pad
        ], dtype=np.int32)

        assert np.array_equal(input_tensor, expected_output), f"Expected {expected_output}, but got {input_tensor}"

    finally:
        # Clean up the temporary batch file
        if os.path.exists(file_path):
            os.remove(file_path)
