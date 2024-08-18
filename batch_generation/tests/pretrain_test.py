import os
import numpy as np
import pytest
import tempfile
from batch_generation.pretrain_batch import PretrainBatchGenerator

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

def test_save_batch(temp_output_dir, mock_tokenizer):
    generator = PretrainBatchGenerator(tokenizer=mock_tokenizer, output_directory=temp_output_dir, file_prefix='test',
                                       max_sequence_length=5, batch_size=2, print_summaries=False)
    
    batch_data = [[1, 2, 3], [4, 5]]
    file_path = os.path.join(temp_output_dir, "test_batch.npz")

    input_tensor = generator.save_batch(batch_data, file_path)

    assert input_tensor.shape == (2, generator.max_sequence_length)
    assert os.path.exists(file_path)

    data = np.load(file_path)

    assert 'input_tensor' in data
    assert 'target_tensor' in data
    assert data['input_tensor'].shape == (2, generator.max_sequence_length)

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
                                       max_sequence_length=5, batch_size=2, print_summaries=False)

    batch_data = [[1, 2, 3], [4, 5], [6]]

    with tempfile.NamedTemporaryFile(delete=False) as temp_batch_file:
        file_path = temp_batch_file.name

    try:
        input_tensor = generator.save_batch(batch_data, file_path)

        expected_output = np.array([
            [1, 2, 3, generator.tokenizer.eos_token_id, generator.tokenizer.pad_token_id],
            [4, 5, generator.tokenizer.eos_token_id, generator.tokenizer.pad_token_id, generator.tokenizer.pad_token_id],
            [6, generator.tokenizer.eos_token_id, generator.tokenizer.pad_token_id, generator.tokenizer.pad_token_id, generator.tokenizer.pad_token_id]
        ], dtype=np.int32)

        assert np.array_equal(input_tensor, expected_output), f"Expected {expected_output}, but got {input_tensor}"

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

def test_padding_and_eos_short_sequence(mock_tokenizer):
    generator = PretrainBatchGenerator(tokenizer=mock_tokenizer, output_directory=tempfile.gettempdir(), file_prefix='test',
                                       max_sequence_length=5, batch_size=2, print_summaries=False)

    batch_data = [[1, 2, 3]]  # Length 3
    file_path = os.path.join(tempfile.gettempdir(), "test_batch.npz")

    try:
        input_tensor = generator.save_batch(batch_data, file_path)

        expected_output = np.array([
            [1, 2, 3, generator.tokenizer.eos_token_id, generator.tokenizer.pad_token_id]  
        ], dtype=np.int32)

        assert np.array_equal(input_tensor, expected_output), f"Expected {expected_output}, but got {input_tensor}"

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

def test_padding_and_eos_exact_sequence(mock_tokenizer):
    generator = PretrainBatchGenerator(tokenizer=mock_tokenizer, output_directory=tempfile.gettempdir(), file_prefix='test',
                                       max_sequence_length=5, batch_size=2, print_summaries=False)

    batch_data = [[1, 2, 3, 4]]  # Length 4
    file_path = os.path.join(tempfile.gettempdir(), "test_batch.npz")

    try:
        input_tensor = generator.save_batch(batch_data, file_path)

        expected_output = np.array([
            [1, 2, 3, 4, generator.tokenizer.eos_token_id]  
        ], dtype=np.int32)

        assert np.array_equal(input_tensor, expected_output), f"Expected {expected_output}, but got {input_tensor}"

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

def test_padding_and_eos_max_length_sequence(mock_tokenizer):
    generator = PretrainBatchGenerator(tokenizer=mock_tokenizer, output_directory=tempfile.gettempdir(), file_prefix='test',
                                       max_sequence_length=5, batch_size=2, print_summaries=False)

    batch_data = [[1, 2, 3, 4, 5]]  # Length 5 (max_sequence_length)
    file_path = os.path.join(tempfile.gettempdir(), "test_batch.npz")

    try:
        input_tensor = generator.save_batch(batch_data, file_path)

        expected_output = np.array([
            [1, 2, 3, 4, generator.tokenizer.eos_token_id]  
        ], dtype=np.int32)

        assert np.array_equal(input_tensor, expected_output), f"Expected {expected_output}, but got {input_tensor}"

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

def test_truncate_and_eos_at_max_length(mock_tokenizer):
    generator = PretrainBatchGenerator(tokenizer=mock_tokenizer, output_directory=tempfile.gettempdir(), file_prefix='test',
                                       max_sequence_length=5, batch_size=2, print_summaries=False)

    batch_data = [[1, 2, 3, 4, 5, 6]]  
    file_path = os.path.join(tempfile.gettempdir(), "test_batch.npz")

    try:
        input_tensor = generator.save_batch(batch_data, file_path)

        expected_output = np.array([
            [1, 2, 3, 4, generator.tokenizer.eos_token_id]  
        ], dtype=np.int32)

        assert np.array_equal(input_tensor, expected_output), f"Expected {expected_output}, but got {input_tensor}"

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

def test_pad_when_sequence_short(mock_tokenizer):
    generator = PretrainBatchGenerator(tokenizer=mock_tokenizer, output_directory=tempfile.gettempdir(), file_prefix='test',
                                       max_sequence_length=5, batch_size=2, print_summaries=False)

    batch_data = [[1, 2]]  
    file_path = os.path.join(tempfile.gettempdir(), "test_batch.npz")

    try:
        input_tensor = generator.save_batch(batch_data, file_path)

        expected_output = np.array([
            [1, 2, generator.tokenizer.eos_token_id, generator.tokenizer.pad_token_id, generator.tokenizer.pad_token_id]  
        ], dtype=np.int32)

        assert np.array_equal(input_tensor, expected_output), f"Expected {expected_output}, but got {input_tensor}"

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

def test_exact_max_length_without_eos(mock_tokenizer):
    generator = PretrainBatchGenerator(tokenizer=mock_tokenizer, output_directory=tempfile.gettempdir(), file_prefix='test',
                                       max_sequence_length=5, batch_size=2, print_summaries=False)

    batch_data = [[1, 2, 3, 4, 5]]  
    file_path = os.path.join(tempfile.gettempdir(), "test_batch.npz")

    try:
        input_tensor = generator.save_batch(batch_data, file_path)

        expected_output = np.array([
            [1, 2, 3, 4, generator.tokenizer.eos_token_id]  
        ], dtype=np.int32)

        assert np.array_equal(input_tensor, expected_output), f"Expected {expected_output}, but got {input_tensor}"

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

def test_exact_max_length_with_eos(mock_tokenizer):
    generator = PretrainBatchGenerator(tokenizer=mock_tokenizer, output_directory=tempfile.gettempdir(), file_prefix='test',
                                       max_sequence_length=5, batch_size=2, print_summaries=False)

    batch_data = [[1, 2, 3, 4, 1]]  
    file_path = os.path.join(tempfile.gettempdir(), "test_batch.npz")

    try:
        input_tensor = generator.save_batch(batch_data, file_path)

        expected_output = np.array([
            [1, 2, 3, 4, generator.tokenizer.eos_token_id]  
        ], dtype=np.int32)

        assert np.array_equal(input_tensor, expected_output), f"Expected {expected_output}, but got {input_tensor}"

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

def test_longer_than_max_length(mock_tokenizer):
    generator = PretrainBatchGenerator(tokenizer=mock_tokenizer, output_directory=tempfile.gettempdir(), file_prefix='test',
                                       max_sequence_length=5, batch_size=2, print_summaries=False)

    batch_data = [[1, 2, 3, 4, 5, 6]]  
    file_path = os.path.join(tempfile.gettempdir(), "test_batch.npz")

    try:
        input_tensor = generator.save_batch(batch_data, file_path)

        expected_output = np.array([
            [1, 2, 3, 4, generator.tokenizer.eos_token_id]  
        ], dtype=np.int32)

        assert np.array_equal(input_tensor, expected_output), f"Expected {expected_output}, but got {input_tensor}"

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

def test_shorter_than_max_length_without_eos(mock_tokenizer):
    generator = PretrainBatchGenerator(tokenizer=mock_tokenizer, output_directory=tempfile.gettempdir(), file_prefix='test',
                                       max_sequence_length=5, batch_size=2, print_summaries=False)

    batch_data = [[1, 2, 3]]  
    file_path = os.path.join(tempfile.gettempdir(), "test_batch.npz")

    try:
        input_tensor = generator.save_batch(batch_data, file_path)

        expected_output = np.array([
            [1, 2, 3, generator.tokenizer.eos_token_id, generator.tokenizer.pad_token_id]  
        ], dtype=np.int32)

        assert np.array_equal(input_tensor, expected_output), f"Expected {expected_output}, but got {input_tensor}"

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

def test_shorter_than_max_length_with_eos(mock_tokenizer):
    generator = PretrainBatchGenerator(tokenizer=mock_tokenizer, output_directory=tempfile.gettempdir(), file_prefix='test',
                                       max_sequence_length=5, batch_size=2, print_summaries=False)

    batch_data = [[1, 2, 3, 1]]  
    file_path = os.path.join(tempfile.gettempdir(), "test_batch.npz")

    try:
        input_tensor = generator.save_batch(batch_data, file_path)

        expected_output = np.array([
            [1, 2, 3, generator.tokenizer.eos_token_id, generator.tokenizer.pad_token_id]  
        ], dtype=np.int32)

        assert np.array_equal(input_tensor, expected_output), f"Expected {expected_output}, but got {input_tensor}"

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
