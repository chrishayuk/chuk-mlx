import os
import pytest
import numpy as np
import tempfile
import mlx.core as mx
from unittest.mock import patch

from core.dataset.mock_finetune_batch_dataset import MockFineTuneBatchDataset

@pytest.fixture
def mock_dataset():
    with tempfile.TemporaryDirectory() as temp_dir:
        dataset = MockFineTuneBatchDataset(
            batch_output_dir=temp_dir,
            batchfile_prefix="mock_batch",
            num_batches=5,
            batch_size=8,
            seq_length=10,
            sep_token_id=99
        )
        yield dataset

def test_length(mock_dataset):
    # Test that the length of the dataset is correct
    assert len(mock_dataset) == 5

def test_batch_structure(mock_dataset):
    # Test that batches have the correct shape and separator token
    concatenated_tensor, lengths = mock_dataset[0]
    
    # Check shape of the tensor
    expected_shape = (8, mock_dataset.seq_length)  # Should be (batch_size, seq_length)
    assert concatenated_tensor.shape == expected_shape

    # Check that the separator token is correctly placed
    for seq in concatenated_tensor[:, mock_dataset.seq_length // 2 - 1]:
        assert seq == 99  # Separator token

def test_lengths(mock_dataset):
    # Test that lengths are correctly calculated and returned
    _, lengths = mock_dataset[0]
    
    # All lengths should be equal to seq_length // 2
    assert mx.all(lengths == mock_dataset.seq_length // 2)

def test_saved_batches(mock_dataset):
    # Test that batches are saved correctly as npz files
    for i in range(len(mock_dataset)):
        batch_file = mock_dataset.batch_files[i]
        batch_path = os.path.join(mock_dataset.batch_output_dir, batch_file)
        
        # Check that the file exists
        assert os.path.exists(batch_path)
        
        # Load the batch and check contents
        batch_data = mx.load(batch_path)
        assert 'concatenated_tensor' in batch_data
        assert 'lengths' in batch_data
        
        # Verify tensor shape
        expected_shape = (8, mock_dataset.seq_length)
        assert batch_data['concatenated_tensor'].shape == expected_shape

def test_separator_warning(capfd, mock_dataset):
    # Test the warning for missing separator tokens
    concatenated_tensor, _ = mock_dataset[0]

    # Modify the tensor to remove a separator token
    concatenated_tensor[0, mock_dataset.seq_length // 2 - 1] = 0  # Manually remove separator for first sequence

    # Simulate reloading and triggering the warning
    with patch.object(mock_dataset, '_load_tensor', return_value={'concatenated_tensor': concatenated_tensor, 'lengths': mx.full((8,), 5)}):
        mock_dataset[0]

    # Capture and verify warning output
    captured = capfd.readouterr()
    assert "Warning: No separator found" in captured.out
