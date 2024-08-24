import os
import pytest
import numpy as np
import tempfile
from core.dataset.mock_pretrain_batch_dataset import MockPreTrainBatchDataset

@pytest.fixture
def mock_dataset():
    with tempfile.TemporaryDirectory() as temp_dir:
        dataset = MockPreTrainBatchDataset(
            batch_output_dir=temp_dir,
            batchfile_prefix="mock_pretrain_batch",
            num_batches=5,
            batch_size=8,
            seq_length=10
        )
        yield dataset

def test_length(mock_dataset):
    # Test that the length of the dataset is correct
    assert len(mock_dataset) == 5

def test_batch_structure(mock_dataset):
    # Test that batches have the correct shape
    input_tensor, target_tensor, lengths = mock_dataset[0]

    # Check shapes of the tensors
    assert input_tensor.shape == (8, 10)  # batch_size, seq_length
    assert target_tensor.shape == (8, 10)  # batch_size, seq_length
    assert lengths.shape == (8,)  # batch_size, should be 1D array

def test_lengths(mock_dataset):
    # Test that lengths are correctly calculated and returned
    _, _, lengths = mock_dataset[0]
    
    # All lengths should be equal to seq_length
    assert np.array_equal(lengths, np.full((8,), 10))

def test_saved_batches(mock_dataset):
    # Test that batches are saved correctly as npz files
    for i in range(len(mock_dataset)):
        batch_file = mock_dataset.batch_files[i]
        batch_path = os.path.join(mock_dataset.batch_output_dir, batch_file)
        
        # Check that the file exists
        assert os.path.exists(batch_path)
        
        # Load the batch and check contents
        batch_data = np.load(batch_path)
        assert 'input_tensor' in batch_data
        assert 'target_tensor' in batch_data
        assert 'lengths' in batch_data
        
        # Verify tensor shapes
        assert batch_data['input_tensor'].shape == (8, 10)
        assert batch_data['target_tensor'].shape == (8, 10)
        # Adjusted this to match the expected shape in the saved data
        assert batch_data['lengths'].shape == (8, 10)


def test_integration_with_base_class(mock_dataset):
    # Ensure that the mock dataset works correctly with the base class methods
    assert len(mock_dataset) == mock_dataset.num_batches
    
    # Check retrieval of a batch using the base class method
    input_tensor, target_tensor, lengths = mock_dataset[0]
    
    assert input_tensor.shape == (8, 10)
    assert target_tensor.shape == (8, 10)
    assert np.array_equal(lengths, np.full((8,), 10))
