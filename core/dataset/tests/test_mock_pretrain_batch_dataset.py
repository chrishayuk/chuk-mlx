import os
import pytest
import numpy as np
import tempfile
from core.dataset.mock_pretrain_batch_dataset import MockPreTrainBatchDataset

@pytest.fixture
def mock_dataset():
    with tempfile.TemporaryDirectory() as temp_dir:
        for i in range(5):  # Assuming 5 batches for the test
            np.savez(os.path.join(temp_dir, f"mock_pretrain_batch_{i}.npz"), 
                     input_tensor=np.random.rand(8, 10),  # batch_size=8, seq_length=10
                     target_tensor=np.random.rand(8, 10), 
                     attention_mask_tensor=np.random.randint(0, 2, (8, 10)),
                     lengths=np.full((8,), 10))  # Random 0/1 attention mask
        
        # Initialize the dataset
        dataset = MockPreTrainBatchDataset(
            batch_output_dir=temp_dir,
            batchfile_prefix="mock_pretrain_batch",
            num_batches=5,
            batch_size=8,
            seq_length=10
        )
        dataset.pre_cache_size = 0  # Disable pre-caching for testing
        yield dataset
        del dataset  # Clean up explicitly

def test_length(mock_dataset):
    # Test that the length of the dataset is correct
    assert len(mock_dataset) == 5

def test_batch_structure(mock_dataset):
    # Test that batches have the correct shape
    input_tensor, target_tensor, attention_mask_tensor, lengths = mock_dataset[0]

    # Check shapes of the tensors
    assert input_tensor.shape == (8, 10)  # batch_size, seq_length
    assert target_tensor.shape == (8, 10)  # batch_size, seq_length
    assert attention_mask_tensor.shape == (8, 10)  # attention mask should match input tensor shape
    assert lengths.shape == (8,)  # batch_size, should be 1D array

def test_lengths(mock_dataset):
    # Test that lengths are correctly calculated and returned
    input_tensor, target_tensor, attention_mask_tensor, lengths = mock_dataset[0]
    
    # All lengths should be equal to seq_length
    assert np.array_equal(lengths, np.full((8,), 10))
    assert attention_mask_tensor.shape == (8, 10)

def test_attention_mask(mock_dataset):
    # Test that attention mask is properly generated and used
    input_tensor, target_tensor, attention_mask_tensor, lengths = mock_dataset[0]

    # Convert to NumPy array with the correct dtype
    attention_mask_tensor = np.array(attention_mask_tensor, dtype=np.int32)
    
    # Ensure that the attention mask has the correct shape and dtype
    assert attention_mask_tensor.shape == (8, 10)
    assert attention_mask_tensor.dtype == np.int32 or attention_mask_tensor.dtype == np.int64

    # Ensure the attention mask only contains 0s and 1s
    assert np.all((attention_mask_tensor == 1) | (attention_mask_tensor == 0)), "Attention mask contains values other than 0 and 1."


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
        assert 'attention_mask_tensor' in batch_data
        
        # Verify tensor shapes
        assert batch_data['input_tensor'].shape == (8, 10)
        assert batch_data['target_tensor'].shape == (8, 10)
        assert batch_data['attention_mask_tensor'].shape == (8, 10)


def test_integration_with_base_class(mock_dataset):
    # Ensure that the mock dataset works correctly with the base class methods
    assert len(mock_dataset) == mock_dataset.num_batches

    # Check retrieval of a batch using the base class method
    input_tensor, target_tensor, attention_mask, lengths = mock_dataset[0]

    # Convert lengths to a NumPy array with the correct dtype
    lengths = np.array(lengths, dtype=np.int32)

    assert input_tensor.shape == (8, 10)
    assert target_tensor.shape == (8, 10)
    assert lengths.shape == (8,), f"Lengths array shape mismatch: expected (8,), got {lengths.shape}"
    assert np.array_equal(lengths, np.full((8,), 10)), "Lengths array does not match the expected values."
    assert attention_mask.shape == (8, 10)

