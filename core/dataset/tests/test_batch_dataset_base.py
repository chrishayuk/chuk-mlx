import os
import pytest
import numpy as np
import tempfile
import mlx.core as mx
from unittest.mock import patch
from core.dataset.batch_dataset_base import BatchDatasetBase

@pytest.fixture
def batch_dataset():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create some dummy npz batch files
        for i in range(10):
            np.savez(os.path.join(temp_dir, f"batch_{i}.npz"), 
                     input_tensor=np.random.rand(10, 10), 
                     target_tensor=np.random.rand(10), 
                     input_lengths=np.array([10]*10))
        
        # Initialize the dataset with pre-caching disabled for controlled testing
        dataset = BatchDatasetBase(batch_output_dir=temp_dir, batchfile_prefix="batch_", shuffle=True)
        dataset.pre_cache_size = 0  # Disable pre-caching during testing
        yield dataset

def test_length(batch_dataset):
    # Test that the length of the dataset is correct
    assert len(batch_dataset) == 10

def test_getitem(batch_dataset):
    # Test that __getitem__ returns the correct shapes
    input_tensor, target_tensor, lengths = batch_dataset[0]
    assert input_tensor.shape == (10, 10)
    assert target_tensor.shape == (10,)
    assert lengths.shape == (10,)

def test_getitem_out_of_range(batch_dataset):
    # Test that __getitem__ raises IndexError when out of range
    with pytest.raises(IndexError):
        _ = batch_dataset[20]

def test_cache(batch_dataset):
    # Increase cache size to avoid eviction during the test
    batch_dataset.cache_size = 20
    
    # Directly load and cache the first batch
    batch_file = batch_dataset._get_batch_file(0)
    batch_dataset.cache[0] = batch_dataset._load_tensors(batch_file)
    
    # Ensure the item is retrieved from cache, not loaded again
    with patch.object(batch_dataset, '_load_tensors') as mock_load_tensors:
        _ = batch_dataset[0]
        mock_load_tensors.assert_not_called()

def test_cache_eviction(batch_dataset):
    # Set cache size to a smaller number for this test
    batch_dataset.cache_size = 2
    
    # Access multiple items to fill and evict cache
    _ = batch_dataset[0]
    _ = batch_dataset[1]
    _ = batch_dataset[2]
    
    # Access the first one again, it should reload since it was evicted
    with patch.object(batch_dataset, '_load_tensors') as mock_load_tensors:
        _ = batch_dataset[0]
        mock_load_tensors.assert_called_once()

def test_epoch_end(batch_dataset):
    # Shuffle and clear cache on epoch end
    batch_dataset.on_epoch_end()

    assert batch_dataset.epoch == 1
    assert batch_dataset.current_index == -1
    assert not batch_dataset.cache

def test_shuffle(batch_dataset):
    # Ensure shuffling changes the order
    batch_dataset.on_epoch_end()
    initial_order = [batch_dataset._get_batch_file(i) for i in range(len(batch_dataset))]
    batch_dataset.on_epoch_end()
    shuffled_order = [batch_dataset._get_batch_file(i) for i in range(len(batch_dataset))]
    
    assert initial_order != shuffled_order

def test_error_handling(batch_dataset):
    # Test error handling in loading a batch
    with patch.object(batch_dataset, '_load_tensors', side_effect=Exception("Test error")):
        with pytest.raises(Exception):
            _ = batch_dataset[0]

def test_precache_thread(batch_dataset):
    # Ensure that pre-caching thread is running
    assert batch_dataset.pre_cache_thread.is_alive()

def test_cleanup(batch_dataset):
    # Test cleanup and thread joining
    del batch_dataset  # This should trigger __del__ and stop the thread
