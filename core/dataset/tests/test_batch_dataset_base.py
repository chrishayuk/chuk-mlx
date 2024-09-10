import os
import pytest
import numpy as np
import tempfile
from unittest.mock import patch
from core.dataset.batch_dataset_base import BatchDatasetBase
import memory_profiler
import gc
from core.utils.model_adapter import ModelAdapter

@pytest.fixture
def batch_dataset():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create some dummy npz batch files
        for i in range(10):
            np.savez(os.path.join(temp_dir, f"batch_{i}.npz"), 
                     input_tensor=np.random.rand(10, 10), 
                     target_tensor=np.random.rand(10), 
                     attention_mask_tensor=np.random.randint(0, 2, (10, 10)))
        
        # Initialize the model adapter
        adapter = ModelAdapter(framework="mlx")
        
        # Initialize the dataset with pre-caching disabled for controlled testing
        dataset = BatchDatasetBase(batch_output_dir=temp_dir, batchfile_prefix="batch_", pre_cache_size=2, model_adapter=adapter)
        yield dataset

def test_length(batch_dataset):
    # Test that the length of the dataset is correct
    assert len(batch_dataset) == 10

def test_getitem(batch_dataset):
    # Test that __getitem__ returns the correct shapes
    input_tensor, target_tensor, attention_mask_tensor = batch_dataset[0]
    assert input_tensor.shape == (10, 10)
    assert target_tensor.shape == (10,)
    
    # Verify attention mask shape if necessary
    assert attention_mask_tensor.shape == (10, 10), f"Expected attention_mask shape to be (10, 10), got {attention_mask_tensor.shape}"

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
    
    # Disable pre-caching to isolate cache behavior
    with patch.object(batch_dataset, '_queue_next_batches', return_value=None):
        # Ensure the item is retrieved from cache, not loaded again
        with patch.object(batch_dataset.model_adapter, 'load_tensor_from_file') as mock_load_batch_data:
            _ = batch_dataset[0]
            _ = batch_dataset[0]  # Access it again to check if it hits the cache
            mock_load_batch_data.assert_not_called()  # It should not load again if caching works

def test_cache_eviction(batch_dataset):
    # Set cache size to a smaller number for this test
    batch_dataset.cache_size = 2
    
    # Disable pre-caching to avoid multi-threading interference
    batch_dataset.pre_cache_size = 0

    # Access multiple items to fill and evict cache
    _ = batch_dataset[0]
    _ = batch_dataset[1]
    _ = batch_dataset[2]
    
    # Print cache state after initial accesses
    print(f"Cache after accessing batches 0, 1, 2: {list(batch_dataset.cache.keys())}")
    
    # Ensure that the oldest accessed batch (0) has been evicted
    assert 0 not in batch_dataset.cache, "Batch 0 should have been evicted from the cache."
    assert len(batch_dataset.cache) == 2, "Cache should contain exactly 2 items."

def test_epoch_end(batch_dataset):
    # End the epoch and check cache is cleared
    batch_dataset.on_epoch_end()
    assert batch_dataset.epoch == 1
    assert batch_dataset.current_index == -1
    assert not batch_dataset.cache

def test_error_handling(batch_dataset):
    # Test error handling in loading a batch
    with patch.object(batch_dataset, '_load_tensors', side_effect=Exception("Test error")):
        with pytest.raises(Exception):
            _ = batch_dataset[0]

def test_precache_thread(batch_dataset):
    # Ensure that pre-caching thread is running
    assert batch_dataset.pre_cache_thread.is_alive()
    
def test_precache_behavior(batch_dataset):
    # Enable pre-caching for this test with a larger cache size
    batch_dataset.pre_cache_size = 3  # Queue batches 0, 1, and 2
    batch_dataset._queue_next_batches(0)
    
    # Wait for the pre-caching thread to process the batches
    batch_dataset.load_queue.join()
    
    # Allow additional time for cache to be populated
    import time
    time.sleep(1)  # Increased wait time
    
    # Debugging: print cache content
    print("Cache content after pre-caching:", batch_dataset.cache.keys())
    
    # Check if the cache has been populated
    assert 1 in batch_dataset.cache, "Batch 1 should be pre-cached but isn't."
    assert 2 in batch_dataset.cache, "Batch 2 should be pre-cached but isn't."

def test_cache_across_epochs(batch_dataset):
    # Access a batch to populate the cache
    _ = batch_dataset[0]

    # Debugging: print cache content
    print("Cache after accessing batch 0:", batch_dataset.cache.keys())

    # Check if the batch is cached immediately after access
    assert 0 in batch_dataset.cache, "Batch 0 should be in cache."

    # End the epoch and check cache is cleared
    batch_dataset.on_epoch_end()
    assert not batch_dataset.cache, "Cache should be cleared after epoch end."

def test_memory_constraints(batch_dataset):
    # Set a very small cache size to simulate memory constraints
    batch_dataset.cache_size = 1

    # Access the first batch to populate the cache
    _ = batch_dataset[0]

    # Debugging: print cache content after accessing batch 0
    print("Cache after accessing batch 0:", batch_dataset.cache.keys())

    # Access the second batch to trigger eviction of the first
    _ = batch_dataset[1]

    # Debugging: print cache content after accessing batch 1
    print("Cache after accessing batch 1:", batch_dataset.cache.keys())

    # Ensure that the first batch was evicted and second batch is still in cache
    assert 0 not in batch_dataset.cache, "Batch 0 should have been evicted from cache."
    assert 1 in batch_dataset.cache, "Batch 1 should be in cache."


def test_zero_length_batch():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create an empty batch file (no tensors included)
        np.savez(os.path.join(temp_dir, "batch_0.npz"))
    
        # Initialize the model adapter
        adapter = ModelAdapter(framework="mlx")
        
        # Initialize the dataset
        dataset = BatchDatasetBase(batch_output_dir=temp_dir, batchfile_prefix="batch_", model_adapter=adapter)
    
        # Access the batch and expect an error
        with pytest.raises(KeyError, match="input_tensor"):
            _ = dataset[0]

def test_cleanup(batch_dataset):
    # Add some items to the queue
    batch_dataset._queue_next_batches(0)
    
    # Check that the pre-cache thread is running
    assert batch_dataset.pre_cache_thread.is_alive()
    
    # Signal the thread to stop by putting None in the queue
    batch_dataset.load_queue.put(None)
    
    # Wait for the thread to stop
    batch_dataset.pre_cache_thread.join(timeout=2)
    
    # Ensure the thread has stopped
    assert not batch_dataset.pre_cache_thread.is_alive(), "Pre-cache thread did not stop."


def test_memory_leak(batch_dataset):
    # Warm-up: load some batches to initialize everything
    for i in range(3):
        _ = batch_dataset[i]

    # Force a garbage collection before measuring memory
    gc.collect()

    # Measure memory usage before processing
    mem_before = memory_profiler.memory_usage()[0]

    # Load all batches and access them to simulate usage
    for i in range(len(batch_dataset)):
        _ = batch_dataset[i]

    # Force another garbage collection
    gc.collect()

    # Measure memory usage after processing
    mem_after = memory_profiler.memory_usage()[0]

    # Check that memory usage has not increased significantly
    assert mem_after <= mem_before + 10, "Memory usage increased significantly, possible memory leak detected."
