import tracemalloc
from asyncio.log import logger
import gc
import os
import mlx.core as mx
import logging
from threading import Thread
from queue import Queue, Empty
import traceback
from collections import OrderedDict
import psutil

from core.utils.memory_utils import log_memory_usage 

class BatchDatasetBase:
    def __init__(self, batch_output_dir, batchfile_prefix, pre_cache_size=5, shuffle=False):
        # Initialize
        self.batch_output_dir = batch_output_dir
        self.batchfile_prefix = batchfile_prefix
        self.batch_files = []
        self.cache = OrderedDict()  # Use OrderedDict for LRU cache behavior
        self.cache_size = pre_cache_size  # Adjust this value based on your memory constraints
        self.pre_cache_size = pre_cache_size
        self.load_queue = Queue()
        self.current_index = -1
        self.epoch = 0

        # Load the batch files
        self._load_batch_files()

        # Start pre-caching in a separate thread to avoid blocking the main thread
        self._start_pre_caching()

    def _load_batch_files(self):
        # Loop through the batch folder
        for filename in os.listdir(self.batch_output_dir):
            # Check the batch file starts with the correct prefix and ends with .npz
            if filename.startswith(self.batchfile_prefix) and filename.endswith(".npz"):
                # Add the batch
                self.batch_files.append(filename)
        
        # Sort the batch files
        self.batch_files.sort()

    def _load_tensors(self, batch_file):
        # set the batch path
        batch_path = os.path.join(self.batch_output_dir, batch_file)
        
        try:
            # check the batch exists
            if not os.path.exists(batch_path):
                raise FileNotFoundError(f"Batch file {batch_file} does not exist at path {batch_path}")
            
            # debug memory
            #log_memory_usage("dataset loader: pre load batch")
            
            # load the batch
            batch_data = mx.load(batch_path)

            # debug memory
            #log_memory_usage("dataset loader: post load batch")

            # load the tensors
            input_tensor = batch_data['input_tensor']
            target_tensor = batch_data['target_tensor']
            attention_mask_tensor = batch_data['attention_mask_tensor']
            lengths = batch_data.get('input_lengths', mx.array([len(seq) for seq in input_tensor]))

            # debug memory
            #log_memory_usage("dataset loader: post load tensors")

            return input_tensor, target_tensor, attention_mask_tensor, lengths
        except FileNotFoundError as e:
            logger.error(f"Error loading batch file {batch_file}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error loading batch file {batch_file}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise


    def _queue_next_batches(self, start_index):
        for i in range(start_index, min(start_index + self.pre_cache_size, len(self.batch_files))):
            if i not in self.cache:
                self.load_queue.put(i)
                #print(f"Queued batch {i} for pre-caching.")

    def _pre_cache_worker(self):
        while True:
            try:
                index = self.load_queue.get(timeout=1)
                if index is None:
                    break

                if index not in self.cache and index < len(self.batch_files):
                    batch_file = self._get_batch_file(index)
                    tensors = self._load_tensors(batch_file)
                    self.cache[index] = tensors

                    # Move the accessed item to the end of the OrderedDict (LRU behavior)
                    self.cache.move_to_end(index)

                    # Log the current state
                    #print(f"Added batch {index} to cache. Current cache: {list(self.cache.keys())}")

                    # Evict the oldest entry if cache exceeds the size limit
                    while len(self.cache) > self.cache_size:
                        oldest_key = next(iter(self.cache))
                        del self.cache[oldest_key]
                        #print(f"Evicted batch {oldest_key} from cache. Current cache: {list(self.cache.keys())}")

                self.load_queue.task_done()
            except Empty:
                continue

    def _start_pre_caching(self):
        self.pre_cache_thread = Thread(target=self._pre_cache_worker)
        self.pre_cache_thread.daemon = True
        self.pre_cache_thread.start()

    def _get_batch_file(self, index):
        return self.batch_files[index]

    def __getitem__(self, index):
        if index >= len(self.batch_files):
            raise IndexError(f"Batch index {index} out of range")

        # Queue the next set of batches if we are accessing a new batch index
        if index > self.current_index:
            self._queue_next_batches(index + 1)
            self.current_index = index

        # Use psutil to monitor memory usage
        #memory_before = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        #logging.info(f"Memory usage before accessing batch {index}: {memory_before:.2f} MB")

        # Load the batch from cache if available
        if index not in self.cache:
            # logging.info(f"Batch {index} not in cache, loading directly")
            #log_memory_usage("dataset loader: get_item pre load batch")
            batch_file = self._get_batch_file(index)
            #log_memory_usage("dataset loader: get_item post load batch")
            self.cache[index] = self._load_tensors(batch_file)
            #log_memory_usage("dataset loader: get_item post load tensors")
            self.cache.move_to_end(index)  # Ensure LRU order
            #log_memory_usage("dataset loader: get_item post move to end")

            # Evict the oldest entry if cache exceeds the size limit
            if len(self.cache) > self.cache_size:
                evicted_index, _ = self.cache.popitem(last=False)
                #logging.info(f"Evicted batch {evicted_index} from cache.")

                # Queue the next batch to keep the cache full
                next_index = max(index + 1, evicted_index + 1)
                if next_index < len(self.batch_files):
                    self.load_queue.put(next_index)
                    #logging.info(f"Queued batch {next_index} after eviction.")

                #log_memory_usage("dataset loader: get_item post eviction")
            # Use psutil to monitor memory usage after eviction
            #memory_after_eviction = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            #logging.info(f"Memory usage after eviction batch {index}: {memory_after_eviction:.2f} MB")

        # Force garbage collection and log memory usage after collection
        # gc.collect()
        #memory_after_gc = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        # logging.info(f"Memory usage after GC batch {index}: {memory_after_gc:.2f} MB")

        return self.cache[index]

    def __len__(self):
        return len(self.batch_files)

    def __del__(self):
        tracemalloc.stop()  # Stop tracing memory allocations
        if hasattr(self, 'load_queue'):
            self.load_queue.put(None)
            self.pre_cache_thread.join(timeout=5)

    def on_epoch_end(self):
        self.epoch += 1
        self.cache.clear()
        self.current_index = -1
        gc.collect()
