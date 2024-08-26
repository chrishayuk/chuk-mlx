import os
import mlx.core as mx
import logging
from threading import Thread
from queue import Queue, Empty
import random
import traceback

# set the logger
logger = logging.getLogger(__name__)

class BatchDatasetBase:
    def __init__(self, batch_output_dir, batchfile_prefix, pre_cache_size=5, shuffle=False):
        self.batch_output_dir = batch_output_dir
        self.batchfile_prefix = batchfile_prefix
        self.batch_files = []
        self.cache = {}
        # Adjust this value based on your memory constraints
        self.cache_size = 10  
        self.pre_cache_size = pre_cache_size
        self.load_queue = Queue()
        self.current_index = -1
        self.shuffle = shuffle
        self.epoch = 0
        self.shuffled_indices = None

        # load the batch files
        self._load_batch_files()

        # cach in a separate thread to avoid blocking the main thread
        self._start_pre_caching()

    def _load_batch_files(self):
        # loop through the batch files folder
        for filename in os.listdir(self.batch_output_dir):
            # ensure it's a batch file
            if filename.startswith(self.batchfile_prefix) and filename.endswith(".npz"):
                self.batch_files.append(filename)

        # short the batch files
        self.batch_files.sort()

        # shuffle if configured
        self._maybe_shuffle()

    def _maybe_shuffle(self):
        # check if we should shuffle the batch files
        if self.shuffle:
            # log the shuffling
            logger.info(f"Shuffling batch order for epoch {self.epoch}")

            # shuffle
            self.shuffled_indices = list(range(len(self.batch_files)))
            random.shuffle(self.shuffled_indices)
        else:
            self.shuffled_indices = None

    def _load_tensors(self, batch_file):
        # get the path
        batch_path = os.path.join(self.batch_output_dir, batch_file)
        
        try:
            # load the file
            batch_data = mx.load(batch_path)

            # loading the input tensor, target tensor and lengths      
            input_tensor = batch_data['input_tensor']
            target_tensor = batch_data['target_tensor']
            attention_mask_tensor = batch_data['attention_mask_tensor']
            lengths = batch_data.get('input_lengths', mx.array([len(seq) for seq in input_tensor]))
            
            # return the tensors
            return input_tensor, target_tensor, attention_mask_tensor, lengths

        except Exception as e:
            # error
            logger.error(f"Error loading batch file {batch_file}: {str(e)}")
            logger.error(f"File exists: {os.path.exists(batch_path)}")
            logger.error(f"File size: {os.path.getsize(batch_path)} bytes")
            logger.error(f"Absolute file path: {os.path.abspath(batch_path)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def _pre_cache_worker(self):
        # loop
        while True:
            try:
                # Wait for 1 second
                index = self.load_queue.get(timeout=1) 

                # check if we're done
                if index is None:
                    break

                # check if we have the file
                if index not in self.cache and index < len(self.batch_files):
                    # get the file
                    batch_file = self._get_batch_file(index)

                    # load the tensors
                    tensors = self._load_tensors(batch_file)

                    # cache
                    self.cache[index] = tensors
                    
                    # Manage cache size
                    while len(self.cache) > self.cache_size:
                        oldest_key = min(k for k in self.cache.keys() if k != index)
                        del self.cache[oldest_key]
                
                # done
                self.load_queue.task_done()
            except Empty:
                continue  # If queue is empty, continue the loop

    def _start_pre_caching(self):
        # create the caching thread
        self.pre_cache_thread = Thread(target=self._pre_cache_worker)

        # Set as daemon thread
        self.pre_cache_thread.daemon = True  

        # start the thread
        self.pre_cache_thread.start()

    def _queue_next_batches(self, start_index):
        # get the batch files
        for i in range(start_index, min(start_index + self.pre_cache_size, len(self.batch_files))):
            # put the index on the queue
            if i not in self.cache:
                self.load_queue.put(i)

    def _get_batch_file(self, index):
        # check if we shuffled
        if self.shuffled_indices is not None:
            # get from the shuffled index
            return self.batch_files[self.shuffled_indices[index]]
        
        # return the file
        return self.batch_files[index]

    def __getitem__(self, index):
        # check if out range
        if index >= len(self.batch_files):
            raise IndexError(f"Batch index {index} out of range")
        
        # Queue pre-caching of next batches
        if index > self.current_index:
            self._queue_next_batches(index + 1)
            self.current_index = index

        # Try to get the batch from cache, if not available, load it directly
        if index not in self.cache:
            logger.info(f"Batch {index} not in cache, loading directly")
            batch_file = self._get_batch_file(index)
            return self._load_tensors(batch_file)
        
        # Return the batch from cache
        return self.cache[index]

    def __len__(self):
        return len(self.batch_files)

    def __del__(self):
        # Signal the pre-cache thread to exit
        self.load_queue.put(None)
        self.pre_cache_thread.join(timeout=5)  # Wait for up to 5 seconds

    def on_epoch_end(self):
        # Increment epoch
        self.epoch += 1

        # Shuffle the indices
        self._maybe_shuffle()

        # Clear cache
        self.cache.clear()
        self.current_index = -1