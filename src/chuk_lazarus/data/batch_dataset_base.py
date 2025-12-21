import gc
import os
import random
import traceback
from asyncio.log import logger
from collections import OrderedDict
from queue import Empty, Queue
from threading import Event, Lock, Thread

from chuk_lazarus.utils import model_adapter


class BatchDatasetBase:
    def __init__(
        self,
        batch_output_dir,
        batchfile_prefix,
        pre_cache_size=5,
        shuffle=False,
        model_adapter=model_adapter.ModelAdapter(framework="mlx"),
    ):
        # Initialize batch output directory, file prefix, cache settings, and model adapter
        self.batch_output_dir = batch_output_dir
        self.batchfile_prefix = batchfile_prefix
        self.batch_files = []
        self.cache = OrderedDict()  # LRU cache behavior
        self.cache_size = pre_cache_size  # Max number of batches to keep in cache
        self.pre_cache_size = pre_cache_size
        self.load_queue = Queue()
        self.current_index = -1
        self.epoch = 0
        self.stop_event = Event()
        self.shuffle = shuffle
        self.model_adapter = model_adapter  # Framework-specific model adapter

        # cache lock
        self.cache_lock = Lock()

        # Load the list of batch files
        self._load_batch_files()

        # Shuffle batch files if shuffle=True
        if self.shuffle:
            random.shuffle(self.batch_files)

        # Start a thread for pre-caching batches
        self._start_pre_caching()

    def _load_batch_files(self):
        # Collect batch files matching the specified prefix and extension
        self.batch_files = sorted(
            filename
            for filename in os.listdir(self.batch_output_dir)
            if filename.startswith(self.batchfile_prefix) and filename.endswith(".npz")
        )

    def _load_tensors(self, batch_file):
        # Construct full path to the batch file
        batch_path = os.path.join(self.batch_output_dir, batch_file)

        try:
            # Ensure the batch file exists
            if not os.path.exists(batch_path):
                raise FileNotFoundError(
                    f"Batch file {batch_file} does not exist at path {batch_path}"
                )

            # Use the ModelAdapter to load the batch data in the correct framework format
            batch_data = self.model_adapter.load_tensor_from_file(batch_path)

            # Extract and return tensors from the loaded batch data
            input_tensor = batch_data["input_tensor"]
            target_tensor = batch_data["target_tensor"]
            attention_mask_tensor = batch_data["attention_mask_tensor"]

            return input_tensor, target_tensor, attention_mask_tensor
        except FileNotFoundError as e:
            logger.error(f"Error loading batch file {batch_file}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error loading batch file {batch_file}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def _queue_next_batches(self, start_index):
        # Queue the next set of batches for pre-caching
        for i in range(start_index, min(start_index + self.pre_cache_size, len(self.batch_files))):
            if i not in self.cache:
                self.load_queue.put(i)

    def _pre_cache_worker(self):
        # Worker thread for pre-caching batches
        while not self.stop_event.is_set():
            try:
                index = self.load_queue.get(timeout=0.1)
                if index is None:
                    break

                # Log the batch being processed in the pre-cache worker
                logger.debug(f"Pre-cache worker: Attempting to load batch {index}")

                # Load the batch and cache it if not already cached
                if index not in self.cache and index < len(self.batch_files):
                    batch_file = self._get_batch_file(index)
                    tensors = self._load_tensors(batch_file)

                    # Acquire lock to ensure cache synchronization
                    with self.cache_lock:
                        self.cache[index] = tensors
                        self.cache.move_to_end(index)  # Maintain LRU order

                        # Log the cache status
                        logger.debug(
                            f"Pre-cache worker: Cached batch {index}. Current cache keys: {list(self.cache.keys())}"
                        )

                        # Evict the oldest entry if the cache exceeds its size limit
                        if len(self.cache) > self.cache_size:
                            oldest_key = next(iter(self.cache))
                            del self.cache[oldest_key]
                            logger.info(
                                f"Pre-cache worker: Evicted batch {oldest_key} due to cache size limit."
                            )

                self.load_queue.task_done()
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error in pre-cache worker: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
            finally:
                pass  # Placeholder for any necessary cleanup

    def __getitem__(self, index):
        # Fetch the batch at the specified index
        if index >= len(self.batch_files):
            raise IndexError(f"Batch index {index} out of range")

        # Queue subsequent batches for pre-caching
        if index > self.current_index:
            self._queue_next_batches(index + 1)
            self.current_index = index

        # Acquire lock to synchronize cache access
        with self.cache_lock:
            # Load and cache the batch if not already cached
            if index not in self.cache:
                batch_file = self._get_batch_file(index)
                self.cache[index] = self._load_tensors(batch_file)
                self.cache.move_to_end(index)  # Maintain LRU order

                # Evict the oldest entry if the cache exceeds its size limit
                if len(self.cache) > self.cache_size:
                    evicted_index, _ = self.cache.popitem(last=False)
                    next_index = max(index + 1, evicted_index + 1)
                    if next_index < len(self.batch_files):
                        self.load_queue.put(next_index)

            # Log cache status after accessing a batch
            logger.debug(
                f"Main thread: Accessed batch {index}. Current cache keys: {list(self.cache.keys())}"
            )

        return self.cache[index]

    def _start_pre_caching(self):
        # Start the pre-cache worker thread
        self.pre_cache_thread = Thread(target=self._pre_cache_worker)
        self.pre_cache_thread.daemon = True
        self.pre_cache_thread.start()

    def _get_batch_file(self, index):
        # Return the batch file at the given index
        return self.batch_files[index]

    def __len__(self):
        # Return the total number of batches
        return len(self.batch_files)

    def __del__(self):
        # Cleanup resources and stop pre-caching thread
        try:
            if hasattr(self, "load_queue"):
                # Signal the thread to stop
                self.stop_event.set()

                # Unblock the queue
                self.load_queue.put(None)

                # clean up the thread
                if hasattr(self, "pre_cache_thread") and self.pre_cache_thread.is_alive():
                    self.pre_cache_thread.join(timeout=5)  # Wait for the thread to finish
        finally:
            gc.collect()  # Invoke garbage collection to free memory

    def on_epoch_end(self):
        # Stop pre-caching to prevent further cache modifications
        self.stop_event.set()

        # Ensure the pre-cache thread can exit
        self.load_queue.put(None)
        self.pre_cache_thread.join(timeout=5)

        # Move to the next epoch
        self.epoch += 1

        # Shuffle batch files if shuffle=True
        if self.shuffle:
            random.shuffle(self.batch_files)

        # Clear the cache at the end of each epoch
        self.cache.clear()

        # Reset the index
        self.current_index = -1

        # Explicitly invoke garbage collection to free memory
        gc.collect()

        # Restart the pre-caching thread for the next epoch
        self.stop_event.clear()
        self._start_pre_caching()
