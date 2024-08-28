import os
import time
import logging
import tracemalloc
import gc
import mlx.core as mx
from training.training_scheduler import schedule_learning_rate

# Set the logger
logger = logging.getLogger(__name__)

class BatchProcessor:
    def __init__(self, model, tokenizer, optimizer, loss_function, warmup_steps):
        # initalize
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.warmup_steps = warmup_steps

        #tracemalloc.start()  # Ensure tracing is started

    def process_batch(self, batch, batch_index, iteration_count):
        # memory debug
        #logger.debug(f"Processing batch {batch_index} - Starting memory usage: {self._get_memory_usage()} MB")

        # Start the batch timer
        batch_start_time = time.time()

        # Initialize variables
        expected_tokens = 0
        lengths = None

        if isinstance(batch, tuple) and len(batch) == 4:
            # get the tensors from the batch
            input_tensor, target_tensor, attention_mask_tensor, lengths = batch

            # load the tensors into mx.array
            input_tensor = mx.array(input_tensor)
            target_tensor = mx.array(target_tensor)
            attention_mask_tensor = mx.array(attention_mask_tensor)
            lengths = mx.array(lengths)

            # check the lengths
            if lengths.ndim > 1:
                # TODO: this feels like it could be optimized
                # get the lengths
                max_length = mx.max(lengths, axis=1, keepdims=True)
                lengths = mx.reshape(max_length, (-1, 1))
            else:
                # get the lengths
                max_length = mx.max(lengths)
                lengths = mx.reshape(max_length, (1,))

            # get the expected tokens
            expected_tokens = input_tensor.shape[0] * input_tensor.shape[1]

        # calculate the current learning rate
        current_lr = schedule_learning_rate(self.optimizer, iteration_count, self.warmup_steps)

        # store the learning rate before update
        lr_before_update = float(current_lr) if isinstance(current_lr, (int, float)) else current_lr.item()

        try:
            #snapshot_before_loss = tracemalloc.take_snapshot()
            #logger.info(f"Memory before loss calculation for batch {batch_index}: {self._format_top_stats(snapshot_before_loss)}")

            # execute the loss function
            (lvalue, ntoks), grad = self.loss_function(self.model, input_tensor, target_tensor, attention_mask_tensor, lengths)

            #snapshot_after_loss = tracemalloc.take_snapshot()
            #logger.info(f"Memory after loss calculation for batch {batch_index}: {self._format_top_stats(snapshot_after_loss)}")

            # update the optimizer
            self.optimizer.update(self.model, grad)

            #snapshot_after_update = tracemalloc.take_snapshot()
            #logger.info(f"Memory after optimizer update for batch {batch_index}: {self._format_top_stats(snapshot_after_update)}")
        except RuntimeError as e:
            # we had a issue with memory profiling
            logger.error(f"Memory profiling error in batch {batch_index}: {e}")

        # end time
        batch_end_time = time.time()
        batch_time = batch_end_time - batch_start_time

        # garbage collect
        # gc.collect()
        #logger.info(f"Memory usage after GC for batch {batch_index}: {self._get_memory_usage()} MB")

        return {
            "loss": lvalue if isinstance(lvalue, (float, int)) else lvalue.item(),
            "ntoks": expected_tokens,
            "expected_tokens": expected_tokens,
            "batch_time": batch_time,
            "lr_before_update": lr_before_update
        }

    def _format_top_stats(self, snapshot, key_type='lineno'):
        # get the top stats
        top_stats = snapshot.statistics(key_type)

        # return top stats
        return "\n".join([f"{stat}" for stat in top_stats[:5]])

    def _get_memory_usage(self):
        import psutil
        # get current process
        process = psutil.Process(os.getpid())

        # get memory usage
        mem_info = process.memory_info()

        # return memory ussage
        return f"RSS={mem_info.rss / (1024 ** 2):.2f}, VMS={mem_info.vms / (1024 ** 2):.2f} MB"
