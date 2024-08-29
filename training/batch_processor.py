import time
import logging
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

    def process_batch(self, batch, batch_index, iteration_count):
        # Start the batch timer
        batch_start_time = time.time()

        # Initialize variables
        expected_tokens = 0

        if isinstance(batch, tuple) and len(batch) == 4:
            # get the tensors from the batch
            input_tensor, target_tensor, attention_mask_tensor, _ = batch

            # get the expected tokens
            expected_tokens = input_tensor.shape[0] * input_tensor.shape[1]

        # calculate the current learning rate
        current_lr = schedule_learning_rate(self.optimizer, iteration_count, self.warmup_steps)

        # store the learning rate before update
        lr_before_update = float(current_lr) if isinstance(current_lr, (int, float)) else current_lr.item()

        try:
            # execute the loss function
            (lvalue, ntoks), grad = self.loss_function(self.model, input_tensor, target_tensor, attention_mask_tensor)

            # update the optimizer
            self.optimizer.update(self.model, grad)
        except RuntimeError as e:
            # we had a issue with memory profiling
            logger.error(f"Memory profiling error in batch {batch_index}: {e}")

        # end time
        batch_end_time = time.time()
        batch_time = batch_end_time - batch_start_time

        return {
            "loss": lvalue if isinstance(lvalue, (float, int)) else lvalue.item(),
            "ntoks": expected_tokens,
            "expected_tokens": expected_tokens,
            "batch_time": batch_time,
            "lr_before_update": lr_before_update
        }
