import logging
import time

from chuk_lazarus.training.schedulers import schedule_learning_rate

# Set the logger
logger = logging.getLogger(__name__)


class BatchProcessor:
    def __init__(self, model, tokenizer, optimizer, loss_function, warmup_steps):
        # initialize
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.warmup_steps = warmup_steps

    def process_batch(self, batch, batch_index, iteration_count):
        # Start the batch timer
        batch_start_time = time.time()

        if isinstance(batch, tuple) and len(batch) == 3:
            # Get the tensors from the batch
            input_tensor, target_tensor, attention_mask_tensor = batch

        # Calculate the current learning rate
        current_lr = schedule_learning_rate(self.optimizer, iteration_count, self.warmup_steps)

        # Store the learning rate before update
        lr_before_update = (
            float(current_lr) if isinstance(current_lr, (int, float)) else current_lr.item()
        )

        try:
            # Execute the loss function, which returns loss and number of tokens (ntoks)
            (lvalue, ntoks), grad = self.loss_function(
                self.model, input_tensor, target_tensor, attention_mask_tensor
            )

            # Convert ntoks to a scalar if it's an array-like object
            ntoks = ntoks.item() if hasattr(ntoks, "item") else ntoks

            # Update the optimizer
            self.optimizer.update(self.model, grad)

        except RuntimeError as e:
            # Handle memory profiling errors
            logger.error(f"Memory profiling error in batch {batch_index}: {e}")

        # End time
        batch_end_time = time.time()
        batch_time = batch_end_time - batch_start_time

        # Calculate tokens per second (using ntoks for actual processed tokens)
        tokens_per_second = ntoks / batch_time if batch_time > 0 else 0

        return {
            "loss": lvalue.item()
            if hasattr(lvalue, "item")
            else lvalue,  # Convert loss to a scalar if necessary
            "ntoks": ntoks,
            "batch_time": batch_time,
            "tokens_per_second": tokens_per_second,
            "lr_before_update": lr_before_update,
        }
