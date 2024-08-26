import time
import logging
import mlx.core as mx
from training.training_scheduler import schedule_learning_rate

# Set the logger
logger = logging.getLogger(__name__)

class BatchProcessor:
    def __init__(self, model, tokenizer, optimizer, loss_function, warmup_steps):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.warmup_steps = warmup_steps

    def process_batch(self, batch, batch_index, iteration_count):
        # Start the batch timer
        batch_start_time = time.time()

        #try:
        # Initialize variables
        expected_tokens = 0
        lengths = None

        # Check for a valid input batch tuple
        if isinstance(batch, tuple) and len(batch) == 4:
            # Unpack the batch into input, target tensors, attention mask, and lengths
            input_tensor, target_tensor, attention_mask_tensor, lengths = batch

            # Convert to mx arrays
            input_tensor = mx.array(input_tensor)
            target_tensor = mx.array(target_tensor)
            attention_mask_tensor = mx.array(attention_mask_tensor)
            lengths = mx.array(lengths)

            if lengths.ndim > 1:
                # Handle multi-dimensional lengths
                max_length = mx.max(lengths, axis=1, keepdims=True)
                lengths = mx.reshape(max_length, (-1, 1))
            else:
                # Handle single-dimensional lengths
                max_length = mx.max(lengths)
                lengths = mx.reshape(max_length, (1,))

            # Calculate the expected tokens
            expected_tokens = input_tensor.shape[0]

        # Schedule and get the current learning rate
        current_lr = schedule_learning_rate(self.optimizer, iteration_count, self.warmup_steps)
        lr_before_update = float(current_lr) if isinstance(current_lr, (int, float)) else current_lr.item()

        # Calculate loss and gradients
        (lvalue, ntoks), grad = self.loss_function(self.model, input_tensor, target_tensor, attention_mask_tensor, lengths)

        # Update the optimizer with the calculated gradients
        self.optimizer.update(self.model, grad)

        # Calculate the time taken for this batch
        batch_end_time = time.time()
        batch_time = batch_end_time - batch_start_time

        # Return relevant metrics
        return {
            "loss": lvalue if isinstance(lvalue, (float, int)) else lvalue.item(),
            "ntoks": ntoks,
            "expected_tokens": expected_tokens,
            "batch_time": batch_time,
            "lr_before_update": lr_before_update
        }

        # except Exception as e:
        #     logger.error(f"Error in batch {batch_index}: {str(e)}", exc_info=True)

        #     # Return error metrics
        #     return {
        #         "loss": 0,
        #         "ntoks": 0,
        #         "expected_tokens": expected_tokens,
        #         "batch_time": time.time() - batch_start_time,
        #         "lr_before_update": self.optimizer.learning_rate
        #     }
