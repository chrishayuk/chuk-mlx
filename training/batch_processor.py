import time
import logging
import mlx.core as mx
from training.trainer_utils import schedule_learning_rate

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
        # set the batch timer
        batch_start_time = time.time()

        try:
            # clear token counts etx
            expected_tokens = 0
            lengths = None

            # checking for a non concatenated tensors
            if isinstance(batch, tuple) and len(batch) == 3:
                # get the input, target tensors and lengths
                input_tensor, target_tensor, lengths = batch

                # load the lengths
                lengths = mx.array(lengths)

                if lengths.ndim > 1:
                    # If lengths has more than 1 dimension, specify axis=1
                    max_length = mx.max(lengths, axis=1, keepdims=True)

                    # reshape
                    lengths = mx.reshape(mx.max(lengths, axis=1, keepdims=True), (-1, 1))
                else:
                    # If lengths has only 1 dimension, no need to specify axis
                    max_length = mx.max(lengths)

                    # reshape
                    lengths = mx.reshape(max_length, (1,))

                # load the input and target tensor
                input_tensor = mx.array(input_tensor)
                target_tensor = mx.array(target_tensor)

                # calculate expected tokens
                expected_tokens = input_tensor.shape[0]

            # get the current learning rate
            current_lr = schedule_learning_rate(self.optimizer, iteration_count, self.warmup_steps)

            # get the learning rate before we do the update
            lr_before_update = float(current_lr) if isinstance(current_lr, (int, float)) else current_lr.item()

            # Calculate loss
            (lvalue, ntoks), grad = self.loss_function(self.model, input_tensor, target_tensor, lengths)

            # update the optimizer
            self.optimizer.update(self.model, grad)

            # calculate the end of the batch time
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time

            # return the metrics
            return {
                "loss": lvalue if isinstance(lvalue, (float, int)) else lvalue.item(),
                "ntoks": ntoks,
                "expected_tokens": expected_tokens,
                "batch_time": batch_time,
                "lr_before_update": lr_before_update
            }

        except Exception as e:
            # error
            logger.error(f"Error in batch {batch_index}: {e}")

            # return the stats
            return {
                "loss": 0,
                "ntoks": 0,
                "expected_tokens": expected_tokens,
                "batch_time": time.time() - batch_start_time,
                "lr_before_update": self.optimizer.learning_rate
            }