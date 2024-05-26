import time
import logging

from training.trainer_utils import schedule_learning_rate
from .batch_processor_utils import pad_sequences, process_concatenated_batch, process_non_concatenated_batch

# set the logger
logger = logging.getLogger(__name__)

class BatchProcessor:
    def __init__(self, model, tokenizer, optimizer, loss_function, warmup_steps):
        # set the model, tokenizer, optimizer etc
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.warmup_steps = warmup_steps

    def process_batch(self, batch, batch_index, iteration_count):
        try:
            if isinstance(batch, tuple) and len(batch) == 2:
                concatenated_tensor, lengths = batch
                input_tensors, target_tensors = process_concatenated_batch(concatenated_tensor, lengths, self.tokenizer, batch_index)
            else:
                input_tensors, target_tensors = process_non_concatenated_batch(batch, batch_index, self.tokenizer)

            if not input_tensors or not target_tensors:
                logger.error(f"No valid sequences in batch {batch_index}. Skipping this batch.")
                return {
                    "loss": 0,
                    "ntoks": 0,
                    "expected_tokens": 0,
                    "batch_time": 0,
                    "lr_before_update": self.optimizer.learning_rate
                }

            max_input_length = max(tensor.shape[0] for tensor in input_tensors)
            max_target_length = max(tensor.shape[0] for tensor in target_tensors)

            input_tensor = pad_sequences(input_tensors, max_input_length, self.tokenizer.pad_token_id)
            target_tensor = pad_sequences(target_tensors, max_input_length, self.tokenizer.pad_token_id)

            expected_tokens = input_tensor.shape[0] * input_tensor.shape[1]

            batch_start_time = time.time()
            current_lr = schedule_learning_rate(self.optimizer, iteration_count, self.warmup_steps)
            lr_before_update = float(current_lr) if isinstance(current_lr, (int, float)) else current_lr.item()

            (lvalue, ntoks), grad = self.loss_function(self.model, input_tensor, target_tensor, lengths)
            self.optimizer.update(self.model, grad)

            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time

            return {
                "loss": lvalue if isinstance(lvalue, (float, int)) else lvalue.item(),
                "ntoks": ntoks,
                "expected_tokens": expected_tokens,
                "batch_time": batch_time,
                "lr_before_update": lr_before_update
            }

        except Exception as e:
            logger.error(f"Error in batch {batch_index}: {e}")
            return {
                "loss": 0,
                "ntoks": 0,
                "expected_tokens": expected_tokens,
                "batch_time": time.time() - batch_start_time,
                "lr_before_update": self.optimizer.learning_rate
            }
