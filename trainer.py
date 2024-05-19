import mlx.core as mx
import mlx.nn as nn
import time
from tqdm import tqdm
import logging
import os
from math import pi, cos

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, model, optimizer, loss_function, progress_interval=1, checkpoint_dir='checkpoints'):
        # set the parameters
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.progress_interval = progress_interval
        self.checkpoint_dir = checkpoint_dir

        # create the checkpoint directory
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def get_current_lr(self):
        return self.optimizer.learning_rate.item()
    
    def train(self, num_epochs, batch_dataset, num_iterations=None):
        # intialize the batch timeers etc
        total_batch_time = 0
        total_epoch_time = 0
        iteration_count = 0
        total_batches = len(batch_dataset)
        total_tokens = 0
        total_theoretical_tokens = 0

        # loop through each epoch
        for epoch in range(num_epochs):
            # initialiaze the epoch timers etc
            epoch_start_time = time.time()
            batch_times = []
            epoch_loss = 0
            epoch_tokens = 0
            epoch_theoretical_tokens = 0
            batch_progress = tqdm(total=total_batches, desc=f"Epoch [{epoch+1}/{num_epochs}]", unit="batch")

            # loop through the batches
            for batch_index, (input_tensor, target_tensor, lengths) in enumerate(batch_dataset, start=1):
                # if we hit the iterations max, break out the loop
                if num_iterations is not None and iteration_count >= num_iterations:
                    break

                # check the expected number of tokens
                expected_tokens = input_tensor.shape[0] * input_tensor.shape[1]

                # start the batch timer
                batch_start_time = time.time()

                try:
                    # Get the current learning rate before update
                    lr_before_update = self.get_current_lr()

                    # Forward pass
                    (lvalue, ntoks), grad = self.loss_function(self.model, input_tensor, target_tensor, lengths)

                    # Backward pass and optimization
                    self.optimizer.update(self.model, grad)

                    # Evaluate and update metrics
                    mx.eval(self.model.parameters(), self.optimizer.state, lvalue)

                    # make all the calculations
                    epoch_loss += lvalue.item()
                    epoch_tokens += ntoks
                    total_tokens += ntoks
                    epoch_theoretical_tokens += expected_tokens
                    total_theoretical_tokens += expected_tokens

                    # end the batch timer
                    batch_end_time = time.time()
                    batch_time = batch_end_time - batch_start_time
                    batch_times.append(batch_time)

                    # update the batch progress
                    if batch_index % self.progress_interval == 0:
                        actual_tokens_per_second = ntoks / batch_time
                        theoretical_tokens_per_second = expected_tokens / batch_time
                        batch_progress.update(self.progress_interval)
                        batch_progress.set_postfix({
                            "Batch Loss": lvalue.item(),
                            "Batch Tokens": ntoks,
                            "Batch Time": f"{batch_time:.3f}s",
                            "Tokens/s": f"{actual_tokens_per_second:.2f}",
                            "Learning Rate": f"{lr_before_update:.7f}"
                        })

                    # increment the iteration
                    iteration_count += 1

                except Exception as e:
                    logger.error(f"Error in batch {batch_index}: {e}")
                    continue

            # check for remaining batches
            remaining_batches = total_batches % self.progress_interval

            # we still have batches to go
            if remaining_batches > 0:
                # set the batch progress
                actual_tokens_per_second = epoch_tokens / sum(batch_times)
                theoretical_tokens_per_second = epoch_theoretical_tokens / sum(batch_times)
                batch_progress.update(remaining_batches)
                batch_progress.set_postfix({
                    "Epoch Loss": epoch_loss / total_batches,
                    "Epoch Tokens": epoch_tokens,
                    "Avg Batch Time": f"{sum(batch_times) / len(batch_times):.3f}s",
                    "Tokens/s": f"{actual_tokens_per_second:.2f} (Actual) / {theoretical_tokens_per_second:.2f} (Theoretical)"
                })

            # end the epoch timer
            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            total_batch_time += sum(batch_times)
            total_epoch_time += epoch_time

            # log the result for the epoch
            logger.info(f"Epoch [{epoch+1}/{num_epochs}] completed. Loss: {epoch_loss / total_batches:.4f}")

            # check if we're out of iterations
            if num_iterations is not None and iteration_count >= num_iterations:
                break

        # print out the results
        actual_tokens_per_second = total_tokens / total_epoch_time
        theoretical_tokens_per_second = total_theoretical_tokens / total_epoch_time
        logger.info(f"Total training time: {total_epoch_time:.3f}s")
        logger.info(f"Total iterations: {iteration_count}")
        logger.info(f"Average batch time: {total_batch_time / iteration_count:.3f}s")
        logger.info(f"Tokens per second: {actual_tokens_per_second:.2f} (Actual) / {theoretical_tokens_per_second:.2f} (Theoretical)")

