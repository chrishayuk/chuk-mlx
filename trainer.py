import mlx.core as mx
import mlx.nn as nn
import time
from tqdm import tqdm

class Trainer:
    def __init__(self, model, optimizer, loss_function, progress_interval=1):
        # set the model
        self.model = model

        # set the optimizer
        self.optimizer = optimizer

        # set the loss function
        self.loss_function = loss_function
        
        # set the progress update interval
        self.progress_interval = progress_interval

    def train(self, num_epochs, batch_dataset, num_iterations=None):
        # set the timers
        total_batch_time = 0
        total_epoch_time = 0

        # initialize iteration counter
        iteration_count = 0
        total_batches = len(batch_dataset)
        total_tokens = 0  # To track the total number of valid tokens processed
        total_theoretical_tokens = 0  # To track the total number of tokens in the batch including padding

        # loop through the epochs
        for epoch in range(num_epochs):
            # kick off the timer for epoch
            epoch_start_time = time.time()

            # reset the batch settings
            batch_times = []
            epoch_loss = 0
            epoch_tokens = 0
            epoch_theoretical_tokens = 0

            # create a progress bar for the batches
            batch_progress = tqdm(total=total_batches, desc=f"Epoch [{epoch+1}/{num_epochs}]", unit="batch")

            # load each batch file
            for batch_index, (input_tensor, target_tensor, lengths) in enumerate(batch_dataset, start=1):
                # check if we've reached the total number of iterations (if num_iterations is specified)
                if num_iterations is not None and iteration_count >= num_iterations:
                    break

                # Log the shapes of input and target tensors
                print(f"Batch {batch_index}: input_tensor shape = {input_tensor.shape}, target_tensor shape = {target_tensor.shape}")

                # Calculate expected tokens per batch
                expected_tokens = input_tensor.shape[0] * input_tensor.shape[1]
                print(f"Expected tokens per batch: {expected_tokens}")

                # kick off the batch timer
                batch_start_time = time.time()

                # execute the loss function
                (lvalue, ntoks), grad = self.loss_function(self.model, input_tensor, target_tensor, lengths)

                # Verify the ntoks value
                print(f"Reported tokens by loss function: {ntoks}")

                # call the optimizer
                self.optimizer.update(self.model, grad)

                # eval
                mx.eval(self.model.parameters(), self.optimizer.state, lvalue)

                # update the loss value for the epoch
                epoch_loss += lvalue.item()
                epoch_tokens += ntoks
                total_tokens += ntoks  # Update total valid tokens processed
                epoch_theoretical_tokens += expected_tokens
                total_theoretical_tokens += expected_tokens  # Update total theoretical tokens

                # end the batch timers
                batch_end_time = time.time()
                batch_time = batch_end_time - batch_start_time
                batch_times.append(batch_time)

                # update the progress bar at the specified interval
                if batch_index % self.progress_interval == 0:
                    actual_tokens_per_second = ntoks / batch_time
                    theoretical_tokens_per_second = expected_tokens / batch_time
                    batch_progress.update(self.progress_interval)
                    batch_progress.set_postfix({
                        "Batch Loss": lvalue.item(),
                        "Batch Tokens": ntoks,
                        "Batch Time": f"{batch_time:.3f}s",
                        "Tokens/s": f"{actual_tokens_per_second:.2f} (Actual) / {theoretical_tokens_per_second:.2f} (Theoretical)"
                    })

                # increment the iteration count
                iteration_count += 1

            # update the remaining progress
            remaining_batches = total_batches % self.progress_interval
            if remaining_batches > 0:
                actual_tokens_per_second = epoch_tokens / sum(batch_times)
                theoretical_tokens_per_second = epoch_theoretical_tokens / sum(batch_times)
                batch_progress.update(remaining_batches)
                batch_progress.set_postfix({
                    "Epoch Loss": epoch_loss / total_batches,
                    "Epoch Tokens": epoch_tokens,
                    "Avg Batch Time": f"{sum(batch_times) / len(batch_times):.3f}s",
                    "Tokens/s": f"{actual_tokens_per_second:.2f} (Actual) / {theoretical_tokens_per_second:.2f} (Theoretical)"
                })

            # end the epoch timers
            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time

            # update the batch timers
            total_batch_time += sum(batch_times)
            total_epoch_time += epoch_time

            # check if we've reached the total number of iterations (if num_iterations is specified)
            if num_iterations is not None and iteration_count >= num_iterations:
                break

        # Print summary of training
        actual_tokens_per_second = total_tokens / total_epoch_time
        theoretical_tokens_per_second = total_theoretical_tokens / total_epoch_time
        print(f"Total training time: {total_epoch_time:.3f}s")
        print(f"Total iterations: {iteration_count}")
        print(f"Average batch time: {total_batch_time / iteration_count:.3f}s")
        print(f"Tokens per second: {actual_tokens_per_second:.2f} (Actual) / {theoretical_tokens_per_second:.2f} (Theoretical)")