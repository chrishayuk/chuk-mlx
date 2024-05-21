import time
from tqdm import tqdm
import logging
import os
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, model, optimizer, loss_function, progress_interval=1, checkpoint_dir='checkpoints', checkpoint_freq=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.progress_interval = progress_interval
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_freq = checkpoint_freq

        if self.checkpoint_freq is not None and not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def get_current_lr(self):
        lr = self.optimizer.learning_rate
        return float(lr) if isinstance(lr, (int, float)) else lr.item()

    def save_checkpoint(self, identifier):
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_{identifier}.npz')
        self.model.save_weights(checkpoint_path)
        logger.info(f'Saved checkpoint: {checkpoint_path}')

    def train(self, num_epochs, batch_dataset, num_iterations=None):
        total_batch_time = 0
        total_epoch_time = 0
        iteration_count = 0
        total_tokens = 0
        total_theoretical_tokens = 0
        checkpoint_interval = None

        if self.checkpoint_freq is not None:
            checkpoint_interval = max(1, (num_epochs * 100) // self.checkpoint_freq)

        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            batch_times = []
            epoch_loss = 0
            epoch_tokens = 0
            epoch_theoretical_tokens = 0
            batch_count = 0
            batch_progress = tqdm(desc=f"Epoch [{epoch+1}/{num_epochs}]", unit="batch")

            for batch_index, batch in enumerate(batch_dataset, start=1):
                if num_iterations is not None and iteration_count >= num_iterations:
                    break

                if len(batch) == 3:
                    input_tensor, target_tensor, lengths = batch
                else:
                    concatenated_tensor, lengths = batch
                    # Split concatenated_tensor into input_tensor and target_tensor
                    split_index = concatenated_tensor.shape[1] // 2
                    input_tensor = concatenated_tensor[:, :split_index]
                    target_tensor = concatenated_tensor[:, split_index:]

                expected_tokens = input_tensor.shape[0] * input_tensor.shape[1]
                batch_start_time = time.time()

                try:
                    lr_before_update = self.get_current_lr()
                    (lvalue, ntoks), grad = self.loss_function(self.model, input_tensor, target_tensor, lengths)
                    if np.isnan(lvalue):
                        logger.warning(f"NaN loss detected in batch {batch_index}. Skipping this batch.")
                        continue
                    self.optimizer.update(self.model, grad)
                    epoch_loss += lvalue if isinstance(lvalue, (float, int)) else lvalue.item()
                    epoch_tokens += ntoks
                    total_tokens += ntoks
                    epoch_theoretical_tokens += expected_tokens
                    total_theoretical_tokens += expected_tokens

                    batch_end_time = time.time()
                    batch_time = batch_end_time - batch_start_time
                    batch_times.append(batch_time)
                    batch_count += 1

                    if batch_index % self.progress_interval == 0:
                        actual_tokens_per_second = ntoks / batch_time
                        theoretical_tokens_per_second = expected_tokens / batch_time
                        batch_progress.update(self.progress_interval)
                        batch_progress.set_postfix({
                            "Batch Loss": lvalue if isinstance(lvalue, (float, int)) else lvalue.item(),
                            "Batch Tokens": ntoks,
                            "Batch Time": f"{batch_time:.3f}s",
                            "Tokens/s": f"{actual_tokens_per_second:.2f}",
                            "LR": f"{lr_before_update:.7f}"
                        })

                    iteration_count += 1

                    if self.checkpoint_freq and iteration_count % self.checkpoint_freq == 0:
                        logger.info(f'Checkpointing at iteration {iteration_count}')
                        self.save_checkpoint(f'iter_{iteration_count}')

                except Exception as e:
                    logger.error(f"Error in batch {batch_index}: {e}")
                    continue

            remaining_batches = batch_count % self.progress_interval

            if remaining_batches > 0:
                actual_tokens_per_second = epoch_tokens / sum(batch_times)
                theoretical_tokens_per_second = epoch_theoretical_tokens / sum(batch_times)
                batch_progress.update(remaining_batches)
                batch_progress.set_postfix({
                    "Epoch Loss": epoch_loss / batch_count if batch_count > 0 else float('inf'),
                    "Epoch Tokens": epoch_tokens,
                    "Avg Batch Time": f"{sum(batch_times) / len(batch_times):.3f}s",
                    "Tokens/s": f"{actual_tokens_per_second:.2f} (Actual) / {theoretical_tokens_per_second:.2f} (Theoretical)"
                })

            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            total_batch_time += sum(batch_times)
            total_epoch_time += epoch_time

            if batch_count > 0:
                logger.info(f"Epoch [{epoch+1}/{num_epochs}] completed. Loss: {epoch_loss / batch_count:.4f}")
            else:
                logger.info(f"Epoch [{epoch+1}/{num_epochs}] completed. No batches processed.")

            if self.checkpoint_freq is not None:
                logger.info(f'Checkpointing at end of epoch {epoch+1}')
                self.save_checkpoint(f'epoch_{epoch+1}')

            if num_iterations is not None and iteration_count >= num_iterations:
                break

        actual_tokens_per_second = total_tokens / total_epoch_time
        theoretical_tokens_per_second = total_theoretical_tokens / total_epoch_time
        logger.info(f"Total training time: {total_epoch_time:.3f}s")
        logger.info(f"Total iterations: {iteration_count}")
        logger.info(f"Average batch time: {total_batch_time / iteration_count:.3f}s")
        logger.info(f"Tokens per second: {actual_tokens_per_second:.2f} (Actual) / {theoretical_tokens_per_second:.2f} (Theoretical)")
