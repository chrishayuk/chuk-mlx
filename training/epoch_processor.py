import logging
import os
import time
from tqdm import tqdm
from .epoch_processor_utils import update_progress_bar, calculate_epoch_metrics

logger = logging.getLogger(__name__)

class EpochProcessor:
    def __init__(self, model, tokenizer, optimizer, loss_function, batch_processor, progress_interval, checkpoint_freq, checkpoint_dir):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.batch_processor = batch_processor
        self.progress_interval = progress_interval
        self.checkpoint_freq = checkpoint_freq
        self.checkpoint_dir = checkpoint_dir

        # Ensure checkpoint directory exists
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def process_epoch(self, epoch, num_epochs, batch_dataset, num_iterations, iteration_count):
        epoch_start_time = time.time()
        batch_times = []
        epoch_loss = 0
        epoch_tokens = 0
        epoch_theoretical_tokens = 0
        batch_count = 0

        num_batches = len(batch_dataset)

        with tqdm(total=num_batches, desc=f"Epoch [{epoch+1}/{num_epochs}]", unit="batch") as batch_progress:
            for batch_index in range(num_batches):
                try:
                    batch = batch_dataset[batch_index]
                    batch_metrics = self.batch_processor.process_batch(batch, batch_index, iteration_count)

                    epoch_loss += batch_metrics["loss"]
                    epoch_tokens += batch_metrics["ntoks"]
                    epoch_theoretical_tokens += batch_metrics["expected_tokens"]
                    batch_times.append(batch_metrics["batch_time"])

                    batch_count += 1
                    iteration_count += 1

                    update_progress_bar(batch_progress, batch_index, batch_metrics, self.progress_interval)

                except IOError as io_err:
                    logger.error(f"IOError processing batch at index {batch_index}: {str(io_err)}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing batch at index {batch_index}: {str(e)}")
                    continue

            epoch_metrics = calculate_epoch_metrics(epoch_start_time, batch_times, epoch_tokens, epoch_theoretical_tokens)

            batch_progress.set_postfix({
                "Epoch Loss": f"{epoch_loss / batch_count:.4f}" if batch_count > 0 else "N/A",
                "Tokens": epoch_tokens,
                "Avg Batch Time": f"{epoch_metrics['average_batch_time']:.3f}s",
                "Tokens/s": f"{epoch_metrics['actual_tokens_per_second']:.2f} (Actual) / {epoch_metrics['theoretical_tokens_per_second']:.2f} (Theoretical)"
            })

            batch_progress.close()

            if self.checkpoint_freq is not None and (epoch + 1) % self.checkpoint_freq == 0:
                logger.info(f'Checkpointing at end of epoch {epoch+1}')
                self.save_checkpoint(f'epoch_{epoch+1}')

        return {
            "iteration_count": iteration_count,
            "epoch_tokens": epoch_tokens,
            "epoch_theoretical_tokens": epoch_theoretical_tokens,
            "total_batch_time": sum(batch_times),
            "epoch_time": epoch_metrics['epoch_time']
        }

    def save_checkpoint(self, identifier):
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_{identifier}.npz')
        try:
            self.model.save_weights(checkpoint_path)
            logger.info(f'Saved checkpoint: {checkpoint_path}')
        except Exception as e:
            logger.error(f"Failed to save checkpoint {checkpoint_path}: {str(e)}")
