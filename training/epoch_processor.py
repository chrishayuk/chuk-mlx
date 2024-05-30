import logging
import os
import time
from tqdm import tqdm
from .epoch_processor_utils import update_progress_bar, calculate_epoch_metrics

logger = logging.getLogger(__name__)

class EpochProcessor:
    def __init__(self, model, tokenizer, optimizer, loss_function, batch_processor, progress_interval, checkpoint_freq, checkpoint_dir):
        # initialize
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.batch_processor = batch_processor
        self.progress_interval = progress_interval
        self.checkpoint_freq = checkpoint_freq
        self.checkpoint_dir = checkpoint_dir

    def process_epoch(self, epoch, num_epochs, batch_dataset, num_iterations, iteration_count):
        # intialize epoch starts
        epoch_start_time = time.time()
        batch_times = []
        epoch_loss = 0
        epoch_tokens = 0
        epoch_theoretical_tokens = 0
        batch_count = 0

        # get the number of batches
        num_batches = len(batch_dataset)

        # setup the epoch progress bar
        with tqdm(total=num_batches, desc=f"Epoch [{epoch+1}/{num_epochs}]", unit="batch") as batch_progress:
            # loop through each batch in the dataset
            for batch_index, batch in enumerate(batch_dataset, start=1):
                
                # check if we have busted number of iterations
                if num_iterations is not None and iteration_count >= num_iterations:
                    break

                # process the batch
                batch_metrics = self.batch_processor.process_batch(batch, batch_index, iteration_count)

                # update the meterics
                epoch_loss += batch_metrics["loss"]
                epoch_tokens += batch_metrics["ntoks"]
                epoch_theoretical_tokens += batch_metrics["expected_tokens"]
                batch_times.append(batch_metrics["batch_time"])

                # increment batch and iteration count
                batch_count += 1
                iteration_count += 1

                # update the batch progress bar
                update_progress_bar(batch_progress, batch_index, batch_metrics, self.progress_interval)

            # calculate the epoch metrics
            epoch_metrics = calculate_epoch_metrics(epoch_start_time, batch_times, epoch_tokens, epoch_theoretical_tokens)

            # update the epoch metrics
            batch_progress.set_postfix({
                "Epoch Loss": f"{epoch_loss / batch_count:.4f}" if batch_count > 0 else "N/A",
                "Tokens": epoch_tokens,
                "Avg Batch Time": f"{epoch_metrics['average_batch_time']:.3f}s",
                "Tokens/s": f"{epoch_metrics['actual_tokens_per_second']:.2f} (Actual) / {epoch_metrics['theoretical_tokens_per_second']:.2f} (Theoretical)"
            })

            # close the progress bar
            batch_progress.close()

            # check if we need to checkpoint
            if self.checkpoint_freq is not None and (epoch + 1) % self.checkpoint_freq == 0:
                logger.info(f'Checkpointing at end of epoch {epoch+1}')

                # checkpoint
                self.save_checkpoint(f'epoch_{epoch+1}')

        # return the stats
        return {
            "iteration_count": iteration_count,
            "epoch_tokens": epoch_tokens,
            "epoch_theoretical_tokens": epoch_theoretical_tokens,
            "total_batch_time": sum(batch_times),
            "epoch_time": epoch_metrics['epoch_time']
        }

    def save_checkpoint(self, identifier):
        # get the checkpoint path
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_{identifier}.npz')

        # save the weights
        self.model.save_weights(checkpoint_path)

        # save the checkpoint
        logger.info(f'Saved checkpoint: {checkpoint_path}')
