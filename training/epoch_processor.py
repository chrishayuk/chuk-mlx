import logging
import os
import time
from tqdm import tqdm
from .epoch_processor_utils import update_progress_bar, calculate_epoch_metrics, log_epoch_metrics

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

    def process_epoch(self, epoch, num_epochs, batch_dataset, num_iterations, iteration_count):
        # Initialize batch statistics
        epoch_start_time = time.time()
        batch_times = []
        epoch_loss = 0
        epoch_tokens = 0
        epoch_theoretical_tokens = 0
        batch_count = 0

        # Initialize the progress bar
        with tqdm(total=len(batch_dataset), desc=f"Epoch [{epoch+1}/{num_epochs}]", unit="batch") as batch_progress:
            # Loop through the batches
            for batch_index, batch in enumerate(batch_dataset, start=1):
                # Check we haven't exceeded iterations
                if num_iterations is not None and iteration_count >= num_iterations:
                    break

                # Process the batch
                batch_metrics = self.batch_processor.process_batch(batch, batch_index, iteration_count)

                # Calculate the batch metrics
                epoch_loss += batch_metrics["loss"]
                epoch_tokens += batch_metrics["ntoks"]
                epoch_theoretical_tokens += batch_metrics["expected_tokens"]
                batch_times.append(batch_metrics["batch_time"])
                batch_count += 1
                iteration_count += 1

                # Update the progress bar
                update_progress_bar(batch_progress, batch_index, batch_metrics, self.progress_interval)

        # Calculate epoch metrics
        epoch_metrics = calculate_epoch_metrics(epoch_start_time, batch_times, epoch_tokens, epoch_theoretical_tokens)

        # log epoch metrics
        log_epoch_metrics(
            epoch, num_epochs, batch_count, epoch_loss, epoch_tokens,
            epoch_metrics['average_batch_time'], epoch_metrics['actual_tokens_per_second'],
            epoch_metrics['theoretical_tokens_per_second']
        )

        # checkpoint
        self.checkpoint_if_necessary(epoch)

        # return the stats
        return {
            "iteration_count": iteration_count,
            "epoch_tokens": epoch_tokens,
            "epoch_theoretical_tokens": epoch_theoretical_tokens,
            "total_batch_time": sum(batch_times),
            "epoch_time": epoch_metrics['epoch_time']
        }

    def checkpoint_if_necessary(self, epoch):
        # check, if we're checkpointing
        if self.checkpoint_freq is not None:
            # checkpointing
            logger.info(f'Checkpointing at end of epoch {epoch+1}')

            # save the checkpoint
            self.save_checkpoint(f'epoch_{epoch+1}')

    def save_checkpoint(self, identifier):
        # get the checkpoint path
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_{identifier}.npz')

        # save the weights
        self.model.save_weights(checkpoint_path)

        # save the checkpoint
        logger.info(f'Saved checkpoint: {checkpoint_path}')
