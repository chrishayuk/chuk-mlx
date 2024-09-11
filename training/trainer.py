import logging
import os
from training.batch_processor import BatchProcessor
from training.epoch_processor import EpochProcessor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, model, tokenizer, optimizer, loss_function, progress_interval=1, checkpoint_dir='checkpoints', checkpoint_freq_epochs=None, checkpoint_freq_iterations=None, warmup_steps=0):
        # Initialize
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.progress_interval = progress_interval
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_freq_epochs = checkpoint_freq_epochs
        self.checkpoint_freq_iterations = checkpoint_freq_iterations
        self.warmup_steps = warmup_steps

        # Setup the batch processor
        self.batch_processor = BatchProcessor(model, tokenizer, optimizer, loss_function, warmup_steps)

        # Setup the epoch processor
        self.epoch_processor = EpochProcessor(model, tokenizer, optimizer, loss_function, self.batch_processor, progress_interval, checkpoint_freq_epochs, checkpoint_freq_iterations, checkpoint_dir)

        # Ensure checkpoint directory exists
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def get_current_learning_rate(self):
        return float(self.optimizer.learning_rate) if isinstance(self.optimizer.learning_rate, (int, float)) else self.optimizer.learning_rate.item()

    def train(self, num_epochs, batch_dataset, num_iterations=None):
        # Initialize the batch timings
        total_batch_time = 0
        total_epoch_time = 0
        iteration_count = 0
        total_tokens = 0

        # Loop through each epoch
        for epoch in range(num_epochs):
            # Process the epoch and return metrics
            epoch_metrics = self.epoch_processor.process_epoch(epoch, num_epochs, batch_dataset, num_iterations, iteration_count)

            # Update epoch metrics
            iteration_count = epoch_metrics["iteration_count"]
            total_tokens += epoch_metrics["epoch_tokens"]
            total_batch_time += epoch_metrics["total_batch_time"]
            total_epoch_time += epoch_metrics["epoch_time"]

            # Check if the number of iterations has exceeded the max iterations
            if num_iterations is not None and iteration_count >= num_iterations:
                break

        # Calculate the actual tokens per second, and theoretical tokens per second
        actual_tokens_per_second = total_tokens / total_epoch_time if total_epoch_time > 0 else 0

        # Log out the results
        logger.info(f"Total training time: {total_epoch_time:.3f}s")
        logger.info(f"Total iterations: {iteration_count}")
        logger.info(f"Average batch time: {total_batch_time / iteration_count if iteration_count > 0 else 0:.3f}s")
        logger.info(f"Tokens per second: {actual_tokens_per_second:.2f}")

        # Save the final model
        self.save_final_model()

    def save_final_model(self):
        if self.checkpoint_dir:
            # Get the final checkpoint path
            final_checkpoint_path = os.path.join(self.checkpoint_dir, 'final_model_checkpoint.npz')

            try:
                # Save the weights
                self.model.save_weights(final_checkpoint_path)

                # Log it
                logger.info(f'Saved final model checkpoint: {final_checkpoint_path}')
            except Exception as e:
                # Log error
                logger.error(f"Failed to save final model checkpoint {final_checkpoint_path}: {str(e)}")
