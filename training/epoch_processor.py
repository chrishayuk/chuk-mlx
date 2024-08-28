import logging
import os
import time
from tqdm import tqdm
import traceback
import mlx.core as mx
from training.epoch_processor_utils import update_progress_bar, calculate_epoch_metrics
import psutil  # Import psutil for memory profiling

logger = logging.getLogger(__name__)

class EpochProcessor:
    def __init__(self, model, tokenizer, optimizer, loss_function, batch_processor, progress_interval, checkpoint_freq_epochs, checkpoint_freq_iterations, checkpoint_dir):
        # initialize
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.batch_processor = batch_processor
        self.progress_interval = progress_interval
        self.checkpoint_freq_epochs = checkpoint_freq_epochs
        self.checkpoint_freq_iterations = checkpoint_freq_iterations
        self.checkpoint_dir = checkpoint_dir

        # Ensure checkpoint directory exists
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def process_epoch(self, epoch, num_epochs, batch_dataset, num_iterations, iteration_count):
        epoch_start_time = time.time()
        batch_times = []
        data_loading_times = []
        overhead_times = []
        epoch_loss = 0
        epoch_tokens = 0
        epoch_theoretical_tokens = 0
        batch_count = 0

        num_batches = len(batch_dataset)

        with tqdm(total=num_batches, desc=f"Epoch [{epoch+1}/{num_epochs}]", unit="batch") as batch_progress:
            for batch_index in range(num_batches):
                try:
                    # memory debug
                    #self.log_memory_usage(f"Before loading batch {batch_index}")

                    # laod the dataset and capture the loading times
                    data_loading_start = time.time()
                    batch = batch_dataset[batch_index]
                    data_loading_time = time.time() - data_loading_start
                    data_loading_times.append(data_loading_time)

                    # memory debug
                    #self.log_memory_usage(f"After loading batch {batch_index}")

                    # process the batch (capturing timings)
                    process_start = time.time()
                    batch_metrics = self.batch_processor.process_batch(batch, batch_index, iteration_count)
                    process_time = time.time() - process_start

                    # memory debug
                    #self.log_memory_usage(f"After processing batch {batch_index}")

                    # calculate the epoc stats
                    epoch_loss += batch_metrics["loss"]
                    epoch_tokens += batch_metrics["ntoks"]
                    epoch_theoretical_tokens += batch_metrics["expected_tokens"]
                    batch_times.append(process_time)

                    # increment the batch count and iteration
                    batch_count += 1
                    iteration_count += 1

                    # update the progress bar
                    overhead_start = time.time()
                    update_progress_bar(batch_progress, batch_index, batch_metrics, self.progress_interval)

                    # check if we need to checkpoint
                    if self.checkpoint_freq_iterations is not None and iteration_count % self.checkpoint_freq_iterations == 0:
                        logger.info(f'Checkpointing at iteration {iteration_count}')
                        self.save_checkpoint(f'iteration_{iteration_count}')

                    # calculate overhead times
                    overhead_time = time.time() - overhead_start
                    overhead_times.append(overhead_time)

                    # calculate batch times
                    total_batch_time = data_loading_time + process_time + overhead_time
                    batch_progress.set_postfix({
                        "Batch Loss": f"{batch_metrics['loss']:.2f}",
                        "Tokens": batch_metrics["ntoks"],
                        "Batch Time": f"{process_time:.3f}s",
                        "Total Time": f"{total_batch_time:.3f}s",
                        "Tokens/s": f"{batch_metrics['ntoks']/process_time:.2f}"
                    })

                except Exception as e:
                    # error
                    logger.error(f"Error processing batch at index {batch_index}: {str(e)}")
                    logger.error(f"Batch type: {type(batch)}")
                    logger.error(f"Batch content: {batch}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    continue

            # calculate the epoch metrics
            epoch_metrics = calculate_epoch_metrics(epoch_start_time, batch_times, epoch_tokens, epoch_theoretical_tokens)

            # calculate the average epoch loss across the batch
            avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0

            # set the progress bar
            batch_progress.set_postfix({
                "Epoch Loss": f"{avg_epoch_loss:.4f}",
                "Tokens": epoch_tokens,
                "Avg Batch Time": f"{epoch_metrics['average_batch_time']:.3f}s",
                "Tokens/s": f"{epoch_metrics['actual_tokens_per_second']:.2f} (Actual) / {epoch_metrics['theoretical_tokens_per_second']:.2f} (Theoretical)"
            })

            # close the batch progress
            batch_progress.close()

            # check if we need to checkpoint
            if self.checkpoint_freq_epochs is not None and (epoch + 1) % self.checkpoint_freq_epochs == 0:
                # checkpointing
                logger.info(f'Checkpointing at end of epoch {epoch+1}')
                self.save_checkpoint(f'epoch_{epoch+1}')

        return {
            "iteration_count": iteration_count,
            "epoch_tokens": epoch_tokens,
            "epoch_theoretical_tokens": epoch_theoretical_tokens,
            "total_batch_time": sum(batch_times),
            "epoch_time": epoch_metrics['epoch_time'],
            "epoch_loss": avg_epoch_loss  # Return average loss
        }

    def log_memory_usage(self, label=""):
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        logger.info(f"{label} - Memory Usage: RSS={mem_info.rss / (1024 * 1024):.2f} MB, VMS={mem_info.vms / (1024 * 1024):.2f} MB")

    def save_checkpoint(self, identifier):
        # Log memory before saving checkpoint
        self.log_memory_usage(f"Before saving checkpoint {identifier}")
        
        # setup the checkpoint path
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_{identifier}.npz')
        
        try:
            # save the weights
            model_state = self.model.state_dict()
            optimizer_state = self.optimizer.state_dict()

            # save the checkpoint
            mx.save(checkpoint_path, {'model': model_state, 'optimizer': optimizer_state})

            # log it
            logger.info(f'Saved checkpoint: {checkpoint_path}')

            # Manually delete state dictionaries, necessary, otherwise we memory leak
            del model_state
            del optimizer_state
        except Exception as e:
            logger.error(f"Failed to save checkpoint {checkpoint_path}: {str(e)}")

        # Force garbage collection after checkpointing
        import gc
        gc.collect()
        
        # Log memory after garbage collection
        self.log_memory_usage(f"After GC post-checkpoint {identifier}")


