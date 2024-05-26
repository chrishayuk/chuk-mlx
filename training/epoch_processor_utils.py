import time
import logging

# setup the logger
logger = logging.getLogger(__name__)

def calculate_epoch_metrics(epoch_start_time, batch_times, epoch_tokens, epoch_theoretical_tokens):
    # set the epoch end time
    epoch_end_time = time.time()
    epoch_time = epoch_end_time - epoch_start_time

    # calculate the average batch times
    average_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0

    # calculate the actual tokens per second
    actual_tokens_per_second = epoch_tokens / sum(batch_times) if sum(batch_times) > 0 else 0

    # calculate the theoretical tokens per second
    theoretical_tokens_per_second = epoch_theoretical_tokens / sum(batch_times) if sum(batch_times) > 0 else 0

    return {
        "epoch_time": epoch_time,
        "average_batch_time": average_batch_time,
        "actual_tokens_per_second": actual_tokens_per_second,
        "theoretical_tokens_per_second": theoretical_tokens_per_second
    }

def update_progress_bar(batch_progress, batch_index, batch_metrics, progress_interval):
    # Ensure we only log when we hit the progress interval
    if batch_index % progress_interval == 0:
        # Calculate the actual tokens per second
        actual_tokens_per_second = batch_metrics["ntoks"] / batch_metrics["batch_time"] if batch_metrics["batch_time"] > 0 else 0

        # Calculate the theoretical tokens per second
        theoretical_tokens_per_second = batch_metrics["expected_tokens"] / batch_metrics["batch_time"] if batch_metrics["batch_time"] > 0 else 0

        # Set the postfix before updating the progress bar
        batch_progress.set_postfix({
            "Batch Loss": batch_metrics["loss"],
            "Batch Tokens": batch_metrics["ntoks"],
            "Batch Time": f"{batch_metrics['batch_time']:.3f}s",
            "Tokens/s": f"{actual_tokens_per_second:.2f}",
            "LR": f"{batch_metrics['lr_before_update']:.7f}"
        })

        # Update the progress bar by the progress interval
        batch_progress.update(progress_interval)



def log_epoch_metrics(epoch, num_epochs, batch_count, epoch_loss, epoch_tokens, average_batch_time, actual_tokens_per_second, theoretical_tokens_per_second):
    # check the batch count
    if batch_count > 0:
        logger.info(f"Epoch [{epoch+1}/{num_epochs}] completed. Loss: {epoch_loss / batch_count:.4f}")
    else:
        logger.info(f"Epoch [{epoch+1}/{num_epochs}] completed. No batches processed.")

    # log the epoch tokens, batch time etc
    logger.info(f"Epoch Tokens: {epoch_tokens}")
    logger.info(f"Average Batch Time: {average_batch_time:.3f}s")
    logger.info(f"Tokens/s: {actual_tokens_per_second:.2f} (Actual) / {theoretical_tokens_per_second:.2f} (Theoretical)")