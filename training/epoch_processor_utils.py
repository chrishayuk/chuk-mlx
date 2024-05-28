import time
import logging

logger = logging.getLogger(__name__)

def calculate_epoch_metrics(epoch_start_time, batch_times, epoch_tokens, epoch_theoretical_tokens):
    # get the end time
    epoch_end_time = time.time()

    # calculate the total epoch time
    epoch_time = epoch_end_time - epoch_start_time

    # figure out the average
    average_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0

    # calculate actual tokens per second
    actual_tokens_per_second = epoch_tokens / sum(batch_times) if sum(batch_times) > 0 else 0

    # calculate theoretical tokens per second
    theoretical_tokens_per_second = epoch_theoretical_tokens / sum(batch_times) if sum(batch_times) > 0 else 0

    # return the stats
    return {
        "epoch_time": epoch_time,
        "average_batch_time": average_batch_time,
        "actual_tokens_per_second": actual_tokens_per_second,
        "theoretical_tokens_per_second": theoretical_tokens_per_second
    }

def update_progress_bar(batch_progress, batch_index, batch_metrics, progress_interval):
    # check the progress interval
    if batch_index % progress_interval == 0:
        # calculate the tokens per second
        actual_tokens_per_second = batch_metrics["ntoks"] / batch_metrics["batch_time"] if batch_metrics["batch_time"] > 0 else 0
        theoretical_tokens_per_second = batch_metrics["expected_tokens"] / batch_metrics["batch_time"] if batch_metrics["batch_time"] > 0 else 0

        # set the progress bar settings
        batch_progress.set_postfix({
            "Batch Loss": batch_metrics["loss"],
            "Batch Tokens": batch_metrics["ntoks"],
            "Batch Time": f"{batch_metrics['batch_time']:.3f}s",
            "Tokens/s": f"{actual_tokens_per_second:.2f}",
            "LR": f"{batch_metrics['lr_before_update']:.7f}"
        })

    # update the progress bar
    batch_progress.update(1)
