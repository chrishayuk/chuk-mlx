import time
import pytest
from unittest.mock import MagicMock
from training.epoch_processor_utils import calculate_epoch_metrics, update_progress_bar

# Test for calculate_epoch_metrics function
def test_calculate_epoch_metrics():
    # Setup
    epoch_start_time = time.time()
    batch_times = [1.0, 1.2, 1.3, 1.1]  # Example batch processing times
    epoch_tokens = 10000
    epoch_theoretical_tokens = 12000
    
    # Simulate the passage of time by sleeping for each batch time
    for bt in batch_times:
        time.sleep(bt)
    
    # Call the function
    metrics = calculate_epoch_metrics(epoch_start_time, batch_times, epoch_tokens, epoch_theoretical_tokens)
    
    # Expected calculations
    expected_epoch_time = sum(batch_times)
    expected_average_batch_time = sum(batch_times) / len(batch_times)
    expected_actual_tokens_per_second = epoch_tokens / sum(batch_times)
    expected_theoretical_tokens_per_second = epoch_theoretical_tokens / sum(batch_times)

    # Assertions
    assert metrics['epoch_time'] >= expected_epoch_time  # The actual epoch time will be slightly more due to additional overhead
    assert pytest.approx(metrics['average_batch_time'], 0.01) == expected_average_batch_time
    assert pytest.approx(metrics['actual_tokens_per_second'], 0.01) == expected_actual_tokens_per_second
    assert pytest.approx(metrics['theoretical_tokens_per_second'], 0.01) == expected_theoretical_tokens_per_second

# Test for update_progress_bar function
def test_update_progress_bar():
    # Setup
    batch_progress = MagicMock()  # Mock the progress bar
    batch_index = 4
    progress_interval = 2
    
    batch_metrics = {
        "loss": 0.5,
        "ntoks": 1024,
        "expected_tokens": 2048,
        "batch_time": 2.0,
        "lr_before_update": 0.001
    }
    
    # Expected calculations
    expected_actual_tokens_per_second = batch_metrics["ntoks"] / batch_metrics["batch_time"]
    expected_theoretical_tokens_per_second = batch_metrics["expected_tokens"] / batch_metrics["batch_time"]
    
    # Call the function
    update_progress_bar(batch_progress, batch_index, batch_metrics, progress_interval)
    
    # Check that progress bar was updated
    batch_progress.update.assert_called_once_with(1)
    
    # Check that the progress bar's postfix was set correctly
    batch_progress.set_postfix.assert_called_once_with({
        "Batch Loss": batch_metrics["loss"],
        "Batch Tokens": batch_metrics["ntoks"],
        "Batch Time": f"{batch_metrics['batch_time']:.3f}s",
        "Tokens/s": f"{expected_actual_tokens_per_second:.2f}",
        "LR": f"{batch_metrics['lr_before_update']:.7f}"
    })

# Running tests
if __name__ == "__main__":
    pytest.main()
