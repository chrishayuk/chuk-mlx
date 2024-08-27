import pytest
from unittest.mock import MagicMock, patch
import os
from training.epoch_processor import EpochProcessor

# Helper function to generate increasing time values
def time_generator():
    time = 0
    while True:
        yield time
        time += 1  # or some increment based on your expected time differences

# Test for the process_epoch method
def test_process_epoch():
    # Setup
    model = MagicMock()
    tokenizer = MagicMock()
    optimizer = MagicMock()
    loss_function = MagicMock()
    batch_processor = MagicMock()
    batch_dataset = [MagicMock() for _ in range(3)]  # Mock three batches
    progress_interval = 1
    checkpoint_freq_epochs = 1
    checkpoint_freq_iterations = 2
    checkpoint_dir = "test_checkpoints"

    # Ensure the checkpoint directory is mocked and does not interfere with real FS
    os.makedirs(checkpoint_dir, exist_ok=True)

    epoch_processor = EpochProcessor(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        loss_function=loss_function,
        batch_processor=batch_processor,
        progress_interval=progress_interval,
        checkpoint_freq_epochs=checkpoint_freq_epochs,
        checkpoint_freq_iterations=checkpoint_freq_iterations,
        checkpoint_dir=checkpoint_dir
    )

    # Mock the return value of process_batch for each batch
    batch_processor.process_batch.side_effect = [
        {"loss": 1.0, "ntoks": 100, "expected_tokens": 100, "batch_time": 1.0, "lr_before_update": 0.001},
        {"loss": 0.8, "ntoks": 100, "expected_tokens": 100, "batch_time": 1.0, "lr_before_update": 0.001},
        {"loss": 0.6, "ntoks": 100, "expected_tokens": 100, "batch_time": 1.0, "lr_before_update": 0.001},
    ]

    # Mock the save_checkpoint method
    epoch_processor.save_checkpoint = MagicMock()

    # Mock time.time() to simulate controlled timing with a generator
    with patch('training.epoch_processor.tqdm', MagicMock()), patch('time.time', side_effect=time_generator()):

        # Run the process_epoch method
        result = epoch_processor.process_epoch(epoch=0, num_epochs=1, batch_dataset=batch_dataset, num_iterations=None, iteration_count=0)

    # Assertions
    assert result["iteration_count"] == 3
    assert result["epoch_tokens"] == 300
    assert result["epoch_theoretical_tokens"] == 300
    assert result["total_batch_time"] == 3.0  # Should match the mocked total time
    assert result["epoch_time"] >= 3.0  # The epoch time should be at least the total time

    # Ensure save_checkpoint was called correctly
    epoch_processor.save_checkpoint.assert_called_with('epoch_1')

def test_save_checkpoint():
    # Setup
    model = MagicMock()
    tokenizer = MagicMock()
    optimizer = MagicMock()
    loss_function = MagicMock()
    batch_processor = MagicMock()
    checkpoint_dir = "test_checkpoints"

    # Ensure the checkpoint directory is mocked and does not interfere with real FS
    os.makedirs(checkpoint_dir, exist_ok=True)

    epoch_processor = EpochProcessor(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        loss_function=loss_function,
        batch_processor=batch_processor,
        progress_interval=1,
        checkpoint_freq_epochs=1,
        checkpoint_freq_iterations=2,
        checkpoint_dir=checkpoint_dir
    )

    # Test successful save
    epoch_processor.save_checkpoint("test_identifier")
    model.save_weights.assert_called_once()

    # Test save with exception
    model.save_weights.side_effect = Exception("Test exception")
    with patch('logging.Logger.error') as mock_logger_error:
        epoch_processor.save_checkpoint("test_identifier")
        mock_logger_error.assert_called_once_with(f"Failed to save checkpoint {os.path.join(checkpoint_dir, 'checkpoint_test_identifier.npz')}: Test exception")

def test_epoch_loss_calculation():
    model = MagicMock()
    tokenizer = MagicMock()
    optimizer = MagicMock()
    loss_function = MagicMock()
    batch_processor = MagicMock()
    batch_dataset = [MagicMock() for _ in range(3)]
    progress_interval = 1
    checkpoint_freq_epochs = 1
    checkpoint_freq_iterations = 2
    checkpoint_dir = "test_checkpoints"

    epoch_processor = EpochProcessor(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        loss_function=loss_function,
        batch_processor=batch_processor,
        progress_interval=progress_interval,
        checkpoint_freq_epochs=checkpoint_freq_epochs,
        checkpoint_freq_iterations=checkpoint_freq_iterations,
        checkpoint_dir=checkpoint_dir
    )

    # Mock return values for each batch
    batch_processor.process_batch.side_effect = [
        {"loss": 1.0, "ntoks": 100, "expected_tokens": 100, "batch_time": 1.0, "lr_before_update": 0.001},
        {"loss": 0.8, "ntoks": 100, "expected_tokens": 100, "batch_time": 1.0, "lr_before_update": 0.001},
        {"loss": 0.6, "ntoks": 100, "expected_tokens": 100, "batch_time": 1.0, "lr_before_update": 0.001},
    ]

    with patch('training.epoch_processor.tqdm', MagicMock()), patch('time.time', side_effect=time_generator()):
        result = epoch_processor.process_epoch(epoch=0, num_epochs=1, batch_dataset=batch_dataset, num_iterations=None, iteration_count=0)
    
    # Calculate expected average epoch loss
    expected_avg_epoch_loss = (1.0 + 0.8 + 0.6) / 3

    # Check that the average epoch loss is calculated correctly
    assert result["epoch_loss"] == expected_avg_epoch_loss

# Cleanup after tests
@pytest.fixture(scope="module", autouse=True)
def cleanup(request):
    def remove_test_dirs():
        if os.path.exists("test_checkpoints"):
            import shutil
            shutil.rmtree("test_checkpoints")
    request.addfinalizer(remove_test_dirs)
