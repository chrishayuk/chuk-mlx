"""Tests for epoch processor utilities."""

import time
from unittest.mock import MagicMock

from chuk_lazarus.training.epoch_processor_utils import (
    calculate_epoch_metrics,
    update_progress_bar,
)


class TestCalculateEpochMetrics:
    """Tests for calculate_epoch_metrics function."""

    def test_calculate_metrics_basic(self):
        """Test basic metrics calculation."""
        epoch_start_time = time.time() - 10  # 10 seconds ago
        batch_times = [1.0, 1.0, 1.0, 1.0, 1.0]  # 5 batches, 1 second each
        epoch_tokens = 1000
        epoch_theoretical_tokens = 1200

        result = calculate_epoch_metrics(
            epoch_start_time,
            batch_times,
            epoch_tokens,
            epoch_theoretical_tokens,
        )

        assert "epoch_time" in result
        assert "average_batch_time" in result
        assert "actual_tokens_per_second" in result
        assert "theoretical_tokens_per_second" in result
        assert result["average_batch_time"] == 1.0
        # 1000 tokens / 5 seconds = 200 tokens/sec
        assert result["actual_tokens_per_second"] == 200.0
        # 1200 tokens / 5 seconds = 240 tokens/sec
        assert result["theoretical_tokens_per_second"] == 240.0

    def test_calculate_metrics_empty_batches(self):
        """Test metrics with empty batch list."""
        epoch_start_time = time.time()
        batch_times = []
        epoch_tokens = 0
        epoch_theoretical_tokens = 0

        result = calculate_epoch_metrics(
            epoch_start_time,
            batch_times,
            epoch_tokens,
            epoch_theoretical_tokens,
        )

        assert result["average_batch_time"] == 0
        assert result["actual_tokens_per_second"] == 0
        assert result["theoretical_tokens_per_second"] == 0

    def test_calculate_metrics_zero_batch_time(self):
        """Test metrics with zero batch time."""
        epoch_start_time = time.time()
        batch_times = [0.0, 0.0]
        epoch_tokens = 100
        epoch_theoretical_tokens = 200

        result = calculate_epoch_metrics(
            epoch_start_time,
            batch_times,
            epoch_tokens,
            epoch_theoretical_tokens,
        )

        assert result["actual_tokens_per_second"] == 0
        assert result["theoretical_tokens_per_second"] == 0


class TestUpdateProgressBar:
    """Tests for update_progress_bar function."""

    def test_update_progress_bar_on_interval(self):
        """Test progress bar update on interval."""
        batch_progress = MagicMock()
        batch_metrics = {
            "loss": 0.5,
            "ntoks": 256,
            "batch_time": 1.0,
            "lr_before_update": 1e-4,
        }

        update_progress_bar(
            batch_progress,
            batch_index=0,
            batch_metrics=batch_metrics,
            progress_interval=1,
        )

        batch_progress.set_postfix.assert_called_once()
        batch_progress.update.assert_called_once_with(1)

    def test_update_progress_bar_not_on_interval(self):
        """Test progress bar update not on interval."""
        batch_progress = MagicMock()
        batch_metrics = {
            "loss": 0.5,
            "ntoks": 256,
            "batch_time": 1.0,
            "lr_before_update": 1e-4,
        }

        update_progress_bar(
            batch_progress,
            batch_index=1,
            batch_metrics=batch_metrics,
            progress_interval=10,
        )

        # set_postfix should not be called (index 1 % 10 != 0)
        batch_progress.set_postfix.assert_not_called()
        batch_progress.update.assert_called_once_with(1)

    def test_update_progress_bar_with_tensor_values(self):
        """Test progress bar with tensor-like values."""
        batch_progress = MagicMock()

        # Mock tensor-like objects with .item() method
        ntoks_tensor = MagicMock()
        ntoks_tensor.item.return_value = 256
        batch_time_tensor = MagicMock()
        batch_time_tensor.item.return_value = 1.0
        lr_tensor = MagicMock()
        lr_tensor.item.return_value = 1e-4

        batch_metrics = {
            "loss": 0.5,
            "ntoks": ntoks_tensor,
            "batch_time": batch_time_tensor,
            "lr_before_update": lr_tensor,
        }

        update_progress_bar(
            batch_progress,
            batch_index=0,
            batch_metrics=batch_metrics,
            progress_interval=1,
        )

        batch_progress.set_postfix.assert_called_once()
        batch_progress.update.assert_called_once_with(1)
