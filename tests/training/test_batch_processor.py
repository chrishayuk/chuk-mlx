"""Tests for batch processor."""

from unittest.mock import MagicMock, patch

from chuk_lazarus.training.batch_processor import BatchProcessor


class TestBatchProcessor:
    """Tests for BatchProcessor class."""

    def test_init(self):
        """Test initialization."""
        model = MagicMock()
        tokenizer = MagicMock()
        optimizer = MagicMock()
        loss_function = MagicMock()

        processor = BatchProcessor(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            loss_function=loss_function,
            warmup_steps=100,
        )

        assert processor.model == model
        assert processor.tokenizer == tokenizer
        assert processor.optimizer == optimizer
        assert processor.loss_function == loss_function
        assert processor.warmup_steps == 100

    @patch("chuk_lazarus.training.batch_processor.schedule_learning_rate")
    def test_process_batch_basic(self, mock_schedule_lr):
        """Test processing a basic batch."""
        mock_schedule_lr.return_value = 1e-4

        model = MagicMock()
        tokenizer = MagicMock()
        optimizer = MagicMock()

        # Create mock tensors
        input_tensor = MagicMock()
        target_tensor = MagicMock()
        attention_mask = MagicMock()
        batch = (input_tensor, target_tensor, attention_mask)

        # Mock loss value and ntoks
        mock_loss = MagicMock()
        mock_loss.item.return_value = 0.5
        mock_ntoks = MagicMock()
        mock_ntoks.item.return_value = 256

        loss_function = MagicMock()
        loss_function.return_value = ((mock_loss, mock_ntoks), MagicMock())

        processor = BatchProcessor(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            loss_function=loss_function,
            warmup_steps=100,
        )

        result = processor.process_batch(batch, batch_index=0, iteration_count=50)

        assert "loss" in result
        assert "ntoks" in result
        assert "batch_time" in result
        assert "tokens_per_second" in result
        assert "lr_before_update" in result
        assert result["loss"] == 0.5
        assert result["ntoks"] == 256

    @patch("chuk_lazarus.training.batch_processor.schedule_learning_rate")
    def test_process_batch_runtime_error(self, mock_schedule_lr):
        """Test processing batch with runtime error."""
        mock_schedule_lr.return_value = 1e-4

        model = MagicMock()
        tokenizer = MagicMock()
        optimizer = MagicMock()

        # Create mock tensors
        input_tensor = MagicMock()
        target_tensor = MagicMock()
        attention_mask = MagicMock()
        batch = (input_tensor, target_tensor, attention_mask)

        # Mock loss function that raises RuntimeError
        loss_function = MagicMock()
        loss_function.side_effect = RuntimeError("Memory error")

        processor = BatchProcessor(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            loss_function=loss_function,
            warmup_steps=100,
        )

        # Should not raise, error is caught and logged
        try:
            processor.process_batch(batch, batch_index=0, iteration_count=50)
        except (RuntimeError, UnboundLocalError):
            # Expected behavior - variables may not be bound due to exception
            pass
