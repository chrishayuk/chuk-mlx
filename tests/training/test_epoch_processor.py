"""Tests for epoch processor."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from chuk_lazarus.training.epoch_processor import EpochProcessor


class TestEpochProcessor:
    """Tests for EpochProcessor class."""

    def test_init(self):
        """Test initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = EpochProcessor(
                model=MagicMock(),
                tokenizer=MagicMock(),
                optimizer=MagicMock(),
                loss_function=MagicMock(),
                batch_processor=MagicMock(),
                progress_interval=10,
                checkpoint_freq_epochs=1,
                checkpoint_freq_iterations=100,
                checkpoint_dir=tmpdir,
            )

            assert processor.progress_interval == 10
            assert processor.checkpoint_freq_epochs == 1
            assert processor.checkpoint_freq_iterations == 100
            assert processor.checkpoint_dir == tmpdir

    def test_init_creates_checkpoint_dir(self):
        """Test that init creates checkpoint directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "new_checkpoints"

            processor = EpochProcessor(
                model=MagicMock(),
                tokenizer=MagicMock(),
                optimizer=MagicMock(),
                loss_function=MagicMock(),
                batch_processor=MagicMock(),
                progress_interval=10,
                checkpoint_freq_epochs=None,
                checkpoint_freq_iterations=None,
                checkpoint_dir=str(checkpoint_path),
            )

            assert checkpoint_path.exists()

    @patch("chuk_lazarus.training.epoch_processor.tqdm")
    def test_process_epoch_basic(self, mock_tqdm):
        """Test basic epoch processing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mocks
            model = MagicMock()
            batch_processor = MagicMock()
            batch_processor.process_batch.return_value = {
                "loss": 0.5,
                "ntoks": 256,
                "batch_time": 1.0,
                "lr_before_update": 1e-4,
            }

            # Create mock dataset
            mock_dataset = MagicMock()
            mock_dataset.__len__ = MagicMock(return_value=3)
            mock_dataset.__getitem__ = MagicMock(return_value=(MagicMock(), MagicMock()))

            # Mock progress bar
            mock_progress = MagicMock()
            mock_tqdm.return_value = mock_progress

            processor = EpochProcessor(
                model=model,
                tokenizer=MagicMock(),
                optimizer=MagicMock(),
                loss_function=MagicMock(),
                batch_processor=batch_processor,
                progress_interval=1,
                checkpoint_freq_epochs=None,
                checkpoint_freq_iterations=None,
                checkpoint_dir=tmpdir,
            )

            result = processor.process_epoch(
                epoch=0,
                num_epochs=1,
                batch_dataset=mock_dataset,
                num_iterations=3,
                iteration_count=0,
            )

            assert "iteration_count" in result
            assert "epoch_tokens" in result
            assert "epoch_loss" in result
            assert result["iteration_count"] == 3

    @patch("chuk_lazarus.training.epoch_processor.mx")
    def test_save_checkpoint(self, mock_mx):
        """Test checkpoint saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = MagicMock()
            model.state_dict.return_value = {"weight": MagicMock()}
            optimizer = MagicMock()
            optimizer.state_dict.return_value = {"lr": 1e-4}

            processor = EpochProcessor(
                model=model,
                tokenizer=MagicMock(),
                optimizer=optimizer,
                loss_function=MagicMock(),
                batch_processor=MagicMock(),
                progress_interval=10,
                checkpoint_freq_epochs=None,
                checkpoint_freq_iterations=None,
                checkpoint_dir=tmpdir,
            )

            processor.save_checkpoint("test_id")

            mock_mx.save.assert_called_once()
            model.state_dict.assert_called_once()
            optimizer.state_dict.assert_called_once()
