"""Tests for SFT training command."""

import logging
from unittest.mock import MagicMock, patch

import pytest

from chuk_lazarus.cli.commands.train._types import SFTConfig, TrainMode
from chuk_lazarus.cli.commands.train.sft import train_sft, train_sft_cmd

LOAD_MODEL_PATCH = "chuk_lazarus.models.load_model"
SFT_DATASET_PATCH = "chuk_lazarus.data.SFTDataset"
SFT_TRAINER_PATCH = "chuk_lazarus.training.SFTTrainer"
SFT_CONFIG_PATCH = "chuk_lazarus.training.losses.SFTConfig"


class TestTrainSFT:
    """Tests for train_sft async command."""

    @pytest.fixture
    def basic_config(self, sft_args):
        """Create basic SFT config."""
        return SFTConfig.from_args(sft_args)

    @pytest.fixture
    def mock_training_components(self, mock_model, mock_trainer, mock_dataset):
        """Create mocks for training components."""
        return {
            "model": mock_model,
            "dataset": mock_dataset,
            "trainer": mock_trainer,
            "config": MagicMock(),
        }

    @pytest.mark.asyncio
    async def test_train_sft_basic(self, basic_config, mock_training_components, caplog):
        """Test basic SFT training."""
        with (
            patch(LOAD_MODEL_PATCH, create=True) as mock_load,
            patch(SFT_DATASET_PATCH, create=True) as mock_dataset_cls,
            patch(SFT_TRAINER_PATCH, create=True) as mock_trainer_cls,
            patch(SFT_CONFIG_PATCH, create=True) as mock_config_cls,
            caplog.at_level(logging.INFO),
        ):
            mock_load.return_value = mock_training_components["model"]
            mock_dataset_cls.return_value = mock_training_components["dataset"]
            mock_trainer_cls.return_value = mock_training_components["trainer"]
            mock_config_cls.return_value = mock_training_components["config"]

            result = await train_sft(basic_config)

            # Verify model was loaded
            mock_load.assert_called_once_with(
                "test-model",
                use_lora=False,
                lora_rank=8,
            )

            # Verify dataset was created
            mock_dataset_cls.assert_called_once()

            # Verify result
            assert result.mode == TrainMode.SFT
            assert result.epochs_completed == 3

            # Check logging
            assert "Loading model: test-model" in caplog.text

    @pytest.mark.asyncio
    async def test_train_sft_with_lora(self, sft_args, mock_training_components):
        """Test SFT training with LoRA."""
        sft_args.use_lora = True
        sft_args.lora_rank = 16
        config = SFTConfig.from_args(sft_args)

        with (
            patch(LOAD_MODEL_PATCH, create=True) as mock_load,
            patch(SFT_DATASET_PATCH, create=True) as mock_dataset_cls,
            patch(SFT_TRAINER_PATCH, create=True) as mock_trainer_cls,
            patch(SFT_CONFIG_PATCH, create=True) as mock_config_cls,
        ):
            mock_load.return_value = mock_training_components["model"]
            mock_dataset_cls.return_value = mock_training_components["dataset"]
            mock_trainer_cls.return_value = mock_training_components["trainer"]
            mock_config_cls.return_value = mock_training_components["config"]

            await train_sft(config)

            mock_load.assert_called_once_with(
                "test-model",
                use_lora=True,
                lora_rank=16,
            )

    @pytest.mark.asyncio
    async def test_train_sft_with_eval_data(self, sft_args, mock_training_components):
        """Test SFT training with evaluation dataset."""
        sft_args.eval_data = "/path/to/eval.jsonl"
        config = SFTConfig.from_args(sft_args)

        with (
            patch(LOAD_MODEL_PATCH, create=True) as mock_load,
            patch(SFT_DATASET_PATCH, create=True) as mock_dataset_cls,
            patch(SFT_TRAINER_PATCH, create=True) as mock_trainer_cls,
            patch(SFT_CONFIG_PATCH, create=True) as mock_config_cls,
        ):
            mock_load.return_value = mock_training_components["model"]
            mock_eval_dataset = MagicMock()
            mock_dataset_cls.side_effect = [
                mock_training_components["dataset"],
                mock_eval_dataset,
            ]
            mock_trainer_cls.return_value = mock_training_components["trainer"]
            mock_config_cls.return_value = mock_training_components["config"]

            await train_sft(config)

            # Verify both datasets were created
            assert mock_dataset_cls.call_count == 2

            # Verify trainer.train was called with both datasets
            mock_training_components["trainer"].train.assert_called_once_with(
                mock_training_components["dataset"],
                mock_eval_dataset,
            )

    @pytest.mark.asyncio
    async def test_train_sft_with_mask_prompt(self, sft_args, mock_training_components):
        """Test SFT training with prompt masking."""
        sft_args.mask_prompt = True
        config = SFTConfig.from_args(sft_args)

        with (
            patch(LOAD_MODEL_PATCH, create=True) as mock_load,
            patch(SFT_DATASET_PATCH, create=True) as mock_dataset_cls,
            patch(SFT_TRAINER_PATCH, create=True) as mock_trainer_cls,
            patch(SFT_CONFIG_PATCH, create=True) as mock_config_cls,
        ):
            mock_load.return_value = mock_training_components["model"]
            mock_dataset_cls.return_value = mock_training_components["dataset"]
            mock_trainer_cls.return_value = mock_training_components["trainer"]
            mock_config_cls.return_value = mock_training_components["config"]

            await train_sft(config)

            # Verify mask_prompt was passed to dataset
            call_kwargs = mock_dataset_cls.call_args[1]
            assert call_kwargs["mask_prompt"] is True


class TestTrainSFTCmd:
    """Tests for train_sft_cmd CLI entry point."""

    @pytest.fixture
    def mock_training_components(self, mock_model, mock_trainer, mock_dataset):
        """Create mocks for training components."""
        return {
            "model": mock_model,
            "dataset": mock_dataset,
            "trainer": mock_trainer,
            "config": MagicMock(),
        }

    @pytest.mark.asyncio
    async def test_train_sft_cmd(self, sft_args, mock_training_components, capsys):
        """Test CLI entry point."""
        with (
            patch(LOAD_MODEL_PATCH, create=True) as mock_load,
            patch(SFT_DATASET_PATCH, create=True) as mock_dataset_cls,
            patch(SFT_TRAINER_PATCH, create=True) as mock_trainer_cls,
            patch(SFT_CONFIG_PATCH, create=True) as mock_config_cls,
        ):
            mock_load.return_value = mock_training_components["model"]
            mock_dataset_cls.return_value = mock_training_components["dataset"]
            mock_trainer_cls.return_value = mock_training_components["trainer"]
            mock_config_cls.return_value = mock_training_components["config"]

            await train_sft_cmd(sft_args)

            captured = capsys.readouterr()
            assert "SFT Training Complete" in captured.out
