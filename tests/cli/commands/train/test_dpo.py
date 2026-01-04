"""Tests for DPO training command."""

import logging
from unittest.mock import MagicMock, patch

import pytest

from chuk_lazarus.cli.commands.train._types import DPOConfig, TrainMode
from chuk_lazarus.cli.commands.train.dpo import train_dpo, train_dpo_cmd

LOAD_MODEL_PATCH = "chuk_lazarus.models.load_model"
PREFERENCE_DATASET_PATCH = "chuk_lazarus.data.PreferenceDataset"
DPO_TRAINER_PATCH = "chuk_lazarus.training.DPOTrainer"
DPO_TRAINER_CONFIG_PATCH = "chuk_lazarus.training.DPOTrainerConfig"
DPO_LOSS_CONFIG_PATCH = "chuk_lazarus.training.losses.DPOConfig"


class TestTrainDPO:
    """Tests for train_dpo async command."""

    @pytest.fixture
    def basic_config(self, dpo_args):
        """Create basic DPO config."""
        return DPOConfig.from_args(dpo_args)

    @pytest.fixture
    def mock_dpo_components(self, mock_model, mock_trainer, mock_dataset):
        """Create mocks for DPO training components."""
        mock_ref_model = MagicMock()
        mock_ref_model.model = MagicMock()

        return {
            "policy_model": mock_model,
            "ref_model": mock_ref_model,
            "dataset": mock_dataset,
            "trainer": mock_trainer,
            "config": MagicMock(),
            "dpo_config": MagicMock(),
        }

    @pytest.mark.asyncio
    async def test_train_dpo_basic(self, basic_config, mock_dpo_components, caplog):
        """Test basic DPO training."""
        with (
            patch(LOAD_MODEL_PATCH, create=True) as mock_load,
            patch(PREFERENCE_DATASET_PATCH, create=True) as mock_dataset_cls,
            patch(DPO_TRAINER_PATCH, create=True) as mock_trainer_cls,
            patch(DPO_TRAINER_CONFIG_PATCH, create=True) as mock_config_cls,
            patch(DPO_LOSS_CONFIG_PATCH, create=True) as mock_dpo_config_cls,
            caplog.at_level(logging.INFO),
        ):
            mock_load.side_effect = [
                mock_dpo_components["policy_model"],
                mock_dpo_components["ref_model"],
            ]
            mock_dataset_cls.return_value = mock_dpo_components["dataset"]
            mock_trainer_cls.return_value = mock_dpo_components["trainer"]
            mock_config_cls.return_value = mock_dpo_components["config"]
            mock_dpo_config_cls.return_value = mock_dpo_components["dpo_config"]

            result = await train_dpo(basic_config)

            # Verify both models were loaded
            assert mock_load.call_count == 2

            # Verify result
            assert result.mode == TrainMode.DPO
            assert result.epochs_completed == 3

            # Check logging
            assert "Loading policy model: test-model" in caplog.text
            assert "Loading reference model: test-model" in caplog.text

    @pytest.mark.asyncio
    async def test_train_dpo_with_separate_ref_model(self, dpo_args, mock_dpo_components, caplog):
        """Test DPO training with separate reference model."""
        dpo_args.ref_model = "ref-model"
        config = DPOConfig.from_args(dpo_args)

        with (
            patch(LOAD_MODEL_PATCH, create=True) as mock_load,
            patch(PREFERENCE_DATASET_PATCH, create=True) as mock_dataset_cls,
            patch(DPO_TRAINER_PATCH, create=True) as mock_trainer_cls,
            patch(DPO_TRAINER_CONFIG_PATCH, create=True) as mock_config_cls,
            patch(DPO_LOSS_CONFIG_PATCH, create=True) as mock_dpo_config_cls,
            caplog.at_level(logging.INFO),
        ):
            mock_load.side_effect = [
                mock_dpo_components["policy_model"],
                mock_dpo_components["ref_model"],
            ]
            mock_dataset_cls.return_value = mock_dpo_components["dataset"]
            mock_trainer_cls.return_value = mock_dpo_components["trainer"]
            mock_config_cls.return_value = mock_dpo_components["config"]
            mock_dpo_config_cls.return_value = mock_dpo_components["dpo_config"]

            await train_dpo(config)

            # Verify reference model was loaded with correct name
            assert mock_load.call_args_list[1] == (
                ("ref-model",),
                {"use_lora": False},
            )

            assert "Loading reference model: ref-model" in caplog.text

    @pytest.mark.asyncio
    async def test_train_dpo_with_lora(self, dpo_args, mock_dpo_components):
        """Test DPO training with LoRA."""
        dpo_args.use_lora = True
        dpo_args.lora_rank = 16
        config = DPOConfig.from_args(dpo_args)

        with (
            patch(LOAD_MODEL_PATCH, create=True) as mock_load,
            patch(PREFERENCE_DATASET_PATCH, create=True) as mock_dataset_cls,
            patch(DPO_TRAINER_PATCH, create=True) as mock_trainer_cls,
            patch(DPO_TRAINER_CONFIG_PATCH, create=True) as mock_config_cls,
            patch(DPO_LOSS_CONFIG_PATCH, create=True) as mock_dpo_config_cls,
        ):
            mock_load.side_effect = [
                mock_dpo_components["policy_model"],
                mock_dpo_components["ref_model"],
            ]
            mock_dataset_cls.return_value = mock_dpo_components["dataset"]
            mock_trainer_cls.return_value = mock_dpo_components["trainer"]
            mock_config_cls.return_value = mock_dpo_components["config"]
            mock_dpo_config_cls.return_value = mock_dpo_components["dpo_config"]

            await train_dpo(config)

            # Verify policy model was loaded with LoRA
            assert mock_load.call_args_list[0] == (
                ("test-model",),
                {"use_lora": True, "lora_rank": 16},
            )
            # Reference model should never use LoRA
            assert mock_load.call_args_list[1] == (
                ("test-model",),
                {"use_lora": False},
            )

    @pytest.mark.asyncio
    async def test_train_dpo_with_eval_data(self, dpo_args, mock_dpo_components):
        """Test DPO training with evaluation dataset."""
        dpo_args.eval_data = "/path/to/eval_preferences.jsonl"
        config = DPOConfig.from_args(dpo_args)

        with (
            patch(LOAD_MODEL_PATCH, create=True) as mock_load,
            patch(PREFERENCE_DATASET_PATCH, create=True) as mock_dataset_cls,
            patch(DPO_TRAINER_PATCH, create=True) as mock_trainer_cls,
            patch(DPO_TRAINER_CONFIG_PATCH, create=True) as mock_config_cls,
            patch(DPO_LOSS_CONFIG_PATCH, create=True) as mock_dpo_config_cls,
        ):
            mock_load.side_effect = [
                mock_dpo_components["policy_model"],
                mock_dpo_components["ref_model"],
            ]
            mock_eval_dataset = MagicMock()
            mock_dataset_cls.side_effect = [
                mock_dpo_components["dataset"],
                mock_eval_dataset,
            ]
            mock_trainer_cls.return_value = mock_dpo_components["trainer"]
            mock_config_cls.return_value = mock_dpo_components["config"]
            mock_dpo_config_cls.return_value = mock_dpo_components["dpo_config"]

            await train_dpo(config)

            # Verify both datasets were created
            assert mock_dataset_cls.call_count == 2

            # Verify trainer.train was called with both datasets
            mock_dpo_components["trainer"].train.assert_called_once_with(
                mock_dpo_components["dataset"],
                mock_eval_dataset,
            )

    @pytest.mark.asyncio
    async def test_train_dpo_custom_beta(self, dpo_args, mock_dpo_components):
        """Test DPO training with custom beta."""
        dpo_args.beta = 0.2
        config = DPOConfig.from_args(dpo_args)

        with (
            patch(LOAD_MODEL_PATCH, create=True) as mock_load,
            patch(PREFERENCE_DATASET_PATCH, create=True) as mock_dataset_cls,
            patch(DPO_TRAINER_PATCH, create=True) as mock_trainer_cls,
            patch(DPO_TRAINER_CONFIG_PATCH, create=True) as mock_config_cls,
            patch(DPO_LOSS_CONFIG_PATCH, create=True) as mock_dpo_config_cls,
        ):
            mock_load.side_effect = [
                mock_dpo_components["policy_model"],
                mock_dpo_components["ref_model"],
            ]
            mock_dataset_cls.return_value = mock_dpo_components["dataset"]
            mock_trainer_cls.return_value = mock_dpo_components["trainer"]
            mock_config_cls.return_value = mock_dpo_components["config"]
            mock_dpo_config_cls.return_value = mock_dpo_components["dpo_config"]

            await train_dpo(config)

            mock_dpo_config_cls.assert_called_once_with(beta=0.2)


class TestTrainDPOCmd:
    """Tests for train_dpo_cmd CLI entry point."""

    @pytest.fixture
    def mock_dpo_components(self, mock_model, mock_trainer, mock_dataset):
        """Create mocks for DPO training components."""
        mock_ref_model = MagicMock()
        mock_ref_model.model = MagicMock()

        return {
            "policy_model": mock_model,
            "ref_model": mock_ref_model,
            "dataset": mock_dataset,
            "trainer": mock_trainer,
            "config": MagicMock(),
            "dpo_config": MagicMock(),
        }

    @pytest.mark.asyncio
    async def test_train_dpo_cmd(self, dpo_args, mock_dpo_components, capsys):
        """Test CLI entry point."""
        with (
            patch(LOAD_MODEL_PATCH, create=True) as mock_load,
            patch(PREFERENCE_DATASET_PATCH, create=True) as mock_dataset_cls,
            patch(DPO_TRAINER_PATCH, create=True) as mock_trainer_cls,
            patch(DPO_TRAINER_CONFIG_PATCH, create=True) as mock_config_cls,
            patch(DPO_LOSS_CONFIG_PATCH, create=True) as mock_dpo_config_cls,
        ):
            mock_load.side_effect = [
                mock_dpo_components["policy_model"],
                mock_dpo_components["ref_model"],
            ]
            mock_dataset_cls.return_value = mock_dpo_components["dataset"]
            mock_trainer_cls.return_value = mock_dpo_components["trainer"]
            mock_config_cls.return_value = mock_dpo_components["config"]
            mock_dpo_config_cls.return_value = mock_dpo_components["dpo_config"]

            await train_dpo_cmd(dpo_args)

            captured = capsys.readouterr()
            assert "DPO Training Complete" in captured.out
