"""Tests for training command handlers."""

import sys
from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True, scope="module")
def setup_mock_modules():
    """Set up mock modules for imports."""
    # Create mock modules if they don't exist
    modules_to_mock = [
        "chuk_lazarus.models",
        "chuk_lazarus.data",
        "chuk_lazarus.training",
        "chuk_lazarus.training.losses",
        "chuk_lazarus.data.generators",
    ]

    original_modules = {}
    for module_name in modules_to_mock:
        if module_name not in sys.modules:
            original_modules[module_name] = None
            sys.modules[module_name] = MagicMock()

    yield

    # Clean up
    for module_name, original in original_modules.items():
        if original is None and module_name in sys.modules:
            del sys.modules[module_name]


class TestTrainSFT:
    """Tests for train_sft command."""

    @pytest.fixture
    def sft_args(self):
        """Create SFT training arguments."""
        return Namespace(
            model="test-model",
            data="/path/to/train.jsonl",
            eval_data=None,
            output="./checkpoints/sft",
            epochs=3,
            batch_size=4,
            learning_rate=1e-5,
            max_length=512,
            use_lora=False,
            lora_rank=8,
            mask_prompt=False,
            log_interval=10,
        )

    @pytest.fixture
    def mock_training_components(self):
        """Create mocks for all training components."""
        mock_model = MagicMock()
        mock_model.model = MagicMock()
        mock_model.tokenizer = MagicMock()

        mock_dataset = MagicMock()
        mock_trainer = MagicMock()
        mock_config = MagicMock()

        return {
            "model": mock_model,
            "dataset": mock_dataset,
            "trainer": mock_trainer,
            "config": mock_config,
        }

    def test_train_sft_basic(self, sft_args, mock_training_components, caplog):
        """Test basic SFT training."""
        import logging

        from chuk_lazarus.cli.commands.train import train_sft

        with (
            patch("chuk_lazarus.models.load_model") as mock_load_model,
            patch("chuk_lazarus.data.SFTDataset") as mock_dataset_cls,
            patch("chuk_lazarus.training.SFTTrainer") as mock_trainer_cls,
            patch("chuk_lazarus.training.losses.SFTConfig") as mock_config_cls,
            caplog.at_level(logging.INFO),
        ):
            mock_load_model.return_value = mock_training_components["model"]
            mock_dataset_cls.return_value = mock_training_components["dataset"]
            mock_trainer_cls.return_value = mock_training_components["trainer"]
            mock_config_cls.return_value = mock_training_components["config"]

            train_sft(sft_args)

            # Verify model was loaded correctly
            mock_load_model.assert_called_once_with(
                "test-model",
                use_lora=False,
                lora_rank=8,
            )

            # Verify dataset was created
            mock_dataset_cls.assert_called_once_with(
                "/path/to/train.jsonl",
                mock_training_components["model"].tokenizer,
                max_length=512,
                mask_prompt=False,
            )

            # Verify config was created
            mock_config_cls.assert_called_once_with(
                num_epochs=3,
                batch_size=4,
                learning_rate=1e-5,
                max_seq_length=512,
                checkpoint_dir="./checkpoints/sft",
                log_interval=10,
            )

            # Verify trainer was created and train was called
            mock_trainer_cls.assert_called_once_with(
                mock_training_components["model"].model,
                mock_training_components["model"].tokenizer,
                mock_training_components["config"],
            )
            mock_training_components["trainer"].train.assert_called_once_with(
                mock_training_components["dataset"],
                None,
            )

            # Check logging
            assert "Loading model: test-model" in caplog.text
            assert "Loading dataset: /path/to/train.jsonl" in caplog.text
            assert "Training complete. Checkpoints saved to ./checkpoints/sft" in caplog.text

    def test_train_sft_with_lora(self, sft_args, mock_training_components):
        """Test SFT training with LoRA."""
        from chuk_lazarus.cli.commands.train import train_sft

        sft_args.use_lora = True
        sft_args.lora_rank = 16

        with (
            patch("chuk_lazarus.models.load_model") as mock_load_model,
            patch("chuk_lazarus.data.SFTDataset") as mock_dataset_cls,
            patch("chuk_lazarus.training.SFTTrainer") as mock_trainer_cls,
            patch("chuk_lazarus.training.losses.SFTConfig") as mock_config_cls,
        ):
            mock_load_model.return_value = mock_training_components["model"]
            mock_dataset_cls.return_value = mock_training_components["dataset"]
            mock_trainer_cls.return_value = mock_training_components["trainer"]
            mock_config_cls.return_value = mock_training_components["config"]

            train_sft(sft_args)

            # Verify LoRA parameters were passed
            mock_load_model.assert_called_once_with(
                "test-model",
                use_lora=True,
                lora_rank=16,
            )

    def test_train_sft_with_eval_data(self, sft_args, mock_training_components):
        """Test SFT training with evaluation dataset."""
        from chuk_lazarus.cli.commands.train import train_sft

        sft_args.eval_data = "/path/to/eval.jsonl"

        with (
            patch("chuk_lazarus.models.load_model") as mock_load_model,
            patch("chuk_lazarus.data.SFTDataset") as mock_dataset_cls,
            patch("chuk_lazarus.training.SFTTrainer") as mock_trainer_cls,
            patch("chuk_lazarus.training.losses.SFTConfig") as mock_config_cls,
        ):
            mock_load_model.return_value = mock_training_components["model"]
            mock_eval_dataset = MagicMock()
            mock_dataset_cls.side_effect = [
                mock_training_components["dataset"],
                mock_eval_dataset,
            ]
            mock_trainer_cls.return_value = mock_training_components["trainer"]
            mock_config_cls.return_value = mock_training_components["config"]

            train_sft(sft_args)

            # Verify both datasets were created
            assert mock_dataset_cls.call_count == 2

            # Verify trainer.train was called with both datasets
            mock_training_components["trainer"].train.assert_called_once_with(
                mock_training_components["dataset"],
                mock_eval_dataset,
            )

    def test_train_sft_with_mask_prompt(self, sft_args, mock_training_components):
        """Test SFT training with prompt masking."""
        from chuk_lazarus.cli.commands.train import train_sft

        sft_args.mask_prompt = True

        with (
            patch("chuk_lazarus.models.load_model") as mock_load_model,
            patch("chuk_lazarus.data.SFTDataset") as mock_dataset_cls,
            patch("chuk_lazarus.training.SFTTrainer") as mock_trainer_cls,
            patch("chuk_lazarus.training.losses.SFTConfig") as mock_config_cls,
        ):
            mock_load_model.return_value = mock_training_components["model"]
            mock_dataset_cls.return_value = mock_training_components["dataset"]
            mock_trainer_cls.return_value = mock_training_components["trainer"]
            mock_config_cls.return_value = mock_training_components["config"]

            train_sft(sft_args)

            # Verify mask_prompt was passed to dataset
            mock_dataset_cls.assert_called_once_with(
                "/path/to/train.jsonl",
                mock_training_components["model"].tokenizer,
                max_length=512,
                mask_prompt=True,
            )

    def test_train_sft_custom_hyperparameters(self, sft_args, mock_training_components):
        """Test SFT training with custom hyperparameters."""
        from chuk_lazarus.cli.commands.train import train_sft

        sft_args.epochs = 5
        sft_args.batch_size = 8
        sft_args.learning_rate = 2e-5
        sft_args.max_length = 1024
        sft_args.log_interval = 20

        with (
            patch("chuk_lazarus.models.load_model") as mock_load_model,
            patch("chuk_lazarus.data.SFTDataset") as mock_dataset_cls,
            patch("chuk_lazarus.training.SFTTrainer") as mock_trainer_cls,
            patch("chuk_lazarus.training.losses.SFTConfig") as mock_config_cls,
        ):
            mock_load_model.return_value = mock_training_components["model"]
            mock_dataset_cls.return_value = mock_training_components["dataset"]
            mock_trainer_cls.return_value = mock_training_components["trainer"]
            mock_config_cls.return_value = mock_training_components["config"]

            train_sft(sft_args)

            # Verify custom hyperparameters were used
            mock_config_cls.assert_called_once_with(
                num_epochs=5,
                batch_size=8,
                learning_rate=2e-5,
                max_seq_length=1024,
                checkpoint_dir="./checkpoints/sft",
                log_interval=20,
            )


class TestTrainDPO:
    """Tests for train_dpo command."""

    @pytest.fixture
    def dpo_args(self):
        """Create DPO training arguments."""
        return Namespace(
            model="test-model",
            ref_model=None,
            data="/path/to/preferences.jsonl",
            eval_data=None,
            output="./checkpoints/dpo",
            epochs=3,
            batch_size=4,
            learning_rate=1e-6,
            beta=0.1,
            max_length=512,
            use_lora=False,
            lora_rank=8,
        )

    @pytest.fixture
    def mock_dpo_components(self):
        """Create mocks for DPO training components."""
        mock_policy_model = MagicMock()
        mock_policy_model.model = MagicMock()
        mock_policy_model.tokenizer = MagicMock()

        mock_ref_model = MagicMock()
        mock_ref_model.model = MagicMock()

        mock_dataset = MagicMock()
        mock_trainer = MagicMock()
        mock_config = MagicMock()
        mock_dpo_config = MagicMock()

        return {
            "policy_model": mock_policy_model,
            "ref_model": mock_ref_model,
            "dataset": mock_dataset,
            "trainer": mock_trainer,
            "config": mock_config,
            "dpo_config": mock_dpo_config,
        }

    def test_train_dpo_basic(self, dpo_args, mock_dpo_components, caplog):
        """Test basic DPO training."""
        import logging

        from chuk_lazarus.cli.commands.train import train_dpo

        with (
            patch("chuk_lazarus.models.load_model") as mock_load_model,
            patch("chuk_lazarus.data.PreferenceDataset") as mock_dataset_cls,
            patch("chuk_lazarus.training.DPOTrainer") as mock_trainer_cls,
            patch("chuk_lazarus.training.DPOTrainerConfig") as mock_config_cls,
            patch("chuk_lazarus.training.losses.DPOConfig") as mock_dpo_config_cls,
            caplog.at_level(logging.INFO),
        ):
            # load_model is called twice: once for policy, once for ref
            mock_load_model.side_effect = [
                mock_dpo_components["policy_model"],
                mock_dpo_components["ref_model"],
            ]
            mock_dataset_cls.return_value = mock_dpo_components["dataset"]
            mock_trainer_cls.return_value = mock_dpo_components["trainer"]
            mock_config_cls.return_value = mock_dpo_components["config"]
            mock_dpo_config_cls.return_value = mock_dpo_components["dpo_config"]

            train_dpo(dpo_args)

            # Verify both models were loaded
            assert mock_load_model.call_count == 2
            # First call: policy model with LoRA settings
            assert mock_load_model.call_args_list[0] == (
                ("test-model",),
                {"use_lora": False, "lora_rank": 8},
            )
            # Second call: reference model without LoRA
            assert mock_load_model.call_args_list[1] == (
                ("test-model",),
                {"use_lora": False},
            )

            # Verify dataset was created
            mock_dataset_cls.assert_called_once_with(
                "/path/to/preferences.jsonl",
                mock_dpo_components["policy_model"].tokenizer,
                max_length=512,
            )

            # Verify DPO config was created
            mock_dpo_config_cls.assert_called_once_with(beta=0.1)

            # Verify trainer config was created
            mock_config_cls.assert_called_once_with(
                dpo=mock_dpo_components["dpo_config"],
                num_epochs=3,
                batch_size=4,
                learning_rate=1e-6,
                checkpoint_dir="./checkpoints/dpo",
            )

            # Verify trainer was created and train was called
            mock_trainer_cls.assert_called_once_with(
                mock_dpo_components["policy_model"].model,
                mock_dpo_components["ref_model"].model,
                mock_dpo_components["policy_model"].tokenizer,
                mock_dpo_components["config"],
            )
            mock_dpo_components["trainer"].train.assert_called_once_with(
                mock_dpo_components["dataset"],
                None,
            )

            # Check logging
            assert "Loading policy model: test-model" in caplog.text
            assert "Loading reference model: test-model" in caplog.text
            assert "Loading dataset: /path/to/preferences.jsonl" in caplog.text
            assert "Training complete. Checkpoints saved to ./checkpoints/dpo" in caplog.text

    def test_train_dpo_with_separate_ref_model(self, dpo_args, mock_dpo_components, caplog):
        """Test DPO training with separate reference model."""
        import logging

        from chuk_lazarus.cli.commands.train import train_dpo

        dpo_args.ref_model = "ref-model"

        with (
            patch("chuk_lazarus.models.load_model") as mock_load_model,
            patch("chuk_lazarus.data.PreferenceDataset") as mock_dataset_cls,
            patch("chuk_lazarus.training.DPOTrainer") as mock_trainer_cls,
            patch("chuk_lazarus.training.DPOTrainerConfig") as mock_config_cls,
            patch("chuk_lazarus.training.losses.DPOConfig") as mock_dpo_config_cls,
            caplog.at_level(logging.INFO),
        ):
            mock_load_model.side_effect = [
                mock_dpo_components["policy_model"],
                mock_dpo_components["ref_model"],
            ]
            mock_dataset_cls.return_value = mock_dpo_components["dataset"]
            mock_trainer_cls.return_value = mock_dpo_components["trainer"]
            mock_config_cls.return_value = mock_dpo_components["config"]
            mock_dpo_config_cls.return_value = mock_dpo_components["dpo_config"]

            train_dpo(dpo_args)

            # Verify reference model was loaded with correct name
            assert mock_load_model.call_args_list[1] == (
                ("ref-model",),
                {"use_lora": False},
            )

            assert "Loading reference model: ref-model" in caplog.text

    def test_train_dpo_with_lora(self, dpo_args, mock_dpo_components):
        """Test DPO training with LoRA."""
        from chuk_lazarus.cli.commands.train import train_dpo

        dpo_args.use_lora = True
        dpo_args.lora_rank = 16

        with (
            patch("chuk_lazarus.models.load_model") as mock_load_model,
            patch("chuk_lazarus.data.PreferenceDataset") as mock_dataset_cls,
            patch("chuk_lazarus.training.DPOTrainer") as mock_trainer_cls,
            patch("chuk_lazarus.training.DPOTrainerConfig") as mock_config_cls,
            patch("chuk_lazarus.training.losses.DPOConfig") as mock_dpo_config_cls,
        ):
            mock_load_model.side_effect = [
                mock_dpo_components["policy_model"],
                mock_dpo_components["ref_model"],
            ]
            mock_dataset_cls.return_value = mock_dpo_components["dataset"]
            mock_trainer_cls.return_value = mock_dpo_components["trainer"]
            mock_config_cls.return_value = mock_dpo_components["config"]
            mock_dpo_config_cls.return_value = mock_dpo_components["dpo_config"]

            train_dpo(dpo_args)

            # Verify policy model was loaded with LoRA
            assert mock_load_model.call_args_list[0] == (
                ("test-model",),
                {"use_lora": True, "lora_rank": 16},
            )
            # Reference model should never use LoRA
            assert mock_load_model.call_args_list[1] == (
                ("test-model",),
                {"use_lora": False},
            )

    def test_train_dpo_with_eval_data(self, dpo_args, mock_dpo_components):
        """Test DPO training with evaluation dataset."""
        from chuk_lazarus.cli.commands.train import train_dpo

        dpo_args.eval_data = "/path/to/eval_preferences.jsonl"

        with (
            patch("chuk_lazarus.models.load_model") as mock_load_model,
            patch("chuk_lazarus.data.PreferenceDataset") as mock_dataset_cls,
            patch("chuk_lazarus.training.DPOTrainer") as mock_trainer_cls,
            patch("chuk_lazarus.training.DPOTrainerConfig") as mock_config_cls,
            patch("chuk_lazarus.training.losses.DPOConfig") as mock_dpo_config_cls,
        ):
            mock_load_model.side_effect = [
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

            train_dpo(dpo_args)

            # Verify both datasets were created
            assert mock_dataset_cls.call_count == 2

            # Verify trainer.train was called with both datasets
            mock_dpo_components["trainer"].train.assert_called_once_with(
                mock_dpo_components["dataset"],
                mock_eval_dataset,
            )

    def test_train_dpo_custom_hyperparameters(self, dpo_args, mock_dpo_components):
        """Test DPO training with custom hyperparameters."""
        from chuk_lazarus.cli.commands.train import train_dpo

        dpo_args.epochs = 5
        dpo_args.batch_size = 8
        dpo_args.learning_rate = 5e-6
        dpo_args.beta = 0.2
        dpo_args.max_length = 1024

        with (
            patch("chuk_lazarus.models.load_model") as mock_load_model,
            patch("chuk_lazarus.data.PreferenceDataset") as mock_dataset_cls,
            patch("chuk_lazarus.training.DPOTrainer") as mock_trainer_cls,
            patch("chuk_lazarus.training.DPOTrainerConfig") as mock_config_cls,
            patch("chuk_lazarus.training.losses.DPOConfig") as mock_dpo_config_cls,
        ):
            mock_load_model.side_effect = [
                mock_dpo_components["policy_model"],
                mock_dpo_components["ref_model"],
            ]
            mock_dataset_cls.return_value = mock_dpo_components["dataset"]
            mock_trainer_cls.return_value = mock_dpo_components["trainer"]
            mock_config_cls.return_value = mock_dpo_components["config"]
            mock_dpo_config_cls.return_value = mock_dpo_components["dpo_config"]

            train_dpo(dpo_args)

            # Verify custom hyperparameters were used
            mock_dpo_config_cls.assert_called_once_with(beta=0.2)
            mock_config_cls.assert_called_once_with(
                dpo=mock_dpo_components["dpo_config"],
                num_epochs=5,
                batch_size=8,
                learning_rate=5e-6,
                checkpoint_dir="./checkpoints/dpo",
            )


class TestGenerateData:
    """Tests for generate_data command."""

    @pytest.fixture
    def gen_args(self):
        """Create data generation arguments."""
        return Namespace(
            type="math",
            output="./data/generated",
            sft_samples=10000,
            dpo_samples=5000,
            seed=42,
        )

    def test_generate_data_math(self, gen_args, caplog):
        """Test generating math dataset."""
        import logging

        from chuk_lazarus.cli.commands.train import generate_data

        with (
            patch("chuk_lazarus.data.generators.generate_lazarus_dataset") as mock_generate,
            caplog.at_level(logging.INFO),
        ):
            generate_data(gen_args)

            # Verify generate function was called with correct parameters
            mock_generate.assert_called_once_with(
                output_dir="./data/generated",
                sft_samples=10000,
                dpo_samples=5000,
                seed=42,
            )

            # Check logging
            assert "Generating math dataset with 10000 SFT samples" in caplog.text
            assert "Dataset saved to ./data/generated" in caplog.text

    def test_generate_data_custom_samples(self, gen_args):
        """Test generating dataset with custom sample counts."""
        from chuk_lazarus.cli.commands.train import generate_data

        gen_args.sft_samples = 5000
        gen_args.dpo_samples = 2500

        with patch("chuk_lazarus.data.generators.generate_lazarus_dataset") as mock_generate:
            generate_data(gen_args)

            mock_generate.assert_called_once_with(
                output_dir="./data/generated",
                sft_samples=5000,
                dpo_samples=2500,
                seed=42,
            )

    def test_generate_data_custom_seed(self, gen_args):
        """Test generating dataset with custom random seed."""
        from chuk_lazarus.cli.commands.train import generate_data

        gen_args.seed = 123

        with patch("chuk_lazarus.data.generators.generate_lazarus_dataset") as mock_generate:
            generate_data(gen_args)

            mock_generate.assert_called_once_with(
                output_dir="./data/generated",
                sft_samples=10000,
                dpo_samples=5000,
                seed=123,
            )

    def test_generate_data_custom_output_dir(self, gen_args):
        """Test generating dataset to custom output directory."""
        from chuk_lazarus.cli.commands.train import generate_data

        gen_args.output = "/custom/path/data"

        with patch("chuk_lazarus.data.generators.generate_lazarus_dataset") as mock_generate:
            generate_data(gen_args)

            mock_generate.assert_called_once_with(
                output_dir="/custom/path/data",
                sft_samples=10000,
                dpo_samples=5000,
                seed=42,
            )

    def test_generate_data_unknown_type(self, gen_args, caplog):
        """Test generating dataset with unknown type exits with error."""
        import logging

        from chuk_lazarus.cli.commands.train import generate_data

        gen_args.type = "unknown"

        with (
            patch("chuk_lazarus.data.generators.generate_lazarus_dataset") as mock_generate,
            caplog.at_level(logging.ERROR),
            pytest.raises(SystemExit) as exc_info,
        ):
            generate_data(gen_args)

        # Verify it exits with error code 1
        assert exc_info.value.code == 1

        # Verify generate was never called
        mock_generate.assert_not_called()

        # Check error logging
        assert "Unknown data type: unknown" in caplog.text
