"""Tests for DPO trainer."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import mlx.core as mx

from chuk_lazarus.training.losses.dpo_loss import DPOConfig
from chuk_lazarus.training.trainers.dpo_trainer import (
    DPOTrainer,
    DPOTrainerConfig,
    DPOTrainingConfig,
    DPOTrainingResult,
)


class TestDPOTrainingConfig:
    """Tests for DPOTrainingConfig."""

    def test_required_fields(self):
        """Test required fields."""
        config = DPOTrainingConfig(model="test-model", data_path=Path("data.jsonl"))

        assert config.model == "test-model"
        assert config.data_path == Path("data.jsonl")

    def test_default_values(self):
        """Test default values."""
        config = DPOTrainingConfig(model="test-model", data_path=Path("data.jsonl"))

        assert config.ref_model is None
        assert config.use_lora is False
        assert config.lora_rank == 8
        assert config.lora_alpha == 16.0
        assert config.num_epochs == 3
        assert config.batch_size == 4
        assert config.learning_rate == 1e-6
        assert config.beta == 0.1
        assert config.max_steps is None
        assert config.log_interval == 10
        assert config.checkpoint_interval == 500

    def test_reference_model_property(self):
        """Test reference_model property."""
        config = DPOTrainingConfig(model="test-model", data_path=Path("data.jsonl"))
        assert config.reference_model == "test-model"

        config2 = DPOTrainingConfig(
            model="test-model", ref_model="ref-model", data_path=Path("data.jsonl")
        )
        assert config2.reference_model == "ref-model"

    def test_lora_targets_default(self):
        """Test LoRA targets default."""
        config = DPOTrainingConfig(model="test-model", data_path=Path("data.jsonl"))

        assert "q_proj" in config.lora_targets
        assert "k_proj" in config.lora_targets
        assert "v_proj" in config.lora_targets
        assert "o_proj" in config.lora_targets


class TestDPOTrainingResult:
    """Tests for DPOTrainingResult."""

    def test_create_result(self):
        """Test creating result."""
        result = DPOTrainingResult(
            output_dir=Path("./output"),
            epochs_completed=3,
            final_loss=0.5,
            adapter_path=Path("./adapters"),
        )

        assert result.output_dir == Path("./output")
        assert result.epochs_completed == 3
        assert result.final_loss == 0.5
        assert result.adapter_path == Path("./adapters")

    def test_optional_fields(self):
        """Test optional fields."""
        result = DPOTrainingResult(output_dir=Path("./output"), epochs_completed=2)

        assert result.final_loss is None
        assert result.adapter_path is None


class TestDPOTrainerConfig:
    """Tests for DPOTrainerConfig."""

    def test_default_values(self):
        """Test default values."""
        config = DPOTrainerConfig()

        assert isinstance(config.dpo, DPOConfig)
        assert config.num_epochs == 3
        assert config.batch_size == 4
        assert config.learning_rate == 1e-6
        assert config.weight_decay == 0.0
        assert config.warmup_steps == 100
        assert config.max_grad_norm == 1.0
        assert config.log_interval == 10
        assert config.eval_interval == 100
        assert config.checkpoint_interval == 500
        assert config.checkpoint_dir == "./checkpoints/dpo"
        assert config.max_steps is None
        assert config.target_reward_margin == 2.0

    def test_custom_values(self):
        """Test custom values."""
        dpo_config = DPOConfig(beta=0.2, label_smoothing=0.1)
        config = DPOTrainerConfig(
            dpo=dpo_config,
            num_epochs=5,
            batch_size=8,
            learning_rate=1e-5,
            target_reward_margin=3.0,
        )

        assert config.dpo.beta == 0.2
        assert config.dpo.label_smoothing == 0.1
        assert config.num_epochs == 5
        assert config.batch_size == 8
        assert config.learning_rate == 1e-5
        assert config.target_reward_margin == 3.0


class TestDPOTrainer:
    """Tests for DPOTrainer class."""

    def test_init_basic(self):
        """Test basic initialization."""
        policy_model = MagicMock()
        ref_model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0

        trainer = DPOTrainer(policy_model, ref_model, tokenizer)

        assert trainer.policy_model is policy_model
        assert trainer.reference_model is ref_model
        assert trainer.best_reward_margin == float("-inf")
        ref_model.freeze.assert_called_once()

    def test_init_with_config(self):
        """Test initialization with config."""
        policy_model = MagicMock()
        ref_model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        config = DPOTrainerConfig(num_epochs=5, batch_size=8)

        trainer = DPOTrainer(policy_model, ref_model, tokenizer, config)

        assert trainer.dpo_config.num_epochs == 5
        assert trainer.dpo_config.batch_size == 8

    def test_dpo_config_property(self):
        """Test dpo_config property."""
        policy_model = MagicMock()
        ref_model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        config = DPOTrainerConfig()

        trainer = DPOTrainer(policy_model, ref_model, tokenizer, config)

        assert trainer.dpo_config is config

    @patch("chuk_lazarus.training.trainers.dpo_trainer.dpo_loss")
    def test_compute_loss(self, mock_dpo_loss):
        """Test compute_loss method."""
        mock_dpo_loss.return_value = (mx.array(0.5), {"loss": mx.array(0.5)})

        policy_model = MagicMock()
        ref_model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0

        trainer = DPOTrainer(policy_model, ref_model, tokenizer)

        batch = {
            "chosen_input_ids": mx.array([[1, 2, 3]]),
            "rejected_input_ids": mx.array([[4, 5, 6]]),
            "chosen_attention_mask": mx.array([[1, 1, 1]]),
            "rejected_attention_mask": mx.array([[1, 1, 1]]),
        }

        loss, metrics = trainer.compute_loss(batch)

        mock_dpo_loss.assert_called_once()
        assert float(loss) == 0.5

    def test_get_train_batches(self):
        """Test get_train_batches method."""
        policy_model = MagicMock()
        ref_model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        config = DPOTrainerConfig(batch_size=4)

        trainer = DPOTrainer(policy_model, ref_model, tokenizer, config)

        mock_dataset = MagicMock()
        mock_iterator = iter([{"test": "batch"}])
        mock_dataset.iter_batches.return_value = mock_iterator

        _result = trainer.get_train_batches(mock_dataset)

        mock_dataset.iter_batches.assert_called_once_with(
            batch_size=4, shuffle=True, pad_token_id=0
        )

    @patch.object(DPOTrainer, "__init__", lambda x, *args, **kwargs: None)
    def test_create_epoch_metrics(self):
        """Test _create_epoch_metrics method."""
        trainer = DPOTrainer.__new__(DPOTrainer)

        metrics = trainer._create_epoch_metrics()

        assert "loss" in metrics
        assert "chosen_reward" in metrics
        assert "rejected_reward" in metrics
        assert "reward_margin" in metrics
        assert "accuracy" in metrics

    @patch("chuk_lazarus.training.trainers.dpo_trainer.time")
    @patch("chuk_lazarus.training.trainers.dpo_trainer.logger")
    def test_log_metrics(self, mock_logger, mock_time):
        """Test _log_metrics method."""
        mock_time.time.return_value = 100.0

        policy_model = MagicMock()
        ref_model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0

        trainer = DPOTrainer(policy_model, ref_model, tokenizer)
        trainer._start_time = 90.0
        trainer.global_step = 100

        metrics = {"loss": 0.5, "reward_margin": 0.3, "accuracy": 0.8}
        trainer._log_metrics(metrics)

        assert len(trainer.metrics_history) == 1
        assert trainer.metrics_history[0]["step"] == 100

    @patch("chuk_lazarus.training.trainers.dpo_trainer.logger")
    def test_log_eval_metrics(self, mock_logger):
        """Test _log_eval_metrics method."""
        policy_model = MagicMock()
        ref_model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0

        trainer = DPOTrainer(policy_model, ref_model, tokenizer)

        metrics = {"reward_margin": 0.5, "accuracy": 0.9}
        trainer._log_eval_metrics(metrics)

        mock_logger.info.assert_called()

    def test_should_stop_early_true(self):
        """Test _should_stop_early returns True."""
        policy_model = MagicMock()
        ref_model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        config = DPOTrainerConfig(target_reward_margin=1.0)

        trainer = DPOTrainer(policy_model, ref_model, tokenizer, config)

        metrics = {"reward_margin": 1.5}
        assert trainer._should_stop_early(metrics) is True

    def test_should_stop_early_false(self):
        """Test _should_stop_early returns False."""
        policy_model = MagicMock()
        ref_model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        config = DPOTrainerConfig(target_reward_margin=2.0)

        trainer = DPOTrainer(policy_model, ref_model, tokenizer, config)

        metrics = {"reward_margin": 1.5}
        assert trainer._should_stop_early(metrics) is False

    @patch("chuk_lazarus.training.trainers.dpo_trainer.mx")
    @patch("chuk_lazarus.training.trainers.dpo_trainer.logger")
    def test_save_checkpoint(self, mock_logger, mock_mx):
        """Test save_checkpoint method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            policy_model = MagicMock()
            policy_model.parameters.return_value = {"weight": mx.array([1.0])}
            ref_model = MagicMock()
            tokenizer = MagicMock()
            tokenizer.pad_token_id = 0
            config = DPOTrainerConfig(checkpoint_dir=tmpdir)

            trainer = DPOTrainer(policy_model, ref_model, tokenizer, config)
            trainer.save_checkpoint("test_checkpoint")

            mock_mx.save_safetensors.assert_called_once()

    @patch("chuk_lazarus.training.trainers.dpo_trainer.mx")
    @patch("chuk_lazarus.training.trainers.dpo_trainer.logger")
    def test_load_checkpoint(self, mock_logger, mock_mx):
        """Test load_checkpoint method."""
        mock_mx.load.return_value = {"weight": mx.array([1.0])}

        policy_model = MagicMock()
        ref_model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0

        trainer = DPOTrainer(policy_model, ref_model, tokenizer)
        trainer.load_checkpoint("/path/to/checkpoint.safetensors")

        mock_mx.load.assert_called_once_with("/path/to/checkpoint.safetensors")
        policy_model.load_weights.assert_called_once()

    @patch("chuk_lazarus.training.trainers.dpo_trainer.dpo_loss")
    def test_evaluate(self, mock_dpo_loss):
        """Test evaluate method."""
        mock_dpo_loss.return_value = (
            mx.array(0.5),
            {
                "loss": mx.array(0.5),
                "chosen_reward": mx.array(1.0),
                "rejected_reward": mx.array(0.5),
                "reward_margin": mx.array(0.5),
                "accuracy": mx.array(0.8),
            },
        )

        policy_model = MagicMock()
        ref_model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        config = DPOTrainerConfig(batch_size=4)

        trainer = DPOTrainer(policy_model, ref_model, tokenizer, config)

        mock_dataset = MagicMock()
        mock_dataset.iter_batches.return_value = [
            {
                "chosen_input_ids": mx.array([[1, 2, 3]]),
                "rejected_input_ids": mx.array([[4, 5, 6]]),
                "chosen_attention_mask": mx.array([[1, 1, 1]]),
                "rejected_attention_mask": mx.array([[1, 1, 1]]),
            }
        ]

        results = trainer.evaluate(mock_dataset)

        assert "loss" in results
        assert "chosen_reward" in results
        assert "rejected_reward" in results
        assert "reward_margin" in results
        assert "accuracy" in results


class TestDPOTrainerRun:
    """Tests for DPOTrainer.run class method."""

    @patch("chuk_lazarus.training.trainers.dpo_trainer.PreferenceDataset")
    @patch.object(DPOTrainer, "train")
    def test_run_basic(self, mock_train, mock_dataset_cls):
        """Test run method with basic config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "data.jsonl"
            data_path.write_text('{"chosen": "a", "rejected": "b"}\n')

            mock_dataset = MagicMock()
            mock_dataset.__len__ = MagicMock(return_value=100)
            mock_dataset_cls.return_value = mock_dataset

            with patch("chuk_lazarus.models_v2.load_model") as mock_load:
                mock_result = MagicMock()
                mock_result.model = MagicMock()
                mock_result.tokenizer = MagicMock()
                mock_result.tokenizer.pad_token_id = 0
                mock_load.return_value = mock_result

                config = DPOTrainingConfig(
                    model="test-model",
                    data_path=data_path,
                    output_dir=Path(tmpdir) / "output",
                    num_epochs=1,
                )

                result = DPOTrainer.run(config)

                assert isinstance(result, DPOTrainingResult)
                assert result.epochs_completed == 1

    @patch("chuk_lazarus.training.trainers.dpo_trainer.PreferenceDataset")
    @patch.object(DPOTrainer, "train")
    def test_run_with_lora(self, mock_train, mock_dataset_cls):
        """Test run method with LoRA."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "data.jsonl"
            data_path.write_text('{"chosen": "a", "rejected": "b"}\n')

            mock_dataset = MagicMock()
            mock_dataset.__len__ = MagicMock(return_value=100)
            mock_dataset_cls.return_value = mock_dataset

            with patch("chuk_lazarus.models_v2.load_model_with_lora") as mock_load_lora:
                with patch("chuk_lazarus.models_v2.load_model") as mock_load:
                    with patch("chuk_lazarus.models_v2.save_adapter") as mock_save:
                        mock_lora_result = MagicMock()
                        mock_lora_result.model = MagicMock()
                        mock_lora_result.tokenizer = MagicMock()
                        mock_lora_result.tokenizer.pad_token_id = 0
                        mock_lora_result.lora_layers = {"layer1": MagicMock()}
                        mock_lora_result.lora_parameter_count = 1000
                        mock_load_lora.return_value = mock_lora_result

                        mock_ref_result = MagicMock()
                        mock_ref_result.model = MagicMock()
                        mock_load.return_value = mock_ref_result

                        config = DPOTrainingConfig(
                            model="test-model",
                            data_path=data_path,
                            output_dir=Path(tmpdir) / "output",
                            use_lora=True,
                            lora_rank=16,
                        )

                        result = DPOTrainer.run(config)

                        assert result.adapter_path is not None
                        mock_save.assert_called()

    @patch("chuk_lazarus.training.trainers.dpo_trainer.PreferenceDataset")
    @patch.object(DPOTrainer, "train")
    def test_run_with_eval_data(self, mock_train, mock_dataset_cls):
        """Test run method with eval data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "data.jsonl"
            data_path.write_text('{"chosen": "a", "rejected": "b"}\n')
            eval_path = Path(tmpdir) / "eval.jsonl"
            eval_path.write_text('{"chosen": "c", "rejected": "d"}\n')

            mock_dataset = MagicMock()
            mock_dataset.__len__ = MagicMock(return_value=100)
            mock_dataset_cls.return_value = mock_dataset

            with patch("chuk_lazarus.models_v2.load_model") as mock_load:
                mock_result = MagicMock()
                mock_result.model = MagicMock()
                mock_result.tokenizer = MagicMock()
                mock_result.tokenizer.pad_token_id = 0
                mock_load.return_value = mock_result

                config = DPOTrainingConfig(
                    model="test-model",
                    data_path=data_path,
                    eval_data_path=eval_path,
                    output_dir=Path(tmpdir) / "output",
                )

                _result = DPOTrainer.run(config)

                # Dataset created twice (train and eval)
                assert mock_dataset_cls.call_count == 2
