"""Tests for SFT trainer."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import mlx.core as mx

from chuk_lazarus.training.trainers.sft_trainer import (
    SFTConfig,
    SFTTrainer,
    SFTTrainingConfig,
    SFTTrainingResult,
)


class TestSFTTrainingConfig:
    """Tests for SFTTrainingConfig."""

    def test_required_fields(self):
        """Test required fields."""
        config = SFTTrainingConfig(model="test-model", data_path=Path("data.jsonl"))

        assert config.model == "test-model"
        assert config.data_path == Path("data.jsonl")

    def test_default_values(self):
        """Test default values."""
        config = SFTTrainingConfig(model="test-model", data_path=Path("data.jsonl"))

        assert config.use_lora is False
        assert config.lora_rank == 8
        assert config.lora_alpha == 16.0
        assert config.max_length == 512
        assert config.mask_prompt is False
        assert config.num_epochs == 3
        assert config.batch_size == 4
        assert config.learning_rate == 1e-5
        assert config.max_steps is None
        assert config.log_interval == 10
        assert config.checkpoint_interval == 500

    def test_lora_targets_default(self):
        """Test LoRA targets default."""
        config = SFTTrainingConfig(model="test-model", data_path=Path("data.jsonl"))

        assert "q_proj" in config.lora_targets
        assert "k_proj" in config.lora_targets
        assert "v_proj" in config.lora_targets
        assert "o_proj" in config.lora_targets


class TestSFTTrainingResult:
    """Tests for SFTTrainingResult."""

    def test_create_result(self):
        """Test creating result."""
        result = SFTTrainingResult(
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
        result = SFTTrainingResult(output_dir=Path("./output"), epochs_completed=2)

        assert result.final_loss is None
        assert result.adapter_path is None


class TestSFTConfig:
    """Tests for SFTConfig."""

    def test_default_values(self):
        """Test default values."""
        config = SFTConfig()

        assert config.num_epochs == 3
        assert config.batch_size == 4
        assert config.learning_rate == 1e-5
        assert config.weight_decay == 0.0
        assert config.max_grad_norm == 1.0
        assert config.warmup_steps == 100
        assert config.log_interval == 10
        assert config.eval_interval == 100
        assert config.checkpoint_interval == 500
        assert config.checkpoint_dir == "./checkpoints/sft"
        assert config.max_steps is None
        assert config.min_loss is None

    def test_custom_values(self):
        """Test custom values."""
        config = SFTConfig(num_epochs=5, batch_size=8, learning_rate=1e-4, min_loss=0.1)

        assert config.num_epochs == 5
        assert config.batch_size == 8
        assert config.learning_rate == 1e-4
        assert config.min_loss == 0.1


class TestSFTTrainer:
    """Tests for SFTTrainer class."""

    def test_init_basic(self):
        """Test basic initialization."""
        model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0

        trainer = SFTTrainer(model, tokenizer)

        assert trainer.model is model
        assert trainer.tokenizer is tokenizer

    def test_init_with_config(self):
        """Test initialization with config."""
        model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        config = SFTConfig(num_epochs=5, batch_size=8)

        trainer = SFTTrainer(model, tokenizer, config)

        assert trainer.sft_config.num_epochs == 5
        assert trainer.sft_config.batch_size == 8

    def test_sft_config_property(self):
        """Test sft_config property."""
        model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        config = SFTConfig()

        trainer = SFTTrainer(model, tokenizer, config)

        assert trainer.sft_config is config

    @patch("chuk_lazarus.training.trainers.sft_trainer.sft_loss")
    def test_compute_loss_logits_only(self, mock_sft_loss):
        """Test compute_loss with logits output."""
        mock_sft_loss.return_value = (mx.array(0.5), {"loss": mx.array(0.5)})

        model = MagicMock()
        model.return_value = mx.array([[[0.1, 0.2, 0.3]]])  # Just logits
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0

        trainer = SFTTrainer(model, tokenizer)

        batch = {
            "input_ids": mx.array([[1, 2, 3]]),
            "labels": mx.array([[2, 3, 4]]),
            "loss_mask": mx.array([[1, 1, 1]]),
        }

        loss, metrics = trainer.compute_loss(batch)

        mock_sft_loss.assert_called_once()
        assert float(loss) == 0.5

    @patch("chuk_lazarus.training.trainers.sft_trainer.sft_loss")
    def test_compute_loss_tuple_output(self, mock_sft_loss):
        """Test compute_loss with tuple output."""
        mock_sft_loss.return_value = (mx.array(0.5), {"loss": mx.array(0.5)})

        model = MagicMock()
        logits = mx.array([[[0.1, 0.2, 0.3]]])
        model.return_value = (logits, None)  # (logits, cache) tuple
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0

        trainer = SFTTrainer(model, tokenizer)

        batch = {
            "input_ids": mx.array([[1, 2, 3]]),
            "labels": mx.array([[2, 3, 4]]),
            "loss_mask": mx.array([[1, 1, 1]]),
        }

        loss, metrics = trainer.compute_loss(batch)

        mock_sft_loss.assert_called_once()

    def test_get_train_batches(self):
        """Test get_train_batches method."""
        model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        config = SFTConfig(batch_size=4)

        trainer = SFTTrainer(model, tokenizer, config)

        mock_dataset = MagicMock()
        mock_iterator = iter([{"test": "batch"}])
        mock_dataset.iter_batches.return_value = mock_iterator

        _result = trainer.get_train_batches(mock_dataset)

        mock_dataset.iter_batches.assert_called_once_with(
            batch_size=4, shuffle=True, pad_token_id=0
        )

    @patch.object(SFTTrainer, "__init__", lambda x, *args, **kwargs: None)
    def test_create_epoch_metrics(self):
        """Test _create_epoch_metrics method."""
        trainer = SFTTrainer.__new__(SFTTrainer)

        metrics = trainer._create_epoch_metrics()

        assert "loss" in metrics
        assert "perplexity" in metrics
        assert "num_tokens" in metrics

    @patch("chuk_lazarus.training.trainers.sft_trainer.logger")
    def test_log_metrics(self, mock_logger):
        """Test _log_metrics method."""
        import time as real_time

        model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0

        trainer = SFTTrainer(model, tokenizer)
        trainer._start_time = real_time.time() - 10  # 10 seconds ago
        trainer.global_step = 100
        trainer.current_epoch = 0
        trainer._current_epoch_metrics = {"num_tokens": [100, 200]}

        metrics = {"loss": 0.5, "perplexity": 2.5}
        trainer._log_metrics(metrics)

        assert len(trainer.metrics_history) == 1
        assert trainer.metrics_history[0]["step"] == 100
        assert trainer.metrics_history[0]["epoch"] == 0

    @patch("chuk_lazarus.training.trainers.sft_trainer.logger")
    def test_log_eval_metrics(self, mock_logger):
        """Test _log_eval_metrics method."""
        model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0

        trainer = SFTTrainer(model, tokenizer)

        metrics = {"loss": 0.5, "perplexity": 2.5}
        trainer._log_eval_metrics(metrics)

        mock_logger.info.assert_called()

    def test_should_stop_early_true(self):
        """Test _should_stop_early returns True."""
        model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        config = SFTConfig(min_loss=0.5)

        trainer = SFTTrainer(model, tokenizer, config)

        metrics = {"loss": 0.4}
        assert trainer._should_stop_early(metrics) is True

    def test_should_stop_early_false(self):
        """Test _should_stop_early returns False."""
        model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        config = SFTConfig(min_loss=0.5)

        trainer = SFTTrainer(model, tokenizer, config)

        metrics = {"loss": 0.6}
        assert trainer._should_stop_early(metrics) is False

    def test_should_stop_early_no_min_loss(self):
        """Test _should_stop_early with no min_loss."""
        model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        config = SFTConfig()

        trainer = SFTTrainer(model, tokenizer, config)

        metrics = {"loss": 0.1}
        assert trainer._should_stop_early(metrics) is False

    @patch("chuk_lazarus.training.trainers.sft_trainer.sft_loss")
    def test_evaluate(self, mock_sft_loss):
        """Test evaluate method."""
        mock_sft_loss.return_value = (
            mx.array(0.5),
            {
                "loss": mx.array(0.5),
                "perplexity": mx.array(2.5),
                "num_tokens": mx.array(100),
            },
        )

        model = MagicMock()
        model.return_value = mx.array([[[0.1, 0.2, 0.3]]])
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        config = SFTConfig(batch_size=4)

        trainer = SFTTrainer(model, tokenizer, config)

        mock_dataset = MagicMock()
        mock_dataset.iter_batches.return_value = [
            {
                "input_ids": mx.array([[1, 2, 3]]),
                "labels": mx.array([[2, 3, 4]]),
                "loss_mask": mx.array([[1, 1, 1]]),
            }
        ]

        results = trainer.evaluate(mock_dataset)

        assert "loss" in results
        assert "perplexity" in results
        assert "num_tokens" in results

    @patch("chuk_lazarus.training.trainers.sft_trainer.sft_loss")
    def test_evaluate_with_tuple_output(self, mock_sft_loss):
        """Test evaluate method with tuple model output."""
        mock_sft_loss.return_value = (
            mx.array(0.5),
            {
                "loss": mx.array(0.5),
                "perplexity": mx.array(2.5),
                "num_tokens": mx.array(100),
            },
        )

        model = MagicMock()
        logits = mx.array([[[0.1, 0.2, 0.3]]])
        model.return_value = (logits, None)
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        config = SFTConfig(batch_size=4)

        trainer = SFTTrainer(model, tokenizer, config)

        mock_dataset = MagicMock()
        mock_dataset.iter_batches.return_value = [
            {
                "input_ids": mx.array([[1, 2, 3]]),
                "labels": mx.array([[2, 3, 4]]),
                "loss_mask": mx.array([[1, 1, 1]]),
            }
        ]

        results = trainer.evaluate(mock_dataset)

        assert "loss" in results


class TestSFTTrainerRun:
    """Tests for SFTTrainer.run class method."""

    @patch("chuk_lazarus.training.trainers.sft_trainer.SFTDataset")
    @patch.object(SFTTrainer, "train")
    def test_run_basic(self, mock_train, mock_dataset_cls):
        """Test run method with basic config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "data.jsonl"
            data_path.write_text('{"prompt": "a", "response": "b"}\n')

            mock_dataset = MagicMock()
            mock_dataset.__len__ = MagicMock(return_value=100)
            mock_dataset_cls.return_value = mock_dataset

            with patch("chuk_lazarus.models_v2.load_model") as mock_load:
                mock_result = MagicMock()
                mock_result.model = MagicMock()
                mock_result.tokenizer = MagicMock()
                mock_result.tokenizer.pad_token_id = 0
                mock_load.return_value = mock_result

                config = SFTTrainingConfig(
                    model="test-model",
                    data_path=data_path,
                    output_dir=Path(tmpdir) / "output",
                    num_epochs=1,
                )

                result = SFTTrainer.run(config)

                assert isinstance(result, SFTTrainingResult)
                assert result.epochs_completed == 1

    @patch("chuk_lazarus.training.trainers.sft_trainer.SFTDataset")
    @patch.object(SFTTrainer, "train")
    def test_run_with_lora(self, mock_train, mock_dataset_cls):
        """Test run method with LoRA."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "data.jsonl"
            data_path.write_text('{"prompt": "a", "response": "b"}\n')

            mock_dataset = MagicMock()
            mock_dataset.__len__ = MagicMock(return_value=100)
            mock_dataset_cls.return_value = mock_dataset

            with patch("chuk_lazarus.models_v2.load_model_with_lora") as mock_load_lora:
                with patch("chuk_lazarus.models_v2.save_adapter") as mock_save:
                    mock_lora_result = MagicMock()
                    mock_lora_result.model = MagicMock()
                    mock_lora_result.tokenizer = MagicMock()
                    mock_lora_result.tokenizer.pad_token_id = 0
                    mock_lora_result.lora_layers = {"layer1": MagicMock()}
                    mock_lora_result.lora_parameter_count = 1000
                    mock_load_lora.return_value = mock_lora_result

                    config = SFTTrainingConfig(
                        model="test-model",
                        data_path=data_path,
                        output_dir=Path(tmpdir) / "output",
                        use_lora=True,
                        lora_rank=16,
                    )

                    result = SFTTrainer.run(config)

                    assert result.adapter_path is not None
                    mock_save.assert_called()

    @patch("chuk_lazarus.training.trainers.sft_trainer.SFTDataset")
    @patch.object(SFTTrainer, "train")
    def test_run_with_eval_data(self, mock_train, mock_dataset_cls):
        """Test run method with eval data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "data.jsonl"
            data_path.write_text('{"prompt": "a", "response": "b"}\n')
            eval_path = Path(tmpdir) / "eval.jsonl"
            eval_path.write_text('{"prompt": "c", "response": "d"}\n')

            mock_dataset = MagicMock()
            mock_dataset.__len__ = MagicMock(return_value=100)
            mock_dataset_cls.return_value = mock_dataset

            with patch("chuk_lazarus.models_v2.load_model") as mock_load:
                mock_result = MagicMock()
                mock_result.model = MagicMock()
                mock_result.tokenizer = MagicMock()
                mock_result.tokenizer.pad_token_id = 0
                mock_load.return_value = mock_result

                config = SFTTrainingConfig(
                    model="test-model",
                    data_path=data_path,
                    eval_data_path=eval_path,
                    output_dir=Path(tmpdir) / "output",
                )

                _result = SFTTrainer.run(config)

                # Dataset created twice (train and eval)
                assert mock_dataset_cls.call_count == 2

    @patch("chuk_lazarus.training.trainers.sft_trainer.SFTDataset")
    @patch.object(SFTTrainer, "train")
    def test_run_with_metrics_history(self, mock_train, mock_dataset_cls):
        """Test run captures final loss from metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "data.jsonl"
            data_path.write_text('{"prompt": "a", "response": "b"}\n')

            mock_dataset = MagicMock()
            mock_dataset.__len__ = MagicMock(return_value=100)
            mock_dataset_cls.return_value = mock_dataset

            with patch("chuk_lazarus.models_v2.load_model") as mock_load:
                mock_result = MagicMock()
                mock_result.model = MagicMock()
                mock_result.tokenizer = MagicMock()
                mock_result.tokenizer.pad_token_id = 0
                mock_load.return_value = mock_result

                config = SFTTrainingConfig(
                    model="test-model",
                    data_path=data_path,
                    output_dir=Path(tmpdir) / "output",
                )

                result = SFTTrainer.run(config)

                assert isinstance(result, SFTTrainingResult)
