"""Tests for dual reward trainer."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest

from chuk_lazarus.training.trainers.dual_reward_trainer import (
    DualRewardTrainer,
    DualRewardTrainerConfig,
)


class TestDualRewardTrainerConfig:
    """Tests for DualRewardTrainerConfig."""

    def test_default_values(self):
        """Test default values."""
        config = DualRewardTrainerConfig()

        assert config.num_epochs == 1
        assert config.batch_size == 1
        assert config.learning_rate == 1e-3
        assert config.max_steps == 500
        assert config.classifier_layer == -1
        assert config.classifier_weight == 0.4
        assert config.lora_rank == 16
        assert config.log_interval == 50
        assert config.checkpoint_interval == 100
        assert config.checkpoint_dir == "./checkpoints/dual_reward"

    def test_classifier_targets_default(self):
        """Test classifier targets default."""
        config = DualRewardTrainerConfig()

        assert "multiply" in config.classifier_targets
        assert "add" in config.classifier_targets
        assert "subtract" in config.classifier_targets
        assert "divide" in config.classifier_targets

    def test_lora_targets_default(self):
        """Test LoRA targets default."""
        config = DualRewardTrainerConfig()

        assert "v_proj" in config.lora_targets
        assert "o_proj" in config.lora_targets

    def test_custom_values(self):
        """Test custom values."""
        config = DualRewardTrainerConfig(
            num_epochs=5,
            batch_size=4,
            learning_rate=1e-4,
            classifier_layer=12,
            classifier_weight=0.6,
            lora_rank=32,
        )

        assert config.num_epochs == 5
        assert config.batch_size == 4
        assert config.learning_rate == 1e-4
        assert config.classifier_layer == 12
        assert config.classifier_weight == 0.6
        assert config.lora_rank == 32


class TestDualRewardTrainer:
    """Tests for DualRewardTrainer class."""

    def _create_mock_model(self, num_layers=24):
        """Create a mock model with proper structure."""
        model = MagicMock()
        model.model = MagicMock()
        model.model.layers = [MagicMock() for _ in range(num_layers)]
        model.model.embed_tokens = MagicMock()
        model.model.embed_tokens.weight = mx.random.uniform(shape=(1000, 128))
        model.model.norm = MagicMock()
        model.lm_head = MagicMock()

        # Make layers return proper output
        for layer in model.model.layers:
            layer.return_value = mx.random.uniform(shape=(1, 10, 128))

        return model

    def _create_mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.encode.return_value = [1, 2, 3]
        return tokenizer

    @patch("chuk_lazarus.training.trainers.dual_reward_trainer.apply_lora")
    @patch("chuk_lazarus.training.trainers.dual_reward_trainer.count_lora_parameters")
    def test_init_basic(self, mock_count, mock_apply_lora):
        """Test basic initialization."""
        mock_apply_lora.return_value = {"layer1": MagicMock()}
        mock_count.return_value = 1000

        model = self._create_mock_model()
        tokenizer = self._create_mock_tokenizer()
        config = DualRewardTrainerConfig()

        trainer = DualRewardTrainer(model, tokenizer, config)

        assert trainer.model is model
        assert trainer.tokenizer is tokenizer
        assert trainer.num_layers == 24
        mock_apply_lora.assert_called_once()

    @patch("chuk_lazarus.training.trainers.dual_reward_trainer.apply_lora")
    @patch("chuk_lazarus.training.trainers.dual_reward_trainer.count_lora_parameters")
    def test_init_classifier_layer_auto(self, mock_count, mock_apply_lora):
        """Test classifier layer auto-calculation."""
        mock_apply_lora.return_value = {"layer1": MagicMock()}
        mock_count.return_value = 1000

        model = self._create_mock_model(num_layers=20)
        tokenizer = self._create_mock_tokenizer()
        config = DualRewardTrainerConfig(classifier_layer=-1)

        trainer = DualRewardTrainer(model, tokenizer, config)

        # 55% of 20 = 11
        assert trainer.classifier_layer == 11

    @patch("chuk_lazarus.training.trainers.dual_reward_trainer.apply_lora")
    @patch("chuk_lazarus.training.trainers.dual_reward_trainer.count_lora_parameters")
    def test_init_classifier_layer_explicit(self, mock_count, mock_apply_lora):
        """Test explicit classifier layer."""
        mock_apply_lora.return_value = {"layer1": MagicMock()}
        mock_count.return_value = 1000

        model = self._create_mock_model()
        tokenizer = self._create_mock_tokenizer()
        config = DualRewardTrainerConfig(classifier_layer=10)

        trainer = DualRewardTrainer(model, tokenizer, config)

        assert trainer.classifier_layer == 10

    @patch("chuk_lazarus.training.trainers.dual_reward_trainer.apply_lora")
    @patch("chuk_lazarus.training.trainers.dual_reward_trainer.count_lora_parameters")
    def test_init_classifier_token_ids(self, mock_count, mock_apply_lora):
        """Test classifier token ID mapping."""
        mock_apply_lora.return_value = {"layer1": MagicMock()}
        mock_count.return_value = 1000

        model = self._create_mock_model()
        tokenizer = self._create_mock_tokenizer()
        tokenizer.encode.side_effect = lambda x, **kwargs: {
            "multiply": [100],
            "add": [101],
            "subtract": [102],
            "divide": [103],
        }.get(x, [0])

        config = DualRewardTrainerConfig()

        trainer = DualRewardTrainer(model, tokenizer, config)

        assert "multiply" in trainer.classifier_token_ids
        assert "add" in trainer.classifier_token_ids

    @patch("chuk_lazarus.training.trainers.dual_reward_trainer.apply_lora")
    @patch("chuk_lazarus.training.trainers.dual_reward_trainer.count_lora_parameters")
    def test_get_num_layers_nested(self, mock_count, mock_apply_lora):
        """Test _get_num_layers with nested model."""
        mock_apply_lora.return_value = {"layer1": MagicMock()}
        mock_count.return_value = 1000

        model = self._create_mock_model(num_layers=16)
        tokenizer = self._create_mock_tokenizer()
        config = DualRewardTrainerConfig()

        trainer = DualRewardTrainer(model, tokenizer, config)

        assert trainer._get_num_layers() == 16

    @patch("chuk_lazarus.training.trainers.dual_reward_trainer.apply_lora")
    @patch("chuk_lazarus.training.trainers.dual_reward_trainer.count_lora_parameters")
    def test_get_num_layers_flat(self, mock_count, mock_apply_lora):
        """Test _get_num_layers with flat model structure."""
        mock_apply_lora.return_value = {"layer1": MagicMock()}
        mock_count.return_value = 1000

        model = MagicMock()
        model.layers = [MagicMock() for _ in range(12)]
        del model.model  # Remove nested model attribute

        tokenizer = self._create_mock_tokenizer()
        config = DualRewardTrainerConfig()

        with patch.object(DualRewardTrainer, "_setup_embeddings"):
            trainer = DualRewardTrainer.__new__(DualRewardTrainer)
            trainer.model = model
            trainer.tokenizer = tokenizer
            trainer.config = config

            assert trainer._get_num_layers() == 12

    def test_get_num_layers_error(self):
        """Test _get_num_layers raises error for invalid model."""
        model = MagicMock()
        del model.model
        del model.layers

        trainer = DualRewardTrainer.__new__(DualRewardTrainer)
        trainer.model = model

        with pytest.raises(ValueError, match="Cannot determine number of layers"):
            trainer._get_num_layers()

    @patch("chuk_lazarus.training.trainers.dual_reward_trainer.apply_lora")
    @patch("chuk_lazarus.training.trainers.dual_reward_trainer.count_lora_parameters")
    def test_get_layers_nested(self, mock_count, mock_apply_lora):
        """Test _get_layers with nested model."""
        mock_apply_lora.return_value = {"layer1": MagicMock()}
        mock_count.return_value = 1000

        model = self._create_mock_model()
        tokenizer = self._create_mock_tokenizer()
        config = DualRewardTrainerConfig()

        trainer = DualRewardTrainer(model, tokenizer, config)

        layers = trainer._get_layers()
        assert layers is model.model.layers

    def test_get_layers_error(self):
        """Test _get_layers raises error for invalid model."""
        model = MagicMock()
        del model.model
        del model.layers

        trainer = DualRewardTrainer.__new__(DualRewardTrainer)
        trainer.model = model

        with pytest.raises(ValueError, match="Cannot access layers"):
            trainer._get_layers()

    @patch("chuk_lazarus.training.trainers.dual_reward_trainer.apply_lora")
    @patch("chuk_lazarus.training.trainers.dual_reward_trainer.count_lora_parameters")
    def test_create_lora_optimizer(self, mock_count, mock_apply_lora):
        """Test _create_lora_optimizer."""
        mock_apply_lora.return_value = {"layer1": MagicMock()}
        mock_count.return_value = 1000

        model = self._create_mock_model()
        tokenizer = self._create_mock_tokenizer()
        config = DualRewardTrainerConfig(learning_rate=1e-4)

        trainer = DualRewardTrainer(model, tokenizer, config)

        assert trainer.optimizer is not None

    @patch("chuk_lazarus.training.trainers.dual_reward_trainer.apply_lora")
    @patch("chuk_lazarus.training.trainers.dual_reward_trainer.count_lora_parameters")
    def test_get_lora_params(self, mock_count, mock_apply_lora):
        """Test _get_lora_params."""
        mock_lora_layer = MagicMock()
        mock_lora_layer.lora_A = mx.zeros((128, 16))
        mock_lora_layer.lora_B = mx.zeros((16, 128))
        mock_apply_lora.return_value = {"layer1": mock_lora_layer}
        mock_count.return_value = 1000

        model = self._create_mock_model()
        tokenizer = self._create_mock_tokenizer()
        config = DualRewardTrainerConfig()

        trainer = DualRewardTrainer(model, tokenizer, config)

        params = trainer._get_lora_params()
        assert len(params) == 2  # A and B for one layer

    @patch("chuk_lazarus.training.trainers.dual_reward_trainer.apply_lora")
    @patch("chuk_lazarus.training.trainers.dual_reward_trainer.count_lora_parameters")
    def test_set_lora_params(self, mock_count, mock_apply_lora):
        """Test _set_lora_params."""
        mock_lora_layer = MagicMock()
        mock_lora_layer.lora_A = mx.zeros((128, 16))
        mock_lora_layer.lora_B = mx.zeros((16, 128))
        mock_apply_lora.return_value = {"layer1": mock_lora_layer}
        mock_count.return_value = 1000

        model = self._create_mock_model()
        tokenizer = self._create_mock_tokenizer()
        config = DualRewardTrainerConfig()

        trainer = DualRewardTrainer(model, tokenizer, config)

        new_A = mx.ones((128, 16))
        new_B = mx.ones((16, 128))
        trainer._set_lora_params([new_A, new_B])

        assert trainer.lora_layers["layer1"].lora_A is new_A
        assert trainer.lora_layers["layer1"].lora_B is new_B


class TestDualRewardTrainerForward:
    """Tests for DualRewardTrainer forward methods."""

    def _create_mock_model_for_forward(self, num_layers=24, hidden_size=128, vocab_size=1000):
        """Create a mock model for forward testing."""
        model = MagicMock()
        model.model = MagicMock()
        model.model.layers = []

        for _ in range(num_layers):
            layer = MagicMock()
            layer.return_value = mx.random.uniform(shape=(1, 10, hidden_size))
            model.model.layers.append(layer)

        model.model.embed_tokens = MagicMock()
        model.model.embed_tokens.return_value = mx.random.uniform(shape=(1, 10, hidden_size))
        model.model.embed_tokens.weight = mx.random.uniform(shape=(vocab_size, hidden_size))

        model.model.norm = MagicMock()
        model.model.norm.return_value = mx.random.uniform(shape=(1, 10, hidden_size))

        model.lm_head = MagicMock()
        model.lm_head.return_value = mx.random.uniform(shape=(1, 10, vocab_size))

        return model

    @patch("chuk_lazarus.training.trainers.dual_reward_trainer.apply_lora")
    @patch("chuk_lazarus.training.trainers.dual_reward_trainer.count_lora_parameters")
    def test_forward_with_intermediate(self, mock_count, mock_apply_lora):
        """Test _forward_with_intermediate method."""
        mock_apply_lora.return_value = {"layer1": MagicMock()}
        mock_count.return_value = 1000

        model = self._create_mock_model_for_forward()
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.encode.return_value = [1, 2, 3]
        config = DualRewardTrainerConfig(classifier_layer=10)

        trainer = DualRewardTrainer(model, tokenizer, config)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        final_logits, classifier_logits = trainer._forward_with_intermediate(input_ids)

        assert final_logits.shape[-1] == 1000  # vocab_size
        assert classifier_logits.shape[-1] == 1000

    @patch("chuk_lazarus.training.trainers.dual_reward_trainer.apply_lora")
    @patch("chuk_lazarus.training.trainers.dual_reward_trainer.count_lora_parameters")
    def test_forward_with_intermediate_tuple_output(self, mock_count, mock_apply_lora):
        """Test _forward_with_intermediate with tuple layer output."""
        mock_apply_lora.return_value = {"layer1": MagicMock()}
        mock_count.return_value = 1000

        model = self._create_mock_model_for_forward()
        # Make layers return tuple
        for layer in model.model.layers:
            layer.return_value = (mx.random.uniform(shape=(1, 10, 128)), None)

        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.encode.return_value = [1, 2, 3]
        config = DualRewardTrainerConfig(classifier_layer=10)

        trainer = DualRewardTrainer(model, tokenizer, config)

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        final_logits, classifier_logits = trainer._forward_with_intermediate(input_ids)

        assert final_logits is not None
        assert classifier_logits is not None


class TestDualRewardTrainerCompute:
    """Tests for DualRewardTrainer compute methods."""

    @patch("chuk_lazarus.training.trainers.dual_reward_trainer.apply_lora")
    @patch("chuk_lazarus.training.trainers.dual_reward_trainer.count_lora_parameters")
    @patch("chuk_lazarus.training.trainers.dual_reward_trainer.dual_reward_loss")
    def test_compute_loss(self, mock_loss, mock_count, mock_apply_lora):
        """Test compute_loss method."""
        mock_apply_lora.return_value = {"layer1": MagicMock()}
        mock_count.return_value = 1000
        mock_loss.return_value = (
            mx.array(0.5),
            {
                "loss": mx.array(0.5),
                "classifier_loss": mx.array(0.2),
                "answer_loss": mx.array(0.3),
                "classifier_accuracy": mx.array(0.8),
            },
        )

        model = MagicMock()
        model.model = MagicMock()
        model.model.layers = [MagicMock() for _ in range(24)]
        for layer in model.model.layers:
            layer.return_value = mx.random.uniform(shape=(1, 5, 128))
        model.model.embed_tokens = MagicMock()
        model.model.embed_tokens.return_value = mx.random.uniform(shape=(1, 5, 128))
        model.model.embed_tokens.weight = mx.random.uniform(shape=(1000, 128))
        model.model.norm = MagicMock()
        model.model.norm.return_value = mx.random.uniform(shape=(1, 5, 128))
        model.lm_head = MagicMock()
        model.lm_head.return_value = mx.random.uniform(shape=(1, 5, 1000))

        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.encode.return_value = [1, 2, 3]
        config = DualRewardTrainerConfig()

        trainer = DualRewardTrainer(model, tokenizer, config)

        batch = {
            "input_ids": mx.array([[1, 2, 3, 4, 5]]),
            "labels": mx.array([[2, 3, 4, 5, 6]]),
            "loss_mask": mx.array([[1.0, 1.0, 1.0, 1.0, 1.0]]),
            "classifier_labels": mx.array([100]),
        }

        loss, metrics = trainer.compute_loss(batch)

        assert float(loss) == 0.5
        assert "classifier_loss" in metrics
        assert "answer_loss" in metrics


class TestDualRewardTrainerBatches:
    """Tests for DualRewardTrainer batch methods."""

    @patch("chuk_lazarus.training.trainers.dual_reward_trainer.apply_lora")
    @patch("chuk_lazarus.training.trainers.dual_reward_trainer.count_lora_parameters")
    def test_get_train_batches(self, mock_count, mock_apply_lora):
        """Test get_train_batches method."""
        mock_apply_lora.return_value = {"layer1": MagicMock()}
        mock_count.return_value = 1000

        model = MagicMock()
        model.model = MagicMock()
        model.model.layers = [MagicMock() for _ in range(24)]
        model.model.embed_tokens = MagicMock()
        model.model.embed_tokens.weight = mx.random.uniform(shape=(1000, 128))
        model.model.norm = MagicMock()
        model.lm_head = MagicMock()

        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.encode.side_effect = lambda x, **kwargs: [1, 2, 3]
        config = DualRewardTrainerConfig()

        trainer = DualRewardTrainer(model, tokenizer, config)
        trainer.classifier_token_ids = {"add": 100, "multiply": 101}

        dataset = [
            {"prompt": "1+1=", "response": "2", "operation": "add"},
            {"prompt": "2*3=", "response": "6", "classification_target": "multiply"},
        ]

        batches = list(trainer.get_train_batches(dataset))

        assert len(batches) == 2
        assert "input_ids" in batches[0]
        assert "labels" in batches[0]
        assert "loss_mask" in batches[0]
        assert "classifier_labels" in batches[0]


class TestDualRewardTrainerTrain:
    """Tests for DualRewardTrainer.train method."""

    @patch("chuk_lazarus.training.trainers.dual_reward_trainer.time")
    @patch("chuk_lazarus.training.trainers.dual_reward_trainer.logger")
    @patch("chuk_lazarus.training.trainers.dual_reward_trainer.apply_lora")
    @patch("chuk_lazarus.training.trainers.dual_reward_trainer.count_lora_parameters")
    @patch.object(DualRewardTrainer, "compute_loss")
    @patch.object(DualRewardTrainer, "save_checkpoint")
    @patch.object(DualRewardTrainer, "get_train_batches")
    def test_train_basic(
        self,
        mock_get_batches,
        mock_save,
        mock_compute_loss,
        mock_count,
        mock_apply_lora,
        mock_logger,
        mock_time,
    ):
        """Test basic training loop."""
        mock_time.time.return_value = 100.0

        mock_lora_layer = MagicMock()
        mock_lora_layer.lora_A = mx.zeros((128, 16))
        mock_lora_layer.lora_B = mx.zeros((16, 128))
        mock_apply_lora.return_value = {"layer1": mock_lora_layer}
        mock_count.return_value = 1000

        mock_compute_loss.return_value = (
            mx.array(0.5),
            {
                "loss": mx.array(0.5),
                "classifier_loss": mx.array(0.2),
                "answer_loss": mx.array(0.3),
                "classifier_accuracy": mx.array(0.8),
            },
        )

        # Mock batches
        mock_get_batches.return_value = iter(
            [
                {
                    "input_ids": mx.array([[1, 2, 3]]),
                    "labels": mx.array([[2, 3, 4]]),
                    "loss_mask": mx.array([[1.0, 1.0, 1.0]]),
                    "classifier_labels": mx.array([100]),
                },
                {
                    "input_ids": mx.array([[4, 5, 6]]),
                    "labels": mx.array([[5, 6, 7]]),
                    "loss_mask": mx.array([[1.0, 1.0, 1.0]]),
                    "classifier_labels": mx.array([100]),
                },
                {
                    "input_ids": mx.array([[7, 8, 9]]),
                    "labels": mx.array([[8, 9, 10]]),
                    "loss_mask": mx.array([[1.0, 1.0, 1.0]]),
                    "classifier_labels": mx.array([100]),
                },
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            model = MagicMock()
            model.model = MagicMock()
            model.model.layers = [MagicMock() for _ in range(24)]
            model.model.embed_tokens = MagicMock()
            model.model.embed_tokens.weight = mx.random.uniform(shape=(1000, 128))
            model.model.norm = MagicMock()
            model.lm_head = MagicMock()

            tokenizer = MagicMock()
            tokenizer.pad_token_id = 0
            tokenizer.encode.return_value = [1, 2, 3]

            config = DualRewardTrainerConfig(
                num_epochs=1,
                max_steps=3,
                checkpoint_dir=tmpdir,
                checkpoint_interval=2,
                log_interval=1,
            )

            trainer = DualRewardTrainer(model, tokenizer, config)

            # Mock _forward_with_intermediate for grad_fn
            with patch.object(trainer, "_forward_with_intermediate") as mock_forward:
                mock_forward.return_value = (
                    mx.random.uniform(shape=(1, 3, 1000)),
                    mx.random.uniform(shape=(1, 3, 1000)),
                )

                dataset = [
                    {"prompt": "1+1=", "response": "2", "operation": "add"},
                    {"prompt": "2+3=", "response": "5", "operation": "add"},
                    {"prompt": "3+4=", "response": "7", "operation": "add"},
                ]

                trainer.train(dataset)

            assert trainer.global_step == 3


class TestDualRewardTrainerCheckpoint:
    """Tests for DualRewardTrainer checkpoint methods."""

    @patch("chuk_lazarus.training.trainers.dual_reward_trainer.apply_lora")
    @patch("chuk_lazarus.training.trainers.dual_reward_trainer.count_lora_parameters")
    @patch("chuk_lazarus.training.trainers.dual_reward_trainer.logger")
    def test_save_checkpoint(self, mock_logger, mock_count, mock_apply_lora):
        """Test save_checkpoint method."""
        mock_lora_layer = MagicMock()
        mock_lora_layer.lora_A = mx.zeros((128, 16))
        mock_lora_layer.lora_B = mx.zeros((16, 128))
        mock_apply_lora.return_value = {"layer1": mock_lora_layer}
        mock_count.return_value = 1000

        with tempfile.TemporaryDirectory() as tmpdir:
            model = MagicMock()
            model.model = MagicMock()
            model.model.layers = [MagicMock() for _ in range(24)]
            model.model.embed_tokens = MagicMock()
            model.model.embed_tokens.weight = mx.random.uniform(shape=(1000, 128))
            model.model.norm = MagicMock()
            model.lm_head = MagicMock()

            tokenizer = MagicMock()
            tokenizer.pad_token_id = 0
            tokenizer.encode.return_value = [1, 2, 3]

            config = DualRewardTrainerConfig(checkpoint_dir=tmpdir)

            trainer = DualRewardTrainer(model, tokenizer, config)
            trainer.classifier_token_ids = {"add": 100}
            trainer.global_step = 100

            with patch("chuk_lazarus.models_v2.loader.save_adapter") as mock_save:
                trainer.save_checkpoint("test_checkpoint")

                mock_save.assert_called_once()
                # Check dual_reward_config.json was created
                config_path = Path(tmpdir) / "test_checkpoint" / "dual_reward_config.json"
                assert config_path.exists()


class TestDualRewardTrainerEvaluate:
    """Tests for DualRewardTrainer evaluate methods."""

    @patch("chuk_lazarus.training.trainers.dual_reward_trainer.apply_lora")
    @patch("chuk_lazarus.training.trainers.dual_reward_trainer.count_lora_parameters")
    def test_evaluate_classifier(self, mock_count, mock_apply_lora):
        """Test evaluate_classifier method."""
        mock_lora_layer = MagicMock()
        mock_lora_layer.lora_A = mx.zeros((128, 16))
        mock_lora_layer.lora_B = mx.zeros((16, 128))
        mock_apply_lora.return_value = {"layer1": mock_lora_layer}
        mock_count.return_value = 1000

        model = MagicMock()
        model.model = MagicMock()
        model.model.layers = [MagicMock() for _ in range(24)]
        for layer in model.model.layers:
            layer.return_value = mx.random.uniform(shape=(1, 10, 128))
        model.model.embed_tokens = MagicMock()
        model.model.embed_tokens.return_value = mx.random.uniform(shape=(1, 10, 128))
        model.model.embed_tokens.weight = mx.random.uniform(shape=(1000, 128))
        model.model.norm = MagicMock()
        model.model.norm.return_value = mx.random.uniform(shape=(1, 10, 128))
        model.lm_head = MagicMock()

        # Make lm_head return high probability for token 100
        def mock_lm_head(x):
            logits = mx.zeros((x.shape[0], x.shape[1], 1000))
            # Create logits with high value at position 100 using index assignment
            logits_list = logits.tolist()
            for i in range(len(logits_list)):
                for j in range(len(logits_list[i])):
                    logits_list[i][j][100] = 10.0
            return mx.array(logits_list)

        model.lm_head.side_effect = mock_lm_head

        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.encode.return_value = [1, 2, 3]

        config = DualRewardTrainerConfig()

        trainer = DualRewardTrainer(model, tokenizer, config)
        trainer.classifier_token_ids = {"add": 100, "multiply": 101}

        test_prompts = [
            ("1+1=", "add"),
            ("2+2=", "add"),
            ("3*3=", "multiply"),
        ]

        results = trainer.evaluate_classifier(test_prompts)

        assert "accuracy" in results
        assert "correct" in results
        assert "total" in results
        assert "results" in results
        assert results["total"] == 3
        assert len(results["results"]) == 3
