"""Tests for GRPO trainer."""

import tempfile
from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest

from chuk_lazarus.training.losses.grpo_loss import GRPOConfig
from chuk_lazarus.training.trainers.grpo_trainer import (
    GRPOTrainer,
    GRPOTrainerConfig,
)


class TestGRPOTrainerConfig:
    """Tests for GRPOTrainerConfig."""

    def test_default_values(self):
        """Test default values."""
        config = GRPOTrainerConfig()

        assert isinstance(config.grpo, GRPOConfig)
        assert config.num_iterations == 1000
        assert config.prompts_per_iteration == 16
        assert config.learning_rate == 1e-6
        assert config.weight_decay == 0.0
        assert config.max_grad_norm == 1.0
        assert config.max_response_length == 256
        assert config.temperature == 1.0
        assert config.log_interval == 1
        assert config.checkpoint_interval == 50
        assert config.checkpoint_dir == "./checkpoints/grpo"
        assert config.max_steps is None
        assert config.target_reward is None

    def test_custom_values(self):
        """Test custom values."""
        grpo_config = GRPOConfig(group_size=8, kl_coef=0.2)
        config = GRPOTrainerConfig(
            grpo=grpo_config,
            num_iterations=500,
            prompts_per_iteration=8,
            learning_rate=1e-5,
            target_reward=10.0,
        )

        assert config.grpo.group_size == 8
        assert config.num_iterations == 500
        assert config.prompts_per_iteration == 8
        assert config.learning_rate == 1e-5
        assert config.target_reward == 10.0


class TestGRPOTrainer:
    """Tests for GRPOTrainer class."""

    def test_init_basic(self):
        """Test basic initialization."""
        policy_model = MagicMock()
        ref_model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        reward_fn = MagicMock()

        trainer = GRPOTrainer(policy_model, ref_model, tokenizer, reward_fn)

        assert trainer.policy_model is policy_model
        assert trainer.reference_model is ref_model
        assert trainer.reward_fn is reward_fn
        assert trainer.iteration == 0
        assert trainer.best_reward == float("-inf")
        ref_model.freeze.assert_called_once()

    def test_init_with_config(self):
        """Test initialization with config."""
        policy_model = MagicMock()
        ref_model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        reward_fn = MagicMock()
        config = GRPOTrainerConfig(num_iterations=500)

        trainer = GRPOTrainer(policy_model, ref_model, tokenizer, reward_fn, config)

        assert trainer.grpo_config.num_iterations == 500

    def test_grpo_config_property(self):
        """Test grpo_config property."""
        policy_model = MagicMock()
        ref_model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        reward_fn = MagicMock()
        config = GRPOTrainerConfig()

        trainer = GRPOTrainer(policy_model, ref_model, tokenizer, reward_fn, config)

        assert trainer.grpo_config is config

    def test_compute_loss_raises(self):
        """Test compute_loss raises NotImplementedError."""
        policy_model = MagicMock()
        ref_model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        reward_fn = MagicMock()

        trainer = GRPOTrainer(policy_model, ref_model, tokenizer, reward_fn)

        with pytest.raises(NotImplementedError, match="GRPO uses custom training loop"):
            trainer.compute_loss({})

    def test_get_train_batches_raises(self):
        """Test get_train_batches raises NotImplementedError."""
        policy_model = MagicMock()
        ref_model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        reward_fn = MagicMock()

        trainer = GRPOTrainer(policy_model, ref_model, tokenizer, reward_fn)

        with pytest.raises(NotImplementedError, match="GRPO generates samples"):
            list(trainer.get_train_batches(None))

    def test_sample_token(self):
        """Test _sample_token method."""
        policy_model = MagicMock()
        ref_model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        reward_fn = MagicMock()

        trainer = GRPOTrainer(policy_model, ref_model, tokenizer, reward_fn)

        probs = mx.array([0.1, 0.7, 0.2])
        token = trainer._sample_token(probs)

        assert isinstance(token, mx.array)
        assert int(token) in [0, 1, 2]

    def test_generate_response(self):
        """Test _generate_response method."""
        policy_model = MagicMock()
        # Model returns (logits, cache) tuple
        logits = mx.random.uniform(shape=(1, 1, 100))
        policy_model.return_value = (logits, None)

        ref_model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 2
        tokenizer.encode.return_value = [1, 2, 3]
        tokenizer.decode.return_value = "response"
        reward_fn = MagicMock()
        config = GRPOTrainerConfig(max_response_length=5, temperature=1.0)

        trainer = GRPOTrainer(policy_model, ref_model, tokenizer, reward_fn, config)

        response = trainer._generate_response("test prompt")

        assert isinstance(response, str)
        tokenizer.encode.assert_called_with("test prompt")
        tokenizer.decode.assert_called()

    def test_generate_response_with_model_output_attr(self):
        """Test _generate_response with model that has .logits attribute."""
        policy_model = MagicMock()
        # Model returns object with .logits attribute
        output = MagicMock()
        output.logits = mx.random.uniform(shape=(1, 1, 100))
        policy_model.return_value = output

        ref_model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 2
        tokenizer.encode.return_value = [1, 2, 3]
        tokenizer.decode.return_value = "response"
        reward_fn = MagicMock()
        config = GRPOTrainerConfig(max_response_length=3, temperature=1.0)

        trainer = GRPOTrainer(policy_model, ref_model, tokenizer, reward_fn, config)

        response = trainer._generate_response("test prompt")

        assert isinstance(response, str)

    def test_generate_response_with_set_mode(self):
        """Test _generate_response calls set_mode if available."""
        policy_model = MagicMock()
        policy_model.set_mode = MagicMock()
        logits = mx.random.uniform(shape=(1, 1, 100))
        policy_model.return_value = logits  # Direct logits

        ref_model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.encode.return_value = [1, 2, 3]
        tokenizer.decode.return_value = "response"
        reward_fn = MagicMock()
        config = GRPOTrainerConfig(max_response_length=3, temperature=1.0)

        trainer = GRPOTrainer(policy_model, ref_model, tokenizer, reward_fn, config)

        # Remove eos_token_id to test path without EOS check
        del tokenizer.eos_token_id

        _response = trainer._generate_response("test prompt")

        policy_model.set_mode.assert_called_with("INFERENCE")

    def test_generate_grpo_batch(self):
        """Test _generate_grpo_batch method."""
        policy_model = MagicMock()
        logits = mx.random.uniform(shape=(1, 1, 100))
        policy_model.return_value = (logits, None)

        ref_model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 2
        tokenizer.encode.return_value = [1, 2, 3]
        tokenizer.decode.return_value = "response"
        reward_fn = MagicMock(return_value=1.0)
        grpo_config = GRPOConfig(group_size=2)
        config = GRPOTrainerConfig(grpo=grpo_config, max_response_length=3)

        trainer = GRPOTrainer(policy_model, ref_model, tokenizer, reward_fn, config)

        batch = trainer._generate_grpo_batch(["prompt1", "prompt2"])

        assert batch.group_size == 2
        assert len(batch.prompts) == 2
        assert len(batch.responses) == 2
        # 2 prompts x 2 responses each
        assert reward_fn.call_count == 4

    @patch("chuk_lazarus.training.trainers.grpo_trainer.extract_log_probs")
    @patch("chuk_lazarus.training.trainers.grpo_trainer.compute_sequence_log_prob")
    @patch("chuk_lazarus.training.trainers.grpo_trainer.grpo_loss")
    @patch("chuk_lazarus.training.trainers.grpo_trainer.nn")
    def test_grpo_update(self, mock_nn, mock_grpo_loss, mock_seq_log_prob, mock_extract_log_probs):
        """Test _grpo_update method."""
        mock_extract_log_probs.return_value = (mx.zeros((4, 10)), mx.zeros((4, 10)))
        mock_seq_log_prob.return_value = mx.zeros((4,))
        mock_grpo_loss.return_value = (
            mx.array(0.5),
            {
                "mean_reward": mx.array(1.0),
                "policy_loss": mx.array(0.3),
                "kl_penalty": mx.array(0.1),
            },
        )
        mock_nn.value_and_grad.return_value = lambda model: (mx.array(0.5), {})

        policy_model = MagicMock()
        policy_model.parameters.return_value = {}
        ref_model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.encode.return_value = [1, 2, 3]
        reward_fn = MagicMock()
        grpo_config = GRPOConfig(group_size=2)
        config = GRPOTrainerConfig(grpo=grpo_config)

        trainer = GRPOTrainer(policy_model, ref_model, tokenizer, reward_fn, config)
        trainer.optimizer = MagicMock()

        # Create mock batch
        from chuk_lazarus.training.losses.grpo_loss import GRPOBatch

        batch = GRPOBatch(group_size=2)
        batch.add_prompt_group("prompt1", ["resp1", "resp2"], [1.0, 0.5])
        batch.add_prompt_group("prompt2", ["resp3", "resp4"], [0.8, 0.2])

        metrics = trainer._grpo_update(batch)

        assert "mean_reward" in metrics
        assert "policy_loss" in metrics
        assert "kl_penalty" in metrics

    @patch("chuk_lazarus.training.trainers.grpo_trainer.mx")
    @patch("chuk_lazarus.training.trainers.grpo_trainer.logger")
    def test_save_checkpoint_full_model(self, mock_logger, mock_mx):
        """Test save_checkpoint with full model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            policy_model = MagicMock()
            policy_model.parameters.return_value = {"weight": mx.array([1.0])}
            ref_model = MagicMock()
            tokenizer = MagicMock()
            tokenizer.pad_token_id = 0
            reward_fn = MagicMock()
            config = GRPOTrainerConfig(checkpoint_dir=tmpdir)

            trainer = GRPOTrainer(policy_model, ref_model, tokenizer, reward_fn, config)
            trainer.save_checkpoint("test_checkpoint")

            mock_mx.save_safetensors.assert_called_once()

    @patch("chuk_lazarus.training.trainers.grpo_trainer.logger")
    def test_save_checkpoint_lora(self, mock_logger):
        """Test save_checkpoint with LoRA layers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            policy_model = MagicMock()
            ref_model = MagicMock()
            tokenizer = MagicMock()
            tokenizer.pad_token_id = 0
            reward_fn = MagicMock()
            config = GRPOTrainerConfig(checkpoint_dir=tmpdir)

            trainer = GRPOTrainer(policy_model, ref_model, tokenizer, reward_fn, config)
            trainer.lora_layers = {"layer1": MagicMock()}
            trainer.lora_config = MagicMock()

            with patch("chuk_lazarus.models_v2.loader.save_adapter") as mock_save:
                trainer.save_checkpoint("test_checkpoint")
                mock_save.assert_called_once()

    @patch("chuk_lazarus.training.trainers.grpo_trainer.mx")
    @patch("chuk_lazarus.training.trainers.grpo_trainer.logger")
    def test_load_checkpoint(self, mock_logger, mock_mx):
        """Test load_checkpoint method."""
        mock_mx.load.return_value = {"weight": mx.array([1.0])}

        policy_model = MagicMock()
        ref_model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        reward_fn = MagicMock()

        trainer = GRPOTrainer(policy_model, ref_model, tokenizer, reward_fn)
        trainer.load_checkpoint("/path/to/checkpoint.safetensors")

        mock_mx.load.assert_called_once_with("/path/to/checkpoint.safetensors")
        policy_model.load_weights.assert_called_once()


class TestGRPOTrainerTrain:
    """Tests for GRPOTrainer.train method."""

    @patch("chuk_lazarus.training.trainers.grpo_trainer.time")
    @patch("chuk_lazarus.training.trainers.grpo_trainer.logger")
    @patch.object(GRPOTrainer, "_generate_grpo_batch")
    @patch.object(GRPOTrainer, "_grpo_update")
    @patch.object(GRPOTrainer, "save_checkpoint")
    def test_train_basic(
        self,
        mock_save,
        mock_update,
        mock_generate,
        mock_logger,
        mock_time,
    ):
        """Test basic training loop."""
        mock_time.time.return_value = 100.0

        from chuk_lazarus.training.losses.grpo_loss import GRPOBatch

        mock_batch = GRPOBatch(group_size=2)
        mock_batch.add_prompt_group("p1", ["r1", "r2"], [1.0, 0.5])
        mock_generate.return_value = mock_batch
        mock_update.return_value = {
            "mean_reward": 1.0,
            "policy_loss": 0.5,
            "kl_penalty": 0.1,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            policy_model = MagicMock()
            ref_model = MagicMock()
            tokenizer = MagicMock()
            tokenizer.pad_token_id = 0
            reward_fn = MagicMock()
            config = GRPOTrainerConfig(
                num_iterations=3,
                prompts_per_iteration=2,
                checkpoint_dir=tmpdir,
                checkpoint_interval=2,
            )

            trainer = GRPOTrainer(policy_model, ref_model, tokenizer, reward_fn, config)

            prompt_source = MagicMock(return_value=["prompt1", "prompt2"])

            trainer.train(prompt_source)

            assert trainer.iteration == 3
            assert mock_generate.call_count == 3
            assert mock_update.call_count == 3

    @patch("chuk_lazarus.training.trainers.grpo_trainer.time")
    @patch("chuk_lazarus.training.trainers.grpo_trainer.logger")
    @patch.object(GRPOTrainer, "_generate_grpo_batch")
    @patch.object(GRPOTrainer, "_grpo_update")
    @patch.object(GRPOTrainer, "save_checkpoint")
    def test_train_early_stopping(
        self,
        mock_save,
        mock_update,
        mock_generate,
        mock_logger,
        mock_time,
    ):
        """Test training with early stopping on target reward."""
        mock_time.time.return_value = 100.0

        from chuk_lazarus.training.losses.grpo_loss import GRPOBatch

        mock_batch = GRPOBatch(group_size=2)
        mock_batch.add_prompt_group("p1", ["r1", "r2"], [1.0, 0.5])
        mock_generate.return_value = mock_batch
        mock_update.return_value = {
            "mean_reward": 15.0,  # Above target
            "policy_loss": 0.5,
            "kl_penalty": 0.1,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            policy_model = MagicMock()
            ref_model = MagicMock()
            tokenizer = MagicMock()
            tokenizer.pad_token_id = 0
            reward_fn = MagicMock()
            config = GRPOTrainerConfig(
                num_iterations=100,
                checkpoint_dir=tmpdir,
                target_reward=10.0,
            )

            trainer = GRPOTrainer(policy_model, ref_model, tokenizer, reward_fn, config)

            prompt_source = MagicMock(return_value=["prompt1"])

            trainer.train(prompt_source)

            # Should stop after first iteration due to target reward
            assert trainer.iteration == 1

    @patch("chuk_lazarus.training.trainers.grpo_trainer.time")
    @patch("chuk_lazarus.training.trainers.grpo_trainer.logger")
    @patch.object(GRPOTrainer, "_generate_grpo_batch")
    @patch.object(GRPOTrainer, "_grpo_update")
    @patch.object(GRPOTrainer, "save_checkpoint")
    def test_train_best_checkpoint(
        self,
        mock_save,
        mock_update,
        mock_generate,
        mock_logger,
        mock_time,
    ):
        """Test that best checkpoint is saved when reward improves."""
        mock_time.time.return_value = 100.0

        from chuk_lazarus.training.losses.grpo_loss import GRPOBatch

        mock_batch = GRPOBatch(group_size=2)
        mock_batch.add_prompt_group("p1", ["r1", "r2"], [1.0, 0.5])
        mock_generate.return_value = mock_batch

        # Reward improves each iteration
        mock_update.side_effect = [
            {"mean_reward": 1.0, "policy_loss": 0.5, "kl_penalty": 0.1},
            {"mean_reward": 2.0, "policy_loss": 0.4, "kl_penalty": 0.1},
            {"mean_reward": 3.0, "policy_loss": 0.3, "kl_penalty": 0.1},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            policy_model = MagicMock()
            ref_model = MagicMock()
            tokenizer = MagicMock()
            tokenizer.pad_token_id = 0
            reward_fn = MagicMock()
            config = GRPOTrainerConfig(
                num_iterations=3,
                checkpoint_dir=tmpdir,
                checkpoint_interval=100,  # High so regular checkpoints don't trigger
            )

            trainer = GRPOTrainer(policy_model, ref_model, tokenizer, reward_fn, config)

            prompt_source = MagicMock(return_value=["prompt1"])

            trainer.train(prompt_source)

            # Should save "best" 3 times (each iteration improves) + "final"
            best_calls = [c for c in mock_save.call_args_list if c[0][0] == "best"]
            assert len(best_calls) == 3
