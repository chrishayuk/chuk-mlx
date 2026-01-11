"""Tests for PPO trainer."""

import tempfile
from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest

from chuk_lazarus.training.losses.ppo_loss import PPOConfig
from chuk_lazarus.training.trainers.ppo_trainer import (
    PPOTrainer,
    PPOTrainerConfig,
)


class TestPPOTrainerConfig:
    """Tests for PPOTrainerConfig."""

    def test_default_values(self):
        """Test default values."""
        config = PPOTrainerConfig()

        assert isinstance(config.ppo, PPOConfig)
        assert config.rollout_steps == 2048
        assert config.num_envs == 1
        assert config.gamma == 0.99
        assert config.gae_lambda == 0.95
        assert config.num_epochs_per_rollout == 4
        assert config.batch_size == 64
        assert config.learning_rate == 3e-4
        assert config.weight_decay == 0.0
        assert config.total_timesteps == 1_000_000
        assert config.warmup_steps == 0
        assert config.log_interval == 1
        assert config.checkpoint_interval == 10
        assert config.checkpoint_dir == "./checkpoints/ppo"
        assert config.max_steps is None
        assert config.target_reward is None

    def test_custom_values(self):
        """Test custom values."""
        ppo_config = PPOConfig(clip_epsilon=0.3, value_loss_coef=0.5)
        config = PPOTrainerConfig(
            ppo=ppo_config,
            rollout_steps=1024,
            num_envs=4,
            gamma=0.95,
            target_reward=100.0,
        )

        assert config.ppo.clip_epsilon == 0.3
        assert config.ppo.value_loss_coef == 0.5
        assert config.rollout_steps == 1024
        assert config.num_envs == 4
        assert config.gamma == 0.95
        assert config.target_reward == 100.0


class TestPPOTrainer:
    """Tests for PPOTrainer class."""

    def test_init_basic(self):
        """Test basic initialization."""
        policy = MagicMock()
        env = MagicMock()
        config = PPOTrainerConfig(rollout_steps=100, num_envs=1)

        trainer = PPOTrainer(policy, env, config)

        assert trainer.policy is policy
        assert trainer.env is env
        assert trainer.num_rollouts == 0
        assert trainer.best_reward == float("-inf")
        assert trainer.buffer is not None

    def test_init_with_buffer_settings(self):
        """Test initialization sets buffer correctly."""
        policy = MagicMock()
        env = MagicMock()
        config = PPOTrainerConfig(rollout_steps=512, num_envs=2, gamma=0.98, gae_lambda=0.9)

        trainer = PPOTrainer(policy, env, config)

        assert trainer.buffer.buffer_size == 512
        assert trainer.buffer.num_envs == 2
        assert trainer.buffer.gamma == 0.98
        assert trainer.buffer.gae_lambda == 0.9

    def test_ppo_config_property(self):
        """Test ppo_config property."""
        policy = MagicMock()
        env = MagicMock()
        config = PPOTrainerConfig()

        trainer = PPOTrainer(policy, env, config)

        assert trainer.ppo_config is config

    def test_compute_loss_raises(self):
        """Test compute_loss raises NotImplementedError."""
        policy = MagicMock()
        env = MagicMock()

        trainer = PPOTrainer(policy, env)

        with pytest.raises(NotImplementedError, match="PPO uses custom training loop"):
            trainer.compute_loss({})

    def test_get_train_batches_raises(self):
        """Test get_train_batches raises NotImplementedError."""
        policy = MagicMock()
        env = MagicMock()

        trainer = PPOTrainer(policy, env)

        with pytest.raises(NotImplementedError, match="PPO generates rollouts"):
            list(trainer.get_train_batches(None))

    def test_obs_to_tensor_list(self):
        """Test _obs_to_tensor with list input."""
        policy = MagicMock()
        env = MagicMock()

        trainer = PPOTrainer(policy, env)

        result = trainer._obs_to_tensor([1.0, 2.0, 3.0])

        assert result == [1.0, 2.0, 3.0]

    def test_obs_to_tensor_tuple(self):
        """Test _obs_to_tensor with tuple input."""
        policy = MagicMock()
        env = MagicMock()

        trainer = PPOTrainer(policy, env)

        result = trainer._obs_to_tensor((1.0, 2.0, 3.0))

        assert result == [1.0, 2.0, 3.0]

    def test_obs_to_tensor_mx_array(self):
        """Test _obs_to_tensor with mx.array input."""
        policy = MagicMock()
        env = MagicMock()

        trainer = PPOTrainer(policy, env)

        obs = mx.array([1.0, 2.0, 3.0])
        result = trainer._obs_to_tensor(obs)

        assert result == [1.0, 2.0, 3.0]

    def test_obs_to_tensor_dict(self):
        """Test _obs_to_tensor with dict input."""
        policy = MagicMock()
        env = MagicMock()

        trainer = PPOTrainer(policy, env)

        obs = {"pos": [1.0, 2.0], "vel": 3.0}
        result = trainer._obs_to_tensor(obs)

        assert 1.0 in result
        assert 2.0 in result
        assert 3.0 in result

    def test_obs_to_tensor_unknown(self):
        """Test _obs_to_tensor with unknown input type."""
        policy = MagicMock()
        env = MagicMock()

        trainer = PPOTrainer(policy, env)

        result = trainer._obs_to_tensor("unknown")

        # Should return default
        assert len(result) == 10

    @patch("chuk_lazarus.training.trainers.ppo_trainer.mx")
    @patch("chuk_lazarus.training.trainers.ppo_trainer.logger")
    def test_save_checkpoint(self, mock_logger, mock_mx):
        """Test save_checkpoint method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            policy = MagicMock()
            policy.parameters.return_value = {"weight": mx.array([1.0])}
            env = MagicMock()
            config = PPOTrainerConfig(checkpoint_dir=tmpdir)

            trainer = PPOTrainer(policy, env, config)
            trainer.save_checkpoint("test_checkpoint")

            mock_mx.save_safetensors.assert_called_once()

    @patch("chuk_lazarus.training.trainers.ppo_trainer.mx")
    @patch("chuk_lazarus.training.trainers.ppo_trainer.logger")
    def test_load_checkpoint(self, mock_logger, mock_mx):
        """Test load_checkpoint method."""
        mock_mx.load.return_value = {"weight": mx.array([1.0])}

        policy = MagicMock()
        env = MagicMock()

        trainer = PPOTrainer(policy, env)
        trainer.load_checkpoint("/path/to/checkpoint.safetensors")

        mock_mx.load.assert_called_once_with("/path/to/checkpoint.safetensors")
        policy.load_weights.assert_called_once()


class TestPPOTrainerCollectRollout:
    """Tests for PPOTrainer._collect_rollout method."""

    def test_collect_rollout_basic(self):
        """Test basic rollout collection."""
        policy = MagicMock()
        policy.return_value = {
            "action": mx.array([1]),
            "log_prob": mx.array([0.5]),
            "value": mx.array([0.3]),
        }

        env = MagicMock()
        env.step.return_value = ([1.0, 2.0], 1.0, False, {})
        env.reset.return_value = [0.0, 0.0]

        config = PPOTrainerConfig(rollout_steps=5, num_envs=1)
        trainer = PPOTrainer(policy, env, config)
        trainer.global_step = 0

        initial_obs = [0.0, 0.0]
        _final_obs = trainer._collect_rollout(initial_obs)

        assert trainer.global_step == 5
        assert len(trainer.buffer) == 5

    def test_collect_rollout_with_done(self):
        """Test rollout collection with episode done."""
        policy = MagicMock()
        policy.return_value = {
            "action": mx.array([1]),
            "log_prob": mx.array([0.5]),
            "value": mx.array([0.3]),
        }

        env = MagicMock()
        # Third step is done
        env.step.side_effect = [
            ([1.0, 2.0], 1.0, False, {}),
            ([2.0, 3.0], 2.0, False, {}),
            ([3.0, 4.0], 3.0, True, {}),
            ([1.0, 2.0], 1.0, False, {}),
            ([2.0, 3.0], 2.0, False, {}),
        ]
        env.reset.return_value = [0.0, 0.0]

        config = PPOTrainerConfig(rollout_steps=5, num_envs=1)
        trainer = PPOTrainer(policy, env, config)
        trainer.global_step = 0

        initial_obs = [0.0, 0.0]
        trainer._collect_rollout(initial_obs)

        # Should have reset once
        env.reset.assert_called()

    def test_collect_rollout_with_reset_hidden(self):
        """Test rollout collection with policy that has reset_hidden."""
        policy = MagicMock()
        policy.reset_hidden = MagicMock()
        policy.return_value = {
            "action": mx.array([1]),
            "log_prob": mx.array([0.5]),
            "value": mx.array([0.3]),
        }

        env = MagicMock()
        env.step.return_value = ([1.0, 2.0], 1.0, False, {})

        config = PPOTrainerConfig(rollout_steps=3, num_envs=1)
        trainer = PPOTrainer(policy, env, config)
        trainer.global_step = 0

        initial_obs = [0.0, 0.0]
        trainer._collect_rollout(initial_obs)

        policy.reset_hidden.assert_called_with(batch_size=1)


class TestPPOTrainerPPOUpdate:
    """Tests for PPOTrainer._ppo_update method."""

    @patch("chuk_lazarus.training.trainers.ppo_trainer.ppo_loss")
    @patch("chuk_lazarus.training.trainers.ppo_trainer.nn")
    def test_ppo_update_basic(self, mock_nn, mock_ppo_loss):
        """Test basic PPO update."""
        mock_ppo_loss.return_value = (
            mx.array(0.5),
            {
                "policy_loss": mx.array(0.3),
                "value_loss": mx.array(0.2),
                "entropy_loss": mx.array(0.1),
                "approx_kl": mx.array(0.01),
                "clip_fraction": mx.array(0.1),
            },
        )
        mock_nn.value_and_grad.return_value = lambda model: (mx.array(0.5), {})

        policy = MagicMock()
        policy.parameters.return_value = {}
        policy.get_action_and_value.return_value = {
            "log_prob": mx.zeros((2,)),
            "value": mx.zeros((2,)),
            "entropy": mx.zeros((2,)),
        }

        env = MagicMock()
        config = PPOTrainerConfig(
            rollout_steps=10,
            num_epochs_per_rollout=1,
            batch_size=2,
        )

        trainer = PPOTrainer(policy, env, config)
        trainer.optimizer = MagicMock()

        # Add some data to buffer
        for i in range(10):
            trainer.buffer.add(
                observation=[float(i)],
                action=i % 2,
                reward=1.0,
                done=i == 9,
                log_prob=-0.5,
                value=0.5,
            )

        metrics = trainer._ppo_update()

        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy_loss" in metrics
        assert "approx_kl" in metrics
        assert "clip_fraction" in metrics

    @patch("chuk_lazarus.training.trainers.ppo_trainer.ppo_loss")
    @patch("chuk_lazarus.training.trainers.ppo_trainer.nn")
    @patch("chuk_lazarus.training.trainers.ppo_trainer.logger")
    def test_ppo_update_early_stopping_kl(self, mock_logger, mock_nn, mock_ppo_loss):
        """Test PPO update early stopping on high KL."""
        # Return high KL on second call to trigger early stopping
        call_count = [0]

        def loss_side_effect(*args, **kwargs):
            call_count[0] += 1
            kl_value = 0.5 if call_count[0] >= 2 else 0.01  # High KL on second call
            return (
                mx.array(0.5),
                {
                    "policy_loss": mx.array(0.3),
                    "value_loss": mx.array(0.2),
                    "entropy_loss": mx.array(0.1),
                    "approx_kl": mx.array(kl_value),
                    "clip_fraction": mx.array(0.1),
                },
            )

        mock_ppo_loss.side_effect = loss_side_effect
        mock_nn.value_and_grad.return_value = lambda model: (mx.array(0.5), {})

        policy = MagicMock()
        policy.parameters.return_value = {}
        policy.get_action_and_value.return_value = {
            "log_prob": mx.zeros((2,)),
            "value": mx.zeros((2,)),
            "entropy": mx.zeros((2,)),
        }

        env = MagicMock()
        ppo_config = PPOConfig(target_kl=0.1)
        config = PPOTrainerConfig(
            ppo=ppo_config,
            rollout_steps=4,
            num_epochs_per_rollout=2,
            batch_size=2,
        )

        trainer = PPOTrainer(policy, env, config)
        trainer.optimizer = MagicMock()

        # Add some data to buffer
        for i in range(4):
            trainer.buffer.add(
                observation=[float(i)],
                action=i % 2,
                reward=1.0,
                done=i == 3,
                log_prob=-0.5,
                value=0.5,
            )

        metrics = trainer._ppo_update()

        # Should have computed some metrics
        assert "policy_loss" in metrics


class TestPPOTrainerTrain:
    """Tests for PPOTrainer.train method."""

    @patch("chuk_lazarus.training.trainers.ppo_trainer.time")
    @patch("chuk_lazarus.training.trainers.ppo_trainer.logger")
    @patch.object(PPOTrainer, "_collect_rollout")
    @patch.object(PPOTrainer, "_ppo_update")
    @patch.object(PPOTrainer, "save_checkpoint")
    def test_train_basic(
        self,
        mock_save,
        mock_update,
        mock_collect,
        mock_logger,
        mock_time,
    ):
        """Test basic training loop."""
        # Return increasing time values to avoid division by zero
        time_values = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
        mock_time.time.side_effect = lambda: time_values.pop(0) if time_values else 110.0
        mock_collect.return_value = [0.0, 0.0]
        mock_update.return_value = {
            "policy_loss": 0.5,
            "value_loss": 0.3,
            "entropy_loss": 0.1,
            "approx_kl": 0.01,
            "clip_fraction": 0.1,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            policy = MagicMock()
            policy_result = MagicMock()
            policy_result.get.return_value = mx.zeros((1,))
            policy.return_value = policy_result

            env = MagicMock()
            env.reset.return_value = [0.0, 0.0]

            config = PPOTrainerConfig(
                rollout_steps=100,
                total_timesteps=250,  # 2.5 rollouts
                checkpoint_dir=tmpdir,
                checkpoint_interval=2,
            )

            trainer = PPOTrainer(policy, env, config)

            # Mock buffer methods
            trainer.buffer.get_episode_stats = MagicMock(
                return_value={
                    "mean_reward": 10.0,
                    "mean_length": 50.0,
                    "num_episodes": 2,
                }
            )
            trainer.buffer.compute_advantages = MagicMock()
            trainer.buffer.reset = MagicMock()

            def advance_step(obs):
                trainer.global_step += 100
                return obs

            mock_collect.side_effect = advance_step

            trainer.train()

            assert trainer.num_rollouts >= 2

    @patch("chuk_lazarus.training.trainers.ppo_trainer.time")
    @patch("chuk_lazarus.training.trainers.ppo_trainer.logger")
    @patch.object(PPOTrainer, "_collect_rollout")
    @patch.object(PPOTrainer, "_ppo_update")
    @patch.object(PPOTrainer, "save_checkpoint")
    def test_train_target_reward(
        self,
        mock_save,
        mock_update,
        mock_collect,
        mock_logger,
        mock_time,
    ):
        """Test training stops on target reward."""
        # Return increasing time values to avoid division by zero
        time_values = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
        mock_time.time.side_effect = lambda: time_values.pop(0) if time_values else 110.0
        mock_collect.return_value = [0.0, 0.0]
        mock_update.return_value = {
            "policy_loss": 0.5,
            "value_loss": 0.3,
            "entropy_loss": 0.1,
            "approx_kl": 0.01,
            "clip_fraction": 0.1,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            policy = MagicMock()
            policy_result = MagicMock()
            policy_result.get.return_value = mx.zeros((1,))
            policy.return_value = policy_result

            env = MagicMock()
            env.reset.return_value = [0.0, 0.0]

            config = PPOTrainerConfig(
                rollout_steps=100,
                total_timesteps=10000,
                checkpoint_dir=tmpdir,
                target_reward=50.0,
            )

            trainer = PPOTrainer(policy, env, config)

            # Mock buffer methods
            trainer.buffer.get_episode_stats = MagicMock(
                return_value={
                    "mean_reward": 100.0,  # Above target
                    "mean_length": 50.0,
                    "num_episodes": 2,
                }
            )
            trainer.buffer.compute_advantages = MagicMock()
            trainer.buffer.reset = MagicMock()

            def advance_step(obs):
                trainer.global_step += 100
                return obs

            mock_collect.side_effect = advance_step

            trainer.train()

            # Should stop after first rollout
            assert trainer.num_rollouts == 1

    @patch("chuk_lazarus.training.trainers.ppo_trainer.time")
    @patch("chuk_lazarus.training.trainers.ppo_trainer.logger")
    @patch.object(PPOTrainer, "_collect_rollout")
    @patch.object(PPOTrainer, "_ppo_update")
    @patch.object(PPOTrainer, "save_checkpoint")
    def test_train_best_checkpoint(
        self,
        mock_save,
        mock_update,
        mock_collect,
        mock_logger,
        mock_time,
    ):
        """Test best checkpoint is saved."""
        # Return increasing time values to avoid division by zero
        time_values = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0]
        mock_time.time.side_effect = lambda: time_values.pop(0) if time_values else 110.0
        mock_collect.return_value = [0.0, 0.0]
        mock_update.return_value = {
            "policy_loss": 0.5,
            "value_loss": 0.3,
            "entropy_loss": 0.1,
            "approx_kl": 0.01,
            "clip_fraction": 0.1,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            policy = MagicMock()
            policy_result = MagicMock()
            policy_result.get.return_value = mx.zeros((1,))
            policy.return_value = policy_result

            env = MagicMock()
            env.reset.return_value = [0.0, 0.0]

            config = PPOTrainerConfig(
                rollout_steps=100,
                total_timesteps=250,
                checkpoint_dir=tmpdir,
                checkpoint_interval=100,  # High so regular checkpoints don't trigger
            )

            trainer = PPOTrainer(policy, env, config)

            # Increasing rewards
            rewards = [10.0, 20.0, 30.0]
            call_count = [0]

            def get_stats():
                idx = min(call_count[0], len(rewards) - 1)
                call_count[0] += 1
                return {
                    "mean_reward": rewards[idx],
                    "mean_length": 50.0,
                    "num_episodes": 2,
                }

            trainer.buffer.get_episode_stats = get_stats
            trainer.buffer.compute_advantages = MagicMock()
            trainer.buffer.reset = MagicMock()

            def advance_step(obs):
                trainer.global_step += 100
                return obs

            mock_collect.side_effect = advance_step

            trainer.train()

            # Should save "best" multiple times
            best_calls = [c for c in mock_save.call_args_list if c[0][0] == "best"]
            assert len(best_calls) >= 1
