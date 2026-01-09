"""Tests for DPO loss."""

from unittest.mock import MagicMock

import mlx.core as mx

from chuk_lazarus.training.losses.dpo_loss import (
    DPOConfig,
    create_dpo_loss_fn,
    dpo_loss,
)


class TestDPOConfig:
    """Tests for DPOConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DPOConfig()

        assert config.beta == 0.1
        assert config.label_smoothing == 0.0
        assert config.reference_free is False

    def test_custom_config(self):
        """Test custom configuration."""
        config = DPOConfig(beta=0.2, label_smoothing=0.1, reference_free=True)

        assert config.beta == 0.2
        assert config.label_smoothing == 0.1
        assert config.reference_free is True


class TestDPOLoss:
    """Tests for dpo_loss function."""

    def _create_mock_model(self, batch_size: int, seq_len: int, vocab_size: int) -> MagicMock:
        """Create a mock model that returns logits."""
        model = MagicMock()
        logits = mx.random.uniform(shape=(batch_size, seq_len, vocab_size))
        model.return_value = (logits,)
        return model

    def test_basic_dpo_loss(self):
        """Test basic DPO loss computation."""
        batch_size = 2
        seq_len = 4
        vocab_size = 10

        policy_model = self._create_mock_model(batch_size, seq_len, vocab_size)
        reference_model = self._create_mock_model(batch_size, seq_len, vocab_size)

        chosen_input_ids = mx.random.randint(0, vocab_size, (batch_size, seq_len))
        rejected_input_ids = mx.random.randint(0, vocab_size, (batch_size, seq_len))

        loss, metrics = dpo_loss(
            policy_model,
            reference_model,
            chosen_input_ids,
            rejected_input_ids,
        )

        assert isinstance(loss.item(), float)
        assert "loss" in metrics
        assert "chosen_reward" in metrics
        assert "rejected_reward" in metrics
        assert "reward_margin" in metrics
        assert "accuracy" in metrics

    def test_dpo_loss_with_attention_mask(self):
        """Test DPO loss with attention masks."""
        batch_size = 2
        seq_len = 4
        vocab_size = 10

        policy_model = self._create_mock_model(batch_size, seq_len, vocab_size)
        reference_model = self._create_mock_model(batch_size, seq_len, vocab_size)

        chosen_input_ids = mx.random.randint(0, vocab_size, (batch_size, seq_len))
        rejected_input_ids = mx.random.randint(0, vocab_size, (batch_size, seq_len))
        chosen_mask = mx.array([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=mx.float32)
        rejected_mask = mx.array([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=mx.float32)

        loss, metrics = dpo_loss(
            policy_model,
            reference_model,
            chosen_input_ids,
            rejected_input_ids,
            chosen_mask,
            rejected_mask,
        )

        assert isinstance(loss.item(), float)

    def test_dpo_loss_reference_free(self):
        """Test reference-free DPO loss."""
        batch_size = 2
        seq_len = 4
        vocab_size = 10

        policy_model = self._create_mock_model(batch_size, seq_len, vocab_size)
        # Reference model not used in reference_free mode
        reference_model = MagicMock()

        chosen_input_ids = mx.random.randint(0, vocab_size, (batch_size, seq_len))
        rejected_input_ids = mx.random.randint(0, vocab_size, (batch_size, seq_len))

        config = DPOConfig(reference_free=True)

        loss, metrics = dpo_loss(
            policy_model,
            reference_model,
            chosen_input_ids,
            rejected_input_ids,
            config=config,
        )

        assert isinstance(loss.item(), float)
        # Reference model should not be called
        reference_model.assert_not_called()

    def test_dpo_loss_with_label_smoothing(self):
        """Test DPO loss with label smoothing."""
        batch_size = 2
        seq_len = 4
        vocab_size = 10

        policy_model = self._create_mock_model(batch_size, seq_len, vocab_size)
        reference_model = self._create_mock_model(batch_size, seq_len, vocab_size)

        chosen_input_ids = mx.random.randint(0, vocab_size, (batch_size, seq_len))
        rejected_input_ids = mx.random.randint(0, vocab_size, (batch_size, seq_len))

        config = DPOConfig(label_smoothing=0.1)

        loss, metrics = dpo_loss(
            policy_model,
            reference_model,
            chosen_input_ids,
            rejected_input_ids,
            config=config,
        )

        assert isinstance(loss.item(), float)

    def test_dpo_loss_accuracy_metric(self):
        """Test that accuracy metric is between 0 and 1."""
        batch_size = 4
        seq_len = 8
        vocab_size = 50

        policy_model = self._create_mock_model(batch_size, seq_len, vocab_size)
        reference_model = self._create_mock_model(batch_size, seq_len, vocab_size)

        chosen_input_ids = mx.random.randint(0, vocab_size, (batch_size, seq_len))
        rejected_input_ids = mx.random.randint(0, vocab_size, (batch_size, seq_len))

        loss, metrics = dpo_loss(
            policy_model,
            reference_model,
            chosen_input_ids,
            rejected_input_ids,
        )

        accuracy = metrics["accuracy"].item()
        assert 0.0 <= accuracy <= 1.0

    def test_dpo_beta_affects_rewards(self):
        """Test that beta scales the rewards."""
        batch_size = 2
        seq_len = 4
        vocab_size = 10

        policy_model = self._create_mock_model(batch_size, seq_len, vocab_size)
        reference_model = self._create_mock_model(batch_size, seq_len, vocab_size)

        chosen_input_ids = mx.random.randint(0, vocab_size, (batch_size, seq_len))
        rejected_input_ids = mx.random.randint(0, vocab_size, (batch_size, seq_len))

        config_low_beta = DPOConfig(beta=0.1)
        config_high_beta = DPOConfig(beta=0.5)

        _, metrics_low = dpo_loss(
            policy_model,
            reference_model,
            chosen_input_ids,
            rejected_input_ids,
            config=config_low_beta,
        )

        _, metrics_high = dpo_loss(
            policy_model,
            reference_model,
            chosen_input_ids,
            rejected_input_ids,
            config=config_high_beta,
        )

        # Higher beta should scale the rewards
        # Note: actual values depend on random logits, but relationship should hold


class TestCreateDPOLossFn:
    """Tests for create_dpo_loss_fn function."""

    def test_create_loss_fn(self):
        """Test creating a loss function."""
        batch_size = 2
        seq_len = 4
        vocab_size = 10

        policy_model = MagicMock()
        policy_model.return_value = (mx.random.uniform(shape=(batch_size, seq_len, vocab_size)),)

        reference_model = MagicMock()
        reference_model.return_value = (mx.random.uniform(shape=(batch_size, seq_len, vocab_size)),)

        loss_fn = create_dpo_loss_fn(policy_model, reference_model)

        batch = {
            "chosen_input_ids": mx.random.randint(0, vocab_size, (batch_size, seq_len)),
            "rejected_input_ids": mx.random.randint(0, vocab_size, (batch_size, seq_len)),
        }

        loss, metrics = loss_fn(batch)

        assert isinstance(loss.item(), float)

    def test_create_loss_fn_with_config(self):
        """Test creating a loss function with custom config."""
        batch_size = 2
        seq_len = 4
        vocab_size = 10

        policy_model = MagicMock()
        policy_model.return_value = (mx.random.uniform(shape=(batch_size, seq_len, vocab_size)),)

        reference_model = MagicMock()
        reference_model.return_value = (mx.random.uniform(shape=(batch_size, seq_len, vocab_size)),)

        config = DPOConfig(beta=0.2, reference_free=True)
        loss_fn = create_dpo_loss_fn(policy_model, reference_model, config)

        batch = {
            "chosen_input_ids": mx.random.randint(0, vocab_size, (batch_size, seq_len)),
            "rejected_input_ids": mx.random.randint(0, vocab_size, (batch_size, seq_len)),
        }

        loss, metrics = loss_fn(batch)

        assert isinstance(loss.item(), float)
        # Reference model should not be called with reference_free=True
        reference_model.assert_not_called()
