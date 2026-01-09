"""Tests for SFT loss."""

import mlx.core as mx
import pytest

from chuk_lazarus.training.losses.sft_loss import SFTLossConfig, sft_loss


class TestSFTLossConfig:
    """Tests for SFTLossConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SFTLossConfig()

        assert config.mask_prompt is True
        assert config.max_seq_length == 512

    def test_custom_config(self):
        """Test custom configuration."""
        config = SFTLossConfig(mask_prompt=False, max_seq_length=1024)

        assert config.mask_prompt is False
        assert config.max_seq_length == 1024


class TestSFTLoss:
    """Tests for sft_loss function."""

    def test_basic_loss_computation(self):
        """Test basic loss computation."""
        batch_size = 2
        seq_len = 4
        vocab_size = 10

        # Create logits with high value for specific tokens using a different approach
        # Start with uniform and add to specific indices
        logits = mx.zeros((batch_size, seq_len, vocab_size))
        labels = mx.array([[1, 2, 3, 4], [5, 6, 7, 8]])

        # Simpler approach: just use uniform logits and verify basic functionality
        logits = mx.random.uniform(shape=(batch_size, seq_len, vocab_size))
        loss_mask = mx.ones((batch_size, seq_len))

        loss, metrics = sft_loss(logits, labels, loss_mask)

        # Loss should be positive
        assert loss.item() > 0
        assert "loss" in metrics
        assert "perplexity" in metrics
        assert "num_tokens" in metrics

    def test_loss_with_mask(self):
        """Test that mask correctly excludes tokens from loss."""
        batch_size = 2
        seq_len = 4
        vocab_size = 10

        logits = mx.zeros((batch_size, seq_len, vocab_size))
        labels = mx.array([[1, 2, 3, 4], [5, 6, 7, 8]])

        # Mask out first two tokens in each sequence
        loss_mask = mx.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=mx.float32)

        loss, metrics = sft_loss(logits, labels, loss_mask)

        # Only 4 tokens should contribute (2 per sequence)
        assert metrics["num_tokens"].item() == pytest.approx(4.0, rel=1e-3)

    def test_perplexity_is_exp_of_loss(self):
        """Test that perplexity is exponential of loss."""
        batch_size = 1
        seq_len = 2
        vocab_size = 5

        logits = mx.zeros((batch_size, seq_len, vocab_size))
        labels = mx.zeros((batch_size, seq_len), dtype=mx.int32)
        loss_mask = mx.ones((batch_size, seq_len))

        loss, metrics = sft_loss(logits, labels, loss_mask)

        expected_perplexity = mx.exp(loss).item()
        assert metrics["perplexity"].item() == pytest.approx(expected_perplexity, rel=1e-3)

    def test_random_logits_produce_positive_loss(self):
        """Test that random logits produce positive loss."""
        batch_size = 2
        seq_len = 4
        vocab_size = 100

        # Random logits (uniform)
        random_logits = mx.random.uniform(shape=(batch_size, seq_len, vocab_size))
        labels = mx.random.randint(0, vocab_size, (batch_size, seq_len))
        loss_mask = mx.ones((batch_size, seq_len))

        loss_random, metrics = sft_loss(random_logits, labels, loss_mask)

        # Random should have positive loss
        assert loss_random.item() > 0
        # Perplexity should be > 1 for random predictions
        assert metrics["perplexity"].item() > 1.0

    def test_empty_mask_handling(self):
        """Test handling of all-zero mask."""
        batch_size = 1
        seq_len = 2
        vocab_size = 5

        logits = mx.zeros((batch_size, seq_len, vocab_size))
        labels = mx.zeros((batch_size, seq_len), dtype=mx.int32)
        loss_mask = mx.zeros((batch_size, seq_len))  # All zeros

        loss, metrics = sft_loss(logits, labels, loss_mask)

        # Loss should be 0 or very small (due to epsilon)
        assert metrics["num_tokens"].item() < 1e-5
