"""Tests for log probability utilities."""

from unittest.mock import MagicMock

import mlx.core as mx
import pytest

from chuk_lazarus.training.utils.log_probs import (
    compute_log_probs_from_logits,
    compute_sequence_log_prob,
    extract_log_probs,
)


class TestComputeLogProbsFromLogits:
    """Tests for compute_log_probs_from_logits function."""

    def test_basic_log_probs(self):
        """Test basic log probability computation."""
        batch_size = 2
        seq_len = 3
        vocab_size = 10

        # Create random logits
        logits = mx.random.uniform(shape=(batch_size, seq_len, vocab_size))
        actions = mx.array([[0, 1, 2], [3, 4, 5]])

        log_probs = compute_log_probs_from_logits(logits, actions)

        assert log_probs.shape == (batch_size, seq_len)
        # Log probs should be negative (prob <= 1)
        assert mx.all(log_probs <= 0.0).item()

    def test_log_probs_shape(self):
        """Test output shape matches input."""
        batch_size = 4
        seq_len = 8
        vocab_size = 100

        logits = mx.random.uniform(shape=(batch_size, seq_len, vocab_size))
        actions = mx.random.randint(0, vocab_size, (batch_size, seq_len))

        log_probs = compute_log_probs_from_logits(logits, actions)

        assert log_probs.shape == (batch_size, seq_len)

    def test_log_probs_are_negative(self):
        """Test log probabilities are negative (or zero for prob=1)."""
        batch_size = 2
        seq_len = 3
        vocab_size = 5

        logits = mx.random.uniform(shape=(batch_size, seq_len, vocab_size))
        actions = mx.random.randint(0, vocab_size, (batch_size, seq_len))

        log_probs = compute_log_probs_from_logits(logits, actions)

        # Log probs should be <= 0
        assert mx.all(log_probs <= 0.0).item()

    def test_uniform_logits_give_log_vocab_size(self):
        """Test uniform logits give log(1/vocab_size)."""
        batch_size = 1
        seq_len = 1
        vocab_size = 10

        # Uniform logits
        logits = mx.zeros((batch_size, seq_len, vocab_size))
        actions = mx.array([[5]])

        log_probs = compute_log_probs_from_logits(logits, actions)

        # With uniform probs, log prob should be log(1/vocab_size)
        expected = mx.log(mx.array(1.0 / vocab_size)).item()
        assert log_probs[0, 0].item() == pytest.approx(expected, rel=1e-2)


class TestExtractLogProbs:
    """Tests for extract_log_probs function."""

    def test_extract_with_model_output_tuple(self):
        """Test extracting log probs when model returns tuple."""
        batch_size = 2
        seq_len = 4
        vocab_size = 10

        # Mock model that returns tuple
        model = MagicMock()
        logits = mx.random.uniform(shape=(batch_size, seq_len, vocab_size))
        model.return_value = (logits,)

        input_ids = mx.random.randint(0, vocab_size, (batch_size, seq_len))

        log_probs, output_logits = extract_log_probs(model, input_ids)

        # Output should be shifted by 1
        assert log_probs.shape == (batch_size, seq_len - 1)
        assert output_logits.shape == (batch_size, seq_len - 1, vocab_size)

    def test_extract_with_model_output_object(self):
        """Test extracting log probs when model returns object with .logits."""
        batch_size = 2
        seq_len = 4
        vocab_size = 10

        # Mock model that returns object with .logits
        model = MagicMock()
        output = MagicMock()
        output.logits = mx.random.uniform(shape=(batch_size, seq_len, vocab_size))
        model.return_value = output

        input_ids = mx.random.randint(0, vocab_size, (batch_size, seq_len))

        log_probs, output_logits = extract_log_probs(model, input_ids)

        assert log_probs.shape == (batch_size, seq_len - 1)

    def test_extract_with_attention_mask(self):
        """Test extracting log probs with attention mask."""
        batch_size = 2
        seq_len = 4
        vocab_size = 10

        model = MagicMock()
        logits = mx.random.uniform(shape=(batch_size, seq_len, vocab_size))
        model.return_value = (logits,)

        input_ids = mx.random.randint(0, vocab_size, (batch_size, seq_len))
        # Mask out last token in each sequence
        attention_mask = mx.array([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=mx.float32)

        log_probs, _ = extract_log_probs(model, input_ids, attention_mask)

        assert log_probs.shape == (batch_size, seq_len - 1)
        # Masked positions should be zero
        assert log_probs[0, 2].item() == 0.0  # Position 3 was masked
        assert log_probs[1, 1].item() == 0.0  # Position 2 was masked


class TestComputeSequenceLogProb:
    """Tests for compute_sequence_log_prob function."""

    def test_basic_sequence_log_prob(self):
        """Test basic sequence log probability computation."""
        log_probs = mx.array([[-1.0, -2.0, -1.5], [-0.5, -1.0, -0.5]])

        seq_log_probs = compute_sequence_log_prob(log_probs)

        assert seq_log_probs.shape == (2,)
        # Sum should be sum of log probs
        assert seq_log_probs[0].item() == pytest.approx(-4.5, rel=1e-3)
        assert seq_log_probs[1].item() == pytest.approx(-2.0, rel=1e-3)

    def test_sequence_log_prob_with_mask(self):
        """Test sequence log probability with mask."""
        log_probs = mx.array([[-1.0, -2.0, -1.5]])
        attention_mask = mx.array([[1.0, 1.0, 0.0]])  # Only first two

        seq_log_prob = compute_sequence_log_prob(log_probs, attention_mask)

        # Should only sum first two: -1.0 + -2.0 = -3.0
        assert seq_log_prob[0].item() == pytest.approx(-3.0, rel=1e-3)

    def test_sequence_log_prob_batch(self):
        """Test sequence log probability with batch dimension."""
        batch_size = 4
        seq_len = 8
        log_probs = mx.random.uniform(shape=(batch_size, seq_len)) * -5

        seq_log_probs = compute_sequence_log_prob(log_probs)

        assert seq_log_probs.shape == (batch_size,)

    def test_sequence_log_prob_all_masked(self):
        """Test sequence log probability with all tokens masked."""
        log_probs = mx.array([[-1.0, -2.0, -1.5]])
        attention_mask = mx.zeros((1, 3))

        seq_log_prob = compute_sequence_log_prob(log_probs, attention_mask)

        # All masked, so sum should be 0
        assert seq_log_prob[0].item() == pytest.approx(0.0, abs=1e-6)
