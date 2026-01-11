"""Tests for training schedulers."""

from unittest.mock import MagicMock

from chuk_lazarus.training.schedulers import (
    SchedulerType,
    schedule_learning_rate,
)


class TestSchedulerType:
    """Tests for SchedulerType enum."""

    def test_warmup(self):
        """Test WARMUP scheduler type."""
        assert SchedulerType.WARMUP.value == "warmup"

    def test_linear_decay(self):
        """Test LINEAR_DECAY scheduler type."""
        assert SchedulerType.LINEAR_DECAY.value == "linear_decay"

    def test_exponential_decay(self):
        """Test EXPONENTIAL_DECAY scheduler type."""
        assert SchedulerType.EXPONENTIAL_DECAY.value == "exponential_decay"

    def test_cosine_annealing(self):
        """Test COSINE_ANNEALING scheduler type."""
        assert SchedulerType.COSINE_ANNEALING.value == "cosine_annealing"

    def test_cosine_decay_with_warmup(self):
        """Test COSINE_DECAY_WITH_WARMUP scheduler type."""
        assert SchedulerType.COSINE_DECAY_WITH_WARMUP.value == "cosine_decay_with_warmup"


class TestScheduleLearningRate:
    """Tests for schedule_learning_rate function."""

    def test_warmup_at_start(self):
        """Test warmup at iteration 0."""
        optimizer = MagicMock()
        optimizer.learning_rate = 1e-4

        lr = schedule_learning_rate(
            optimizer,
            iteration_count=0,
            warmup_steps=100,
            scheduler_type=SchedulerType.WARMUP,
        )

        assert lr == 1e-4 * (1 / 100)
        assert optimizer.learning_rate == lr

    def test_warmup_at_midpoint(self):
        """Test warmup at midpoint."""
        optimizer = MagicMock()
        optimizer.learning_rate = 1e-4
        optimizer.initial_lr = 1e-4

        lr = schedule_learning_rate(
            optimizer,
            iteration_count=50,
            warmup_steps=100,
            scheduler_type=SchedulerType.WARMUP,
        )

        assert lr == 1e-4 * (51 / 100)

    def test_warmup_after_warmup(self):
        """Test warmup after warmup period."""
        optimizer = MagicMock()
        optimizer.learning_rate = 1e-4
        optimizer.initial_lr = 1e-4

        lr = schedule_learning_rate(
            optimizer,
            iteration_count=100,
            warmup_steps=100,
            scheduler_type=SchedulerType.WARMUP,
        )

        assert lr == 1e-4

    def test_linear_decay(self):
        """Test linear decay schedule."""
        optimizer = MagicMock()
        optimizer.learning_rate = 1e-4
        optimizer.initial_lr = 1e-4

        lr = schedule_learning_rate(
            optimizer,
            iteration_count=5000,
            warmup_steps=100,
            scheduler_type=SchedulerType.LINEAR_DECAY,
            total_steps=10000,
            min_lr=1e-6,
        )

        # At midpoint, lr should be halfway between initial and min
        expected = 1e-4 * (1 - 5000 / 10000)
        assert abs(lr - expected) < 1e-10

    def test_linear_decay_respects_min_lr(self):
        """Test linear decay respects minimum lr."""
        optimizer = MagicMock()
        optimizer.learning_rate = 1e-4
        optimizer.initial_lr = 1e-4

        lr = schedule_learning_rate(
            optimizer,
            iteration_count=20000,  # Beyond total steps
            warmup_steps=100,
            scheduler_type=SchedulerType.LINEAR_DECAY,
            total_steps=10000,
            min_lr=1e-6,
        )

        assert lr == 1e-6

    def test_exponential_decay(self):
        """Test exponential decay schedule."""
        optimizer = MagicMock()
        optimizer.learning_rate = 1e-4
        optimizer.initial_lr = 1e-4

        lr = schedule_learning_rate(
            optimizer,
            iteration_count=1000,
            warmup_steps=100,
            scheduler_type=SchedulerType.EXPONENTIAL_DECAY,
            decay_rate=0.96,
            decay_steps=1000,
        )

        expected = 1e-4 * (0.96**1)
        assert abs(lr - expected) < 1e-10

    def test_cosine_annealing(self):
        """Test cosine annealing schedule."""
        optimizer = MagicMock()
        optimizer.learning_rate = 1e-4
        optimizer.initial_lr = 1e-4

        lr = schedule_learning_rate(
            optimizer,
            iteration_count=0,
            warmup_steps=0,
            scheduler_type=SchedulerType.COSINE_ANNEALING,
            total_steps=10000,
            min_lr=0,
        )

        # At start, lr should be initial_lr
        assert lr == 1e-4

    def test_cosine_annealing_at_end(self):
        """Test cosine annealing at end of training."""
        optimizer = MagicMock()
        optimizer.learning_rate = 1e-4
        optimizer.initial_lr = 1e-4

        lr = schedule_learning_rate(
            optimizer,
            iteration_count=10000,
            warmup_steps=0,
            scheduler_type=SchedulerType.COSINE_ANNEALING,
            total_steps=10000,
            min_lr=0,
        )

        # At end, lr should be close to min_lr
        assert lr < 1e-8

    def test_cosine_decay_with_warmup(self):
        """Test cosine decay with warmup."""
        optimizer = MagicMock()
        optimizer.learning_rate = 1e-4
        optimizer.initial_lr = 1e-4

        # During warmup
        lr = schedule_learning_rate(
            optimizer,
            iteration_count=50,
            warmup_steps=100,
            scheduler_type=SchedulerType.COSINE_DECAY_WITH_WARMUP,
            total_steps=10000,
            min_lr=0,
        )

        expected_warmup = 1e-4 * (51 / 100)
        assert abs(lr - expected_warmup) < 1e-10

    def test_cosine_decay_with_warmup_after_warmup(self):
        """Test cosine decay after warmup period."""
        optimizer = MagicMock()
        optimizer.learning_rate = 1e-4
        optimizer.initial_lr = 1e-4

        lr = schedule_learning_rate(
            optimizer,
            iteration_count=100,
            warmup_steps=100,
            scheduler_type=SchedulerType.COSINE_DECAY_WITH_WARMUP,
            total_steps=10000,
            min_lr=0,
        )

        # Just after warmup, should be close to initial_lr
        assert lr > 0.99 * 1e-4

    def test_unsupported_scheduler_type(self):
        """Test unsupported scheduler type defaults to warmup."""
        optimizer = MagicMock()
        optimizer.learning_rate = 1e-4
        optimizer.initial_lr = 1e-4

        lr = schedule_learning_rate(
            optimizer,
            iteration_count=50,
            warmup_steps=100,
            scheduler_type="unsupported_type",
        )

        # Should default to warmup behavior
        expected_warmup = 1e-4 * (51 / 100)
        assert abs(lr - expected_warmup) < 1e-10

    def test_string_scheduler_type(self):
        """Test scheduler type as string."""
        optimizer = MagicMock()
        optimizer.learning_rate = 1e-4
        optimizer.initial_lr = 1e-4

        lr = schedule_learning_rate(
            optimizer,
            iteration_count=100,
            warmup_steps=100,
            scheduler_type="warmup",
        )

        assert lr == 1e-4
