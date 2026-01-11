"""Tests for learning rate schedulers."""

from unittest.mock import MagicMock

import pytest

from chuk_lazarus.training.schedulers import SchedulerType, schedule_learning_rate


class TestSchedulerType:
    """Tests for SchedulerType enum."""

    def test_scheduler_type_values(self):
        """Test all scheduler types have string values."""
        assert SchedulerType.WARMUP.value == "warmup"
        assert SchedulerType.LINEAR_DECAY.value == "linear_decay"
        assert SchedulerType.EXPONENTIAL_DECAY.value == "exponential_decay"
        assert SchedulerType.COSINE_ANNEALING.value == "cosine_annealing"
        assert SchedulerType.COSINE_DECAY_WITH_WARMUP.value == "cosine_decay_with_warmup"


class TestScheduleLearningRate:
    """Tests for schedule_learning_rate function."""

    def _create_mock_optimizer(self, initial_lr: float = 0.001) -> MagicMock:
        """Create a mock optimizer."""
        optimizer = MagicMock()
        optimizer.learning_rate = initial_lr
        return optimizer

    def test_warmup_initial_iteration(self):
        """Test warmup scheduler at iteration 0."""
        optimizer = self._create_mock_optimizer(0.001)

        lr = schedule_learning_rate(
            optimizer,
            iteration_count=0,
            warmup_steps=100,
            scheduler_type=SchedulerType.WARMUP,
        )

        # At iteration 0, warmup factor is 1/100
        assert lr == pytest.approx(0.00001, rel=1e-3)
        assert optimizer.initial_lr == 0.001

    def test_warmup_mid_warmup(self):
        """Test warmup scheduler at mid warmup."""
        optimizer = self._create_mock_optimizer(0.001)
        optimizer.initial_lr = 0.001  # Set initial_lr as it would be after iteration 0

        lr = schedule_learning_rate(
            optimizer,
            iteration_count=50,
            warmup_steps=100,
            scheduler_type=SchedulerType.WARMUP,
        )

        # At iteration 50, warmup factor is 51/100
        assert lr == pytest.approx(0.00051, rel=1e-3)

    def test_warmup_after_warmup(self):
        """Test warmup scheduler after warmup phase."""
        optimizer = self._create_mock_optimizer(0.001)
        optimizer.initial_lr = 0.001

        lr = schedule_learning_rate(
            optimizer,
            iteration_count=150,
            warmup_steps=100,
            scheduler_type=SchedulerType.WARMUP,
        )

        # After warmup, lr should be initial_lr
        assert lr == pytest.approx(0.001, rel=1e-3)

    def test_linear_decay(self):
        """Test linear decay scheduler."""
        optimizer = self._create_mock_optimizer(0.001)
        optimizer.initial_lr = 0.001

        lr = schedule_learning_rate(
            optimizer,
            iteration_count=500,
            warmup_steps=0,
            scheduler_type=SchedulerType.LINEAR_DECAY,
            total_steps=1000,
            min_lr=0.0001,
        )

        # At iteration 500/1000, should be 50% decayed
        # lr = initial * (1 - iteration/total) = 0.001 * (1 - 0.5) = 0.0005
        assert lr == pytest.approx(0.0005, rel=1e-3)

    def test_linear_decay_respects_min_lr(self):
        """Test linear decay doesn't go below min_lr."""
        optimizer = self._create_mock_optimizer(0.001)
        optimizer.initial_lr = 0.001

        lr = schedule_learning_rate(
            optimizer,
            iteration_count=2000,  # Beyond total_steps
            warmup_steps=0,
            scheduler_type=SchedulerType.LINEAR_DECAY,
            total_steps=1000,
            min_lr=0.0001,
        )

        assert lr >= 0.0001

    def test_exponential_decay(self):
        """Test exponential decay scheduler."""
        optimizer = self._create_mock_optimizer(0.001)
        optimizer.initial_lr = 0.001

        lr = schedule_learning_rate(
            optimizer,
            iteration_count=1000,
            warmup_steps=0,
            scheduler_type=SchedulerType.EXPONENTIAL_DECAY,
            decay_rate=0.96,
            decay_steps=1000,
        )

        # lr = initial * (decay_rate ^ (iteration/decay_steps))
        # lr = 0.001 * (0.96 ^ 1) = 0.00096
        assert lr == pytest.approx(0.00096, rel=1e-3)

    def test_cosine_annealing(self):
        """Test cosine annealing scheduler."""
        optimizer = self._create_mock_optimizer(0.001)
        optimizer.initial_lr = 0.001

        # At half way, cosine should be at mean
        lr = schedule_learning_rate(
            optimizer,
            iteration_count=500,
            warmup_steps=0,
            scheduler_type=SchedulerType.COSINE_ANNEALING,
            total_steps=1000,
            min_lr=0.0001,
        )

        # At 50%, cos(pi * 0.5) = 0, so lr = min + (max-min) * (1 + 0) / 2
        expected = 0.0001 + (0.001 - 0.0001) * 0.5
        assert lr == pytest.approx(expected, rel=1e-2)

    def test_cosine_decay_with_warmup_during_warmup(self):
        """Test cosine decay with warmup during warmup phase."""
        optimizer = self._create_mock_optimizer(0.001)

        lr = schedule_learning_rate(
            optimizer,
            iteration_count=0,
            warmup_steps=100,
            scheduler_type=SchedulerType.COSINE_DECAY_WITH_WARMUP,
            total_steps=1000,
            min_lr=0.0001,
        )

        # During warmup, same as regular warmup
        assert lr == pytest.approx(0.00001, rel=1e-3)

    def test_cosine_decay_with_warmup_after_warmup(self):
        """Test cosine decay with warmup after warmup phase."""
        optimizer = self._create_mock_optimizer(0.001)
        optimizer.initial_lr = 0.001

        lr = schedule_learning_rate(
            optimizer,
            iteration_count=100,  # Just at end of warmup
            warmup_steps=100,
            scheduler_type=SchedulerType.COSINE_DECAY_WITH_WARMUP,
            total_steps=1000,
            min_lr=0.0001,
        )

        # At start of cosine phase, should be at initial_lr
        # cos(0) = 1, so lr = min + (max-min) * (1 + 1) / 2 = max
        assert lr == pytest.approx(0.001, rel=1e-2)

    def test_unsupported_scheduler_falls_back_to_warmup(self):
        """Test that unsupported scheduler type falls back to warmup."""
        optimizer = self._create_mock_optimizer(0.001)

        lr = schedule_learning_rate(
            optimizer,
            iteration_count=0,
            warmup_steps=100,
            scheduler_type="unknown_scheduler",
        )

        # Should fall back to warmup behavior
        assert lr == pytest.approx(0.00001, rel=1e-3)

    def test_scheduler_type_as_string(self):
        """Test scheduler type can be passed as string."""
        optimizer = self._create_mock_optimizer(0.001)
        optimizer.initial_lr = 0.001

        lr = schedule_learning_rate(
            optimizer, iteration_count=150, warmup_steps=100, scheduler_type="warmup"
        )

        assert lr == pytest.approx(0.001, rel=1e-3)

    def test_learning_rate_is_set_on_optimizer(self):
        """Test that learning rate is set on optimizer."""
        optimizer = self._create_mock_optimizer(0.001)
        optimizer.initial_lr = 0.001

        schedule_learning_rate(
            optimizer,
            iteration_count=150,
            warmup_steps=100,
            scheduler_type=SchedulerType.WARMUP,
        )

        assert optimizer.learning_rate == pytest.approx(0.001, rel=1e-3)
