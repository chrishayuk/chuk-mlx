"""Tests for optimizer loader."""

from unittest.mock import MagicMock, patch

import pytest

from chuk_lazarus.utils.optimizer_loader import (
    linear_warmup_schedule,
    load_optimizer,
    piecewise_scheduler,
)


class TestLinearWarmupSchedule:
    """Tests for linear_warmup_schedule function."""

    def test_warmup_at_step_zero(self):
        """Test warmup at step 0."""
        schedule = linear_warmup_schedule(initial_lr=1.0, warmup_steps=10)
        # At step 0, lr = 1.0 * (0 + 1) / 10 = 0.1
        assert schedule(0) == 0.1

    def test_warmup_at_midpoint(self):
        """Test warmup at midpoint."""
        schedule = linear_warmup_schedule(initial_lr=1.0, warmup_steps=10)
        # At step 4, lr = 1.0 * (4 + 1) / 10 = 0.5
        assert schedule(4) == 0.5

    def test_warmup_at_last_step(self):
        """Test warmup at last warmup step."""
        schedule = linear_warmup_schedule(initial_lr=1.0, warmup_steps=10)
        # At step 9, lr = 1.0 * (9 + 1) / 10 = 1.0
        assert schedule(9) == 1.0

    def test_after_warmup(self):
        """Test after warmup period."""
        schedule = linear_warmup_schedule(initial_lr=1.0, warmup_steps=10)
        # After warmup, returns initial_lr
        assert schedule(10) == 1.0
        assert schedule(100) == 1.0

    def test_warmup_different_lr(self):
        """Test warmup with different initial lr."""
        schedule = linear_warmup_schedule(initial_lr=0.001, warmup_steps=100)
        # At step 49, lr = 0.001 * (49 + 1) / 100 = 0.0005
        assert schedule(49) == 0.0005


class TestPiecewiseScheduler:
    """Tests for piecewise_scheduler function."""

    def test_single_scheduler(self):
        """Test piecewise with single scheduler."""

        def sched1(x):
            return 1.0 - x * 0.1

        schedule = piecewise_scheduler([sched1], [])

        assert schedule(0) == 1.0
        assert schedule(5) == 0.5

    def test_two_schedulers(self):
        """Test piecewise with two schedulers."""

        def sched1(x):
            return 1.0  # constant 1.0

        def sched2(x):
            return 0.5  # constant 0.5

        schedule = piecewise_scheduler([sched1, sched2], [10])

        # Before milestone 10, use sched1
        assert schedule(0) == 1.0
        assert schedule(9) == 1.0
        # After milestone 10, use sched2
        assert schedule(10) == 0.5
        assert schedule(20) == 0.5


class TestLoadOptimizer:
    """Tests for load_optimizer function."""

    @patch("chuk_lazarus.utils.optimizer_loader.optim")
    def test_load_adamw_cosine_decay(self, mock_optim):
        """Test loading AdamW with cosine decay."""
        mock_optimizer = MagicMock()
        mock_optim.AdamW.return_value = mock_optimizer
        mock_optim.cosine_decay.return_value = lambda x: 1e-4

        config = {
            "name": "AdamW",
            "initial_lr": 1e-4,
            "lr_schedule": {
                "type": "cosine_decay",
                "minimum": 1e-6,
            },
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
        }

        result = load_optimizer(config, total_iterations=1000)

        mock_optim.AdamW.assert_called_once()
        assert result == mock_optimizer

    @patch("chuk_lazarus.utils.optimizer_loader.optim")
    def test_load_adamw_with_warmup(self, mock_optim):
        """Test loading AdamW with warmup."""
        mock_optimizer = MagicMock()
        mock_optim.AdamW.return_value = mock_optimizer
        mock_optim.cosine_decay.return_value = lambda x: 1e-4

        config = {
            "name": "AdamW",
            "initial_lr": 1e-4,
            "lr_schedule": {
                "type": "cosine_decay",
                "warmup_steps": 100,
                "minimum": 1e-6,
            },
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
        }

        result = load_optimizer(config, total_iterations=1000)

        mock_optim.AdamW.assert_called_once()
        assert result == mock_optimizer

    @patch("chuk_lazarus.utils.optimizer_loader.optim")
    def test_load_adamw_exponential_decay(self, mock_optim):
        """Test loading AdamW with exponential decay."""
        mock_optimizer = MagicMock()
        mock_optim.AdamW.return_value = mock_optimizer
        mock_optim.exponential_decay.return_value = lambda x: 1e-4

        config = {
            "name": "AdamW",
            "initial_lr": 1e-4,
            "lr_schedule": {
                "type": "exponential_decay",
                "decay_rate": 0.96,
            },
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
        }

        result = load_optimizer(config, total_iterations=1000)

        mock_optim.exponential_decay.assert_called_once()
        mock_optim.AdamW.assert_called_once()
        assert result == mock_optimizer

    @patch("chuk_lazarus.utils.optimizer_loader.optim")
    def test_unsupported_lr_schedule(self, mock_optim):
        """Test unsupported learning rate schedule."""
        config = {
            "name": "AdamW",
            "initial_lr": 1e-4,
            "lr_schedule": {
                "type": "unsupported_schedule",
            },
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
        }

        with pytest.raises(ValueError, match="Unsupported learning rate schedule"):
            load_optimizer(config, total_iterations=1000)

    @patch("chuk_lazarus.utils.optimizer_loader.optim")
    def test_unsupported_optimizer(self, mock_optim):
        """Test unsupported optimizer."""
        mock_optim.cosine_decay.return_value = lambda x: 1e-4

        config = {
            "name": "UnsupportedOptimizer",
            "initial_lr": 1e-4,
            "lr_schedule": {
                "type": "cosine_decay",
                "minimum": 1e-6,
            },
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
        }

        with pytest.raises(ValueError, match="Unsupported optimizer"):
            load_optimizer(config, total_iterations=1000)
