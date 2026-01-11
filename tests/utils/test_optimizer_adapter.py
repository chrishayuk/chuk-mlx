"""Tests for optimizer adapter."""

from unittest.mock import MagicMock, patch

import pytest

from chuk_lazarus.utils.optimizer_adapter import OptimizerAdapter


class TestOptimizerAdapter:
    """Tests for OptimizerAdapter class."""

    def test_init_mlx_framework(self):
        """Test initialization with MLX framework."""
        adapter = OptimizerAdapter(framework="mlx")
        assert adapter.framework == "mlx"
        assert adapter.optimizer is None

    def test_init_torch_without_torch(self):
        """Test initialization with torch when not available."""
        with patch("chuk_lazarus.utils.optimizer_adapter.HAS_TORCH", False):
            with pytest.raises(ImportError):
                OptimizerAdapter(framework="torch")

    @patch("chuk_lazarus.utils.optimizer_adapter.mlx_optim")
    def test_create_optimizer_mlx_adam(self, mock_mlx_optim):
        """Test creating MLX Adam optimizer."""
        mock_optimizer = MagicMock()
        mock_mlx_optim.Adam.return_value = mock_optimizer

        adapter = OptimizerAdapter(framework="mlx")
        params = MagicMock()
        result = adapter.create_optimizer(params, optimizer_name="Adam", learning_rate=1e-4)

        mock_mlx_optim.Adam.assert_called_once_with(params, learning_rate=1e-4)
        assert result == mock_optimizer
        assert adapter.optimizer == mock_optimizer

    @patch("chuk_lazarus.utils.optimizer_adapter.mlx_optim")
    def test_create_optimizer_mlx_sgd(self, mock_mlx_optim):
        """Test creating MLX SGD optimizer."""
        mock_optimizer = MagicMock()
        mock_mlx_optim.SGD.return_value = mock_optimizer

        adapter = OptimizerAdapter(framework="mlx")
        params = MagicMock()
        result = adapter.create_optimizer(params, optimizer_name="SGD", learning_rate=0.01)

        mock_mlx_optim.SGD.assert_called_once_with(params, learning_rate=0.01)
        assert result == mock_optimizer

    def test_step_no_optimizer(self):
        """Test step when no optimizer is set."""
        adapter = OptimizerAdapter(framework="mlx")
        # Should not raise
        adapter.step()

    @patch("chuk_lazarus.utils.optimizer_adapter.mlx_optim")
    def test_step_with_optimizer(self, mock_mlx_optim):
        """Test step with optimizer."""
        mock_optimizer = MagicMock()
        mock_mlx_optim.Adam.return_value = mock_optimizer

        adapter = OptimizerAdapter(framework="mlx")
        adapter.create_optimizer(MagicMock(), optimizer_name="Adam")
        adapter.step()

        mock_optimizer.step.assert_called_once()

    def test_zero_grad_no_optimizer(self):
        """Test zero_grad when no optimizer is set."""
        adapter = OptimizerAdapter(framework="mlx")
        # Should not raise
        adapter.zero_grad()

    @patch("chuk_lazarus.utils.optimizer_adapter.mlx_optim")
    def test_zero_grad_with_optimizer(self, mock_mlx_optim):
        """Test zero_grad with optimizer."""
        mock_optimizer = MagicMock()
        mock_mlx_optim.Adam.return_value = mock_optimizer

        adapter = OptimizerAdapter(framework="mlx")
        adapter.create_optimizer(MagicMock(), optimizer_name="Adam")
        adapter.zero_grad()

        mock_optimizer.zero_grad.assert_called_once()
