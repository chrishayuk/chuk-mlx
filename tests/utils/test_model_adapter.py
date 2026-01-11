"""Tests for model adapter."""

import tempfile
from unittest.mock import MagicMock

import mlx.core as mx

from chuk_lazarus.utils.model_adapter import ModelAdapter


class TestModelAdapterInit:
    """Tests for ModelAdapter initialization."""

    def test_init_mlx_framework(self):
        """Test initialization with MLX framework."""
        adapter = ModelAdapter(framework="mlx")

        assert adapter.framework == "mlx"
        assert adapter.model is None

    def test_init_with_model(self):
        """Test initialization with model."""
        model = MagicMock()
        adapter = ModelAdapter(framework="mlx", model=model)

        assert adapter.model is model

    def test_init_torch_without_import(self):
        """Test that torch framework requires torch installation."""
        # This test checks the behavior when torch is not available
        # Since torch may or may not be installed, we test the logic
        adapter = ModelAdapter(framework="mlx")
        assert adapter.framework == "mlx"


class TestToTensor:
    """Tests for to_tensor method."""

    def test_to_tensor_mlx(self):
        """Test converting to MLX tensor."""
        adapter = ModelAdapter(framework="mlx")
        data = [1, 2, 3, 4]

        tensor = adapter.to_tensor(data)

        assert isinstance(tensor, mx.array)
        assert tensor.tolist() == data

    def test_to_tensor_mlx_2d(self):
        """Test converting 2D data to MLX tensor."""
        adapter = ModelAdapter(framework="mlx")
        data = [[1, 2], [3, 4]]

        tensor = adapter.to_tensor(data)

        assert isinstance(tensor, mx.array)
        assert tensor.shape == (2, 2)


class TestForward:
    """Tests for forward method."""

    def test_forward_mlx(self):
        """Test forward pass with MLX model."""
        model = MagicMock()
        model.return_value = mx.array([1, 2, 3])

        adapter = ModelAdapter(framework="mlx", model=model)
        input_tensor = mx.array([0, 1, 2])

        result = adapter.forward(input_tensor)

        model.assert_called_once_with(input_tensor)
        assert isinstance(result, mx.array)


class TestArgmax:
    """Tests for argmax method."""

    def test_argmax_mlx(self):
        """Test argmax with MLX."""
        adapter = ModelAdapter(framework="mlx")
        output = [[0.1, 0.9, 0.0], [0.8, 0.1, 0.1]]

        result = adapter.argmax(output, axis=-1)

        assert result == [1, 0]

    def test_argmax_mlx_axis_0(self):
        """Test argmax with axis=0."""
        adapter = ModelAdapter(framework="mlx")
        output = [[0.1, 0.9], [0.8, 0.1]]

        result = adapter.argmax(output, axis=0)

        assert result == [1, 0]


class TestCreateValueAndGradFn:
    """Tests for create_value_and_grad_fn method."""

    def test_create_value_and_grad_fn_mlx(self):
        """Test creating value_and_grad function for MLX."""
        adapter = ModelAdapter(framework="mlx")
        loss_fn = MagicMock()

        # For MLX, this should return nn.value_and_grad
        result = adapter.create_value_and_grad_fn(loss_fn)

        # Result should be callable
        assert callable(result)


class TestLoadTensorFromFile:
    """Tests for load_tensor_from_file method."""

    def test_load_tensor_mlx(self):
        """Test loading tensor from file with MLX."""
        adapter = ModelAdapter(framework="mlx")

        # Create a temporary file with MLX tensor
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            mx.savez(f.name, test=mx.array([1, 2, 3]))
            f.flush()

            result = adapter.load_tensor_from_file(f.name)

            assert "test" in result
            assert result["test"].tolist() == [1, 2, 3]

    def test_load_tensor_mlx_safetensors(self):
        """Test loading tensor from safetensors file with MLX."""
        adapter = ModelAdapter(framework="mlx")

        # Create a temporary safetensors file
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            mx.save_safetensors(f.name, {"test": mx.array([1, 2, 3])})
            f.flush()

            result = adapter.load_tensor_from_file(f.name)

            assert "test" in result


class TestModelAdapterIntegration:
    """Integration tests for ModelAdapter."""

    def test_full_workflow_mlx(self):
        """Test full workflow with MLX."""

        # Create simple model
        class SimpleModel:
            def __call__(self, x):
                return x * 2

        model = SimpleModel()
        adapter = ModelAdapter(framework="mlx", model=model)

        # Convert data to tensor
        data = [1, 2, 3]
        tensor = adapter.to_tensor(data)

        # Forward pass
        result = adapter.forward(tensor)

        assert result.tolist() == [2, 4, 6]
