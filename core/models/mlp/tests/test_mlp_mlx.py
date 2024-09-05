import pytest
import mlx.core as mx
from core.models.model_config import ModelConfig
from core.models.mlp.mlx.swiglu_mlp import MLP as MLXSwiGLUMLP
from core.models.mlp.mlx.gelu_glu_mlp import MLP as MLXGeluGLUMLP
from core.models.mlp.mlx.relu_mlp import MLP as MLXReLUMlp
from core.models.mlp.mlp_factory import create_mlp

@pytest.fixture
def mock_config():
    """Fixture to create a mock configuration for the MLPs."""
    return ModelConfig(
        hidden_size=512,
        intermediate_size=1024,
        mlp_bias=False,
        bos_token_id=0,  # Add required fields with default values
        eos_token_id=1,
        num_hidden_layers=12,
        max_position_embeddings=1024,
        vocab_size=30522,
    )


# Factory Tests for MLX MLPs
def test_create_mlx_swiglu_mlp(mock_config):
    mock_config.hidden_act = "silu"
    mlp = create_mlp(mock_config, framework="mlx")
    assert isinstance(mlp, MLXSwiGLUMLP), "Expected MLX SwiGLU MLP"

def test_create_mlx_gelu_mlp(mock_config):
    mock_config.hidden_act = "gelu"
    mlp = create_mlp(mock_config, framework="mlx")
    assert isinstance(mlp, MLXGeluGLUMLP), "Expected MLX GELU MLP"

def test_create_mlx_relu_mlp(mock_config):
    mock_config.hidden_act = "relu"
    mlp = create_mlp(mock_config, framework="mlx")
    assert isinstance(mlp, MLXReLUMlp), "Expected MLX ReLU MLP"

# Forward Pass Tests for MLX MLPs
def test_mlx_swiglu_mlp_forward(mock_config):
    mock_config.hidden_act = "silu"
    mlp = create_mlp(mock_config, framework="mlx")
    x = mx.random.uniform(low=0.0, high=1.0, shape=(2, 10, 512))
    output = mlp(x)
    assert output.shape == (2, 10, 512), "Incorrect output shape for MLX SwiGLU MLP"

def test_mlx_gelu_mlp_forward(mock_config):
    mock_config.hidden_act = "gelu"
    mlp = create_mlp(mock_config, framework="mlx")
    x = mx.random.uniform(low=0.0, high=1.0, shape=(2, 10, 512))
    output = mlp(x)
    assert output.shape == (2, 10, 512), "Incorrect output shape for MLX GELU MLP"

def test_mlx_relu_mlp_forward(mock_config):
    mock_config.hidden_act = "relu"
    mlp = create_mlp(mock_config, framework="mlx")
    x = mx.random.uniform(low=0.0, high=1.0, shape=(2, 10, 512))
    output = mlp(x)
    assert output.shape == (2, 10, 512), "Incorrect output shape for MLX ReLU MLP"
