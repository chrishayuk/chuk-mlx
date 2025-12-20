import pytest
import torch
from chuk_lazarus.models.model_config import ModelConfig
from chuk_lazarus.models.mlp.torch.swiglu_mlp import MLP as TorchSwiGLUMLP
from chuk_lazarus.models.mlp.torch.gelu_glu_mlp import MLP as TorchGeluGLUMLP
from chuk_lazarus.models.mlp.torch.relu_mlp import MLP as TorchReLUMlp
from chuk_lazarus.models.mlp.mlp_factory import create_mlp

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


# Factory Tests for PyTorch MLPs
def test_create_torch_swiglu_mlp(mock_config):
    mock_config.hidden_act = "silu"
    mlp = create_mlp(mock_config, framework="torch")
    assert isinstance(mlp, TorchSwiGLUMLP), "Expected PyTorch SwiGLU MLP"

def test_create_torch_gelu_mlp(mock_config):
    mock_config.hidden_act = "gelu"
    mlp = create_mlp(mock_config, framework="torch")
    assert isinstance(mlp, TorchGeluGLUMLP), "Expected PyTorch GELU MLP"

def test_create_torch_relu_mlp(mock_config):
    mock_config.hidden_act = "relu"
    mlp = create_mlp(mock_config, framework="torch")
    assert isinstance(mlp, TorchReLUMlp), "Expected PyTorch ReLU MLP"

# Forward Pass Tests for PyTorch MLPs
def test_torch_swiglu_mlp_forward(mock_config):
    mock_config.hidden_act = "silu"
    mlp = create_mlp(mock_config, framework="torch")
    x = torch.rand(2, 10, 512)  # Input tensor
    output = mlp(x)
    assert output.shape == (2, 10, 512), "Incorrect output shape for PyTorch SwiGLU MLP"

def test_torch_gelu_mlp_forward(mock_config):
    mock_config.hidden_act = "gelu"
    mlp = create_mlp(mock_config, framework="torch")
    x = torch.rand(2, 10, 512)  # Input tensor
    output = mlp(x)
    assert output.shape == (2, 10, 512), "Incorrect output shape for PyTorch GELU MLP"

def test_torch_relu_mlp_forward(mock_config):
    mock_config.hidden_act = "relu"
    mlp = create_mlp(mock_config, framework="torch")
    x = torch.rand(2, 10, 512)  # Input tensor
    output = mlp(x)
    assert output.shape == (2, 10, 512), "Incorrect output shape for PyTorch ReLU MLP"
