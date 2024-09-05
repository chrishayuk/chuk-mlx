from core.models.model_config import ModelConfig

# MLX MLP imports
from core.models.mlp.mlx.swiglu_mlp import MLP as MLXSwiGluMLP
from core.models.mlp.mlx.gelu_glu_mlp import MLP as MLXGeluGluMLP
from core.models.mlp.mlx.relu_mlp import MLP as MLXReLUMlp  # MLX ReLU MLP

# Torch MLP imports
from core.models.mlp.torch.swiglu_mlp import MLP as TorchSwiGLUMLP
from core.models.mlp.torch.gelu_glu_mlp import MLP as TorchGeluGLUMLP
from core.models.mlp.torch.relu_mlp import MLP as TorchReLUMlp  # Torch ReLU MLP

def create_mlp(config: ModelConfig, framework: str = "torch"):
    """
    Factory function to create the appropriate MLP based on the model configuration.
    Supports MLX and PyTorch frameworks and activation functions SwiGLU, GELU, and ReLU.
    """
    # Default to GELU if hidden_act is None
    hidden_act = config.hidden_act or config.hidden_activation or "gelu"

    # Check framework and return the appropriate MLP
    if framework == "torch":
        # PyTorch MLPs
        if hidden_act == "silu":
            return TorchSwiGLUMLP(config.hidden_size, config.intermediate_size, config.mlp_bias)
        elif hidden_act in ["gelu", "gelu_pytorch_tanh"]:
            return TorchGeluGLUMLP(config.hidden_size, config.intermediate_size, config.mlp_bias)
        elif hidden_act == "relu":
            return TorchReLUMlp(config.hidden_size, config.intermediate_size, config.mlp_bias)
        else:
            raise ValueError(f"Unsupported activation function for PyTorch: {hidden_act}")

    elif framework == "mlx":
        # MLX MLPs
        if hidden_act == "silu":
            return MLXSwiGluMLP(config.hidden_size, config.intermediate_size, config.mlp_bias)
        elif hidden_act in ["gelu", "gelu_pytorch_tanh"]:
            return MLXGeluGluMLP(config.hidden_size, config.intermediate_size, config.mlp_bias)
        elif hidden_act == "relu":
            return MLXReLUMlp(config.hidden_size, config.intermediate_size, config.mlp_bias)
        else:
            raise ValueError(f"Unsupported activation function for MLX: {hidden_act}")

    else:
        raise ValueError(f"Unsupported framework: {framework}")
