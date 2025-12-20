class NormalizationLayerFactory:
    @staticmethod
    def create_norm_layer(hidden_size, eps):
        try:
            import mlx.nn as nn  # Try importing MLX
            return nn.RMSNorm(hidden_size, eps=eps)
        except ImportError:
            import torch.nn as nn  # Fallback to PyTorch if MLX is not available
            return nn.LayerNorm(hidden_size, eps)
