import mlx.nn as nn
from models.model_config import ModelConfig

class StarCoder2MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        # First linear layer (expansion)
        # Named 'c_fc' to match checkpoint naming convention
        self.c_fc = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        
        # GELU activation function
        self.gelu = nn.GELU()
        
        # Second linear layer (projection back to hidden size)
        # Named 'c_proj' to match checkpoint naming convention
        self.c_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(getattr(config, 'residual_dropout', 0.1))

    def __call__(self, x):
        # Forward pass through the MLP
        x = self.c_fc(x)  # Expand dimensions
        x = self.gelu(x)  # Apply GELU activation
        x = self.c_proj(x)  # Project back to original dimensions
        x = self.dropout(x)  # Apply dropout
        return x