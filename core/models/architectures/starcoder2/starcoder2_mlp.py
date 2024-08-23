import mlx.nn as nn
from core.models.model_config import ModelConfig

class StarCoder2MLP(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.c_fc = nn.Linear(args.hidden_size, args.intermediate_size, bias=True)
        self.c_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=True)

    def __call__(self, x):
        return self.c_proj(nn.gelu(self.c_fc(x)))