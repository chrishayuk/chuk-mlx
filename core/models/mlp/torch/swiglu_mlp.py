import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, bias=False):
        # call parent constructor
        super().__init__()

        # Gated Logic Units (GLU)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)

    def forward(self, x):
        # Applying the swish activation function
        gate = torch.nn.functional.silu(self.gate_proj(x))
        return self.down_proj(gate * self.up_proj(x))
