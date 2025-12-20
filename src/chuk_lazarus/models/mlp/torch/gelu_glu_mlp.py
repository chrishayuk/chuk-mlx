import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, bias=False):
        super().__init__()

        # Gated Linear Units (GLU) in PyTorch
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)

    def forward(self, x):
        # Apply the GELU activation function
        gate = torch.nn.functional.gelu(self.gate_proj(x))
        return self.down_proj(gate * self.up_proj(x))  # Gating mechanism
