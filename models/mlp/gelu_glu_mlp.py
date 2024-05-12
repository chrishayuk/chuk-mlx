import mlx.core as mx
import mlx.nn as nn

class MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        # initialize
        super().__init__()

        # Gated Logic Units (GLU)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)

    def __call__(self, x) -> mx.array:
        # call is the same as forward
        return self.forward(x)
    
    def forward(self, x):
        # Applying the gelu activation function
        gate = nn.gelu(self.gate_proj(x))
        return self.down_proj(gate * self.up_proj(x))