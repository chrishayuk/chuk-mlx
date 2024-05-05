import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(MLP, self).__init__()
        
        # 3 gates
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)

    def forward(self, x):
        # Applying the SiLU (Swish) activation function
        gate = F.silu(self.gate_proj(x))
        return self.down_proj(gate * self.up_proj(x))
