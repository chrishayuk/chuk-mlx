import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, bias=False):
        # call parent constructor
        super().__init__()

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=bias)

        # ReLU activation
        self.relu = nn.ReLU()

    def forward(self, x):
        # Apply ReLU after first layer
        x = self.relu(self.fc1(x))

        # Return after second layer
        return self.fc2(x)  
