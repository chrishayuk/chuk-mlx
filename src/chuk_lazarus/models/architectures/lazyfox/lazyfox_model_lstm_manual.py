import mlx.core as mx
import mlx.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, intermediate_size, bias=False):
        super().__init__()

        # Define layers similar to GLU
        self.gate_proj = nn.Linear(input_size + hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.up_proj = nn.Linear(input_size + hidden_size, intermediate_size, bias=bias)

    def forward(self, x, hidden_state):
        # Concatenate input and hidden state
        combined = mx.concatenate([x, hidden_state], axis=-1)

        # Apply GLU-like operations
        gate = nn.silu(self.gate_proj(combined))
        output = self.down_proj(gate * self.up_proj(combined))

        return output


class CustomModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Set the embedding layer
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        # Initialize the custom recurrent MLP
        self.mlp = MLP(config.hidden_size, config.hidden_size, config.intermediate_size)

        # Linear layer to project the final hidden state to the output space
        self.output_projection = nn.Linear(config.hidden_size, config.vocab_size)

        # Hidden size
        self.hidden_size = config.hidden_size

    def forward(self, x):
        # Embed the input sequence
        embedded_vectors = self.embedding(x)  # Shape: (batch_size, seq_length, hidden_size)

        # Initialize the hidden state
        batch_size = x.shape[0]
        hidden_state = mx.zeros((batch_size, self.hidden_size))  # Shape: (batch_size, hidden_size)

        # Process each time step in the sequence
        for t in range(embedded_vectors.shape[1]):
            # Update the hidden state using the MLP
            hidden_state = self.mlp.forward(
                embedded_vectors[:, t, :], hidden_state
            )  # Shape: (batch_size, hidden_size)

        # Project the final hidden state to the output space
        output = self.output_projection(hidden_state)  # Shape: (batch_size, vocab_size)

        return output

    def __call__(self, x) -> mx.array:
        return self.forward(x)
