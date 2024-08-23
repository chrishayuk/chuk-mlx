import mlx.core as mx
import mlx.nn as nn
from core.models.model_config import ModelConfig

# Define a simple RNN model using LSTM
class CustomModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        # Set the embedding layer
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        # LSTM layer
        self.lstm = nn.RNN(config.hidden_size, config.hidden_size)

        # Output layer
        self.output_projection = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, x):
        # Embed the input sequence
        embedded_vectors = self.embedding(x)  # Shape: (batch_size, seq_length, hidden_size)

        # Pass through LSTM
        lstm_out, _ = self.lstm(embedded_vectors)  # Shape: (batch_size, seq_length, hidden_size)

        # Project to output space
        output = self.output_projection(lstm_out)  # Shape: (batch_size, seq_length, vocab_size)

        return output

    def __call__(self, x) -> mx.array:
        return self.forward(x)
