import mlx.core as mx
import mlx.nn as nn
from models.mlx.mlp import MLP

# Define a simple language model with embedding and MLP
class SimpleLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, intermediate_size):
        super().__init__()

        # set the embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # set the mlp
        self.mlp = MLP(hidden_size, intermediate_size)
    
    def __call__(self, x) -> mx.array:
        # call is the same as forward
        return self.forward(x)

    def forward(self, x):
        # do a forward pass
        embedded_vectors = self.embedding(x)

        # Pass embedded vectors through the GLU
        output = self.mlp(embedded_vectors)

        # return the output
        return output