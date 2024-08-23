import mlx.core as mx
import mlx.nn as nn
from core.models.mlp.swiglu_mlp import MLP
from core.models.model_config import ModelConfig

# Define a simple language model with embedding and MLP
class CustomModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        # set the embedding layer
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        # set the mlp
        self.mlp = MLP(config.hidden_size, config.intermediate_size)
    
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