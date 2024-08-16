import mlx.core as mx
import mlx.nn as nn
from models.mlp.swiglu_mlp import MLP
from models.model_config import ModelConfig

class SimpleLanguageModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        # Embedding layer
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        # MLP layer
        self.mlp = MLP(config.hidden_size, config.intermediate_size)

        # Layer normalization (optional)
        self.layer_norm = nn.LayerNorm(config.hidden_size)

        # Output projection layer to vocab size
        self.output_layer = nn.Linear(config.hidden_size, config.vocab_size)

        # Dropout (optional)
        #self.dropout = nn.Dropout(p=config.dropout_rate)
    
    def forward(self, x):
        # Convert input tokens to embeddings
        embedded_vectors = self.embedding(x)

        # Apply layer normalization
        embedded_vectors = self.layer_norm(embedded_vectors)

        # Apply dropout (if any)
        #embedded_vectors = self.dropout(embedded_vectors)

        # Pass through MLP
        output = self.mlp(embedded_vectors)

        # Apply dropout (if any)
        #output = self.dropout(output)

        # Project to vocabulary size
        logits = self.output_layer(output)

        return logits
    
    def __call__(self, x) -> mx.array:
        # call is the same as forward
        return self.forward(x)
