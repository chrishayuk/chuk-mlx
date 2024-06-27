import mlx.nn as nn
from models.model_config import ModelConfig
from models.architectures.starcoder2.starcoder2_attention import StarCoder2Attention
from models.architectures.starcoder2.starcoder2_mlp import StarCoder2MLP

class StarCoder2TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        # Initialize the parent nn.Module class
        super().__init__()

        # Use rms_norm_eps if available, otherwise default to a small value
        norm_eps = getattr(config, 'rms_norm_eps', 1e-6)
        
        # Input layer normalization
        # This normalizes the input before it goes into the attention layer
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=norm_eps)

        # Multi-head attention layer
        # This is a custom attention mechanism specific to StarCoder2
        self.self_attn = StarCoder2Attention(config)

        # Second layer normalization (pre-MLP)
        # This normalizes the input before it goes into the MLP
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=norm_eps)

        # Multi-layer perceptron (MLP)
        # This is the feed-forward network part of the transformer block
        self.mlp = StarCoder2MLP(config)

    def __call__(self, x, attention_mask=None, cache=None):
        # Apply attention mechanism
        # 1. Normalize input
        # 2. Pass through attention layer
        # 3. Add residual connection
        attn_output, cache = self.attn(self.ln_1(x), attention_mask, cache)
        x = x + attn_output  # Residual connection

        # Apply MLP
        # 1. Normalize input
        # 2. Pass through MLP
        # 3. Add residual connection
        x = x + self.mlp(self.ln_2(x))  # Residual connection

        return x, cache