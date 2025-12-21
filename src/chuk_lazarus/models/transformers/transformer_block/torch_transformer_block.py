import torch
import torch.nn as nn
from typing import Callable, Optional, Tuple, Type
from chuk_lazarus.models.config import ModelConfig
from chuk_lazarus.models.transformers.transformer_block.base_transformer_block import BaseTransformerBlock
from chuk_lazarus.models.mlp.mlp_factory import create_mlp

class TorchTransformerBlock(BaseTransformerBlock, nn.Module):
    def __init__(self, config: ModelConfig, attention_layer: Type[nn.Module], norm_layer: Callable[[int, float], nn.Module]):
        nn.Module.__init__(self)  # Initialize PyTorch nn.Module
        BaseTransformerBlock.__init__(self, config, attention_layer, norm_layer)

    def create_mlp(self, config):
        # PyTorch-specific MLP creation
        return create_mlp(config)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        # Self-attention
        normed_hidden_states = self.input_layernorm(hidden_states)
        attention_output, cache = self.self_attn(normed_hidden_states, attention_mask, cache)
        
        # Residual connection for attention output
        hidden_states = hidden_states + attention_output
        
        # Apply the mask again after the residual connection
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(-1).expand_as(hidden_states)
            hidden_states = hidden_states * attention_mask
        
        # Post-attention normalization and MLP
        normed_hidden_states = self.post_attention_layernorm(hidden_states)
        
        # MLP forward pass
        mlp_output = self.mlp(normed_hidden_states)
        
        # Residual connection for MLP output
        hidden_states = hidden_states + mlp_output
        
        # Apply the mask again after the MLP residual connection
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask

        # Return the outputs
        return hidden_states, cache

    
    def get_activation_function(activation_name: str):
        if activation_name == "relu":
            return torch.nn.ReLU()
        elif activation_name == "gelu":
            return torch.nn.GELU()
        elif activation_name == "tanh":
            return torch.nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {activation_name}")
