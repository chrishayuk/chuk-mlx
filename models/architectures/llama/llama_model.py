import mlx.nn as nn
from models.model_config import ModelConfig
from models.architectures.model import Model
from models.architectures.attention_base import AttentionBase
from models.architectures.transformer_base_model import TransformerBaseModel

class LlamaForCausalLM(Model):
    """
    Llama model for causal language modeling tasks.
    This class wraps the core LlamaModel with a causal language modeling head.
    """
    def __init__(self, args: ModelConfig):
        # Initialize the base Model class
        super().__init__(args)

        # Create and set the core Llama model
        self.model = LlamaModel(args)

    def sanitize(self, weights):
        # Simply return all weights without filtering
        return weights
    
    def set_inv_freq(self, inv_freq):
        for layer in self.model.layers:
            if hasattr(layer.self_attn, 'set_inv_freq'):
                layer.self_attn.set_inv_freq(inv_freq)

class LlamaModel(TransformerBaseModel):
    """
    Core Llama model implementation.
    This class uses the default TransformerBaseModel structure with Llama-specific
    attention and normalization layers.
    """
    def __init__(self, config: ModelConfig):
        # Initialize the TransformerBaseModel with Llama-specific components
        super().__init__(
            config, 
            attention_layer=LlamaAttention,
            # Llama uses RMSNorm for layer normalization
            norm_layer=lambda hidden_size, eps: nn.RMSNorm(hidden_size, eps=eps)
        )
        # Note: Llama does not apply additional embedding scaling,
        # so we don't override the scale_embeddings method here

class LlamaAttention(AttentionBase):
    """
    Llama-specific attention mechanism.
    Uses the default RoPE setup from AttentionBase, which is already
    configured for Llama-style rotary embeddings.
    """
    pass