import mlx.nn as nn
from models.model_config import ModelConfig
from models.architectures.model import Model
from models.architectures.transformer_base_model import TransformerBaseModel
from models.architectures.attention_base import AttentionBase
from models.architectures.gemma.rms import RMSNorm

class GemmaForCausalLM(Model):
    """
    Gemma model for causal language modeling tasks.
    This class wraps the core GemmaModel with a causal language modeling head.
    """
    def __init__(self, args: ModelConfig):
        # Initialize the base Model class
        super().__init__(args)

        # Create and set the core Gemma model
        self.model = GemmaModel(args)

class GemmaModel(TransformerBaseModel):
    """
    Core Gemma model implementation.
    This class extends the TransformerBaseModel with Gemma-specific
    attention, normalization, and embedding scaling.
    """
    def __init__(self, config: ModelConfig):
        # Initialize the TransformerBaseModel with Gemma-specific components
        super().__init__(
            config, 
            attention_layer=GemmaAttention,
            # Gemma uses a custom RMSNorm implementation
            norm_layer=lambda hidden_size, eps: RMSNorm(hidden_size, eps=eps)
        )

    def scale_embeddings(self, embeddings):
        """
        Apply Gemma-specific embedding scaling.
        Gemma scales the embeddings by the square root of the hidden size.
        """
        return embeddings * (self.args.hidden_size**0.5)
    
class GemmaAttention(AttentionBase):
    """
    Gemma-specific attention mechanism.
    Overrides the RoPE setup to use Gemma's specific configuration.
    """

    def _setup_rope(self, config):
        """
        Set up RoPE (Rotary Position Embedding) for Gemma model.
        Gemma uses a specific base (theta) value for RoPE.
        """
        return nn.RoPE(
            self.dimensions_per_head,
            traditional=config.rope_traditional,
            base=config.rope_theta,
        )