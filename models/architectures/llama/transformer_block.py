import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple, Union
from models.mlp.mlp_factory import create_mlp
from models.model_config import ModelConfig
from models.architectures.llama.llama_attention import LlamaAttention


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        # initialize
        super().__init__()

        # set the dimensions
        self.hidden_size = config.hidden_size

        # create the attention layers
        self.self_attn = LlamaAttention(config)
        
        # create the MLP using our factory function
        self.mlp = create_mlp(config)
        
        # llama models normalize the inputs for each sub layer to stablize training
        # Llama-Paper
        # Pre-normalization [GPT3]. 
        # To improve the training stability, 
        # we normalize the input of each transformer sub-layer, 
        # instead of normalizing the outsput. 
        # We use the RMSNorm normalizing func- tion, introduced by Zhang and Sennrich (2019).
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Set the post attention layer normalisation, again, we use RMS
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    # def __call__(   
    #     self,
    #     hidden_states: mx.array,
    #     attention_mask: Optional[mx.array] = None,
    #     cache: Optional[Tuple[mx.array, mx.array]] = None,
    # ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
    #     """
    #     Forward pass for the transformer block.
        
    #     Args:
    #         hidden_states: Input tensor
    #         attention_mask: Optional attention mask
    #         cache: Optional key/value cache for attention

    #     Returns:
    #         Tuple of output tensor and updated cache
    #     """
    #     # Self-attention
    #     normed_hidden_states = self.input_layernorm(hidden_states)
    #     attention_output, cache = self.self_attn(normed_hidden_states, attention_mask, cache)
    #     hidden_states = hidden_states + attention_output

    #     # MLP
    #     normed_hidden_states = self.post_attention_layernorm(hidden_states)
    #     mlp_output = self.mlp(normed_hidden_states)
    #     hidden_states = hidden_states + mlp_output

    #     return hidden_states, cache
    
    def __call__(   
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        #print('llama layer call')
        r, cache = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r

        # execute our mlp, performing normalisation on the input
        r = self.mlp(self.post_attention_layernorm(h))

        # ummm
        out = h + r

        # ummm
        return out, cache