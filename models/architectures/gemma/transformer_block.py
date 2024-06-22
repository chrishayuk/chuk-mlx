import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple
from models.architectures.gemma.rms import RMSNorm
from models.mlp.mlp_factory import create_mlp
from models.model_config import ModelConfig
from models.architectures.gemma.gemma_attention import GemmaAttention

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        # initialize
        super().__init__()

        # set the dimensions
        self.hidden_size = config.hidden_size

        # create the attention layers
        self.self_attn = GemmaAttention(config)
        
        # create the MLP using our factory function
        self.mlp = create_mlp(config)
        
        # llama models normalize the inputs for each sub layer to stablize training
        # Llama-Paper
        # Pre-normalization [GPT3]. 
        # To improve the training stability, 
        # we normalize the input of each transformer sub-layer, 
        # instead of normalizing the outsput. 
        # We use the RMSNorm normalizing func- tion, introduced by Zhang and Sennrich (2019).
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Set the post attention layer normalisation, again, we use RMS
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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