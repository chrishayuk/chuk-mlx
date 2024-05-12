from typing import Optional, Tuple
import mlx.core as mx
import mlx.nn as nn
from models.model_config import ModelConfig
from models.llama.llama_attention import LlamaAttention
from models.mlp.swiglu_mlp import MLP

class LlamaLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        # initialize
        super().__init__()

        # set the dimensions
        self.hidden_size = config.hidden_size

        # create the attention layers
        self.self_attn = LlamaAttention(config)

        # llama models use a SWIGlu activation function instead of RELU
        # Llama-Paper
        # We replace the ReLU non-linearity by the SwiGLU activation function
        # introduced by Shazeer (2020) to improve the performance.
        # We use a dimension 2/3 4d instead of 4d as in PaLM.
        self.mlp = MLP(config.hidden_size, config.intermediate_size)

        # llama models normalize the inputs for each sub layer to stablize training
        # Llama-Paper
        # Pre-normalization [GPT3]. 
        # To improve the training stability, 
        # we normalize the input of each transformer sub-layer, 
        # instead of normalizing the output. 
        # We use the RMSNorm normalizing func- tion, introduced by Zhang and Sennrich (2019).
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Set the post attention layer normalisation, again, we use RMS
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

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
