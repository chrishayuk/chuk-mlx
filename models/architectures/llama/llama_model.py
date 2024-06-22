import mlx.core as mx
import mlx.nn as nn
from models.architectures.transformer_base_model import TransformerBaseModel
from models.architectures.transformer_block import TransformerBlock
from models.architectures.llama.llama_attention import LlamaAttention
from models.model_config import ModelConfig
  
class LlamaModel(TransformerBaseModel):
    def __init__(self, config: ModelConfig):
        # call the base constructor
        super().__init__(
            config, 
            attention_layer=LlamaAttention,
            norm_layer=lambda hidden_size, eps: nn.RMSNorm(hidden_size, eps=eps)
        )