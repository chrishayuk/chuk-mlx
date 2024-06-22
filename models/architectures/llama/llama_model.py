import mlx.nn as nn
from models.model_config import ModelConfig
from models.architectures.model import Model
from models.architectures.transformer_base_model import TransformerBaseModel
from models.architectures.llama.llama_attention import LlamaAttention
class LlamaModel(TransformerBaseModel):
    def __init__(self, config: ModelConfig):
        # call the base constructor
        super().__init__(
            config, 
            attention_layer=LlamaAttention,
            norm_layer=lambda hidden_size, eps: nn.RMSNorm(hidden_size, eps=eps)
        )
class LlamaForCausalLM(Model):
    def __init__(self, args: ModelConfig):
        # initialize
        super().__init__(args)

        # set the model as Llama
        self.model = LlamaModel(args)

