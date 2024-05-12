import mlx.core as mx
import mlx.nn as nn
from models.model_config import ModelConfig
from models.llama.llama_model import LlamaModel

class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        # initialize
        super().__init__()

        # set the model as Llama
        self.model = LlamaModel(config)
        
        # set the head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(self, x) -> mx.array:
        # call is the same as forward
        return self.forward(x)
    
    def forward(self, inputs, cache=False):
        # perform a forward pass
        out, cache = self.model(inputs, cache)
        return self.lm_head(out), cache