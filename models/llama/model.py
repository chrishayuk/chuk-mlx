import mlx.core as mx
import mlx.nn as nn
from models.model_config import ModelConfig
from models.llama.llama_model import LlamaModel

class Model(nn.Module):
    def __init__(self, args: ModelConfig):
        # initialize
        super().__init__()

        # set the model as Llama
        self.model = LlamaModel(args)

        # set the head
        # TODO: Not supported by Google Gemma
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        # perform a forward pass
        out, cache = self.model(inputs, cache)
        return self.lm_head(out), cache
        #return out, cache