import mlx.core as mx
import mlx.nn as nn
from models.model_config import ModelConfig
from models.architectures.llama.llama_model import LlamaModel

class Model(nn.Module):
    def __init__(self, args: ModelConfig):
        # initialize
        super().__init__()

        # set the model as Llama
        self.model = LlamaModel(args)

        # store the config, just in-case we need
        self.config = args

        # check if the weights are tied between the input embeddings and the output embeddings
        if not self.config.tie_word_embeddings:
            # set the head as a language modelling head
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        # perform a forward pass
        out, cache = self.model(inputs, cache)
        
        # check if the weights are tied between the input embeddings and the output embeddings
        if self.config.tie_word_embeddings:
            # execute through the embeddings layer
            return self.model.embed_tokens.as_linear(out), cache
        else:
            # execute through the language modelling head
            return self.lm_head(out), cache
        
    def sanitize(self, weights):
        # nothing to sanitize
        return {
            k: v for k, v in weights.items()
        }