# model.py
import mlx.core as mx
import mlx.nn as nn
from core.models.model_config import ModelConfig

class Model(nn.Module):
    def __init__(self, args: ModelConfig):
        # initialize
        super().__init__()

        # store the config, just in-case we need
        self.config = args

        # The specific model (e.g., LlamaModel) will be set in the subclass

        # check if the weights are tied between the input embeddings and the output embeddings
        if not self.config.tie_word_embeddings:
            # set the head as a language modelling head
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        else:
            self.lm_head = None

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        # perform a forward pass
        out, cache = self.model(inputs, cache)
        
        # check if the weights are tied between the input embeddings and the output embeddings
        if self.lm_head is not None:
            out = self.lm_head(out)
        else:
            out = self.model.embed_tokens.as_linear(out)
        return out, cache
        
    def sanitize(self, weights):
        # nothing to sanitize
        return {
            k: v for k, v in weights.items()
        }