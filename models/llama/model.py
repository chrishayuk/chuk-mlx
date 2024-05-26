import mlx.core as mx
import mlx.nn as nn
from models.load_weights import load_model_weights
from models.model_config import ModelConfig
from models.llama.llama_model import LlamaModel
from utils.huggingface_utils import load_from_hub

class Model(nn.Module):
    def __init__(self, args: ModelConfig):
        # initialize
        super().__init__()

        # set the model as Llama
        self.model = LlamaModel(args)

        # store the config, just in-case we need
        self.config = args

        # set the head
        if not self.config.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        # perform a forward pass
        out, cache = self.model(inputs, cache)

        # set the head
        if self.config.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out), cache
        else:
            return self.lm_head(out), cache
        
    def sanitize(self, weights):
        # Remove unused precomputed rotary freqs
        return {
            k: v for k, v in weights.items() if "self_attn.rotary_emb.inv_freq" not in k
        }