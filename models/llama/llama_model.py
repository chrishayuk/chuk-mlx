import mlx.core as mx
import mlx.nn as nn
from models.load_weights import load_model_weights
from models.model_config import ModelConfig
from models.llama.llama_layer import LlamaLayer
from utils.huggingface_utils import load_from_hub
  
class LlamaModel(nn.Module):
    def __init__(self, config: ModelConfig):
        # initialize
        super().__init__()

        # we store the config for later
        self.args = config

        # set the vocabulary size
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers

        # set the embeddings layer
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # now we create the layers 
        self.layers = [
            # create a llama layer for each hidden layer
            LlamaLayer(config=config) for _ in range(config.num_hidden_layers)
        ]

        # set the normalization layer as using RMS normalization
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        h = self.embed_tokens(inputs)

        mask = None
        if h.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
            mask = mask.astype(h.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        for e, layer in enumerate(self.layers):
            h, cache[e] = layer(h, mask, cache[e])

        # TODO: I don't think we need to normalize, as it's done on input layer
        return self.norm(h), cache
        #return h, cache#self.norm(h), cache

    # @classmethod
    # def load(cls, model_name) -> 'LlamaModel':        
    #     # load the model from huggingface
    #     print(f"Loading Model: {model_name}")

    #     # load the model from huggingface
    #     model_path = load_from_hub(model_name)

    #     # load config
    #     model_config = ModelConfig.load(model_path)

        

    #     # create the model instance
    #     model = cls(model_config)
    #     print(model)

    #     # Model Loaded
    #     print(f"Model Loaded: {model_name}")

        

    #     # return the model
    #     return model

