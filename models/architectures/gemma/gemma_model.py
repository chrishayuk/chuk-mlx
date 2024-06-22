from models.architectures.transformer_base_model import TransformerBaseModel
from models.architectures.gemma.gemma_attention import GemmaAttention
from models.architectures.gemma.rms import RMSNorm
from models.model_config import ModelConfig
  
class GemmaModel(TransformerBaseModel):
    def __init__(self, config: ModelConfig):
        # call the base constructor
        super().__init__(
            config, 
            attention_layer=GemmaAttention,
            norm_layer=lambda hidden_size, eps: RMSNorm(hidden_size, eps=eps)
        )

    def scale_embeddings(self, embeddings):
        # scale the embedding
        return embeddings * (self.args.hidden_size**0.5)
        
