from core.models.architectures.model import Model
from core.models.model_config import ModelConfig
from core.models.architectures.llama.llama_model import LlamaModel

class CustomModel(Model):
    def __init__(self, config: ModelConfig):
        # initialize
        super().__init__(config)

        # set the model as Llama
        self.model = LlamaModel(config)
        
    def sanitize(self, weights):
        # Simply return all weights without filtering
        return weights