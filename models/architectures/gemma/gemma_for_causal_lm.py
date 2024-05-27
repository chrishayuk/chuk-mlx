from models.architectures.model import Model
from models.model_config import ModelConfig
from models.architectures.gemma.gemma_model import GemmaModel

class GemmaForCausalLM(Model):
    def __init__(self, args: ModelConfig):
        # initialize
        super().__init__(args)

        # set the model as Gemma
        self.model = GemmaModel(args)