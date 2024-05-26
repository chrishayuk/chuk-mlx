from models.architectures.model import Model
from models.model_config import ModelConfig
from models.architectures.llama.llama_model import LlamaModel

class GemmaForCausalLM(Model):
    def __init__(self, args: ModelConfig):
        # initialize
        super().__init__(args)

        # set the model as Llama
        self.model = LlamaModel(args)