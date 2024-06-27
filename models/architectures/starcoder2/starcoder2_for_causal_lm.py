from models.model_config import ModelConfig
from models.architectures.model import Model
from models.architectures.starcoder2.starcoder2_model import StarCoder2Model

class Starcoder2ForCausalLM(Model):
    def __init__(self, args: ModelConfig):
        # call the constructor
        super().__init__(args)

        # set the model as starcoder 2
        self.model = StarCoder2Model(args)

    from models.model_config import ModelConfig
from models.architectures.model import Model
from models.architectures.starcoder2.starcoder2_model import StarCoder2Model

class Starcoder2ForCausalLM(Model):
    def __init__(self, args: ModelConfig):
        super().__init__(args)
        self.model = StarCoder2Model(args)

    def sanitize(self, weights):
        sanitized = {}
        for k, v in weights.items():
            if k.startswith('model.'):
                k = k[6:]  # Remove 'model.' prefix
            
            parts = k.split('.')
            if parts[0] == 'layers' and len(parts) > 2:
                layer_num = int(parts[1])
                if layer_num < len(self.model.layers):
                    # Adjust the key structure
                    new_key = f"layers.{layer_num}.{'.'.join(parts[2:])}"
                    sanitized[new_key] = v
            elif hasattr(self.model, parts[0]):
                # Keep other valid top-level keys
                sanitized[k] = v
        
        return sanitized

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)