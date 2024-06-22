from models.architectures.model import Model
from models.model_config import ModelConfig
from models.architectures.llama.llama_model import LlamaModel

class MistralForCausalLM(Model):
    def __init__(self, args: ModelConfig):
        # initialize
        super().__init__(args)

        # set the model as Llama
        self.model = LlamaModel(args)
        
    def sanitize(self, weights):
        # sanitize
        sanitized_weights = {
            k: v for k, v in weights.items()
        }

        # # Handle sharded weights concatenation
        # num_layers = len(self.model.layers)
        # shard_prefixes = [f"shard_{i}" for i in range(len(weights) // num_layers)]
        # layer_names = [f"layers.{i}" for i in range(num_layers)]
        # weight_suffixes = {
        #     "self_attn.o_proj.weight": 1,
        #     "mlp.gate_proj.weight": 0,
        #     "mlp.down_proj.weight": 1,
        #     "mlp.up_proj.weight": 0
        # }

        # for layer_name in layer_names:
        #     for suffix, axis in weight_suffixes.items():
        #         shard_keys = [f"{prefix}.{layer_name}.{suffix}" for prefix in shard_prefixes]
        #         if all(key in weights for key in shard_keys):
        #             concatenated_weight = mx.concatenate([weights[key] for key in shard_keys], axis=axis)
        #             sanitized_weights[f"model.{layer_name}.{suffix}"] = concatenated_weight
        #             for key in shard_keys:
        #                 sanitized_weights.pop(key, None)


        return sanitized_weights