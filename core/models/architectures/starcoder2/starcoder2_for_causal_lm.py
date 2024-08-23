import mlx.nn as nn
from core.models.model_config import ModelConfig
from core.models.architectures.model import Model
from core.models.architectures.starcoder2.starcoder2_model import Starcoder2Model

class Starcoder2ForCausalLM(Model):
    def __init__(self, args: ModelConfig):
        super().__init__(args)
        self.model = Starcoder2Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def sanitize(self, weights):
        sanitized = {}
        for k, v in weights.items():
            if k.startswith('model.'):
                k = k[6:]  # Remove 'model.' prefix
            
            parts = k.split('.')
            if parts[0] == 'layers':
                layer_num = int(parts[1])
                if layer_num < len(self.model.layers):
                    layer = self.model.layers[layer_num]
                    if parts[2] == 'self_attn':
                        if parts[3] in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                            setattr(layer.self_attn, parts[3], v)
                    elif parts[2] == 'mlp':
                        if parts[3] in ['c_fc', 'c_proj']:
                            setattr(layer.mlp, parts[3], v)
                    elif parts[2] in ['input_layernorm', 'post_attention_layernorm']:
                        setattr(layer, parts[2], v)
                    sanitized[k] = v
            elif parts[0] == 'embed_tokens':
                setattr(self.model, parts[0], v)
                sanitized[k] = v
            elif parts[0] == 'norm':
                setattr(self.model, parts[0], v)
                sanitized[k] = v
            elif not self.model.args.tie_word_embeddings and parts[0] == 'lm_head':
                setattr(self.lm_head, parts[1], v)
                sanitized[k] = v
        
        return sanitized

    def __call__(self, inputs, cache=None):
        out, cache = self.model(inputs, cache)
        if self.model.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out, cache

    @property
    def layers(self):
        return self.model.layers