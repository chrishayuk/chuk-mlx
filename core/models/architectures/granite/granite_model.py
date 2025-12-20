import mlx.core as mx
import mlx.nn as nn
from core.models.architectures.normalization_layer_factory import NormalizationLayerFactory
from core.models.model_config import ModelConfig
from core.models.architectures.model import Model
from core.models.architectures.attention_base import AttentionBase
from core.models.architectures.transformer_base_model import TransformerBaseModel

class GraniteForCausalLM(Model):
    """
    Llama model for causal language modeling tasks.
    This class wraps the core Granite model with a causal language modeling head.
    """
    def __init__(self, args: ModelConfig, tokenizer):
        super().__init__(args)
        self.tokenizer = tokenizer  # Attach tokenizer to the model

        # Initialize the core Granite model
        self.model = GraniteModel(args)

        # Define language modeling head with consistent float32 processing
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size)

    def sanitize(self, weights):
        """
        Sanitize model weights by filtering unwanted keys and casting to float32.
        """
        sanitized_weights = {}
        for k, v in weights.items():
            # Filter out unwanted keys and ensure weights are in float32
            if 'rotary_emb.inv_freq' not in k and (self.lm_head or 'lm_head' not in k):
                sanitized_weights[k] = v.astype(mx.float32)

        # Ensure `lm_head.bias` is included if necessary, cast to float32
        if self.lm_head and 'lm_head.bias' not in sanitized_weights:
            sanitized_weights['lm_head.bias'] = self.lm_head.bias.astype(mx.float32)

        print("Sanitized weights:", sanitized_weights.keys())
        return sanitized_weights

    def set_inv_freq(self, inv_freq):
        """
        Set inverse frequency for rotary embeddings across attention layers.
        """
        for layer in self.model.layers:
            if hasattr(layer.self_attn, 'set_inv_freq'):
                layer.self_attn.set_inv_freq(inv_freq)

    def __call__(self, inputs: mx.array, cache=None):
        """
        Perform a forward pass through the model with caching support for inference.
        """
        if self.model is None:
            raise ValueError("The model has not been set. Ensure that a subclass sets the model.")

        # Forward pass through the GraniteModel in float32
        out, cache = self.model(inputs, cache=cache if self.use_cache else None)

        # Apply lm_head in float32 consistency
        if self.lm_head:
            out = self.lm_head(out)
        else:
            out = self.model.embed_tokens.as_linear(out)

        # Log output shape and dtype once per main input
        if cache is None:
            print(f"Output shape after lm_head: {out.shape}, dtype: {out.dtype}")

        return out, cache if self.use_cache else None

    def encode(self, text):
        """
        Encode text to model-compatible token IDs using the attached tokenizer.
        """
        encoded = self.tokenizer.encode(text, return_tensors="pt")
        print("Encoded text:", encoded)
        return encoded

    def decode(self, token_ids):
        """
        Decode token IDs back to text using the attached tokenizer.
        """
        decoded = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        print("Decoded text:", decoded)
        return decoded

    def tokenization_check(self, text):
        """
        Run a tokenization check by encoding and then decoding the input text.
        """
        print("Original text:", text)
        encoded = self.encode(text)
        decoded = self.decode(encoded[0])
        
        if decoded == text:
            print("Tokenization check passed.")
        else:
            print("Tokenization check failed.")
            print(f"Decoded text: '{decoded}' does not match original text: '{text}'")


class GraniteModel(TransformerBaseModel):
    """
    Core Granite model implementation.
    Utilizes TransformerBaseModel with Granite-specific attention and normalization layers.
    """
    def __init__(self, config: ModelConfig):
        super().__init__(
            config,
            attention_layer=GraniteAttention,
            norm_layer=lambda hidden_size, eps: NormalizationLayerFactory.create_norm_layer(hidden_size, eps)
        )


class GraniteAttention(AttentionBase):
    """
    Llama-specific attention mechanism.
    Configured with RoPE setup from AttentionBase for Llama-style rotary embeddings.
    """
    def __init__(self, config: ModelConfig):
        super().__init__(config)
