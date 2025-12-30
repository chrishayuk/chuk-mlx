"""
Model adapter for accessing model internals across architectures.

This module provides a unified interface for accessing layers,
getting/setting component weights, and running generation across
different model architectures.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn


class ModelAdapter:
    """
    Adapter for accessing model internals across different architectures.

    Provides a unified interface for:
    - Accessing layers
    - Getting/setting component weights
    - Running generation
    """

    def __init__(self, model: nn.Module, tokenizer: Any, config: Any):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self._detect_architecture()

    def _detect_architecture(self):
        """Detect model architecture and set accessors."""
        # Try different common patterns
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            self._layers = self.model.model.layers
            self._backbone = self.model.model
        elif hasattr(self.model, "layers"):
            self._layers = self.model.layers
            self._backbone = self.model
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            self._layers = self.model.transformer.h
            self._backbone = self.model.transformer
        else:
            raise ValueError(
                "Cannot detect model architecture. Expected model.model.layers, model.layers, or model.transformer.h"
            )

    @property
    def num_layers(self) -> int:
        """Number of transformer/SSM layers."""
        return len(self._layers)

    @property
    def hidden_size(self) -> int:
        """Hidden dimension size."""
        if hasattr(self.config, "hidden_size"):
            return self.config.hidden_size
        elif hasattr(self.config, "d_model"):
            return self.config.d_model
        raise ValueError("Cannot determine hidden size from config")

    def get_layer(self, idx: int) -> nn.Module:
        """Get layer by index."""
        return self._layers[idx]

    def is_moe_layer(self, layer_idx: int) -> bool:
        """Check if a layer uses Mixture of Experts."""
        layer = self.get_layer(layer_idx)
        if hasattr(layer, "mlp"):
            mlp = layer.mlp
            # Check for MoE patterns
            if hasattr(mlp, "router") or hasattr(mlp, "experts"):
                return True
            # Check class name
            if "MoE" in type(mlp).__name__:
                return True
        return False

    def get_mlp_down_weight(self, layer_idx: int) -> mx.array:
        """Get MLP down projection weight.

        For MoE layers, returns the router weight instead.
        """
        layer = self.get_layer(layer_idx)

        # Check for MoE first
        if hasattr(layer, "mlp"):
            mlp = layer.mlp
            # MoE: return router weight (zeroing this effectively disables MLP)
            if hasattr(mlp, "router"):
                if hasattr(mlp.router, "weight"):
                    return mlp.router.weight
            # Dense MLP patterns
            if hasattr(mlp, "down_proj"):
                return mlp.down_proj.weight
            elif hasattr(mlp, "c_proj"):  # GPT-2 style
                return mlp.c_proj.weight
            elif hasattr(mlp, "w2"):  # Some Llama variants
                return mlp.w2.weight
        elif hasattr(layer, "feed_forward"):
            ff = layer.feed_forward
            if hasattr(ff, "down_proj"):
                return ff.down_proj.weight
            elif hasattr(ff, "w2"):
                return ff.w2.weight

        raise ValueError(f"Cannot find MLP down projection in layer {layer_idx}")

    def set_mlp_down_weight(self, layer_idx: int, weight: mx.array):
        """Set MLP down projection weight.

        For MoE layers, sets the router weight instead.
        """
        layer = self.get_layer(layer_idx)

        if hasattr(layer, "mlp"):
            mlp = layer.mlp
            # MoE: set router weight
            if hasattr(mlp, "router"):
                if hasattr(mlp.router, "weight"):
                    mlp.router.weight = weight
                    mx.eval(weight)
                    return
            # Dense MLP patterns
            if hasattr(mlp, "down_proj"):
                mlp.down_proj.weight = weight
            elif hasattr(mlp, "c_proj"):
                mlp.c_proj.weight = weight
            elif hasattr(mlp, "w2"):
                mlp.w2.weight = weight
            else:
                raise ValueError(f"Cannot find MLP down projection in layer {layer_idx}")
        elif hasattr(layer, "feed_forward"):
            ff = layer.feed_forward
            if hasattr(ff, "down_proj"):
                ff.down_proj.weight = weight
            elif hasattr(ff, "w2"):
                ff.w2.weight = weight
            else:
                raise ValueError(f"Cannot find MLP down projection in layer {layer_idx}")
        else:
            raise ValueError(f"Cannot find MLP in layer {layer_idx}")

        mx.eval(weight)

    def get_attn_o_weight(self, layer_idx: int) -> mx.array:
        """Get attention output projection weight."""
        layer = self.get_layer(layer_idx)

        # Try common patterns
        if hasattr(layer, "self_attn"):
            attn = layer.self_attn
            if hasattr(attn, "o_proj"):
                return attn.o_proj.weight
            elif hasattr(attn, "out_proj"):
                return attn.out_proj.weight
        elif hasattr(layer, "attention"):
            attn = layer.attention
            if hasattr(attn, "o_proj"):
                return attn.o_proj.weight
            elif hasattr(attn, "wo"):  # Llama style
                return attn.wo.weight

        raise ValueError(f"Cannot find attention output projection in layer {layer_idx}")

    def set_attn_o_weight(self, layer_idx: int, weight: mx.array):
        """Set attention output projection weight."""
        layer = self.get_layer(layer_idx)

        if hasattr(layer, "self_attn"):
            attn = layer.self_attn
            if hasattr(attn, "o_proj"):
                attn.o_proj.weight = weight
            elif hasattr(attn, "out_proj"):
                attn.out_proj.weight = weight
            else:
                raise ValueError(f"Cannot find attention output projection in layer {layer_idx}")
        elif hasattr(layer, "attention"):
            attn = layer.attention
            if hasattr(attn, "o_proj"):
                attn.o_proj.weight = weight
            elif hasattr(attn, "wo"):
                attn.wo.weight = weight
            else:
                raise ValueError(f"Cannot find attention output projection in layer {layer_idx}")
        else:
            raise ValueError(f"Cannot find attention in layer {layer_idx}")

        mx.eval(weight)

    def generate(
        self,
        input_ids: mx.array,
        max_new_tokens: int = 60,
        temperature: float = 0.0,
    ) -> str:
        """Generate text from input IDs."""
        # Use model's generate method if available
        if hasattr(self.model, "generate"):
            stop_tokens = []
            if self.tokenizer.eos_token_id is not None:
                stop_tokens = [self.tokenizer.eos_token_id]

            generated = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                stop_tokens=stop_tokens,
            )
            output_ids = generated[0, input_ids.shape[1] :].tolist()
            return self.tokenizer.decode(output_ids, skip_special_tokens=False)

        # Fallback: manual generation
        return self._manual_generate(input_ids, max_new_tokens, temperature)

    def _manual_generate(
        self,
        input_ids: mx.array,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        """Manual autoregressive generation."""
        generated = input_ids

        for _ in range(max_new_tokens):
            output = self.model(generated)

            # Handle different output types
            if hasattr(output, "logits"):
                logits = output.logits
            elif isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output

            # Get last token logits
            next_logits = logits[0, -1, :]

            # Sample
            if temperature == 0:
                next_token = mx.argmax(next_logits, axis=-1, keepdims=True)
            else:
                probs = mx.softmax(next_logits / temperature)
                next_token = mx.random.categorical(mx.log(probs + 1e-10))
                next_token = mx.expand_dims(next_token, axis=0)

            generated = mx.concatenate([generated, next_token[None, :]], axis=1)

            # Check for EOS
            if self.tokenizer.eos_token_id and int(next_token[0]) == self.tokenizer.eos_token_id:
                break

        output_ids = generated[0, input_ids.shape[1] :].tolist()
        return self.tokenizer.decode(output_ids, skip_special_tokens=False)
