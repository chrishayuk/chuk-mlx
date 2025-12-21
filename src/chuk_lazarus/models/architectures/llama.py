"""Llama model architecture."""

import mlx.nn as nn

from ..config import ModelConfig
from .attention import Attention
from .base import BaseModel, TransformerBlock, TransformerModel
from .mlp import MLP


class LlamaBlock(TransformerBlock):
    """Llama transformer block."""

    def __init__(self, config: ModelConfig):
        super().__init__(
            config=config, attention_class=Attention, mlp_class=MLP, norm_class=nn.RMSNorm
        )


class LlamaTransformer(TransformerModel):
    """Llama transformer (without LM head)."""

    def __init__(self, config: ModelConfig):
        super().__init__(config=config, layer_class=LlamaBlock, norm_class=nn.RMSNorm)


class LlamaModel(BaseModel):
    """
    Llama language model.

    Supports:
    - Llama 1, 2, 3
    - TinyLlama
    - Other Llama-architecture models
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model = LlamaTransformer(config)
