"""Llama model architecture."""

from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from ..config import ModelConfig
from .base import BaseModel, TransformerModel, TransformerBlock
from .attention import Attention
from .mlp import MLP


class LlamaBlock(TransformerBlock):
    """Llama transformer block."""

    def __init__(self, config: ModelConfig):
        super().__init__(
            config=config,
            attention_class=Attention,
            mlp_class=MLP,
            norm_class=nn.RMSNorm
        )


class LlamaTransformer(TransformerModel):
    """Llama transformer (without LM head)."""

    def __init__(self, config: ModelConfig):
        super().__init__(
            config=config,
            layer_class=LlamaBlock,
            norm_class=nn.RMSNorm
        )


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
