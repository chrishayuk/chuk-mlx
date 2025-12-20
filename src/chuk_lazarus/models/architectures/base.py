"""Base model classes."""

import logging
from enum import Enum
from typing import Optional, Tuple, Any

import mlx.core as mx
import mlx.nn as nn

from ..config import ModelConfig

logger = logging.getLogger(__name__)


class ModelMode(Enum):
    """Model operating mode."""
    TRAIN = "train"
    INFERENCE = "inference"


class BaseModel(nn.Module):
    """
    Base class for all language models.

    Provides:
    - Common interface for forward pass
    - Mode switching (train/inference)
    - LM head handling
    - Cache management
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.use_cache = False
        self.model = None  # Set by subclasses

        # LM head - either separate or tied to embeddings
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        else:
            self.lm_head = None

    def __call__(
        self,
        inputs: mx.array,
        cache: Any = None
    ) -> Tuple[mx.array, Any]:
        """
        Forward pass.

        Args:
            inputs: Token IDs, shape (batch, seq_len)
            cache: Optional KV cache

        Returns:
            logits: Shape (batch, seq_len, vocab_size)
            cache: Updated cache
        """
        if self.model is None:
            raise ValueError("Model not initialized. Subclass must set self.model")

        # Forward through transformer
        hidden, cache = self.model(inputs, cache=cache if self.use_cache else None)

        # Project to vocabulary
        if self.lm_head is not None:
            logits = self.lm_head(hidden)
        else:
            logits = self.model.embed_tokens.as_linear(hidden)

        return logits, cache if self.use_cache else None

    def set_mode(self, mode: ModelMode):
        """Set training or inference mode."""
        if isinstance(mode, str):
            mode = ModelMode(mode.lower())

        if mode == ModelMode.TRAIN:
            self.use_cache = False
        elif mode == ModelMode.INFERENCE:
            self.use_cache = True
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def reset_cache(self):
        """Reset KV cache."""
        if self.model is not None and hasattr(self.model, 'reset_cache'):
            self.model.reset_cache()


class TransformerModel(nn.Module):
    """
    Base transformer model (encoder stack).

    This is the inner model without the LM head.
    """

    def __init__(
        self,
        config: ModelConfig,
        layer_class,
        norm_class=nn.RMSNorm
    ):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_layers = config.num_hidden_layers

        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Transformer layers
        self.layers = [
            layer_class(config)
            for _ in range(config.num_hidden_layers)
        ]

        # Final normalization
        self.norm = norm_class(config.hidden_size, eps=config.rms_norm_eps)

        # Mask cache for efficiency
        self._mask_cache = {}

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[list] = None
    ) -> Tuple[mx.array, list]:
        """
        Forward pass through transformer.

        Args:
            inputs: Token IDs, shape (batch, seq_len)
            cache: Optional list of KV caches per layer

        Returns:
            hidden: Hidden states, shape (batch, seq_len, hidden_size)
            cache: Updated cache list
        """
        # Embed tokens
        h = self.embed_tokens(inputs)
        h = self._scale_embeddings(h)

        # Create causal mask
        mask = None
        if h.shape[1] > 1:
            mask = self._get_mask(h.shape[1], h.dtype)

        # Initialize cache if needed
        if cache is None:
            cache = [None] * len(self.layers)

        # Forward through layers
        for i, layer in enumerate(self.layers):
            h, cache[i] = layer(h, mask, cache[i])

        # Final normalization
        return self.norm(h), cache

    def _scale_embeddings(self, embeddings: mx.array) -> mx.array:
        """Scale embeddings (override in subclass if needed)."""
        return embeddings

    def _get_mask(self, seq_len: int, dtype) -> mx.array:
        """Get or create causal mask."""
        if seq_len not in self._mask_cache:
            self._mask_cache[seq_len] = nn.MultiHeadAttention.create_additive_causal_mask(
                seq_len
            ).astype(dtype)
        return self._mask_cache[seq_len]


class TransformerBlock(nn.Module):
    """
    Standard transformer block with attention and MLP.
    """

    def __init__(
        self,
        config: ModelConfig,
        attention_class,
        mlp_class,
        norm_class=nn.RMSNorm
    ):
        super().__init__()

        # Attention with pre-norm
        self.input_layernorm = norm_class(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = attention_class(config)

        # MLP with pre-norm
        self.post_attention_layernorm = norm_class(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = mlp_class(config)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        """
        Forward pass.

        Args:
            x: Input hidden states
            mask: Attention mask
            cache: KV cache for this layer

        Returns:
            output: Output hidden states
            cache: Updated KV cache
        """
        # Attention block with residual
        residual = x
        x = self.input_layernorm(x)
        x, cache = self.self_attn(x, mask=mask, cache=cache)
        x = residual + x

        # MLP block with residual
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x, cache
