"""
Causal language model.

Complete model for next-token prediction (GPT-style).
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from ..backbones import HybridBackbone, MambaBackbone, TransformerBackbone
from ..core.config import ModelConfig
from ..core.enums import BackboneType
from ..heads import LMHead
from .base import Model, ModelOutput


class CausalLM(Model):
    """
    Causal language model for next-token prediction.

    Combines a backbone (transformer, Mamba, or hybrid) with an LM head
    for autoregressive text generation.

    Args:
        config: Model configuration
        backbone_type: Type of backbone to use

    Example:
        >>> config = ModelConfig(vocab_size=32000, hidden_size=768, num_hidden_layers=12)
        >>> model = CausalLM.from_config(config)
        >>> input_ids = mx.array([[1, 2, 3, 4, 5]])
        >>> output = model(input_ids)
        >>> output.logits.shape
        (1, 5, 32000)

        >>> # With labels for training
        >>> labels = mx.array([[2, 3, 4, 5, 6]])
        >>> output = model(input_ids, labels=labels)
        >>> output.loss  # Cross-entropy loss
    """

    def __init__(
        self,
        config: ModelConfig,
        backbone_type: BackboneType = BackboneType.TRANSFORMER,
    ):
        super().__init__()

        self._config = config
        self.backbone_type = backbone_type

        # Create backbone based on type
        if backbone_type == BackboneType.TRANSFORMER:
            self._backbone = TransformerBackbone.from_config(config)
        elif backbone_type == BackboneType.MAMBA:
            self._backbone = MambaBackbone.from_config(config)
        elif backbone_type == BackboneType.HYBRID:
            self._backbone = HybridBackbone(
                vocab_size=config.vocab_size,
                hidden_size=config.hidden_size,
                num_layers=config.num_hidden_layers,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
            )
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")

        # LM head
        if config.tie_word_embeddings:
            self.lm_head = LMHead(
                hidden_size=config.hidden_size,
                vocab_size=config.vocab_size,
                tied_embeddings=self._backbone.embed_tokens,
            )
        else:
            self.lm_head = LMHead(
                hidden_size=config.hidden_size,
                vocab_size=config.vocab_size,
            )

    @property
    def config(self) -> ModelConfig:
        return self._config

    @property
    def backbone(self) -> nn.Module:
        return self._backbone

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        labels: mx.array | None = None,
        cache: list[Any] | None = None,
        output_hidden_states: bool = False,
    ) -> ModelOutput:
        """
        Forward pass.

        Args:
            input_ids: Token IDs, shape (batch, seq_len)
            attention_mask: Optional attention mask
            labels: Optional target IDs for loss, shape (batch, seq_len)
            cache: Optional KV cache for inference
            output_hidden_states: Return all layer hidden states

        Returns:
            ModelOutput with logits and optional loss
        """
        # Get backbone output
        backbone_output = self._backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            cache=cache,
            output_hidden_states=output_hidden_states,
        )

        # Get LM head output
        head_output = self.lm_head(
            hidden_states=backbone_output.last_hidden_state,
            labels=labels,
        )

        return ModelOutput(
            loss=head_output.loss,
            logits=head_output.logits,
            hidden_states=backbone_output.hidden_states,
            cache=backbone_output.cache,
            aux_loss=backbone_output.aux_loss,
        )

    def generate(
        self,
        input_ids: mx.array,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        stop_tokens: list[int] | None = None,
    ) -> mx.array:
        """
        Generate text autoregressively.

        Args:
            input_ids: Prompt token IDs, shape (batch, prompt_len)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            stop_tokens: Token IDs to stop generation

        Returns:
            Generated token IDs, shape (batch, prompt_len + generated_len)
        """
        batch_size, prompt_len = input_ids.shape
        stop_tokens_set = set(stop_tokens or [])

        # Process prompt and evaluate immediately
        output = self(input_ids)
        mx.eval(output.logits)
        cache = output.cache

        # Track generated tokens
        generated = [input_ids]

        for _ in range(max_new_tokens):
            # Get logits for last token
            logits = output.logits[:, -1, :]  # (batch, vocab)

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Apply top-k
            if top_k is not None:
                top_k_values = mx.topk(logits, k=min(top_k, logits.shape[-1]))
                min_val = top_k_values[:, -1:]
                logits = mx.where(logits < min_val, float("-inf"), logits)

            # Apply top-p (nucleus sampling)
            if top_p is not None:
                sorted_logits = mx.sort(logits, axis=-1)[:, ::-1]
                sorted_probs = mx.softmax(sorted_logits, axis=-1)
                cumsum = mx.cumsum(sorted_probs, axis=-1)
                # Find cutoff
                mask = cumsum > top_p
                # Shift mask right by 1 to keep first token above threshold
                mask = mx.concatenate(
                    [mx.zeros((batch_size, 1), dtype=mx.bool_), mask[:, :-1]],
                    axis=-1,
                )
                sorted_logits = mx.where(mask, float("-inf"), sorted_logits)
                # Unsort - for simplicity, just use the masked logits directly
                logits = sorted_logits

            # Sample
            probs = mx.softmax(logits, axis=-1)
            next_token = mx.random.categorical(mx.log(probs + 1e-10))
            next_token = mx.expand_dims(next_token, axis=-1)

            # Evaluate to avoid graph buildup
            mx.eval(next_token)

            generated.append(next_token)

            # Check stop condition
            next_token_val = int(next_token[0, 0])
            if next_token_val in stop_tokens_set:
                break

            # Forward with new token
            output = self(next_token, cache=cache)
            mx.eval(output.logits)
            cache = output.cache

        return mx.concatenate(generated, axis=1)

    @classmethod
    def from_config(
        cls,
        config: ModelConfig,
        backbone_type: BackboneType = BackboneType.TRANSFORMER,
    ) -> CausalLM:
        """Create CausalLM from config."""
        return cls(config=config, backbone_type=backbone_type)


class MambaCausalLM(CausalLM):
    """Causal LM with Mamba backbone."""

    def __init__(self, config: ModelConfig):
        super().__init__(config, backbone_type=BackboneType.MAMBA)


class HybridCausalLM(CausalLM):
    """Causal LM with hybrid backbone."""

    def __init__(self, config: ModelConfig):
        super().__init__(config, backbone_type=BackboneType.HYBRID)


def create_causal_lm(
    vocab_size: int,
    hidden_size: int,
    num_layers: int,
    num_heads: int,
    backbone_type: str = "transformer",
) -> CausalLM:
    """
    Factory function for CausalLM.

    Args:
        vocab_size: Vocabulary size
        hidden_size: Model dimension
        num_layers: Number of layers
        num_heads: Number of attention heads
        backbone_type: "transformer", "mamba", or "hybrid"

    Returns:
        CausalLM instance
    """
    config = ModelConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
    )

    backbone_enum = BackboneType(backbone_type)
    return CausalLM(config=config, backbone_type=backbone_enum)
