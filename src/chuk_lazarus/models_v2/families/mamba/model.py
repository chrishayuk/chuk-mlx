"""
Mamba model implementation.

Pure SSM architecture without attention.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ...backbones.base import Backbone, BackboneOutput
from ...components.embeddings import create_token_embedding
from ...components.normalization import RMSNorm
from ...components.ssm import MambaBlock
from ...core.registry import register_model
from ...heads import LMHead
from ...models.base import Model, ModelOutput
from .config import MambaConfig


class MambaModel(Backbone):
    """
    Mamba backbone (without LM head).

    Pure SSM architecture:
    - Token embeddings
    - Stack of Mamba blocks
    - Final normalization
    """

    def __init__(self, config: MambaConfig):
        super().__init__()

        self._config = config
        self._vocab_size = config.vocab_size
        self._hidden_size = config.hidden_size
        self._num_layers = config.num_hidden_layers

        # Token embeddings
        self.embed_tokens = create_token_embedding(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
        )

        # Mamba blocks
        self.layers = [
            MambaBlock(
                d_model=config.hidden_size,
                d_state=config.d_state,
                d_conv=config.d_conv,
                expand=config.expand,
                norm_eps=config.rms_norm_eps,
            )
            for _ in range(config.num_hidden_layers)
        ]

        # Final norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,  # Ignored
        cache: list[tuple[mx.array, mx.array]] | None = None,
        output_hidden_states: bool = False,
    ) -> BackboneOutput:
        """
        Forward pass.

        Note: attention_mask is ignored because Mamba is inherently causal.
        """
        # Embeddings
        hidden_states = self.embed_tokens(input_ids)

        # Track hidden states
        all_hidden_states = (hidden_states,) if output_hidden_states else None
        new_cache = []

        # Process layers
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache else None
            hidden_states, layer_new_cache = layer(hidden_states, layer_cache)
            new_cache.append(layer_new_cache)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        # Final norm
        hidden_states = self.norm(hidden_states)

        return BackboneOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            cache=new_cache,
        )

    def init_cache(
        self,
        batch_size: int,
        max_seq_len: int = 0,  # Not used for Mamba
    ) -> list[tuple[mx.array, mx.array]]:
        """Initialize cache for all layers."""
        return [layer.init_cache(batch_size) for layer in self.layers]

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def set_input_embeddings(self, embeddings: nn.Module) -> None:
        self.embed_tokens = embeddings


@register_model(
    model_type="mamba",
    architectures=["MambaForCausalLM", "MambaLMHeadModel"],
)
class MambaForCausalLM(Model):
    """
    Mamba for causal language modeling.

    Complete SSM model with LM head.
    """

    def __init__(self, config: MambaConfig):
        super().__init__()

        self._config = config

        # Backbone
        self.model = MambaModel(config)

        # LM head
        if config.tie_word_embeddings:
            self.lm_head = LMHead(
                hidden_size=config.hidden_size,
                vocab_size=config.vocab_size,
                tied_embeddings=self.model.embed_tokens,
            )
        else:
            self.lm_head = LMHead(
                hidden_size=config.hidden_size,
                vocab_size=config.vocab_size,
            )

    @property
    def config(self) -> MambaConfig:
        return self._config

    @property
    def backbone(self) -> nn.Module:
        return self.model

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        labels: mx.array | None = None,
        cache: list[tuple[mx.array, mx.array]] | None = None,
        output_hidden_states: bool = False,
    ) -> ModelOutput:
        """Forward pass."""
        # Backbone
        backbone_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            cache=cache,
            output_hidden_states=output_hidden_states,
        )

        # LM head
        head_output = self.lm_head(
            hidden_states=backbone_output.last_hidden_state,
            labels=labels,
        )

        return ModelOutput(
            loss=head_output.loss,
            logits=head_output.logits,
            hidden_states=backbone_output.hidden_states,
            cache=backbone_output.cache,
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

        Mamba generation is efficient because:
        - O(1) memory per step (no KV cache growth)
        - O(1) time per step (no attention over history)
        """
        stop_tokens = stop_tokens or []
        generated = input_ids

        # Process prompt
        output = self(input_ids)
        cache = output.cache

        for _ in range(max_new_tokens):
            # Get logits for last position
            logits = output.logits[:, -1, :]

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Apply top-k
            if top_k is not None and top_k > 0:
                top_k_logits, _ = mx.topk(logits, k=top_k)
                min_val = top_k_logits[:, -1:]
                logits = mx.where(logits < min_val, float("-inf"), logits)

            # Sample
            probs = mx.softmax(logits, axis=-1)
            next_token = mx.random.categorical(mx.log(probs + 1e-10))
            next_token = mx.expand_dims(next_token, axis=-1)

            # Append
            generated = mx.concatenate([generated, next_token], axis=1)

            # Check stop
            if any(int(next_token[0, 0]) == stop for stop in stop_tokens):
                break

            # Forward with cache (single token)
            output = self(next_token, cache=cache)
            cache = output.cache

        return generated

    def init_cache(
        self,
        batch_size: int,
        max_seq_len: int = 0,
    ) -> list[tuple[mx.array, mx.array]]:
        """Initialize cache."""
        return self.model.init_cache(batch_size, max_seq_len)

    @classmethod
    def from_config(cls, config: MambaConfig) -> MambaForCausalLM:
        """Create from config."""
        return cls(config)
