"""
Granite model implementation.

Supports:
- Granite 3.x: Dense transformer with multipliers
- Granite 4.x: Hybrid Mamba-2/Transformer with optional MoE

Reference: https://huggingface.co/ibm-granite
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from ...backbones.base import Backbone, BackboneOutput
from ...blocks.base import Block, BlockOutput
from ...components.embeddings import create_token_embedding
from ...components.ffn import SwiGLU
from ...components.normalization import RMSNorm
from ...core.config import FFNConfig
from ...core.registry import register_model
from ...heads import LMHead
from ...models.base import Model, ModelOutput
from .config import GraniteConfig


class GraniteAttention(nn.Module):
    """
    Granite attention with attention multiplier.

    Similar to GQA but with configurable multiplier on output.
    """

    def __init__(self, config: GraniteConfig, layer_idx: int = 0):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads or config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.layer_idx = layer_idx

        # Attention multiplier
        self.attention_multiplier = config.attention_multiplier

        # Number of query heads per KV head
        self.n_rep = self.num_heads // self.num_kv_heads

        # Projections
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
        )

        # RoPE
        from ...components.embeddings.rope import RoPE
        from ...core.config import RoPEConfig

        rope_config = RoPEConfig(
            theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
        )
        self.rope = RoPE(rope_config, dims=self.head_dim)

        # Attention scaling
        self.scale = self.head_dim**-0.5

        # Dropout
        self.dropout_rate = config.attention_dropout

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
        """Compute attention with multiplier."""
        batch_size, seq_len, _ = x.shape

        # Project
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # RoPE
        offset = 0
        if cache is not None:
            offset = cache[0].shape[2]

        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)

        # Update cache
        if cache is not None:
            key_cache, value_cache = cache
            k = mx.concatenate([key_cache, k], axis=2)
            v = mx.concatenate([value_cache, v], axis=2)

        new_cache = (k, v)

        # Repeat KV
        if self.n_rep > 1:
            k = self._repeat_kv(k, self.n_rep)
            v = self._repeat_kv(v, self.n_rep)

        # Attention
        output = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)

        # Reshape
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)

        # Output projection with multiplier
        output = self.o_proj(output)
        output = output * self.attention_multiplier

        return output, new_cache

    def _repeat_kv(self, x: mx.array, n_rep: int) -> mx.array:
        """Repeat KV heads."""
        if n_rep == 1:
            return x
        batch, num_kv_heads, seq_len, head_dim = x.shape
        x = mx.expand_dims(x, axis=2)
        x = mx.repeat(x, n_rep, axis=2)
        x = x.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)
        return x


class GraniteBlock(Block):
    """
    Granite transformer block.

    Pre-norm transformer with:
    - RMSNorm
    - Granite attention (with attention multiplier)
    - SwiGLU FFN
    - Residual multiplier
    """

    def __init__(self, config: GraniteConfig, layer_idx: int = 0):
        super().__init__()

        self._hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.residual_multiplier = config.residual_multiplier

        # Pre-attention norm
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Attention
        self.self_attn = GraniteAttention(config, layer_idx=layer_idx)

        # Post-attention norm
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # FFN
        ffn_config = FFNConfig(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=config.mlp_bias,
        )
        self.mlp = SwiGLU(ffn_config)

    @property
    def block_type(self):
        from ...core.enums import BlockType

        return BlockType.TRANSFORMER

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: tuple[mx.array, mx.array] | None = None,
    ) -> BlockOutput:
        """Forward pass with residual multiplier."""
        # Self-attention with residual
        residual = x
        x = self.input_layernorm(x)
        x, new_cache = self.self_attn(x, mask=mask, cache=cache)
        x = residual + x * self.residual_multiplier

        # FFN with residual
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x * self.residual_multiplier

        return BlockOutput(hidden_states=x, cache=new_cache)


class GraniteModel(Backbone):
    """
    Granite backbone (without LM head).

    Token embeddings with multiplier + transformer blocks + final norm.
    """

    def __init__(self, config: GraniteConfig):
        super().__init__()

        self.config = config
        self._vocab_size = config.vocab_size
        self._hidden_size = config.hidden_size
        self._num_layers = config.num_hidden_layers
        self.embedding_multiplier = config.embedding_multiplier

        # Token embeddings
        self.embed_tokens = create_token_embedding(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
        )

        # Transformer blocks
        self.layers = [GraniteBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)]

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
        attention_mask: mx.array | None = None,
        cache: list[Any] | None = None,
        output_hidden_states: bool = False,
    ) -> BackboneOutput:
        """Forward pass."""
        batch_size, seq_len = input_ids.shape

        # Embeddings with multiplier
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = hidden_states * self.embedding_multiplier

        # Create causal mask
        if attention_mask is None:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
            mask = mask.astype(hidden_states.dtype)
        else:
            mask = attention_mask

        # Track hidden states
        all_hidden_states = (hidden_states,) if output_hidden_states else None
        new_cache = []

        # Process layers
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache else None
            output = layer(hidden_states, mask=mask, cache=layer_cache)
            hidden_states = output.hidden_states
            new_cache.append(output.cache)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        # Final norm
        hidden_states = self.norm(hidden_states)

        return BackboneOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            cache=new_cache,
        )

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def set_input_embeddings(self, embeddings: nn.Module) -> None:
        self.embed_tokens = embeddings


@register_model(
    model_type="granite",
    architectures=["GraniteForCausalLM"],
)
class GraniteForCausalLM(Model):
    """
    Granite for causal language modeling.

    Complete model with backbone + LM head + logits scaling.
    """

    def __init__(self, config: GraniteConfig):
        super().__init__()

        self._config = config
        self.logits_scaling = config.logits_scaling

        # Backbone
        self.model = GraniteModel(config)

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
    def config(self) -> GraniteConfig:
        return self._config

    @property
    def backbone(self) -> nn.Module:
        return self.model

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        labels: mx.array | None = None,
        cache: list[Any] | None = None,
        output_hidden_states: bool = False,
    ) -> ModelOutput:
        """Forward pass with logits scaling."""
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

        # Apply logits scaling
        logits = head_output.logits
        if self.logits_scaling != 1.0:
            logits = logits / self.logits_scaling

        return ModelOutput(
            loss=head_output.loss,
            logits=logits,
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
        repetition_penalty: float = 1.0,
        stop_tokens: list[int] | None = None,
    ) -> mx.array:
        """Generate text autoregressively."""
        stop_tokens_set = set(stop_tokens or [])

        # Process prompt
        output = self(input_ids)
        mx.eval(output.logits)
        cache = output.cache

        # Track generated tokens
        generated_tokens = [input_ids]

        for _ in range(max_new_tokens):
            logits = output.logits[:, -1, :]

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                all_tokens = mx.concatenate(generated_tokens, axis=1)
                unique_tokens = set(all_tokens.flatten().tolist())
                vocab_size = logits.shape[-1]
                token_indices = mx.array([t for t in unique_tokens if t < vocab_size])
                if token_indices.size > 0:
                    mask = mx.zeros((vocab_size,))
                    for tok in token_indices.tolist():
                        mask = mask.at[tok].add(1.0)
                    penalty_mask = mx.where(mask > 0, repetition_penalty, 1.0)
                    logits = logits / penalty_mask

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Apply top-k
            if top_k is not None and top_k > 0:
                top_k_values = mx.topk(logits, k=min(top_k, logits.shape[-1]))
                min_val = top_k_values[:, -1:]
                logits = mx.where(logits < min_val, float("-inf"), logits)

            # Sample
            probs = mx.softmax(logits, axis=-1)
            next_token = mx.random.categorical(mx.log(probs + 1e-10))
            next_token = mx.expand_dims(next_token, axis=-1)

            mx.eval(next_token)
            generated_tokens.append(next_token)

            # Check stop
            next_token_val = int(next_token[0, 0])
            if next_token_val in stop_tokens_set:
                break

            # Forward with cache
            output = self(next_token, cache=cache)
            mx.eval(output.logits)
            cache = output.cache

        return mx.concatenate(generated_tokens, axis=1)

    @classmethod
    def from_config(cls, config: GraniteConfig) -> GraniteForCausalLM:
        """Create from config."""
        return cls(config)


# Convenience alias
Granite = GraniteForCausalLM
