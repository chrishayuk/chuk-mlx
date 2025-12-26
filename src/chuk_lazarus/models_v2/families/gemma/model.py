"""
Gemma 3 model implementation.

Gemma 3 has several unique architectural features:
- Alternating sliding window / global attention layers
- Query/key pre-attention normalization
- 4 normalization layers per block (pre/post attention, pre/post FFN)
- Gated GELU activation (not SwiGLU)
- Embedding scaling by sqrt(hidden_size)
"""

from __future__ import annotations

from functools import partial
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from ...backbones.base import Backbone, BackboneOutput
from ...blocks.base import Block, BlockOutput
from ...core.registry import register_model
from ...models.base import Model, ModelOutput
from .config import GemmaConfig


class GemmaRMSNorm(nn.Module):
    """
    Gemma-style RMSNorm.

    Uses (1 + weight) scaling like Gemma implementation.
    """

    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.dims = dims
        self.eps = eps
        self.weight = mx.ones((dims,))

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)


class GemmaMLP(nn.Module):
    """
    Gemma MLP with gated GELU activation.

    Architecture: gate_proj * gelu(up_proj) -> down_proj
    (Note: This is GEGLU-style but with the gate being after activation)
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.gelu_approx(self.gate_proj(x)) * self.up_proj(x))


class GemmaAttention(nn.Module):
    """
    Gemma attention with query/key normalization.

    Features:
    - RMSNorm on queries and keys before attention
    - Separate RoPE bases for sliding vs global layers
    - GQA (grouped query attention)
    """

    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads or config.num_attention_heads
        self.head_dim = config.head_dim
        self.n_rep = self.num_heads // self.num_kv_heads

        # Attention scale using query_pre_attn_scalar
        self.scale = config.query_pre_attn_scalar**-0.5

        # Projections (no bias for Gemma)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Query/key normalization (Gemma-specific)
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Determine if this is a sliding window layer
        self.is_sliding = config.is_sliding_layer(layer_idx)

        # RoPE with appropriate base frequency
        rope_base = config.rope_local_base_freq if self.is_sliding else config.rope_theta
        self.rope = nn.RoPE(dims=self.head_dim, base=rope_base, traditional=False)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
        batch_size, seq_len, _ = x.shape

        # Project Q, K, V
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        # Reshape to (batch, heads, seq, head_dim)
        queries = queries.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        values = values.transpose(0, 2, 1, 3)

        # Apply query/key normalization (Gemma-specific)
        queries = self.q_norm(queries)
        keys = self.k_norm(keys)

        # Apply RoPE
        if cache is not None:
            offset = cache[0].shape[2]
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)
            # Update cache
            keys = mx.concatenate([cache[0], keys], axis=2)
            values = mx.concatenate([cache[1], values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        new_cache = (keys, values)

        # Repeat KV heads for GQA
        if self.n_rep > 1:
            keys = mx.repeat(keys, self.n_rep, axis=1)
            values = mx.repeat(values, self.n_rep, axis=1)

        # Scaled dot-product attention
        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )

        # Reshape and project output
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        output = self.o_proj(output)

        return output, new_cache


@partial(mx.compile, shapeless=True)
def clip_residual(x: mx.array, y: mx.array) -> mx.array:
    """
    Add residual with float16 overflow protection.

    Gemma uses this to prevent overflow in float16 mode.
    """
    if x.dtype != mx.float16:
        return x + y
    bound = mx.finfo(mx.float16).max
    return mx.clip(x.astype(mx.float32) + y.astype(mx.float32), -bound, bound).astype(mx.float16)


class GemmaBlock(Block):
    """
    Gemma transformer block.

    Has 4 normalization layers per block:
    - input_layernorm (pre-attention)
    - post_attention_layernorm (after attention, before residual add)
    - pre_feedforward_layernorm (pre-FFN)
    - post_feedforward_layernorm (after FFN, before residual add)
    """

    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()

        self._hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # Attention
        self.self_attn = GemmaAttention(config, layer_idx)

        # MLP
        self.mlp = GemmaMLP(config.hidden_size, config.intermediate_size)

        # 4 normalization layers (Gemma-specific)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
        # Attention with pre/post norm and residual
        r, new_cache = self.self_attn(self.input_layernorm(x), mask, cache)
        h = clip_residual(x, self.post_attention_layernorm(r))

        # MLP with pre/post norm and residual
        r = self.mlp(self.pre_feedforward_layernorm(h))
        out = clip_residual(h, self.post_feedforward_layernorm(r))

        return BlockOutput(hidden_states=out, cache=new_cache)


class GemmaModel(Backbone):
    """
    Gemma 3 backbone (without LM head).

    Features:
    - Alternating sliding window / global attention
    - Embedding scaling by sqrt(hidden_size)
    """

    def __init__(self, config: GemmaConfig):
        super().__init__()

        self.config = config
        self._vocab_size = config.vocab_size
        self._hidden_size = config.hidden_size
        self._num_layers = config.num_hidden_layers
        self.sliding_window = config.sliding_window
        self.sliding_window_pattern = config.sliding_window_pattern

        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Transformer blocks
        self.layers = [GemmaBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)]

        # Final norm
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def _create_attention_mask(
        self,
        h: mx.array,
        cache: list | None,
        window_size: int | None = None,
    ) -> mx.array:
        """Create causal attention mask, optionally with sliding window."""
        _, seq_len, _ = h.shape

        # Create causal mask
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        mask = mask.astype(h.dtype)

        # Apply sliding window if specified
        if window_size is not None and seq_len > 1:
            # Create sliding window mask
            positions = mx.arange(seq_len)
            window_mask = mx.where(
                positions[:, None] - positions[None, :] >= window_size,
                float("-inf"),
                0.0,
            )
            mask = mask + window_mask.astype(h.dtype)

        return mask

    def __call__(
        self,
        input_ids: mx.array,
        cache: list[Any] | None = None,
        input_embeddings: mx.array | None = None,
        output_hidden_states: bool = False,
    ) -> BackboneOutput:
        """Forward pass."""
        # Get embeddings
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(input_ids)

        # Scale embeddings by sqrt(hidden_size) - Gemma specific
        h = h * mx.array(self.config.hidden_size**0.5, dtype=mx.bfloat16).astype(h.dtype)

        # Initialize cache if needed
        if cache is None:
            cache = [None] * len(self.layers)

        # Create masks for global and sliding window layers
        # Global layers get a reference cache for mask creation
        global_layer_idx = self.sliding_window_pattern - 1
        global_mask = self._create_attention_mask(
            h, [cache[global_layer_idx]] if cache[global_layer_idx] else None
        )

        if self.sliding_window_pattern > 1:
            sliding_mask = self._create_attention_mask(
                h,
                [cache[0]] if cache[0] else None,
                window_size=self.sliding_window,
            )
        else:
            sliding_mask = None

        # Track hidden states
        all_hidden_states = (h,) if output_hidden_states else None
        new_cache = []

        # Process layers
        for i, (layer, layer_cache) in enumerate(zip(self.layers, cache)):
            # Choose mask based on layer type
            is_global = self.config.is_global_layer(i)
            mask = global_mask if is_global else sliding_mask

            output = layer(h, mask=mask, cache=layer_cache)
            h = output.hidden_states
            new_cache.append(output.cache)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (h,)

        # Final norm
        h = self.norm(h)

        return BackboneOutput(
            last_hidden_state=h,
            hidden_states=all_hidden_states,
            cache=new_cache,
        )

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def set_input_embeddings(self, embeddings: nn.Module) -> None:
        self.embed_tokens = embeddings


@register_model(
    model_type="gemma3_text",
    architectures=["Gemma3ForCausalLM"],
)
class GemmaForCausalLM(Model):
    """
    Gemma 3 for causal language modeling.

    Complete model with backbone + LM head.
    Also serves as base for FunctionGemma.
    """

    def __init__(self, config: GemmaConfig):
        super().__init__()

        self._config = config
        self.tie_word_embeddings = False

        # Backbone
        self.model = GemmaModel(config)

        # LM head (typically not tied for Gemma)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    @property
    def config(self) -> GemmaConfig:
        return self._config

    @property
    def backbone(self) -> nn.Module:
        return self.model

    def __call__(
        self,
        input_ids: mx.array,
        cache: list[Any] | None = None,
        input_embeddings: mx.array | None = None,
        labels: mx.array | None = None,
        output_hidden_states: bool = False,
    ) -> ModelOutput:
        """Forward pass."""
        # Backbone
        backbone_output = self.model(
            input_ids=input_ids,
            cache=cache,
            input_embeddings=input_embeddings,
            output_hidden_states=output_hidden_states,
        )

        # LM head
        if self.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(backbone_output.last_hidden_state)
        else:
            logits = self.lm_head(backbone_output.last_hidden_state)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift for causal LM
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]

            # Cross-entropy loss
            loss = mx.mean(
                nn.losses.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.shape[-1]),
                    shift_labels.reshape(-1),
                )
            )

        return ModelOutput(
            loss=loss,
            logits=logits,
            hidden_states=backbone_output.hidden_states,
            cache=backbone_output.cache,
        )

    def sanitize(self, weights: dict) -> dict:
        """Handle weight loading quirks."""
        if "lm_head.weight" not in weights:
            self.tie_word_embeddings = True
        return weights

    @property
    def layers(self):
        """Access transformer layers for compatibility."""
        return self.model.layers

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
            input_ids: Prompt, shape (batch, prompt_len)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            stop_tokens: Tokens that stop generation

        Returns:
            Generated sequence
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
            if temperature != 1.0 and temperature > 0:
                logits = logits / temperature

            # Apply top-k
            if top_k is not None and top_k > 0:
                top_k_logits = mx.topk(logits, k=min(top_k, logits.shape[-1]))
                min_val = top_k_logits[:, -1:]
                logits = mx.where(logits < min_val, float("-inf"), logits)

            # Apply top-p (nucleus sampling)
            if top_p is not None and top_p < 1.0:
                sorted_logits = mx.sort(logits, axis=-1)[:, ::-1]
                sorted_probs = mx.softmax(sorted_logits, axis=-1)
                cumsum_probs = mx.cumsum(sorted_probs, axis=-1)

                # Find cutoff
                cutoff_idx = mx.sum(cumsum_probs < top_p, axis=-1, keepdims=True)
                cutoff_logit = mx.take_along_axis(sorted_logits, cutoff_idx, axis=-1)
                logits = mx.where(logits < cutoff_logit, float("-inf"), logits)

            # Sample
            if temperature == 0:
                next_token = mx.argmax(logits, axis=-1, keepdims=True)
            else:
                probs = mx.softmax(logits, axis=-1)
                next_token = mx.random.categorical(mx.log(probs + 1e-10))
                next_token = mx.expand_dims(next_token, axis=-1)

            # Append
            generated = mx.concatenate([generated, next_token], axis=1)

            # Check stop condition
            if any(int(next_token[0, 0]) == stop for stop in stop_tokens):
                break

            # Forward with cache
            output = self(next_token, cache=cache)
            cache = output.cache

        return generated

    @classmethod
    def from_config(cls, config: GemmaConfig) -> GemmaForCausalLM:
        """Create from config."""
        return cls(config)


# Alias for FunctionGemma
FunctionGemmaForCausalLM = GemmaForCausalLM
