"""
GPT-2 model implementation.

Uses the composable architecture from models_v2.
GPT-2 differs from Llama-style models in:
- Learned positional embeddings (not RoPE)
- Post-layer norm (not pre-norm)
- GELU activation (not SwiGLU)
- Combined QKV projection
- Bias in attention and MLP
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from ...backbones.base import Backbone, BackboneOutput
from ...blocks.base import Block, BlockOutput
from ...components.embeddings import create_token_embedding
from ...components.normalization import LayerNorm
from ...core.registry import register_model
from ...heads import LMHead
from ...models.base import Model, ModelOutput
from .config import GPT2Config


class GPT2Attention(nn.Module):
    """GPT-2 multi-head attention with combined QKV projection."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.hidden_size = config.hidden_size

        # Combined QKV projection (GPT-2 style)
        self.c_attn = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=True)
        self.c_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

        self.scale = self.head_dim**-0.5

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
        batch_size, seq_len, _ = x.shape

        # Combined QKV projection
        qkv = self.c_attn(x)
        q, k, v = mx.split(qkv, 3, axis=-1)

        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Handle KV cache
        if cache is not None:
            k_cache, v_cache = cache
            k = mx.concatenate([k_cache, k], axis=2)
            v = mx.concatenate([v_cache, v], axis=2)

        new_cache = (k, v)

        # Attention
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale

        if mask is not None:
            scores = scores + mask

        attn_weights = mx.softmax(scores, axis=-1)
        attn_output = attn_weights @ v

        # Reshape back
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_len, self.hidden_size
        )

        # Output projection
        return self.c_proj(attn_output), new_cache


class GPT2MLP(nn.Module):
    """GPT-2 MLP with GELU activation."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.c_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.c_fc(x)
        x = nn.gelu_approx(x)  # GPT-2 uses approximate GELU
        return self.c_proj(x)


class GPT2Block(Block):
    """
    GPT-2 transformer block.

    Uses post-layer norm (unlike Llama which uses pre-norm):
    x = x + attn(ln_1(x))
    x = x + mlp(ln_2(x))
    """

    def __init__(self, config: GPT2Config, layer_idx: int = 0):
        super().__init__()
        self._hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # Layer norms
        self.ln_1 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ln_2 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Attention and MLP
        self.attn = GPT2Attention(config)
        self.mlp = GPT2MLP(config)

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
        """Forward pass with pre-norm (GPT-2 uses pre-norm actually, not post)."""
        # Self-attention with residual
        residual = x
        x = self.ln_1(x)
        attn_out, new_cache = self.attn(x, mask=mask, cache=cache)
        x = residual + attn_out

        # MLP with residual
        residual = x
        x = self.ln_2(x)
        x = self.mlp(x)
        x = residual + x

        return BlockOutput(hidden_states=x, cache=new_cache)


class GPT2Model(Backbone):
    """
    GPT-2 backbone (without LM head).

    Embeddings + transformer blocks + final norm.
    """

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self._vocab_size = config.vocab_size
        self._hidden_size = config.hidden_size
        self._num_layers = config.num_hidden_layers

        # Token embeddings
        self.wte = create_token_embedding(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
        )

        # Learned position embeddings
        self.wpe = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # Transformer blocks
        self.h = [GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)]

        # Final layer norm
        self.ln_f = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

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

        # Determine position offset from cache
        if cache is not None and cache[0] is not None:
            past_length = cache[0][0].shape[2]
        else:
            past_length = 0

        # Position IDs
        position_ids = mx.arange(past_length, past_length + seq_len)
        position_ids = mx.broadcast_to(position_ids, (batch_size, seq_len))

        # Embeddings
        hidden_states = self.wte(input_ids) + self.wpe(position_ids)

        # Create causal mask
        if attention_mask is None:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len + past_length)
            mask = mask.astype(hidden_states.dtype)
            if past_length > 0:
                mask = mask[-seq_len:, :]
        else:
            mask = attention_mask

        # Track hidden states
        all_hidden_states = (hidden_states,) if output_hidden_states else None
        new_cache = []

        # Process layers
        for i, layer in enumerate(self.h):
            layer_cache = cache[i] if cache else None
            output = layer(hidden_states, mask=mask, cache=layer_cache)
            hidden_states = output.hidden_states
            new_cache.append(output.cache)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        # Final norm
        hidden_states = self.ln_f(hidden_states)

        return BackboneOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            cache=new_cache,
        )

    def get_input_embeddings(self) -> nn.Module:
        return self.wte

    def set_input_embeddings(self, embeddings: nn.Module) -> None:
        self.wte = embeddings


@register_model(
    model_type="gpt2",
    architectures=["GPT2LMHeadModel"],
)
class GPT2ForCausalLM(Model):
    """
    GPT-2 for causal language modeling.

    Complete model with backbone + LM head.
    """

    def __init__(self, config: GPT2Config):
        super().__init__()
        self._config = config

        # Backbone (GPT-2 calls it "transformer")
        self.transformer = GPT2Model(config)

        # LM head (tied to embeddings by default)
        if config.tie_word_embeddings:
            self.lm_head = LMHead(
                hidden_size=config.hidden_size,
                vocab_size=config.vocab_size,
                tied_embeddings=self.transformer.wte,
            )
        else:
            self.lm_head = LMHead(
                hidden_size=config.hidden_size,
                vocab_size=config.vocab_size,
            )

    @property
    def config(self) -> GPT2Config:
        return self._config

    @property
    def backbone(self) -> nn.Module:
        return self.transformer

    # Aliases for weight loading compatibility
    @property
    def model(self) -> GPT2Model:
        """Alias for transformer (HF compatibility)."""
        return self.transformer

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        labels: mx.array | None = None,
        cache: list[Any] | None = None,
        output_hidden_states: bool = False,
    ) -> ModelOutput:
        """Forward pass."""
        backbone_output = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            cache=cache,
            output_hidden_states=output_hidden_states,
        )

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
        repetition_penalty: float = 1.0,
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
            repetition_penalty: Penalty for repeating tokens
            stop_tokens: Tokens that stop generation

        Returns:
            Generated sequence, shape (batch, total_len)
        """
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

            # Sample next token
            if temperature == 0.0:
                next_token = mx.argmax(logits, axis=-1)
                next_token = mx.expand_dims(next_token, axis=-1)
            else:
                if temperature != 1.0:
                    logits = logits / temperature

                if top_k is not None and top_k > 0:
                    top_k_values = mx.topk(logits, k=min(top_k, logits.shape[-1]))
                    min_val = top_k_values[:, -1:]
                    logits = mx.where(logits < min_val, float("-inf"), logits)

                probs = mx.softmax(logits, axis=-1)
                next_token = mx.random.categorical(mx.log(probs + 1e-10))
                next_token = mx.expand_dims(next_token, axis=-1)

            mx.eval(next_token)
            generated_tokens.append(next_token)

            # Check stop condition
            next_token_val = int(next_token[0, 0])
            if next_token_val in stop_tokens_set:
                break

            # Forward with cache
            output = self(next_token, cache=cache)
            mx.eval(output.logits)
            cache = output.cache

        return mx.concatenate(generated_tokens, axis=1)

    @classmethod
    def from_config(cls, config: GPT2Config) -> GPT2ForCausalLM:
        """Create from config."""
        return cls(config)

    @staticmethod
    def sanitize(weights: dict[str, Any]) -> dict[str, Any]:
        """Sanitize weights for loading.

        Maps HuggingFace weight names to our format.
        GPT-2 uses Conv1D which stores weights as (in, out) instead of (out, in),
        so we need to transpose linear layer weights.
        """
        from .convert import _map_weight_name

        sanitized = {}
        for name, weight in weights.items():
            new_name = _map_weight_name(name)
            if new_name is not None:
                # GPT-2 Conv1D weights need transposition for nn.Linear
                # HF stores as (in_features, out_features), MLX expects (out_features, in_features)
                if ".weight" in name and weight.ndim == 2 and "wte" not in name and "wpe" not in name:
                    weight = weight.T
                sanitized[new_name] = weight
        return sanitized
