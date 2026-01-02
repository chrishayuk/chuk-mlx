"""
Qwen3 model implementation.

Very similar to Llama, but with:
- Bias on QKV projections
- QK normalization (RMSNorm on Q and K before attention)
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
from ...core.config import AttentionConfig, FFNConfig
from ...core.registry import register_model
from ...heads import LMHead
from ...models.base import Model, ModelOutput
from .config import Qwen3Config


class Qwen3Attention(nn.Module):
    """
    Qwen3 attention with QK normalization.

    Wraps GroupedQueryAttention and adds q_norm/k_norm.
    """

    def __init__(self, config: Qwen3Config, layer_idx: int = 0):
        super().__init__()

        head_dim = config.head_dim or (config.hidden_size // config.num_attention_heads)
        num_kv_heads = config.num_key_value_heads or config.num_attention_heads

        # Attention config
        attn_config = AttentionConfig(
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=num_kv_heads,
            hidden_size=config.hidden_size,
            head_dim=head_dim,
            attention_bias=getattr(config, "attention_bias", True),
        )

        # Core attention components (q_proj, k_proj, v_proj, o_proj)
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * head_dim,
            bias=attn_config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size, num_kv_heads * head_dim, bias=attn_config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, num_kv_heads * head_dim, bias=attn_config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * head_dim, config.hidden_size, bias=False
        )

        # QK normalization (Qwen3-specific)
        self.q_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)

        # Attention parameters
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

        # RoPE
        self.rope = nn.RoPE(
            dims=head_dim,
            traditional=False,
            base=config.rope_theta,
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
        """Forward pass with QK normalization."""
        batch, seq_len, _ = x.shape

        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (batch, seq_len, num_heads, head_dim)
        q = q.reshape(batch, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = v.reshape(batch, seq_len, self.num_kv_heads, self.head_dim)

        # Apply QK normalization (Qwen3-specific)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Transpose to (batch, num_heads, seq_len, head_dim) BEFORE RoPE
        # RoPE expects (..., seq_len, head_dim) format
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Apply RoPE
        if cache is not None:
            q = self.rope(q, offset=cache[0].shape[2])
            k = self.rope(k, offset=cache[0].shape[2])
        else:
            q = self.rope(q)
            k = self.rope(k)

        # Update cache (cache is in heads-first format)
        if cache is not None:
            k = mx.concatenate([cache[0], k], axis=2)
            v = mx.concatenate([cache[1], v], axis=2)
        new_cache = (k, v)

        # GQA: repeat K, V heads if needed
        if self.num_kv_heads < self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = mx.repeat(k, n_rep, axis=1)
            v = mx.repeat(v, n_rep, axis=1)

        # Compute attention
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        if mask is not None:
            scores = scores + mask
        weights = mx.softmax(scores, axis=-1)
        output = weights @ v

        # Reshape output back to (batch, seq_len, hidden)
        output = output.transpose(0, 2, 1, 3)
        output = output.reshape(batch, seq_len, -1)

        # Output projection
        output = self.o_proj(output)

        return output, new_cache


class Qwen3Block(Block):
    """
    Qwen3 transformer block.

    Standard pre-norm transformer with:
    - RMSNorm
    - GQA with QK normalization
    - SwiGLU FFN
    """

    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int = 0,
    ):
        super().__init__()

        self._hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # Pre-attention norm
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        # Qwen3 attention with QK normalization
        self.self_attn = Qwen3Attention(config, layer_idx=layer_idx)

        # Post-attention norm
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        # FFN (SwiGLU)
        ffn_config = FFNConfig(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
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
        """Forward pass."""
        # Self-attention with residual
        residual = x
        x = self.input_layernorm(x)
        x, new_cache = self.self_attn(x, mask=mask, cache=cache)
        x = residual + x

        # FFN with residual
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return BlockOutput(hidden_states=x, cache=new_cache)


class Qwen3Model(Backbone):
    """
    Qwen3 backbone (without LM head).
    """

    def __init__(self, config: Qwen3Config):
        super().__init__()

        self.config = config
        self._vocab_size = config.vocab_size
        self._hidden_size = config.hidden_size
        self._num_layers = config.num_hidden_layers

        # Token embeddings
        self.embed_tokens = create_token_embedding(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
        )

        # Transformer blocks
        self.layers = [Qwen3Block(config, layer_idx=i) for i in range(config.num_hidden_layers)]

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

        # Embeddings
        hidden_states = self.embed_tokens(input_ids)

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
    model_type="qwen3",
    architectures=["Qwen3ForCausalLM"],
)
class Qwen3ForCausalLM(Model):
    """
    Qwen3 for causal language modeling.
    """

    def __init__(self, config: Qwen3Config):
        super().__init__()

        self._config = config

        # Backbone
        self.model = Qwen3Model(config)

        # LM head (optionally tied)
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
    def config(self) -> Qwen3Config:
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
        """Forward pass."""
        backbone_output = self.model(
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
        """Generate text autoregressively."""
        stop_tokens_set = set(stop_tokens or [])

        output = self(input_ids)
        mx.eval(output.logits)
        cache = output.cache

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

            next_token_val = int(next_token[0, 0])
            if next_token_val in stop_tokens_set:
                break

            output = self(next_token, cache=cache)
            mx.eval(output.logits)
            cache = output.cache

        return mx.concatenate(generated_tokens, axis=1)

    @classmethod
    def from_config(cls, config: Qwen3Config) -> Qwen3ForCausalLM:
        """Create from config."""
        return cls(config)

    @staticmethod
    def sanitize(
        weights: dict[str, mx.array],
        tie_word_embeddings: bool = True,
    ) -> dict[str, mx.array]:
        """Convert HuggingFace weights to our format.

        Args:
            weights: Raw HuggingFace weights
            tie_word_embeddings: Whether embeddings are tied (skip lm_head weight)

        Returns:
            Sanitized weights for model.update()
        """
        result = {}
        for key, value in weights.items():
            # Map weight names
            new_key = key

            # Embedding mapping (TokenEmbedding wraps nn.Embedding)
            if key == "model.embed_tokens.weight":
                new_key = "model.embed_tokens.weight.weight"
            # lm_head for non-tied models
            elif key == "lm_head.weight":
                if tie_word_embeddings:
                    # Skip - will use embedding weights
                    continue
                new_key = "lm_head.lm_head.weight"

            result[new_key] = value
        return result
