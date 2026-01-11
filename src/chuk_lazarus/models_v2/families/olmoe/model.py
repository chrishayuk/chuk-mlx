"""
OLMoE model implementation.

Allen AI's Open Language Model with Mixture of Experts.
Based on Llama architecture with MoE FFN layers.
"""

from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from ...backbones.base import Backbone, BackboneOutput
from ...blocks.base import Block, BlockOutput
from ...components.embeddings import create_token_embedding
from ...components.normalization import RMSNorm
from ...core.registry import register_model
from ...heads import LMHead
from ...models.base import Model, ModelOutput
from .config import OLMoEConfig


class OLMoEAttention(nn.Module):
    """
    OLMoE Attention with QK normalization.

    OLMoE uses RMSNorm on queries and keys BEFORE reshape (on full Q/K vectors).
    """

    def __init__(self, config: OLMoEConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        # Projections
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        # QK normalization (OLMoE applies on full Q/K before reshape)
        # Q has shape (batch, seq, num_heads * head_dim) = (batch, seq, hidden_size)
        self.q_norm = RMSNorm(config.num_attention_heads * self.head_dim, eps=config.rms_norm_eps)
        # K has shape (batch, seq, num_kv_heads * head_dim)
        self.k_norm = RMSNorm(config.num_key_value_heads * self.head_dim, eps=config.rms_norm_eps)

        # RoPE
        self.rope = nn.RoPE(self.head_dim, base=config.rope_theta)

        self.scale = self.head_dim ** -0.5

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
        batch_size, seq_len, _ = x.shape

        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Apply QK normalization BEFORE reshape (OLMoE style)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Reshape to heads (after normalization)
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Apply RoPE
        if cache is not None:
            k_cache, v_cache = cache
            offset = k_cache.shape[2]
            q = self.rope(q, offset=offset)
            k = self.rope(k, offset=offset)
            k = mx.concatenate([k_cache, k], axis=2)
            v = mx.concatenate([v_cache, v], axis=2)
        else:
            q = self.rope(q)
            k = self.rope(k)

        new_cache = (k, v)

        # Repeat KV heads if needed
        if self.num_kv_heads < self.num_heads:
            repeats = self.num_heads // self.num_kv_heads
            k = mx.repeat(k, repeats, axis=1)
            v = mx.repeat(v, repeats, axis=1)

        # Compute attention
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale

        if mask is not None:
            scores = scores + mask

        weights = mx.softmax(scores, axis=-1)
        output = weights @ v

        # Reshape and project output
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        output = self.o_proj(output)

        return output, new_cache


class OLMoERouter(nn.Linear):
    """
    MoE Router for OLMoE.

    Uses standard softmax routing with top-k selection.
    Inherits from nn.Linear to match HF weight naming (gate.weight).
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        norm_topk_prob: bool = False,
    ):
        # Initialize as Linear: (hidden_size) -> (num_experts)
        super().__init__(hidden_size, num_experts, bias=False)

        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """
        Compute routing weights and indices.

        Args:
            x: Input tensor, shape (batch * seq, hidden_size)

        Returns:
            Tuple of:
            - weights: Routing weights, shape (batch * seq, k)
            - indices: Expert indices, shape (batch * seq, k)
        """
        # Compute router logits using inherited Linear weights
        router_logits = super().__call__(x)  # (batch * seq, num_experts)

        # Get top-k experts
        k = self.num_experts_per_tok
        indices = mx.argpartition(-router_logits, kth=k - 1, axis=-1)[..., :k]

        # Get logits for selected experts
        selected_logits = mx.take_along_axis(router_logits, indices, axis=-1)

        # Apply softmax to get weights
        if self.norm_topk_prob:
            # Normalize over selected experts only
            weights = mx.softmax(selected_logits, axis=-1)
        else:
            # Full softmax then select (more faithful to original)
            all_probs = mx.softmax(router_logits, axis=-1)
            weights = mx.take_along_axis(all_probs, indices, axis=-1)

        return weights, indices


class OLMoEExpert(nn.Module):
    """
    Single expert MLP (SwiGLU).
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass through SwiGLU expert."""
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class OLMoESparseMoEBlock(nn.Module):
    """
    OLMoE Sparse Mixture of Experts block.

    Uses token-level routing to select top-k experts per token.
    Each expert is a SwiGLU MLP.
    """

    def __init__(self, config: OLMoEConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok

        # Router (called 'gate' in HF OLMoE)
        self.gate = OLMoERouter(
            hidden_size=config.hidden_size,
            num_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            norm_topk_prob=config.norm_topk_prob,
        )

        # Experts (as list for HF weight compatibility)
        self.experts = [
            OLMoEExpert(config.hidden_size, config.intermediate_size)
            for _ in range(config.num_experts)
        ]

    @property
    def router(self):
        """Alias for introspection code compatibility."""
        return self.gate

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass through MoE.

        Args:
            x: Input tensor, shape (batch, seq_len, hidden_size)

        Returns:
            Output tensor, shape (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, hidden_size = x.shape

        # Flatten for routing
        x_flat = x.reshape(-1, hidden_size)  # (batch * seq, hidden)

        # Get routing weights and indices (use self.gate for HF compatibility)
        weights, indices = self.gate(x_flat)  # (batch * seq, k), (batch * seq, k)

        # Initialize output
        output = mx.zeros_like(x_flat)

        # Process each expert
        # Note: This is the simple implementation. For better performance,
        # we could use gather_mm like in Llama4MoE.
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = indices == expert_idx  # (batch * seq, k)

            # Get weight for this expert (sum across k in case same expert selected twice)
            expert_weights = mx.where(expert_mask, weights, 0.0).sum(axis=-1, keepdims=True)

            # Only process if any tokens go to this expert
            # Compute expert output (for all tokens, then mask)
            expert_out = self.experts[expert_idx](x_flat)

            # Add weighted expert output
            output = output + expert_out * expert_weights

        return output.reshape(batch_size, seq_len, hidden_size)


class OLMoEBlock(Block):
    """
    OLMoE transformer block.

    Standard pre-norm transformer with:
    - RMSNorm
    - OLMoE Attention (with QK normalization)
    - Sparse MoE FFN
    """

    def __init__(self, config: OLMoEConfig, layer_idx: int = 0):
        super().__init__()

        self._hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # Pre-attention norm
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        # Attention with QK normalization
        self.self_attn = OLMoEAttention(config)

        # Post-attention norm
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        # MoE FFN
        self.mlp = OLMoESparseMoEBlock(config)

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

        # MoE FFN with residual
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return BlockOutput(hidden_states=x, cache=new_cache)


class OLMoEModel(Backbone):
    """
    OLMoE backbone (without LM head).
    """

    def __init__(self, config: OLMoEConfig):
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

        # Transformer blocks with MoE
        self.layers = [
            OLMoEBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)
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


@register_model(
    model_type="olmoe",
    architectures=["OlmoeForCausalLM"],
)
class OLMoEForCausalLM(Model):
    """
    OLMoE for causal language modeling.
    """

    def __init__(self, config: OLMoEConfig):
        super().__init__()

        self._config = config

        # Backbone
        self.model = OLMoEModel(config)

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
    def config(self) -> OLMoEConfig:
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

    @classmethod
    def from_config(cls, config: OLMoEConfig) -> OLMoEForCausalLM:
        """Create from config."""
        return cls(config)

    @staticmethod
    def sanitize(weights: dict[str, mx.array], tie_word_embeddings: bool = False) -> dict[str, mx.array]:
        """
        Convert HuggingFace weight names to our format.

        Key conversions:
        - embed_tokens.weight -> embed_tokens.weight.weight (TokenEmbedding wraps nn.Embedding)
        - lm_head.weight -> lm_head.linear.weight (LMHead has nn.Linear inside)
        """
        sanitized = {}
        for key, value in weights.items():
            new_key = key

            # Handle tied embeddings - skip lm_head if tying
            if tie_word_embeddings and key == "lm_head.weight":
                continue

            # Convert embed_tokens.weight -> embed_tokens.weight.weight
            # Because our TokenEmbedding wraps nn.Embedding
            if key == "model.embed_tokens.weight":
                new_key = "model.embed_tokens.weight.weight"

            # Convert lm_head.weight -> lm_head.lm_head.weight
            # Because our LMHead has self.lm_head = nn.Linear (when not tied)
            if key == "lm_head.weight" and not tie_word_embeddings:
                new_key = "lm_head.lm_head.weight"

            sanitized[new_key] = value

        return sanitized
