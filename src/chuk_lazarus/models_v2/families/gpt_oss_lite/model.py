"""
GPT-OSS-Lite model implementation for Lazarus.

This model supports variable experts per layer with MXFP4 quantized weights.
The lite models have 4-16 experts per layer (vs 32 in original GPT-OSS).

Weight structure (from safetensors):
- model.layers.{i}.mlp.experts.gate_proj.{weight,scales,bias}
- model.layers.{i}.mlp.experts.up_proj.{weight,scales,bias}
- model.layers.{i}.mlp.experts.down_proj.{weight,scales,bias}
- model.layers.{i}.mlp.router.{weight,bias}
"""

from __future__ import annotations

from functools import partial
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from ...backbones.base import Backbone, BackboneOutput
from ...components.normalization import RMSNorm
from ...core.registry import register_model
from ...models.base import Model, ModelOutput
from .config import GptOssLiteConfig


# =============================================================================
# GPT-OSS Custom SwiGLU Activation
# =============================================================================


@partial(mx.compile, shapeless=True)
def _gpt_oss_swiglu(
    x_linear: mx.array, x_glu: mx.array, alpha: float = 1.702, limit: float = 7.0
) -> mx.array:
    """
    GPT-OSS custom SwiGLU activation.

    Key differences from standard SwiGLU:
    - Alpha scaling (1.702) on sigmoid
    - Asymmetric clamping (glu: upper only, linear: both)
    - +1 bias on linear path

    Formula: (x_glu * sigmoid(alpha * x_glu)) * (x_linear + 1)
    """
    x_glu = mx.clip(x_glu, a_min=None, a_max=limit)
    x_linear = mx.clip(x_linear, a_min=-limit, a_max=limit)
    glu_scaled = alpha * x_glu
    sig = mx.sigmoid(glu_scaled)
    out_glu = x_glu * sig
    return out_glu * (x_linear + 1)


# =============================================================================
# MXFP4 Quantized Expert Projections
# =============================================================================


class QuantizedExpertProj(nn.Module):
    """
    MXFP4 quantized projection for batched experts.

    Weights are stored as:
    - weight: (num_experts, out_features, in_features_packed) uint32
    - scales: (num_experts, out_features, num_groups) uint8
    - bias: (num_experts, out_features) bfloat16
    """

    def __init__(self, num_experts: int, in_features: int, out_features: int):
        super().__init__()
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features

        # MXFP4 packing: 8 values per uint32
        in_packed = in_features // 8
        num_groups = in_features // 32  # group_size=32

        # Placeholder weights - will be loaded from checkpoint
        self.weight = mx.zeros((num_experts, out_features, in_packed), dtype=mx.uint32)
        self.scales = mx.zeros((num_experts, out_features, num_groups), dtype=mx.uint8)
        self.bias = mx.zeros((num_experts, out_features), dtype=mx.bfloat16)

    def __call__(self, x: mx.array, expert_idx: int) -> mx.array:
        """Apply quantized matmul for a single expert."""
        out = mx.quantized_matmul(
            x,
            self.weight[expert_idx],
            scales=self.scales[expert_idx],
            biases=None,
            transpose=True,
            group_size=32,
            bits=4,
            mode="mxfp4",
        )
        return out + self.bias[expert_idx]


class GptOssLiteExperts(nn.Module):
    """
    Batched experts for GPT-OSS-Lite with separate gate/up/down projections.

    Each expert has:
    - gate_proj: hidden_size -> intermediate_size (for gating)
    - up_proj: hidden_size -> intermediate_size (for linear path)
    - down_proj: intermediate_size -> hidden_size (output)
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts

        # Separate projections (matches weight structure)
        self.gate_proj = QuantizedExpertProj(num_experts, hidden_size, intermediate_size)
        self.up_proj = QuantizedExpertProj(num_experts, hidden_size, intermediate_size)
        self.down_proj = QuantizedExpertProj(num_experts, intermediate_size, hidden_size)


class GptOssLiteRouter(nn.Module):
    """Router for top-k expert selection."""

    def __init__(self, hidden_size: int, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        self.weight = mx.zeros((num_experts, hidden_size), dtype=mx.bfloat16)
        self.bias = mx.zeros((num_experts,), dtype=mx.bfloat16)

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Compute routing weights and indices."""
        # x: (num_tokens, hidden_size)
        logits = x @ self.weight.T + self.bias  # (num_tokens, num_experts)
        return logits


class GptOssLiteMoE(nn.Module):
    """MoE layer with top-k routing and MXFP4 quantized experts."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int = 4,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_tok = min(num_experts_per_tok, num_experts)

        self.router = GptOssLiteRouter(hidden_size, num_experts)
        self.experts = GptOssLiteExperts(hidden_size, intermediate_size, num_experts)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass through MoE."""
        batch_size, seq_len, hidden_size = x.shape
        x_flat = x.reshape(-1, hidden_size)
        num_tokens = x_flat.shape[0]

        # Router logits
        logits = self.router(x_flat)  # (num_tokens, num_experts)

        # Top-k selection
        k = self.num_experts_per_tok
        if self.num_experts <= k:
            # Use all experts
            indices = mx.broadcast_to(
                mx.arange(self.num_experts)[None, :], (num_tokens, self.num_experts)
            )
            weights = mx.softmax(logits, axis=-1)
        else:
            # Select top-k
            partitioned = mx.argpartition(logits, kth=-k, axis=-1)
            indices = partitioned[:, -k:]
            topk_logits = mx.take_along_axis(logits, indices, axis=-1)
            weights = mx.softmax(topk_logits, axis=-1)

        # Expert computation
        output = mx.zeros((num_tokens, hidden_size), dtype=x.dtype)

        for tok_idx in range(num_tokens):
            token_x = x_flat[tok_idx : tok_idx + 1]

            for k_idx in range(indices.shape[1]):
                exp_idx = int(indices[tok_idx, k_idx])
                w = weights[tok_idx, k_idx]

                # Gate and up projections
                gate_out = self.experts.gate_proj(token_x, exp_idx)
                up_out = self.experts.up_proj(token_x, exp_idx)

                # Custom SwiGLU activation
                hidden_states = _gpt_oss_swiglu(up_out, gate_out)

                # Down projection
                expert_out = self.experts.down_proj(hidden_states, exp_idx)

                output = output.at[tok_idx].add(w * expert_out[0])

        return output.reshape(batch_size, seq_len, hidden_size)


# =============================================================================
# Attention
# =============================================================================


def _create_rope(head_dim: int, config: GptOssLiteConfig) -> nn.Module:
    """Create appropriate RoPE based on config."""
    rope_scaling = config.rope_scaling

    if rope_scaling and rope_scaling.get("type") == "yarn":
        # Use YarnRoPE from mlx_lm if available
        try:
            from mlx_lm.models.rope_utils import YarnRoPE

            return YarnRoPE(
                dims=head_dim,
                traditional=False,
                base=config.rope_theta,
                scaling_factor=rope_scaling.get("factor", 1.0),
                original_max_position_embeddings=rope_scaling.get(
                    "original_max_position_embeddings", 4096
                ),
                beta_fast=rope_scaling.get("beta_fast", 32.0),
                beta_slow=rope_scaling.get("beta_slow", 1.0),
            )
        except ImportError:
            # Fallback to basic RoPE with scale
            return nn.RoPE(
                dims=head_dim,
                traditional=False,
                base=config.rope_theta,
                scale=rope_scaling.get("factor", 1.0),
            )

    # Default RoPE
    return nn.RoPE(dims=head_dim, traditional=False, base=config.rope_theta)


class GptOssLiteAttention(nn.Module):
    """Attention with RoPE for GPT-OSS-Lite.

    Includes attention biases and sinks for compatibility with the original model.
    """

    def __init__(self, config: GptOssLiteConfig, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads

        # Infer head_dim from config or default
        self.head_dim = getattr(config, "head_dim", 64)

        # Q has more heads than K/V for GQA
        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim

        # GPT-OSS uses bias=True for attention projections
        # Biases are initialized to zero - will be overwritten if present in weights
        self.q_proj = nn.Linear(config.hidden_size, q_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, kv_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, kv_dim, bias=True)
        self.o_proj = nn.Linear(q_dim, config.hidden_size, bias=True)

        self.scale = self.head_dim**-0.5

        # Attention sinks for improved long-context attention
        # Initialized to zeros - will be overwritten if present in weights
        self.sinks = mx.zeros((self.num_heads,))

        # RoPE with YaRN scaling if configured
        self.rope = _create_rope(self.head_dim, config)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
        batch, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape: (B, L, num_heads, head_dim) -> (B, num_heads, L, head_dim)
        q = q.reshape(batch, seq_len, self.num_heads, self.head_dim).swapaxes(1, 2)
        k = k.reshape(batch, seq_len, self.num_kv_heads, self.head_dim).swapaxes(1, 2)
        v = v.reshape(batch, seq_len, self.num_kv_heads, self.head_dim).swapaxes(1, 2)

        # Apply RoPE
        if cache is not None:
            offset = cache[0].shape[2]
            q = self.rope(q, offset=offset)
            k = self.rope(k, offset=offset)
            k = mx.concatenate([cache[0], k], axis=2)
            v = mx.concatenate([cache[1], v], axis=2)
        else:
            q = self.rope(q)
            k = self.rope(k)

        new_cache = (k, v)

        # Use mx.fast.scaled_dot_product_attention for efficiency
        # It handles GQA automatically without explicit head repetition
        output = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask, sinks=self.sinks
        )

        output = output.swapaxes(1, 2).reshape(batch, seq_len, -1)
        return self.o_proj(output), new_cache


# =============================================================================
# Transformer Block
# =============================================================================


class GptOssLiteBlock(nn.Module):
    """Transformer block for GPT-OSS-Lite."""

    def __init__(self, config: GptOssLiteConfig, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = GptOssLiteAttention(config, layer_idx=layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # MoE with num_local_experts (uniform across layers for lite models)
        self.mlp = GptOssLiteMoE(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_experts=config.num_local_experts,
            num_experts_per_tok=config.num_experts_per_tok,
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
        # Pre-norm attention
        residual = x
        x = self.input_layernorm(x)
        x, new_cache = self.self_attn(x, mask=mask, cache=cache)
        x = residual + x

        # Post-norm MoE
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x, new_cache


# =============================================================================
# Full Model
# =============================================================================


class GptOssLiteModel(Backbone):
    """GPT-OSS-Lite backbone."""

    def __init__(self, config: GptOssLiteConfig):
        super().__init__()
        self.config = config
        self._vocab_size = config.vocab_size
        self._hidden_size = config.hidden_size
        self._num_layers = config.num_hidden_layers

        # Use direct nn.Embedding to match weight structure
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        self.layers = [
            GptOssLiteBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)
        ]
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
        batch_size, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)

        # Create causal mask
        # Use "causal" string for mx.fast.scaled_dot_product_attention efficiency
        if attention_mask is not None:
            mask = attention_mask
        elif seq_len > 1:
            mask = "causal"
        else:
            mask = None

        all_hidden_states = (hidden_states,) if output_hidden_states else None
        new_cache = []

        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache else None
            hidden_states, layer_new_cache = layer(hidden_states, mask=mask, cache=layer_cache)
            new_cache.append(layer_new_cache)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

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
    model_type="gpt_oss_lite",
    architectures=["GptOssLiteForCausalLM", "GPTOSSLiteForCausalLM"],
)
class GptOssLiteForCausalLM(Model):
    """GPT-OSS-Lite for causal language modeling."""

    def __init__(self, config: GptOssLiteConfig):
        super().__init__()
        self._config = config
        self.model = GptOssLiteModel(config)

        # Use direct nn.Linear to match weight structure
        if config.tie_word_embeddings:
            self.lm_head = None  # Will use embed_tokens.as_linear()
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    @property
    def config(self) -> GptOssLiteConfig:
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
        backbone_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            cache=cache,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = backbone_output.last_hidden_state

        # Compute logits
        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)
        else:
            # Tied embeddings - use transpose of embedding weights
            logits = self.model.embed_tokens.as_linear(hidden_states)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
            shift_labels = labels[:, 1:].reshape(-1)
            loss = nn.losses.cross_entropy(shift_logits, shift_labels)

        return ModelOutput(
            loss=loss,
            logits=logits,
            hidden_states=backbone_output.hidden_states,
            cache=backbone_output.cache,
        )

    @classmethod
    def from_config(cls, config: GptOssLiteConfig) -> GptOssLiteForCausalLM:
        return cls(config)

    @staticmethod
    def sanitize(
        weights: dict[str, mx.array],
        tie_word_embeddings: bool = False,
    ) -> dict[str, mx.array]:
        """
        Convert weights from safetensors format to model format.

        With direct nn.Embedding/nn.Linear, weights match directly:
        - model.embed_tokens.weight -> model.embed_tokens.weight
        - model.norm.weight -> model.norm.weight
        - lm_head.weight -> lm_head.weight
        """
        result = {}
        for key, value in weights.items():
            if key == "lm_head.weight" and tie_word_embeddings:
                continue
            result[key] = value
        return result

    def load_weights(self, weights: dict[str, mx.array]) -> tuple[int, list[str]]:
        """
        Load weights from a dictionary using manual attribute navigation.

        This is more reliable than model.update() for complex nested structures.

        Args:
            weights: Dictionary of weight tensors keyed by full path names

        Returns:
            Tuple of (loaded_count, failed_keys)
        """
        loaded = 0
        failed = []

        for key, value in weights.items():
            parts = key.split(".")
            try:
                obj = self
                for part in parts[:-1]:
                    if part.isdigit():
                        obj = obj[int(part)]
                    else:
                        obj = getattr(obj, part)
                setattr(obj, parts[-1], value)
                loaded += 1
            except (AttributeError, IndexError, TypeError) as e:
                failed.append(key)

        return loaded, failed
