"""
GPT-OSS model implementation.

GPT-OSS is OpenAI's open-source MoE model with:
- Alternating sliding window and full attention layers
- MoE FFN with 32 experts, 4 active per token
- SwiGLU activation
- YaRN RoPE scaling

The native model from OpenAI uses MXFP4 quantized experts with:
- Batched expert weights (all experts in single tensors)
- Fused gate+up projection (gate_up_proj instead of separate gate/up)
- Quantized format with *_blocks, *_scales, *_bias
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from ...backbones.base import Backbone, BackboneOutput
from ...components.embeddings import create_token_embedding
from ...components.ffn import SwiGLU
from ...components.normalization import RMSNorm
from ...core.config import FFNConfig
from ...core.registry import register_model
from ...heads import LMHead
from ...models.base import Model, ModelOutput
from .config import GptOssConfig

# =============================================================================
# GPT-OSS Native MoE Components
# =============================================================================


def _gpt_oss_swiglu(
    x_linear: mx.array, x_glu: mx.array, alpha: float = 1.702, limit: float = 7.0
) -> mx.array:
    """
    GPT-OSS custom SwiGLU activation.

    This is a non-standard SwiGLU variant used by OpenAI's GPT-OSS model:
    - Clamps x_glu on upper bound only (not lower)
    - Clamps x_linear on both sides
    - Uses scaled sigmoid instead of SiLU
    - Adds +1 bias to the linear path

    Note: Follows mlx-lm's exact implementation.

    Args:
        x_linear: Up projection output (the "linear" path)
        x_glu: Gate projection output (the "gated" path)
        alpha: Scaling factor for sigmoid (default 1.702)
        limit: Clipping limit (default 7.0)

    Returns:
        Activated output: (x_glu * sigmoid(alpha * x_glu)) * (x_linear + 1)
    """
    # Clamp x_glu on upper bound only (matching mlx-lm)
    x_glu = mx.clip(x_glu, a_min=None, a_max=limit)
    # Clamp x_linear on both sides
    x_linear = mx.clip(x_linear, a_min=-limit, a_max=limit)

    # Scaled sigmoid gate
    glu_scaled = alpha * x_glu
    sig = mx.sigmoid(glu_scaled)

    # Gate output: x_glu * sigmoid(alpha * x_glu)
    out_glu = x_glu * sig

    # Final: gate_out * (x_linear + 1)
    return out_glu * (x_linear + 1)


def _repack_mxfp4_weights(blocks: mx.array) -> mx.array:
    """
    Repack MXFP4 weights from OpenAI format to MLX format.

    Uses the same approach as mlx-lm: view(mx.uint32).flatten(-2)

    This reinterprets the uint8 bytes as uint32 (4 bytes at a time)
    and flattens the last two dimensions.

    Args:
        blocks: Weight blocks in OpenAI format (uint8)

    Returns:
        Repacked weights in MLX format (uint32)
    """
    return blocks.view(mx.uint32).flatten(-2)


class GptOssBatchedExperts(nn.Module):
    """
    Batched experts for GPT-OSS native format.

    OpenAI's native format stores all expert weights in batched tensors:
    - gate_up_proj_blocks: (num_experts, out_features, num_groups, 16) uint8
    - gate_up_proj_scales: (num_experts, out_features, num_groups) uint8
    - gate_up_proj_bias: (num_experts, out_features) bfloat16
    - down_proj_blocks/scales/bias: same pattern

    After repacking by sanitize, the blocks become:
    - gate_up_proj_blocks: (num_experts, out_features, num_groups * 4) uint32

    This class provides the structure that matches the weight names.
    During forward pass, it uses mx.quantized_matmul with mode='mxfp4'.
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

        # For MXFP4: group_size=32, bits=4
        # After repacking: blocks are (out_features, in_features/8) uint32
        group_size = 32

        # gate_up_proj: (hidden_size -> 2 * intermediate_size)
        gate_up_out = 2 * intermediate_size
        gate_up_in_packed = hidden_size // 8  # 8 values per uint32

        # down_proj: (intermediate_size -> hidden_size)
        down_out = hidden_size
        down_in_packed = intermediate_size // 8

        # Number of scale groups
        num_groups_gate_up = hidden_size // group_size
        num_groups_down = intermediate_size // group_size

        # Initialize with placeholder tensors - will be overwritten by sanitized weights
        # Blocks in MLX format (repacked)
        self.gate_up_proj_blocks = mx.zeros(
            (num_experts, gate_up_out, gate_up_in_packed), dtype=mx.uint32
        )
        self.gate_up_proj_scales = mx.zeros(
            (num_experts, gate_up_out, num_groups_gate_up), dtype=mx.uint8
        )
        self.gate_up_proj_bias = mx.zeros((num_experts, gate_up_out), dtype=mx.bfloat16)

        self.down_proj_blocks = mx.zeros((num_experts, down_out, down_in_packed), dtype=mx.uint32)
        self.down_proj_scales = mx.zeros((num_experts, down_out, num_groups_down), dtype=mx.uint8)
        self.down_proj_bias = mx.zeros((num_experts, down_out), dtype=mx.bfloat16)

    def __call__(self, x: mx.array, expert_indices: mx.array, expert_weights: mx.array) -> mx.array:
        """
        Apply experts to input tokens.

        Args:
            x: Input tensor (batch * seq_len, hidden_size)
            expert_indices: Selected expert indices per token (batch * seq_len, num_experts_per_tok)
            expert_weights: Routing weights per expert (batch * seq_len, num_experts_per_tok)

        Returns:
            Output tensor (batch * seq_len, hidden_size)
        """
        num_tokens = x.shape[0]

        # Initialize output
        output = mx.zeros((num_tokens, self.hidden_size), dtype=x.dtype)

        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find which tokens use this expert and with what weight
            # expert_mask: (num_tokens, num_experts_per_tok)
            expert_mask = expert_indices == expert_idx

            # Get the weight for this expert per token
            # Sum across the num_experts_per_tok dimension to get per-token weight
            token_weights = mx.sum(
                expert_weights * expert_mask.astype(expert_weights.dtype), axis=-1
            )

            # Skip if no tokens routed to this expert
            if not mx.any(token_weights > 0):
                continue

            # Get expert weights for this expert
            gate_up_blocks = self.gate_up_proj_blocks[expert_idx]
            gate_up_scales = self.gate_up_proj_scales[expert_idx]
            gate_up_bias = self.gate_up_proj_bias[expert_idx]

            down_blocks = self.down_proj_blocks[expert_idx]
            down_scales = self.down_proj_scales[expert_idx]
            down_bias = self.down_proj_bias[expert_idx]

            # Apply fused gate+up projection with MXFP4 dequantization
            # gate_up_out: (num_tokens, 2 * intermediate_size)
            # MXFP4 mode with group_size=32
            gate_up_out = mx.quantized_matmul(
                x,
                gate_up_blocks,
                scales=gate_up_scales,
                biases=None,
                transpose=True,
                group_size=32,
                bits=4,
                mode="mxfp4",
            )
            gate_up_out = gate_up_out + gate_up_bias

            # Split into gate and up (interleaved: gate at even indices, up at odd)
            # OpenAI stores them interleaved, not concatenated
            gate_out = gate_up_out[:, 0::2]  # Even indices
            up_out = gate_up_out[:, 1::2]  # Odd indices

            # GPT-OSS custom SwiGLU: (gate * sigmoid(alpha * gate)) * (up + 1)
            hidden = _gpt_oss_swiglu(up_out, gate_out)

            # Apply down projection with MXFP4 dequantization
            expert_out = mx.quantized_matmul(
                hidden,
                down_blocks,
                scales=down_scales,
                biases=None,
                transpose=True,
                group_size=32,
                bits=4,
                mode="mxfp4",
            )
            expert_out = expert_out + down_bias

            # Weight the output and accumulate
            output = output + expert_out * token_weights[:, None]

        return output


class GptOssRouter(nn.Module):
    """
    Router for GPT-OSS MoE.

    Uses standard (non-quantized) linear layer for routing.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

        # Router uses regular linear layer (not quantized per OpenAI config)
        self.weight = mx.zeros((num_experts, hidden_size))
        self.bias = mx.zeros((num_experts,))

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """
        Compute routing weights and indices.

        Uses the same approach as mlx-lm:
        1. Compute router logits
        2. Get top-k expert indices and their logits
        3. Apply softmax only to top-k logits (not all experts)

        Args:
            x: Input (batch, seq_len, hidden_size) or (batch * seq_len, hidden_size)

        Returns:
            weights: (batch * seq_len, num_experts_per_tok)
            indices: (batch * seq_len, num_experts_per_tok)
        """
        # Handle both 2D and 3D inputs
        if x.ndim == 3:
            batch_size, seq_len, hidden_size = x.shape
            x = x.reshape(-1, hidden_size)

        # Compute router logits
        logits = x @ self.weight.T + self.bias

        # Get top-k experts using argpartition (matches mlx-lm's mlx_topk)
        k = self.num_experts_per_tok
        partitioned_indices = mx.argpartition(logits, kth=-k, axis=-1)
        top_k_indices = partitioned_indices[..., -k:]
        top_k_logits = mx.take_along_axis(logits, top_k_indices, axis=-1)

        # Apply softmax only to top-k logits (critical difference from before!)
        expert_weights = mx.softmax(top_k_logits, axis=-1)

        return expert_weights, top_k_indices


class GptOssMoE(nn.Module):
    """
    GPT-OSS Mixture of Experts layer.

    Uses native OpenAI format with batched quantized experts.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

        # Router
        self.router = GptOssRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
        )

        # Batched experts
        self.experts = GptOssBatchedExperts(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass through MoE.

        Args:
            x: Input (batch, seq_len, hidden_size)

        Returns:
            Output (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, hidden_size = x.shape

        # Flatten for processing
        x_flat = x.reshape(-1, hidden_size)

        # Get routing
        weights, indices = self.router(x_flat)

        # Apply experts
        output = self.experts(x_flat, indices, weights)

        # Reshape back
        return output.reshape(batch_size, seq_len, hidden_size)


class GptOssAttention(nn.Module):
    """
    GPT-OSS attention with sliding window or full attention.

    Supports both sliding window attention (for local context) and
    full attention (for global context) based on layer configuration.
    Uses attention sinks (learned per-head biases) for improved attention.
    """

    def __init__(
        self,
        config: GptOssConfig,
        layer_idx: int = 0,
        attention_type: str = "full_attention",
    ):
        super().__init__()

        self.layer_idx = layer_idx
        self.attention_type = attention_type

        head_dim = config.head_dim or (config.hidden_size // config.num_attention_heads)
        num_kv_heads = config.num_key_value_heads or config.num_attention_heads

        # Attention sinks - learned per-head biases (loaded from weights)
        self.sinks = mx.zeros((config.num_attention_heads,))

        # QKV projections with bias
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            num_kv_heads * head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            num_kv_heads * head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * head_dim,
            config.hidden_size,
            bias=config.attention_bias,  # GPT-OSS uses bias for all projections
        )

        # Attention parameters
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.sliding_window = (
            config.sliding_window if attention_type == "sliding_attention" else None
        )

        # RoPE
        self.rope = nn.RoPE(
            dims=head_dim,
            traditional=False,
            base=config.rope_theta,
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
        """Forward pass with attention sinks.

        Uses mx.fast.scaled_dot_product_attention with sinks parameter
        for accurate attention computation.
        """
        batch, seq_len, _ = x.shape

        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (batch, seq_len, num_heads, head_dim)
        q = q.reshape(batch, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = v.reshape(batch, seq_len, self.num_kv_heads, self.head_dim)

        # Transpose to (batch, num_heads, seq_len, head_dim)
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

        # Update cache
        if cache is not None:
            k = mx.concatenate([cache[0], k], axis=2)
            v = mx.concatenate([cache[1], v], axis=2)
        new_cache = (k, v)

        # Use mx.fast.scaled_dot_product_attention with sinks
        # This handles GQA internally and supports attention sinks
        output = mx.fast.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=self.scale,
            mask=mask,
            sinks=self.sinks,
        )

        # Reshape back
        output = output.transpose(0, 2, 1, 3)
        output = output.reshape(batch, seq_len, -1)

        # Output projection
        output = self.o_proj(output)

        return output, new_cache


class GptOssBlock(nn.Module):
    """
    GPT-OSS transformer block with MoE.

    Pre-norm -> Attention -> Residual -> Pre-norm -> MoE -> Residual
    """

    def __init__(
        self,
        config: GptOssConfig,
        layer_idx: int = 0,
    ):
        super().__init__()

        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        # Get attention type for this layer
        attention_type = config.get_layer_type(layer_idx)

        # Pre-attention norm
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Attention
        self.self_attn = GptOssAttention(config, layer_idx=layer_idx, attention_type=attention_type)

        # Post-attention norm
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # MoE FFN (all layers in GPT-OSS use MoE)
        # Use native GptOssMoE that matches OpenAI's weight format
        if config.is_moe:
            self.mlp = GptOssMoE(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_experts=config.num_local_experts,
                num_experts_per_tok=config.num_experts_per_tok,
            )
        else:
            ffn_config = FFNConfig(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
            )
            self.mlp = SwiGLU(ffn_config)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
        """Forward pass."""
        # Self-attention with residual
        residual = x
        x = self.input_layernorm(x)
        x, new_cache = self.self_attn(x, mask=mask, cache=cache)
        x = residual + x

        # MoE with residual
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x, new_cache


class GptOssModel(Backbone):
    """
    GPT-OSS backbone (without LM head).
    """

    def __init__(self, config: GptOssConfig):
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
        self.layers = [GptOssBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)]

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

        # Create mask - use "causal" string for mx.fast.scaled_dot_product_attention
        # when seq_len > 1 and no cache. With cache or seq_len=1, use None.
        if attention_mask is not None:
            mask = attention_mask
        elif cache is None and seq_len > 1:
            # Use "causal" string which mx.fast.scaled_dot_product_attention understands
            mask = "causal"
        else:
            # Single token or using cache - no explicit mask needed
            mask = None

        # Track hidden states
        all_hidden_states = (hidden_states,) if output_hidden_states else None
        new_cache = []

        # Process layers
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache else None
            hidden_states, layer_new_cache = layer(hidden_states, mask=mask, cache=layer_cache)
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

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def set_input_embeddings(self, embeddings: nn.Module) -> None:
        self.embed_tokens = embeddings


@register_model(
    model_type="gpt_oss",
    architectures=["GptOssForCausalLM"],
)
class GptOssForCausalLM(Model):
    """
    GPT-OSS for causal language modeling.
    """

    def __init__(self, config: GptOssConfig):
        super().__init__()

        self._config = config

        # Backbone
        self.model = GptOssModel(config)

        # LM head (typically not tied for GPT-OSS)
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
    def config(self) -> GptOssConfig:
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
    def from_config(cls, config: GptOssConfig) -> GptOssForCausalLM:
        """Create from config."""
        return cls(config)

    @staticmethod
    def sanitize(
        weights: dict[str, mx.array],
        tie_word_embeddings: bool = False,
    ) -> dict[str, mx.array]:
        """Convert HuggingFace weights to our format.

        OpenAI's native MXFP4 format uses:
        - model.layers.X.mlp.experts.gate_up_proj_blocks: (experts, out, groups, 16) uint8
        - model.layers.X.mlp.experts.gate_up_proj_scales: (experts, out, groups) uint8
        - model.layers.X.mlp.experts.gate_up_proj_bias: (experts, out) bfloat16
        - model.layers.X.mlp.experts.down_proj_*: same pattern
        - model.layers.X.mlp.router.weight/bias
        - model.layers.X.self_attn.{q,k,v,o}_proj.weight/bias
        - model.layers.X.self_attn.sinks (attention sinks)

        We need to repack the MXFP4 blocks from uint8 to uint32 format for MLX.

        Args:
            weights: Raw HuggingFace weights
            tie_word_embeddings: Whether embeddings are tied

        Returns:
            Sanitized weights for model.update()
        """
        result = {}
        for key, value in weights.items():
            new_key = key
            new_value = value

            # Embedding mapping (TokenEmbedding wraps nn.Embedding)
            if key == "model.embed_tokens.weight":
                new_key = "model.embed_tokens.weight.weight"
            # lm_head for non-tied models
            elif key == "lm_head.weight":
                if tie_word_embeddings:
                    continue
                new_key = "lm_head.lm_head.weight"
            # Attention sinks - keep them (critical for attention computation)
            # They are loaded as-is without modification
            elif ".self_attn.sinks" in key:
                pass  # Keep the key and value unchanged
            # Repack MXFP4 expert blocks from OpenAI format to MLX format
            elif "_proj_blocks" in key and ".mlp.experts." in key:
                # Repack from (experts, out, groups, 16) uint8 to (experts, out, groups*4) uint32
                new_value = _repack_mxfp4_weights(value)
            # Router weights and other params stay as-is

            result[new_key] = new_value
        return result
